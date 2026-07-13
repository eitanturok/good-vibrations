import threading, time
from datetime import datetime, timezone
import sys
from pathlib import Path

import modal
import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

# when running inside a Modal container, src is mounted at /src but this file
# is copied to /root by Modal's function loader — add /src so imports resolve
if Path("/src").exists() and str(Path("/src")) not in sys.path:
    sys.path.insert(0, "/src")

from utils.io_utils import save, append, symlink, Timing, load, modal_upload, modal_download, fix_symlinks, logger

MIN_FREQ, MAX_FREQ = 50, 1000

#***** 0 capture vibrations *****

def capture_vibrations(cam, speaker, play_audio_fxn, capture_n_frames_fxn, audio_dir:Path, sample_dir:Path, height:int, width:int, n_capture_seconds=3.1, fps:int=2500, verbose=1, do_save=True):
    sample_id = sample_dir.name
    n_frames = int(n_capture_seconds * fps)

    with Timing(f"[sample {sample_id}] record vibrations: ", enabled=verbose >= 2):
        t_start = time.perf_counter()
        play_audio_fxn(audio_dir / 'audio.wav', speaker, wait=False)
        logger.debug(f'[sample {sample_id}] capturing {n_frames} frames')
        try:
            raw_vibrations, _ = capture_n_frames_fxn(cam, n_frames, height, width) # *cam.get_im_size()[::-1]
            logger.debug(f'[sample {sample_id}] {raw_vibrations.shape=}=(frames, height, width)')
        finally:
            # always wait for audio to finish — even if capture throws — so the next
            # sample's play_audio call never overlaps with this one
            remaining = n_capture_seconds - (time.perf_counter() - t_start)
            if remaining > 0: time.sleep(remaining)

    return raw_vibrations

#***** 1 speckle shifts from speckle vibrations *****

def get_shifts(frame_recording:np.ndarray, rois: list[list[int]], batch_size: int, pclk_mode: str = "sequential") -> np.ndarray:
    from data.pclk import compute_shifts_for_roi, compute_shifts_for_all_rois_batched, compute_shifts_for_all_rois_batched_optimized
    if pclk_mode == "batched":
        crops = np.stack([frame_recording[:, y:y+h, x:x+w] for x, y, w, h in rois])  # (L, T, H, W)
        return compute_shifts_for_all_rois_batched(crops, batch_size)                  # (L, T, 2)
    if pclk_mode == "batched_optimized":
        crops = np.stack([frame_recording[:, y:y+h, x:x+w] for x, y, w, h in rois])  # (L, T, H, W)
        return compute_shifts_for_all_rois_batched_optimized(crops, batch_size)        # (L, T, 2)
    else:
        from tqdm import tqdm
        all_shifts = []
        for x, y, w, h in tqdm(rois):
            all_shifts.append(compute_shifts_for_roi(frame_recording[:, y:y+h, x:x+w], batch_size))
        return np.stack(all_shifts, axis=0)  # (L, T, 2)

#***** 2 clean speckle shifts *****

def get_clean_shifts(shifts: np.ndarray, fs: float, lowcut: float = MIN_FREQ, highcut: float = MAX_FREQ, filter_order: int = 5) -> np.ndarray:
    """Set all the frequencies outside of [lowcut, highcut] to 0 but still keep them in the array"""
    B, L, T, C = shifts.shape

    # butterworth filter
    if highcut >= fs / 2: raise ValueError(f"highcut ({highcut} Hz) must be below the Nyquist frequency ({fs / 2} Hz)")
    sos = butter(filter_order, [lowcut, highcut], fs=fs, btype="band", output="sos")
    flat = shifts.transpose(0, 1, 3, 2).reshape(B * L * C, T)   # (B*L*C, T)
    filtered = sosfiltfilt(sos, flat, axis=-1)
    out = filtered.reshape(B, L, C, T).transpose(0, 1, 3, 2)    # (B, L, T, C)

    # hann window filter
    window = np.hanning(T).astype(out.dtype, copy=False)
    out *= window[None, None, :, None]
    return out

#***** 3 fft from speckle shifts *****

def get_fft_shifts(shifts: np.ndarray, fs:float, min_freq:float=MIN_FREQ, max_freq:float=MAX_FREQ) -> tuple[np.ndarray, np.ndarray, int]:
    """After taking the fft, drop all the frequencies outside of [lowcut, highcut], i.e. crop out and physically remove them from the array. This changes the shape.

    Input:  (L, T, C)
    Output: (1, L, F, C) — batch dim added here so all downstream functions see (B, L, F, C)
    """
    n_samples = shifts.shape[2]
    full_fft = np.fft.rfft(shifts, axis=2).astype(np.complex64)
    full_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (full_freqs >= min_freq) & (full_freqs <= max_freq)
    return full_fft[:, :, mask, :], full_freqs[mask], n_samples

#***** 4 recover audio *****

def get_recovered_audio(fft: np.ndarray, n_samples:int, fs: float, audio_sample_rate:int=22050, min_freq: float = MIN_FREQ, max_freq: float = MAX_FREQ, laser_idx: int = 50, xy_idx: int = 0) -> np.ndarray:
    """Return 16-bit PCM audio reconstructed from a cropped FFT via IFFT.

    Reconstructs the full spectrum by zero-filling bins outside [min_freq, max_freq],
    then resamples from fs to audio_sample_rate.
    """
    assert fft.shape[0] == 1, f"Can only recover audio with batch size of 1 but got {fft.shape[0]}"
    full_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (full_freqs >= min_freq) & (full_freqs <= max_freq)
    spectrum = np.zeros(len(full_freqs), dtype=np.complex64)
    spectrum[mask] = fft[0, laser_idx, :, xy_idx]
    signal = np.fft.irfft(spectrum, n=n_samples)

    MAX_INT16_VAL = 32767 # greatest number representable in INT16
    audio = resample(signal, int(audio_sample_rate * len(signal) / fs))
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return (audio * MAX_INT16_VAL).astype(np.int16)

#***** 5 group steps and add timing, save to files ******

def _process_vibrations_local(sample_dir:Path, min_freq:int=MIN_FREQ, max_freq:int=MAX_FREQ, audio_sample_rate:int=22050, signal_mode:str='magnitude', normalize_mode:str='std-sample', patch_size:int=256, pclk_batch_size:int=256, pclk_mode:str='batched_optimized', verbose:int=1, do_save:bool=True):
    sample_id = sample_dir.name
    metadata = {k: v for d in load(sample_dir / 'metadata.jsonl') for k, v in d.items()}
    fps, rois = int(metadata['fps']), metadata['roi']

    # turn vibrations into shifts with pclk algorithm
    with Timing(f"[sample {sample_id}] pclk: ", enabled=verbose >= 2):
        raw_vibrations = load(sample_dir / 'vibration/00_raw_vibrations.npy')
        raw_shifts = get_shifts(raw_vibrations, rois, pclk_batch_size, pclk_mode)  # (L, T, 2)
        logger.debug(f'[sample {sample_id}] {raw_shifts.shape=}=(lasers, frames, x/y)')
        save(raw_shifts, sample_dir / 'vibration/01_raw_shifts.npy', do_save)
        append({"pclk": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

    # clean the shifts
    with Timing(f"[sample {sample_id}] clean shifts: ", enabled=verbose >= 2):
        clean_shifts = get_clean_shifts(raw_shifts[None], fps, min_freq, max_freq)  # (L,T,2) -> (B,L,T,2)
        logger.debug(f'[sample {sample_id}] {clean_shifts.shape=}=(batch, lasers, frames, x/y)')
        save(clean_shifts, sample_dir / 'vibration/02_clean_shifts.npy', do_save)
        append({"clean_shifts": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

    # fft the shifts
    with Timing(f"[sample {sample_id}] fft shifts: ", enabled=verbose >= 2):
        fft, freqs, n_samples = get_fft_shifts(clean_shifts, fps, min_freq, max_freq) # (B,L,T,2) -> (B,L,F,2), (F,) (,)
        logger.debug(f'[sample {sample_id}] {fft.shape=}=(batch, lasers, freq bins, x/y)\n[sample {sample_id}] {freqs.shape=}=(freq bins)\n[sample {sample_id}] {n_samples=}')
        save({'fft': fft, 'freqs': freqs, 'n_samples': n_samples}, sample_dir / 'vibration/03_fft.npz', do_save)
        append({"fft_shifts": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

    # recover audio from fft
    with Timing(f"[sample {sample_id}] recover audio: ", enabled=verbose >= 2):
        recovered_audio = get_recovered_audio(fft, n_samples, fps, audio_sample_rate, min_freq, max_freq)
        save((recovered_audio, audio_sample_rate), sample_dir / 'vibration/04_recovered_audio.wav', do_save)
        symlink(sample_dir / 'vibration/04_recovered_audio.wav', sample_dir / 'recovered_audio.wav', do_save)
        append({"recover_audio": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

    # update tracking
    append({"process_vibrations": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

def save_vibrations(raw_vibrations:np.ndarray, sample_dir:Path, audio_dir:Path, do_save:bool=True, verbose:int=1):
    sample_id = sample_dir.name

    # save raw vibrations RAM->DISK (~2.7 GB per sample)
    raw_vibration_path = sample_dir / 'vibration/00_raw_vibrations.npy'
    with Timing(f'[sample {sample_id}] save raw vibrations RAM->DISK::{raw_vibration_path}: ', enabled=verbose >= 2):
        save(raw_vibrations, raw_vibration_path, do_save)
        timestamp = datetime.now(timezone.utc).isoformat()
        append({"save_vibrations": timestamp}, sample_dir / "times.jsonl", do_save)
        append({"sample_id": sample_id, "time": timestamp}, audio_dir.parent / "samples.jsonl", do_save)

#****** 6 run on modal *****

app = modal.App("pclk")
volume = modal.Volume.from_name("samples", create_if_missing=True)
VOLUME_PATH = Path("/samples")

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .env({"PYTHONUNBUFFERED": "1"})   # flush all prints so they show immedietally in modal logs
    .pip_install("cupy-cuda12x", "numpy", "tqdm", "scipy", "matplotlib", "pillow", "ipython")
    .add_local_dir(Path(__file__).parent, remote_path="/src/data")
    .add_local_dir(Path(__file__).parents[2] / "utils", remote_path="/src/utils")
)

@app.function(
    gpu="A10G",
    image=cuda_image,
    timeout=60*10, # timeout after 10 minutes
    volumes={VOLUME_PATH: volume},
)
def _process_vibrations_modal(sample_dir_name: str, **kwargs):
    import sys
    sys.path.insert(0, "/src")
    volume.reload()
    from data.vibrate import _process_vibrations_local
    _process_vibrations_local(VOLUME_PATH / sample_dir_name, **kwargs)
    volume.commit()

VIBRATION_FILES = ["00_raw_vibrations.npy", "01_raw_shifts.npy", "02_clean_shifts.npy", "03_fft.npz", "04_recovered_audio.wav"]

def process_vibrations(sample_dir:Path, use_modal:bool=False, pclk_mode:str='batched_optimized', pclk_batch_size:int=256, do_save:bool=True, verbose:int=1):
    sample_id = sample_dir.name

    # run locally
    if not use_modal:
        with Timing(f'[sample {sample_id}] process vibrations locally: ', enabled=verbose >= 1):
            return _process_vibrations_local(sample_dir, pclk_mode=pclk_mode, pclk_batch_size=pclk_batch_size, do_save=do_save, verbose=verbose)

    # upload raw vibrations DISK->modal_volume
    with Timing(f'[sample {sample_id}] upload raw vibrations DISK->modal_volume: ', enabled=verbose >= 1):
        modal_upload(volume, sample_dir, verbose=verbose)
        upload_timestamp = datetime.now(timezone.utc).isoformat()

    # process raw vibrations remotely
    with Timing(f'[sample {sample_id}] process vibrations on modal: ', enabled=verbose >= 1):
        _process_vibrations_modal.remote(sample_dir.name, pclk_batch_size=pclk_batch_size, pclk_mode=pclk_mode, verbose=verbose)

    # download processed vibrations modal_volume->DISK
    with Timing(f'[sample {sample_id}] download processed vibrations modal_volume->DISK::{sample_dir}: ', enabled=verbose >= 1):
        for f in VIBRATION_FILES: modal_download(volume, f"{sample_dir.name}/vibration/{f}", sample_dir / f"vibration/{f}")
        fix_symlinks(sample_dir)
        append({"modal_upload": upload_timestamp}, sample_dir / "times.jsonl", do_save)
        append({"modal_download": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

def save_and_process_vibrations(raw_vibrations:np.ndarray, sample_dir:Path, audio_dir:Path, min_freq:int, max_freq:int, use_modal:bool=False, pclk_mode:str='batched_optimized', pclk_batch_size:int=256, do_save:bool=True, verbose:int=1):
    save_vibrations(raw_vibrations, sample_dir, audio_dir, do_save, verbose)
    # delete vibrations as soon as we finish saving it
    del raw_vibrations
    process_vibrations(sample_dir, use_modal, pclk_mode, pclk_batch_size, do_save, verbose)
