import threading, time
from datetime import datetime, timezone
import sys
from pathlib import Path

import modal
import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

# when running inside a Modal container, src3 is mounted at /src3 but this file
# is copied to /root by Modal's function loader — add /src3 so imports resolve
if Path("/src3").exists() and str(Path("/src3")) not in sys.path:
    sys.path.insert(0, "/src3")

from io_utils import save, append, symlink, Timing, load, modal_upload, modal_download, fix_symlinks

MIN_FREQ, MAX_FREQ = 50, 1000

#***** 0 capture vibrations *****

# def capture_vibrations(cam, run_opt, speaker, play_audio_fxn, capture_n_frames_fxn, audio_dir:Path, sample_dir:Path, output_dir:Path, n_capture_seconds:float=3.1, verbose:int=1, do_save:bool=True):
#     sample_id, output_id = sample_dir.name, output_dir.name

#     # symlink audio to sample_dir
#     symlink(audio_dir, sample_dir / "audio.wav", do_save)
#     append({'audio_dir': sample_dir / "audio.wav"}, sample_dir / "metadata.jsonl", do_save)

#     # record vibrations
#     with Timing(f"[sample {sample_id}] record vibrations: ", enabled=verbose >= 2):
#         play_audio_fxn(audio_dir, speaker, wait=False)
#         n_frames = int(n_capture_seconds * run_opt['cam_params']['camera_FPS'])
#         raw_vibrations, _ = capture_n_frames_fxn(cam, n_frames, *cam.get_im_size()[::-1])
#         if verbose >= 2: print(f'[sample {sample_id}] captured {n_frames} frames')
#         append({"capture_vibrations": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

#     # save vibrations
#     with Timing(f"[sample {sample_id}] save vibrations: ", enabled=verbose >= 2):
#         save(raw_vibrations, sample_dir / 'inputs/00_raw_vibrations.npy', do_save)
#         timestamp = datetime.now(timezone.utc).isoformat()
#         append({"save_vibrations": timestamp}, sample_dir / "times.jsonl", do_save)

#     # update tracking status
#     append({"sample_id": sample_id, "output_id": output_id, "time": timestamp}, audio_dir.parent / "samples.jsonl", do_save)

#     return raw_vibrations

# def capture_vibrations_async(cam, run_opt, speaker, play_audio_fxn, capture_n_frames_fxn, audio_dir, sample_dir, n_capture_seconds=3.1, strict:bool=False, verbose=1, do_save=True):
#     """Like capture_vibrations but launches the numpy save in a background thread.

#     Returns (raw_vibrations, save_thread). Caller must join save_thread before
#     the experiment ends to guarantee the file is fully written.
#     """
#     # symlink the audio to the current sample_dir
#     sample_id = sample_dir.name
#     symlink(audio_dir, sample_dir / "audio.wav", do_save)
#     append([{'audio_dir': sample_dir / "audio.wav"}, {'speaker': speaker}], sample_dir / "metadata.jsonl", do_save)

#     # with Timing(f"[sample {sample_id}] record vibrations: ", enabled=verbose >= 2):
#     #     play_audio_fxn(audio_dir, speaker)
#     #     n_frames = int(n_capture_seconds * run_opt['cam_params']['camera_FPS'])
#     #     raw_vibrations, times = capture_n_frames_fxn(cam, n_frames, *cam.get_im_size()[::-1])
#     #     if verbose >= 2: print(f'[sample {sample_id}] captured {n_frames} frames')
#     #     append({"capture_vibrations": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

#     with Timing(f"[sample {sample_id}] record vibrations: ", enabled=verbose >= 2):
#         t_start = time.perf_counter()
#         play_audio_fxn(audio_dir, speaker, wait=False)
#         n_frames = int(n_capture_seconds * run_opt['cam_params']['camera_FPS'])
#         if verbose >= 2: print(f'[sample {sample_id}] capturing {n_frames} frames')
#         try:
#             raw_vibrations, _ = capture_n_frames_fxn(cam, n_frames, *cam.get_im_size()[::-1])
#             if verbose >= 2: print(f'[sample {sample_id}] {raw_vibrations.shape=}=(frames, height, width)')
#         finally:
#             # always wait for audio to finish — even if capture throws — so the next
#             # sample's play_audio call never overlaps with this one
#             remaining = n_capture_seconds - (time.perf_counter() - t_start)
#             if remaining > 0: time.sleep(remaining)

#     # launch numpy save in background so the main loop never has to wait for it
#     npy_path = sample_dir / 'inputs/00_raw_vibrations.npy'
#     save_thread = threading.Thread(target=save, args=(raw_vibrations, npy_path, do_save), daemon=True)
#     save_thread.start()

#     return raw_vibrations, save_thread

def capture_vibrations(cam, speaker, play_audio_fxn, capture_n_frames_fxn, audio_dir:Path, sample_dir:Path, height:int, width:int, n_capture_seconds=3.1, fps:int=2500, verbose=1, do_save=True):
    sample_id = sample_dir.name
    n_frames = int(n_capture_seconds * fps)

    with Timing(f"[sample {sample_id}] record vibrations: ", enabled=verbose >= 2):
        t_start = time.perf_counter()
        play_audio_fxn(audio_dir / 'audio.wav', speaker, wait=False)
        if verbose >= 2: print(f'[sample {sample_id}] capturing {n_frames} frames')
        try:
            raw_vibrations, _ = capture_n_frames_fxn(cam, n_frames, height, width) # *cam.get_im_size()[::-1]
            if verbose >= 2: print(f'[sample {sample_id}] {raw_vibrations.shape=}=(frames, height, width)')
        finally:
            # always wait for audio to finish — even if capture throws — so the next
            # sample's play_audio call never overlaps with this one
            remaining = n_capture_seconds - (time.perf_counter() - t_start)
            if remaining > 0: time.sleep(remaining)

    return raw_vibrations

#***** 1 speckle shifts from speckle vibrations *****

def get_shifts(frame_recording:np.ndarray, rois: list[list[int]], batch_size: int, pclk_mode: str = "sequential") -> np.ndarray:
    from pclk import compute_shifts_for_roi, compute_shifts_for_all_rois_batched, compute_shifts_for_all_rois_batched_optimized
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

#***** 5 process fft (extract signal, normalize signal, tokenize)

def _extract_signal(x: np.ndarray, signal_mode: str) -> np.ndarray:
    # Cast to complex128 before abs/angle: np.abs on complex64 loses precision for large values
    # because sqrt(re²+im²) is computed in float32; PyTorch promotes internally so we match it.
    if signal_mode == "magnitude": return np.abs(x.astype(np.complex128))
    if signal_mode == "complex": return np.concatenate([x.real, x.imag], axis=-1)
    if signal_mode == "mag_phase": return np.concatenate([np.abs(x.astype(np.complex128)), np.angle(x.astype(np.complex128))], axis=-1)
    raise ValueError(f"Unknown signal mode: {signal_mode}")

def _normalize_fft(x: np.ndarray, normalize_mode: str, verbose:int=0) -> np.ndarray:
    if normalize_mode is None: return x
    # Compute stats in float64 with ddof=1 to match PyTorch's std behavior
    x64 = x.astype(np.float64)
    if normalize_mode == 'std-sample':
        std = np.maximum(x64.std(axis=(1, 2, 3), ddof=1, keepdims=True), 1e-8).astype(np.float32)
        if verbose: print(f'Normalize {normalize_mode}\n{std.shape=}\n{std.squeeze()=}')
        return x / std
    if normalize_mode == 'z-sample':
        mean = x64.mean(axis=(1, 2, 3), keepdims=True).astype(np.float32)
        std = np.maximum(x64.std(axis=(1, 2, 3), ddof=1, keepdims=True), 1e-8).astype(np.float32)
        if verbose: print(f'Normalize {normalize_mode}\n{mean.shape=}\t{std.shape=}\n{mean.squeeze()=}\n{std.squeeze()=}')
        return (x - mean) / std
    raise ValueError(f"Unknown normalize mode: {normalize_mode}")

def _tokenize(x: np.ndarray, patch_size:int):
    # Note: unfold drops entries that do not fully fit into patch_size
    if patch_size <= 0: return x
    B, L, F, C = x.shape
    P = F // patch_size
    return x[:, :, :P * patch_size, :].reshape(B, L, P, patch_size, C)  # (B,L,P,PS,C)

def get_processed_fft(fft: np.ndarray, signal_mode:str, normalize_mode:str, patch_size:int, verbose:int=0) -> np.ndarray:
    """Extract the signal from the FFT, normalize the FFT, and tokenize it (turn into patches like in ViTs)"""
    fft = _extract_signal(fft, signal_mode).astype(np.float32)   # (B,L,F_,C) -> (B,L,F,C)
    fft = _normalize_fft(fft, normalize_mode, verbose)           # (B,L,F,C) -> (B,L,F,C)
    return _tokenize(fft, patch_size)                            # (B,L,F,C) -> (B,L,P,PS,C)

#***** 6 process vibrations ******

def process_vibrations_local(sample_dir:Path, min_freq:int=MIN_FREQ, max_freq:int=MAX_FREQ, audio_sample_rate:int=22050, signal_mode:str='magnitude', normalize_mode:str='std-sample', patch_size:int=256, pclk_batch_size:int=1024, pclk_mode:str='sequential', verbose:int=1, do_save:bool=True):
    sample_id = sample_dir.name
    metadata = {k: v for d in load(sample_dir / 'metadata.jsonl') for k, v in d.items()}
    fps, rois = int(metadata['fps']), metadata['roi']

    # turn vibrations into shifts with pclk algorithm
    with Timing(f"[sample {sample_id}] pclk: ", enabled=verbose >= 2):
        raw_vibrations = load(sample_dir / 'inputs/00_raw_vibrations.npy')
        raw_shifts = get_shifts(raw_vibrations, rois, pclk_batch_size, pclk_mode)  # (L, T, 2)
        if verbose >= 2: print(f'[sample {sample_id}] {raw_shifts.shape=}=(lasers, frames, x/y)')
        save(raw_shifts, sample_dir / 'inputs/01_raw_shifts.npy', do_save)
        append({"pclk": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

    # clean the shifts
    with Timing(f"[sample {sample_id}] clean shifts: ", enabled=verbose >= 2):
        clean_shifts = get_clean_shifts(raw_shifts[None], fps, min_freq, max_freq)  # (L,T,2) -> (B,L,T,2)
        if verbose >= 2: print(f'[sample {sample_id}] {clean_shifts.shape=}=(batch, lasers, frames, x/y)')
        save(clean_shifts, sample_dir / 'inputs/02_clean_shifts.npy', do_save)
        append({"clean_shifts": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

    # fft the shifts
    with Timing(f"[sample {sample_id}] fft shifts: ", enabled=verbose >= 2):
        fft, freqs, n_samples = get_fft_shifts(clean_shifts, fps, min_freq, max_freq) # (B,L,T,2) -> (B,L,F,2), (F,) (,)
        if verbose >= 2: print(f'[sample {sample_id}] {fft.shape=}=(batch, lasers, freq bins, x/y)\n[sample {sample_id}] {freqs.shape=}=(freq bins)\n[sample {sample_id}] {n_samples=}')
        save({'fft': fft, 'freqs': freqs, 'n_samples': n_samples}, sample_dir / 'inputs/03_fft_shifts.npz', do_save)
        append({"fft_shifts": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

    # recover audio from fft
    with Timing(f"[sample {sample_id}] recover audio: ", enabled=verbose >= 2):
        recovered_audio = get_recovered_audio(fft, n_samples, fps, audio_sample_rate, min_freq, max_freq)
        save((recovered_audio, audio_sample_rate), sample_dir / 'inputs/04_recovered_audio.wav', do_save)
        symlink(sample_dir / 'inputs/04_recovered_audio.wav', sample_dir / 'recovered_audio.wav', do_save)
        append({"recover_audio": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

    # process fft (extract signal, normalize, and tokenize fft)
    with Timing(f"[sample {sample_id}] process fft: ", enabled=verbose >= 2):
        processed_fft = get_processed_fft(fft, signal_mode, normalize_mode, patch_size) # (B,L,F,2) -> (B,L,P,PS,2)
        if verbose >= 2: print(f'[sample {sample_id}] {processed_fft.shape=}=(batch, lasers, num_patches, patch_size, x/y)')
        save(processed_fft, sample_dir / 'inputs/05_processed_fft.npy', do_save)
        symlink(sample_dir / 'inputs/05_processed_fft.npy', sample_dir / 'X.npy', do_save)
        append({"process_fft": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

    # update tracking
    append({"process_vibrations": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

app = modal.App("pclk")
volume = modal.Volume.from_name("samples", create_if_missing=True)
VOLUME_PATH = Path("/samples")

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .env({"PYTHONUNBUFFERED": "1"})   # flush all prints so they show immedietally in modal logs
    .pip_install("cupy-cuda12x", "numpy", "tqdm", "scipy", "matplotlib", "pillow", "ipython")
    .add_local_dir(Path(__file__).parent, remote_path="/src3")
)

@app.function(
    gpu="A10G",
    image=cuda_image,
    timeout=60*10, # timeout after 10 minutes
    volumes={VOLUME_PATH: volume},
)
def process_vibrations_modal(sample_dir_name: str, **kwargs):
    import sys
    sys.path.insert(0, "/src3")
    volume.reload()
    from vibrations_pipeline import process_vibrations_local
    process_vibrations_local(VOLUME_PATH / sample_dir_name, **kwargs)
    volume.commit()

VIBRATION_FILES = ["01_raw_shifts.npy", "02_clean_shifts.npy", "03_fft_shifts.npz", "04_recovered_audio.wav", "05_processed_fft.npy"]

def process_vibrations(sample_dir:Path, use_modal:bool=True, do_save:bool=True, verbose:int=1):
    sample_id = sample_dir.name

    # run locally
    if not use_modal:
        with Timing(f'[sample {sample_id}] process vibrations locally: ', enabled=verbose >= 1):
            return process_vibrations_local(sample_dir, do_save=do_save, verbose=verbose)

    # upload raw vibrations DISK->modal_volume
    with Timing(f'[sample {sample_id}] upload raw vibrations DISK->modal_volume: ', enabled=verbose >= 1):
        modal_upload(volume, sample_dir)
        upload_timestamp = datetime.now(timezone.utc).isoformat()

    # process raw vibrations remotely
    with Timing(f'[sample {sample_id}] process vibrations on modal: ', enabled=verbose >= 1):
        process_vibrations_modal.remote(sample_dir.name, pclk_batch_size=1024, pclk_mode='sequential', verbose=verbose)

    # download processed vibrations modal_volume->DISK
    with Timing(f'[sample {sample_id}] download processed vibrations modal_volume->DISK::{sample_dir}: ', enabled=verbose >= 1):
        for f in VIBRATION_FILES: modal_download(volume, f"{sample_dir.name}/inputs/{f}", sample_dir / f"inputs/{f}")
        fix_symlinks(sample_dir)
        append({"modal_upload": upload_timestamp}, sample_dir / "times.jsonl", do_save)
        append({"modal_download": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

#***** 7 define the different stages of capturing the vibrations

def save_vibrations(raw_vibrations:np.ndarray, sample_dir:Path, audio_dir:Path, do_save:bool=True, verbose:int=1):
    sample_id = sample_dir.name

    # save raw vibrations RAM->DISK
    raw_vibration_path = sample_dir / 'inputs/00_raw_vibrations.npy'
    with Timing(f'[sample {sample_id}] save raw vibrations RAM->DISK::{raw_vibration_path}: ', enabled=verbose >= 1):
        save(raw_vibrations, raw_vibration_path, do_save)
        timestamp = datetime.now(timezone.utc).isoformat()
        append({"save_vibrations": timestamp}, sample_dir / "times.jsonl", do_save)
        append({"sample_id": sample_id, "time": timestamp}, audio_dir.parent / "samples.jsonl", do_save)

# run this together async
def save_and_process_vibrations(raw_vibrations:np.ndarray, sample_dir:Path, audio_dir:Path, min_freq:int, max_freq:int, use_modal:bool=True, do_save:bool=True, verbose:int=1):
    save_vibrations(raw_vibrations, sample_dir, audio_dir, do_save, verbose)
    del raw_vibrations
    process_vibrations(sample_dir, use_modal, do_save, verbose)