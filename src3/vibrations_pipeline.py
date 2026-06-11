from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

from io_utils import save, load, append, symlink, Timing, copy

MIN_FREQ, MAX_FREQ = 50, 1000

#***** 0 capture vibrations *****


#***** 1 speckle shifts from speckle vibrations *****

def speckle_shifts(vibrations):
    # run pclk
    pass

#***** 2 clean speckle shifts *****

def clean_shifts(shifts: np.ndarray, fs: float, lowcut: float = MIN_FREQ, highcut: float = MAX_FREQ, filter_order: int = 5) -> np.ndarray:
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

def fft_shifts(shifts: np.ndarray, fs:float, min_freq:float=MIN_FREQ, max_freq:float=MAX_FREQ) -> tuple[np.ndarray, np.ndarray, int]:
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

def recover_audio(fft: np.ndarray, n_samples:int, fs: float, audio_sr:int=22050, laser_idx: int = 50, xy_idx: int = 0, min_freq: float = MIN_FREQ, max_freq: float = MAX_FREQ) -> np.ndarray:
    """Return 16-bit PCM audio reconstructed from a cropped FFT via IFFT.

    Reconstructs the full spectrum by zero-filling bins outside [min_freq, max_freq],
    then resamples from fs to audio_sr.
    """
    assert fft.shape[0] == 1, f"Can only recover audio with batch size of 1 but got {fft.shape[0]}"
    full_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (full_freqs >= min_freq) & (full_freqs <= max_freq)
    spectrum = np.zeros(len(full_freqs), dtype=np.complex64)
    spectrum[mask] = fft[0, laser_idx, :, xy_idx]
    signal = np.fft.irfft(spectrum, n=n_samples)

    MAX_INT16_VAL = 32767 # greatest number representable in INT16
    audio = resample(signal, int(audio_sr * len(signal) / fs))
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

def process_fft(fft: np.ndarray, signal_mode:str='magnitude', normalize_mode:str='std-sample', patch_size:int=256, verbose:int=0) -> np.ndarray:
    """Extract the signal from the FFT, normalize the FFT, and tokenize it (turn into patches like in ViTs)"""
    fft = _extract_signal(fft, signal_mode).astype(np.float32)   # (B,L,F_,C) -> (B,L,F,C)
    fft = _normalize_fft(fft, normalize_mode, verbose)           # (B,L,F,C) -> (B,L,F,C)
    return _tokenize(fft, patch_size)                            # (B,L,F,C) -> (B,L,P,PS,C)

#***** 6 define each stage of the pipeline ******

def capture_vibrations(cam, run_opt, speaker, play_audio_fxn, capture_n_frames_fxn, audio_file:Path, sample_dir:Path, output_dir:Path, n_capture_seconds:float=3.1, verbose:int=1, do_save:bool=True):
    sample_id, output_id = sample_dir.name, output_dir.name

    with Timing(f"[sample {sample_id}] record vibrations in "):
        play_audio_fxn(audio_file, speaker)
        n_frames = int(n_capture_seconds * run_opt['cam_params']['camera_FPS'])
        frame_recording, times = capture_n_frames_fxn(cam, n_frames, *cam.get_im_size()[::-1])
        if verbose >= 2: print(f'captured {n_frames} frames')

    # symlink audio to sample_dir
    symlink(audio_file, sample_dir / "audio.wav", do_save)

    # update tracking status
    time = datetime.now(timezone.utc).isoformat()
    append({"sample_id": sample_id, "output_id": output_id, "time": time}, audio_file.parent / "samples.jsonl", do_save)
    append({"capture_vibrations": time}, sample_dir / "times.jsonl", do_save)

    return frame_recording
