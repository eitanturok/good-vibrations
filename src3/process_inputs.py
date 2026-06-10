import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

MIN_FREQ, MAX_FREQ = 50, 1000

#***** 1 speckle shifts from speckle vibrations

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
    if signal_mode == "magnitude": return np.abs(x)
    if signal_mode == "complex": return np.concatenate([x.real, x.imag], axis=-1)
    if signal_mode == "mag_phase": return np.concatenate([np.abs(x), np.angle(x)], axis=-1)
    raise ValueError(f"Unknown signal mode: {signal_mode}")

def _normalize_fft(x: np.ndarray, normalize_mode: str, verbose:int=0) -> np.ndarray:
    if normalize_mode is None: return x
    reduce_axes = (1, 2, 3)  # reduce over L, F, C; keep B
    if normalize_mode == 'std-sample':
        std = np.maximum(x.std(axis=reduce_axes, keepdims=True), 1e-8)
        if verbose: print(f'Normalize {normalize_mode}\n{std.shape=}\n{std.squeeze()=}')
        return x / std
    if normalize_mode == 'z-sample':
        mean = x.mean(axis=reduce_axes, keepdims=True)
        std = np.maximum(x.std(axis=reduce_axes, keepdims=True), 1e-8)
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
