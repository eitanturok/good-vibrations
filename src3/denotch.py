import numpy as np

#***** notch out narrow-band spikes (e.g. 60/100 Hz mains hum) from a complex FFT *****

def find_notch_mask(freqs: np.ndarray, spike_freqs: list[float], half_width_hz: float) -> np.ndarray:
    """Boolean mask over freqs, True where freq falls within half_width_hz of any spike_freq."""
    mask = np.zeros_like(freqs, dtype=bool)
    for f in spike_freqs:
        mask |= np.abs(freqs - f) <= half_width_hz
    return mask

def interpolate_notch(fft: np.ndarray, freqs: np.ndarray, spike_freqs: list[float] = [60.0, 100.0], half_width_hz: float = 3.0) -> np.ndarray:
    """Linearly interpolate the complex FFT across each notch band, using the bins immediately
    outside the band as endpoints. Real and imaginary parts are interpolated independently so
    phase isn't forced to a spurious value. Leaves everything outside the notch bands untouched.

    fft: (B,L,F,2) complex. freqs: (F,) in Hz, same ordering as fft's F axis.
    """
    notch = find_notch_mask(freqs, spike_freqs, half_width_hz)
    if not notch.any(): return fft.copy()

    F = freqs.shape[0]
    real, imag = fft.real.astype(np.float64).copy(), fft.imag.astype(np.float64).copy()

    # find contiguous notch runs so each gets interpolated against its own pair of endpoint bins
    edges = np.flatnonzero(np.diff(np.concatenate(([0], notch.view(np.int8), [0]))))
    starts, ends = edges[0::2], edges[1::2]  # [start, end) index pairs, in bin space

    for start, end in zip(starts, ends):
        lo, hi = start - 1, end  # bin just before / just after the notch
        if lo < 0 or hi >= F:  # spike at spectrum edge: fall back to nearest valid neighbor (flat fill)
            src = hi if lo < 0 else lo
            real[..., start:end, :] = real[..., src:src + 1, :]
            imag[..., start:end, :] = imag[..., src:src + 1, :]
            continue
        # weight goes 0->1 across the band; broadcasts over all leading (B,L) and trailing (C) dims
        w = ((freqs[start:end] - freqs[lo]) / (freqs[hi] - freqs[lo])).reshape(-1, 1)
        real[..., start:end, :] = real[..., lo:lo + 1, :] * (1 - w) + real[..., hi:hi + 1, :] * w
        imag[..., start:end, :] = imag[..., lo:lo + 1, :] * (1 - w) + imag[..., hi:hi + 1, :] * w

    return (real + 1j * imag).astype(fft.dtype)
