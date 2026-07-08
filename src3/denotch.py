import numpy as np

#***** notch out narrow-band spikes (e.g. 60/100 Hz mains hum) from a complex FFT *****

def find_notch_mask(freqs: np.ndarray, spike_freqs: list[float], half_width_hz: float) -> np.ndarray:
    """Boolean mask over freqs, True where freq falls within half_width_hz of any spike_freq."""
    mask = np.zeros_like(freqs, dtype=bool)
    for f in spike_freqs:
        mask |= np.abs(freqs - f) <= half_width_hz
    return mask

def _endpoint(real, imag, freqs, center_hz, side, offset_hz, window_hz, F):
    """Median real/imag over a small window centered offset_hz away from center_hz (side=-1 below, +1 above),
    used as a robust, leakage-clear interpolation endpoint (as opposed to the single bin adjacent to the notch,
    which can itself still be depressed/elevated by spectral leakage from the spike)."""
    target = center_hz + side * offset_hz
    in_window = np.abs(freqs - target) <= window_hz / 2
    idx = np.flatnonzero(in_window)
    if idx.size == 0:  # fall back to the nearest single bin if the window is empty (e.g. near spectrum edge)
        idx = np.array([np.argmin(np.abs(freqs - target))])
    idx = idx[(idx >= 0) & (idx < F)]
    return np.median(real[..., idx, :], axis=-2, keepdims=True), np.median(imag[..., idx, :], axis=-2, keepdims=True)

def interpolate_notch(fft: np.ndarray, freqs: np.ndarray, spike_freqs: list[float] = [60.0, 100.0],
                       half_width_hz: float = 1.0, endpoint_offset_hz: float = 5.0, endpoint_window_hz: float = 2.0) -> np.ndarray:
    """Linearly interpolate the complex FFT across each notch band. The band itself (what gets
    replaced) is +/- half_width_hz around each spike_freq. The interpolation endpoints are NOT the
    bins immediately adjacent to the band -- those can still be slightly depressed/elevated by
    spectral leakage from the spike -- but the median magnitude over a small window centered
    endpoint_offset_hz away on each side, where the floor is clean. Real and imaginary parts are
    interpolated independently so phase isn't forced to a spurious value. Leaves everything outside
    the notch bands untouched.

    fft: (B,L,F,2) complex. freqs: (F,) in Hz, same ordering as fft's F axis.
    """
    notch = find_notch_mask(freqs, spike_freqs, half_width_hz)
    if not notch.any(): return fft.copy()

    F = freqs.shape[0]
    real, imag = fft.real.astype(np.float64).copy(), fft.imag.astype(np.float64).copy()

    # find contiguous notch runs so each gets interpolated against its own pair of endpoints
    edges = np.flatnonzero(np.diff(np.concatenate(([0], notch.view(np.int8), [0]))))
    starts, ends = edges[0::2], edges[1::2]  # [start, end) index pairs, in bin space

    for start, end in zip(starts, ends):
        center_hz = (freqs[start] + freqs[end - 1]) / 2
        lo_val, hi_val = (_endpoint(real, imag, freqs, center_hz, -1, endpoint_offset_hz, endpoint_window_hz, F),
                          _endpoint(real, imag, freqs, center_hz, +1, endpoint_offset_hz, endpoint_window_hz, F))
        (lo_real, lo_imag), (hi_real, hi_imag) = lo_val, hi_val

        band_freqs = freqs[start:end]
        f_lo, f_hi = center_hz - endpoint_offset_hz, center_hz + endpoint_offset_hz
        w = ((band_freqs - f_lo) / (f_hi - f_lo)).clip(0, 1).reshape(-1, 1)
        real[..., start:end, :] = lo_real * (1 - w) + hi_real * w
        imag[..., start:end, :] = lo_imag * (1 - w) + hi_imag * w

    return (real + 1j * imag).astype(fft.dtype)
