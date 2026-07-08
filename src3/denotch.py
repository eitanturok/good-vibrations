import numpy as np

#***** notch out narrow-band spikes (e.g. 60/100 Hz mains hum) from a complex FFT *****

def find_notch_mask(freqs: np.ndarray, spike_freqs: list[float], half_width_hz: float) -> np.ndarray:
    """Boolean mask over freqs, True where freq falls within half_width_hz of any spike_freq."""
    mask = np.zeros_like(freqs, dtype=bool)
    for f in spike_freqs:
        mask |= np.abs(freqs - f) <= half_width_hz
    return mask

def interpolate_notch(fft: np.ndarray, freqs: np.ndarray, spike_freqs: list[float] = [60.0, 100.0],
                       half_width_hz: float = 3.0, seed: int | None = 0) -> np.ndarray:
    """Remove narrow-band spikes (e.g. 60/100Hz mains hum) from a complex FFT by replacing each
    notch band (+/- half_width_hz around each spike_freq) with a smooth fill: MAGNITUDE is linearly
    interpolated between the two bins immediately outside the band (a scalar, always well-behaved),
    and each notch bin gets a phase drawn uniformly at random in [-pi, pi].

    Interpolating real/imag parts directly (a straight line through the complex plane between two
    near-random-phase noise bins) is NOT valid here: it can pass close to the origin in the middle
    of the band even when both endpoints have normal magnitude, collapsing the fill toward 0. Since
    the surrounding noise floor has no coherent phase relationship between bins, randomizing phase
    while interpolating only the (always-positive, well-behaved) magnitude reproduces what genuine
    floor bins look like.

    fft: (B,L,F,2) complex. freqs: (F,) in Hz, same ordering as fft's F axis.
    """
    notch = find_notch_mask(freqs, spike_freqs, half_width_hz)
    if not notch.any(): return fft.copy()

    F = freqs.shape[0]
    mag = np.abs(fft).astype(np.float64)
    out = fft.copy()
    rng = np.random.default_rng(seed)

    # find contiguous notch runs so each gets interpolated against its own pair of endpoint bins
    edges = np.flatnonzero(np.diff(np.concatenate(([0], notch.view(np.int8), [0]))))
    starts, ends = edges[0::2], edges[1::2]  # [start, end) index pairs, in bin space

    for start, end in zip(starts, ends):
        lo, hi = start - 1, end  # bin just before / just after the notch
        n = end - start
        shape = fft.shape[:-2] + (n,) + fft.shape[-1:]  # (...,n,C)
        if lo < 0 or hi >= F:  # spike at spectrum edge: fall back to nearest valid neighbor's magnitude
            src = hi if lo < 0 else lo
            fill_mag = np.broadcast_to(mag[..., src:src + 1, :], shape)
        else:
            w = ((freqs[start:end] - freqs[lo]) / (freqs[hi] - freqs[lo])).reshape(-1, 1)
            fill_mag = mag[..., lo:lo + 1, :] * (1 - w) + mag[..., hi:hi + 1, :] * w

        phase = rng.uniform(-np.pi, np.pi, size=shape)
        out[..., start:end, :] = (fill_mag * np.exp(1j * phase)).astype(fft.dtype)

    return out
