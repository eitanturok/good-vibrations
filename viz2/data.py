"""Loading and numpy. Everything is derived from the experiment dir -- nothing hardcoded.

Sample ids stay strings end to end; viz2 never builds a cross-sample array, so unlike viz/
there is no row space and no id->row conversion to get wrong.
"""

import json
import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, resample

FFT = "vibration/04_ffts.npz"
SHIFTS = {"clean": "vibration/03_clean_shifts.npy", "raw": "vibration/02_raw_shifts.npy"}
PHOTOS = ["image/02_cropped_overhead.png", "image/01_cropped.png"]
MASKS = ["image/03_smask.npy", "image/02_smask.npy"]

DIRS: dict[str, Path] = {}   # sample id -> dir; the only id->path map
META: dict[str, dict] = {}
INFO: dict = {}


def _meta(d: Path) -> dict:
    """metadata.jsonl holds ONE KEY PER LINE, so lines must be merged."""
    m = {}
    for line in (d / "metadata.jsonl").read_text().splitlines():
        if line.strip():
            m.update(json.loads(line))
    return m


def com(v) -> list[float]:
    """[y, x] from either a JSON list or gastronorm's str(ndarray) '[603.1 901.2]'."""
    if isinstance(v, str):
        v = v.replace("[", " ").replace("]", " ").replace(",", " ").split()
    try:
        f = np.asarray(v, dtype=float).reshape(-1)
        return [float(f[0]), float(f[1])]
    except Exception:
        return [-1.0, -1.0]


def _first(d: Path, names):
    return next((n for n in names if (d / n).exists()), None)


def init(exp: Path) -> int:
    """Scan once. Keep only samples that really have an FFT -- this is what drops
    gastronorm's 000009 (images but no vibration data), with no special case."""
    exp = Path(exp)
    for d in sorted((exp / "samples").iterdir()):
        if (d / FFT).exists():
            DIRS[d.name] = d
    if not DIRS:
        raise SystemExit(f"no samples with {FFT} under {exp}/samples")

    for sid, d in DIRS.items():
        m = _meta(d)
        META[sid] = {
            "id": sid,
            "pos": int(m.get("position_id") or 0),      # int here, string on exp-25
            "spk": int(m.get("speaker") or 0),
            "layout": m.get("layout") or "",
            "n": int(m.get("n_objects") or 0),
            "empty": bool(m.get("is_empty_box")),
            "com": com(m.get("avg_com")),
        }

    d0 = DIRS[next(iter(DIRS))]
    m0 = _meta(d0)
    z = np.load(d0 / FFT)
    n_lasers = z["fft"].shape[1]
    rows = m0.get("n_rows") or m0.get("n_laser_rows") or int(round(n_lasers ** 0.5))
    INFO.update(
        rows=int(rows), cols=int(n_lasers // int(rows)), n_lasers=int(n_lasers),
        fps=float(m0.get("fps") or 2500), n_samples=int(z["n_samples"]),
        min_freq=float(m0.get("min_freq") or 50), max_freq=float(m0.get("max_freq") or 1000),
        photo=_first(d0, PHOTOS), mask=_first(d0, MASKS),
    )
    INFO["scale"] = _scales()
    return len(DIRS)


# Signed quantities vary ~5x between samples and log magnitude shifts by ~0.7 decades, so
# per-sample ranges silently rescale every axis as you browse -- identical curve heights
# would then mean different physical values. These global ranges make every plot comparable.
SCALE_N = 160          # measured: converged and stable across seeds by ~80; 160 costs ~1 s


def _scales(n=SCALE_N):
    """One range per quantity, shared by every sample.

    Taken over a random subset (a full scan is not worth ~20 s at boot) and reduced with a
    percentile rather than min/max, so one freak recording cannot stretch every axis in
    the app. Signed quantities still reduce per sample then across; log magnitude pools
    first -- see the note below.
    """
    ids = sorted(DIRS)
    pick = ids if len(ids) <= n else random.Random(0).sample(ids, n)
    lm_all, sig, sh, md = [], [], [], []
    for sid in pick:
        z = chan(fft(sid)[0], "avg")
        # Log magnitude is POOLED across samples rather than reduced per sample first.
        # Taking p99 within a sample and then p90 across them clipped twice over: the top
        # of the range landed below the median sample's own peak, so the tallest resonance
        # ran off the frame on most recordings. One percentile over the pooled values
        # spends the budget where the data actually is.
        lm_all.append(logmag(np.abs(z)).ravel())
        sig.append(np.percentile(np.abs(z.real), 99))
        sig.append(np.percentile(np.abs(z.imag), 99))
        sh.append(np.percentile(np.abs(chan(shifts(sid, "clean"), "avg")), 99))
        md.append(np.abs(z).max())          # peak modal displacement of the sample
    fft.cache_clear()
    shifts.cache_clear()
    m, t = float(np.percentile(sig, 90)), float(np.percentile(sh, 90))
    mode = float(np.percentile(md, 90))
    # Both ends are the extreme actually observed, not a percentile. The tails here are
    # thin enough that trimming buys almost no vertical space while cutting real curve off
    # the frame: p99.9 dropped 0.1% of points yet cut the PEAK off 98 of 160 samples, and
    # p0.5 dropped 0.5% yet cut the TROUGH off all 160. Taking the true extremes clips
    # nothing and still spans only ~6.5 decades.
    lm = np.concatenate(lm_all)
    lm_lo = float(min(x.min() for x in lm_all))
    lm_hi = float(max(x.max() for x in lm_all))
    pad = 0.02 * (lm_hi - lm_lo)             # a hair of headroom so peaks are not flush
    lm_lo, lm_hi = lm_lo - pad, lm_hi + pad
    _mag_hi = float(10 ** np.percentile(lm, 99.9))
    return {
        "logmag": [lm_lo, lm_hi],
        # Linear magnitude, unlike the log axis, is dominated by the single largest peak:
        # 10**lm_hi would leave the typical curve a flat line along the bottom. A high
        # percentile of the pooled values keeps the usual shape readable. Magnitude cannot
        # go below zero, so the axis starts there -- nudged just under it so a curve
        # sitting on the floor is not drawn flush along the frame edge.
        "mag": [-0.02 * _mag_hi, _mag_hi],
        "phase": [-math.pi, math.pi],          # bounded already, so global by definition
        "cosphase": [-1.0, 1.0],
        "re": [-m, m], "im": [-m, m],
        "shifts": [-t, t],
        "mode": [0.0, mode],
    }


def d(sid: str) -> Path:
    """The whole path defense: an id not in DIRS never becomes a path."""
    if sid not in DIRS:
        raise KeyError(sid)
    return DIRS[sid]


# ***** signal *****

@lru_cache(maxsize=64)
def fft(sid):
    z = np.load(d(sid) / FFT)
    return z["fft"][0], z["freqs"].astype(float)      # (L,F,C) complex64, (F,)


@lru_cache(maxsize=16)
def shifts(sid, kind):
    a = np.load(d(sid) / SHIFTS[kind])
    a = a[0] if a.ndim == 4 else a
    return a.astype(np.float32)                        # (L,T,C)


def chan(a, ch):
    """x|y|avg -> drop the trailing channel axis."""
    return a.mean(-1) if ch == "avg" else a[..., 0 if ch == "x" else 1]


def pick(a, laser):
    """avg over lasers, or one laser."""
    return a.mean(0) if laser == "avg" else a[int(laser)]


def logmag(z):
    return np.log10(np.abs(z) + 1e-8)


def peaks(mag, freqs, k=12):
    """The k strongest resonances, as a residual against a smooth baseline
    (notebooks/53 cell 9). Ranked by prominence rather than thresholded: the spectrum is
    genuinely peaky (100+ pass any sane threshold), so the count is the honest knob."""
    db = 20 * np.log10(np.abs(mag) + 1e-8)
    resid = db - median_filter(db, size=201, mode="nearest")
    dist = max(1, int(5.0 / (freqs[1] - freqs[0])))
    idx, props = find_peaks(resid, prominence=3.0, distance=dist)
    best = idx[np.argsort(props["prominences"])[::-1][:k]]
    return sorted(int(i) for i in best)


def mode(sid, fi):
    """Mode shape at one bin: (U,V) real displacement on the laser grid.

    ALWAYS both components. The mode is a 2-D displacement field; zeroing one channel
    would collapse every arrow onto one axis, which is a rendering artefact, not physics.
    Channel choice belongs to the spectra, not here.

    Phase is normalized so taking the real part is a meaningful snapshot rather than an
    arbitrary point in the cycle (03_mark_visualize.ipynb cell 16).
    """
    f, _ = fft(sid)
    z = f[:, int(fi), :]                                  # (L, C)
    z = z * (np.conj(z[0, 0]) / (abs(z[0, 0]) + 1e-12))
    r, c = INFO["rows"], INFO["cols"]
    return z[:, 0].real.reshape(r, c), z[:, 1].real.reshape(r, c)


def envelope(y, n=800):
    """Min/max decimation. Plain subsampling of a noise trace visibly changes shape
    between redraws; this is what audio editors do."""
    if len(y) <= n:
        return y.tolist()
    k = len(y) // n
    b = y[: k * n].reshape(n, k)
    return np.stack([b.min(1), b.max(1)], 1).ravel().tolist()


def audio(sid, ch, laser):
    """Recovered audio by zero-filling outside the band and inverting.

    Reimplemented from src/data/vibrate.py:get_recovered_audio -- that module does
    `import modal` at top level, so it cannot be imported here.
    """
    f, _ = fft(sid)
    n, fs = INFO["n_samples"], INFO["fps"]
    full = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (full >= INFO["min_freq"]) & (full <= INFO["max_freq"])
    spec = np.zeros(len(full), dtype=np.complex64)
    spec[band] = pick(chan(f, ch if ch != "avg" else "x"), laser)
    sig = np.fft.irfft(spec, n=n)
    out = resample(sig, int(22050 * len(sig) / fs))
    return (out / (np.abs(out).max() + 1e-8) * 32767).astype(np.int16), 22050


def surface(U, V):
    """Height field whose gradient is (U,V): Frankot-Chellappa, as in
    figure_signals.ipynb's reconstruct_surface_from_gradients.

    The quiver shows the gradient of the mode; this is the mode itself. Solved in the
    Fourier domain -- one FFT pair, no iteration -- with a screened-Poisson term (lam)
    that keeps the low frequencies from running away, since the DC component of a height
    reconstructed from slopes alone is arbitrary.
    """
    R, C = U.shape
    # U is the x-slope and V the y-slope, but the grid is indexed [row, col] = [y, x], so
    # the ROW frequency pairs with V and the COLUMN frequency with U.
    ky = 2 * np.pi * np.fft.fftfreq(R)[:, None]
    kx = 2 * np.pi * np.fft.fftfreq(C)[None, :]
    # Screened-Poisson term. The notebook's smoothing_length is in metres against a
    # metre-spaced grid; here the spacing is one laser, so the equivalent is a fraction of
    # the field -- a whole field width, which damps only the very longest wavelength (the
    # one the boundary assumption gets wrong anyway) and leaves the mode shape intact.
    lam = 2 * np.pi / max(R, C)
    den = kx**2 + ky**2 + lam**2
    den[0, 0] = np.inf                         # the arbitrary constant offset
    Z = np.fft.ifft2((-1j * kx * np.fft.fft2(U) - 1j * ky * np.fft.fft2(V)) / den).real
    return Z - Z.mean()
