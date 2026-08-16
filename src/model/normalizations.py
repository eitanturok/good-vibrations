"""Candidate normalizations for the complex FFT, for the phase ablation grid.

Everything here operates on the raw complex spectrum `(L, F, C)` loaded from
`vibration/04_fft.npz` and returns a REAL feature array `(L, F, K)`, so the same
function can be plotted in notebook 70 and later dropped into
`dataset.extract_signal`.

Two independent axes, deliberately kept separate (they were confounded in the
first pass at this grid):

  * magnitude recipe  -- what gets subtracted in log space (stage A)
  * phase recipe      -- which phase gauge, if any, rides along (stage B)

Terminology, because two different "relative phase" quantities are easy to mix up:

    relative laser phase   theta_l(f) - theta_ref(f)      difference along LASER
    group delay            theta_l(f+df) - theta_l(f)     difference along FREQ

Both are gauge fixes, but they cancel different things. Any term that is constant
across lasers within a sample -- the chirp's own phase, the speaker transfer
function, trigger jitter, and the -pi/2 + half-sample ramp contributed by the
`cumsum` in pclk.py:120 -- cancels EXACTLY under laser referencing, whatever its
shape in f. Group delay only cancels terms linear in f, which is why the
linear-ramp fit in notebooks/57_phase_ramp_explainer.md could not remove speaker
group delay.
"""
import numpy as np

LOG_EPS = 1e-3   # matches dataset.LOG_EPS; |fft| bottoms out near 1e-9
_DIV_EPS = 1e-20  # guards the unit-modulus divide at dead bins

#***** 0 helpers *****

def _to_LFC(fft: np.ndarray) -> np.ndarray:
    """Accept (1,L,F,C) as saved on disk or (L,F,C), always return (L,F,C)."""
    if fft.ndim == 4:
        assert fft.shape[0] == 1, f"expected batch of 1, got {fft.shape[0]}"
        return fft[0]
    assert fft.ndim == 3, f"expected (L,F,C) or (1,L,F,C), got {fft.shape}"
    return fft

def log_mag(fft: np.ndarray) -> np.ndarray:
    """log|X|. The +LOG_EPS floor keeps dead bins at -6.9 instead of -20."""
    return np.log(np.abs(fft) + LOG_EPS)

def energy_mask(fft: np.ndarray, top_frac: float = 0.10) -> np.ndarray:
    """Boolean (F,) mask of the top `top_frac` bins by laser-mean magnitude.

    Phase outside the excited bins is close to uniform noise -- measured circular
    resultant across lasers falls from 0.98 on the top decile to 0.27 over all
    bins -- so every phase feature below is optionally restricted to this mask.
    """
    fft = _to_LFC(fft)
    mag = np.abs(fft).mean(axis=(0, 2))                  # (F,)
    return mag >= np.quantile(mag, 1.0 - top_frac)

def std_normalize(x: np.ndarray) -> np.ndarray:
    """Divide by the per-sample std, removing per-recording gain (laser power,
    surface reflectivity, speaker volume drift). Applied LAST, after any
    subtraction -- otherwise the std is inflated by the term being removed."""
    return x / max(float(x.std()), 1e-8)

#***** 1 phase gauges *****

def smoothed_cross_spectrum(fft: np.ndarray, width: int = 1, ref: int | str = 'mean') -> np.ndarray:
    """Laser-referenced cross-spectrum, smoothed over `width` frequency bins BEFORE
    the angle is taken.

    Rationale (from a peer session): K=1 here -- one chirp, one rFFT over the whole
    record -- so classical coherence is identically 1 and there is no averaging over
    realizations. Frequency smoothing is the only available variance reduction, and
    modal structure varies over tens of Hz against 0.77 Hz bins, so it should be
    nearly free.

    MEASURED: it is not free, and it does not help. rho vs COM (purple-cube, mean of
    4 speakers) falls monotonically with width:
        1 bin +0.428 | 3 +0.427 | 5 +0.411 | 11 +0.387 | 21 +0.381 | 41 +0.353
    So single-shot estimator noise is NOT what limits this feature, and nb57's
    -0.136 is not explained by variance. Kept because the negative result is worth
    keeping runnable; default width=1 is a plain cross-spectrum.
    """
    fft = _to_LFC(fft)
    z_ref = fft.sum(axis=0, keepdims=True) if ref == 'mean' else fft[int(ref):int(ref) + 1]
    C = fft * np.conj(z_ref)
    if width > 1:
        k = np.ones(width) / width
        C = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 1, C)
    return C / (np.abs(C) + _DIV_EPS)

def neighbor_pair_phase(fft: np.ndarray) -> np.ndarray:
    """Phase differences to right/down neighbours on the 10x10 grid, (180,F,C) complex.

    Rationale (peer session): no global reference at all, so none of the reference
    degeneracies apply -- a single laser can sit on a node, and the complex mean
    cancels exactly on anti-symmetric modes.

    MEASURED: rho +0.383 vs +0.428 for the global magnitude-weighted mean reference,
    consistently across 4 speakers. Locality costs more than the degeneracy it avoids.
    """
    fft = _to_LFC(fft)
    L, F, C = fft.shape
    g = fft.reshape(10, 10, F, C)
    right = (g[:, 1:] * np.conj(g[:, :-1])).reshape(-1, F, C)
    down = (g[1:, :] * np.conj(g[:-1, :])).reshape(-1, F, C)
    z = np.concatenate([right, down], axis=0)
    return z / (np.abs(z) + _DIV_EPS)

def relative_laser_phase(fft: np.ndarray, ref: int | str = 'mean') -> np.ndarray:
    """Unit-modulus exp(i*(theta_l - theta_ref)), shape (L,F,C), complex.

    z_l * conj(z_ref) adds theta_l and -theta_ref in the exponent; dividing by
    |z_l||z_ref| strips the magnitude prefactor, leaving a pure phasor. Doing it
    this way rather than np.angle(z) - np.angle(z_ref) keeps the result on the
    circle with no wrapping to repair.

    ref='mean' uses the magnitude-weighted mean phasor over lasers, which is a
    lower-variance gauge than any single laser (and avoids the failure mode where
    laser 0 happens to sit on a node and contributes only noise).
    """
    fft = _to_LFC(fft)
    if ref == 'mean':
        z_ref = fft.sum(axis=0, keepdims=True)           # magnitude-weighted mean phasor
    else:
        z_ref = fft[int(ref):int(ref) + 1]
    num = fft * np.conj(z_ref)
    return num / (np.abs(fft) * np.abs(z_ref) + _DIV_EPS)

def group_delay_phasor(fft: np.ndarray) -> np.ndarray:
    """exp(i*(theta(f+df) - theta(f))), shape (L,F-1,C), complex. Notebook 68's
    quantity, kept as the control arm."""
    fft = _to_LFC(fft)
    z = fft[:, 1:, :] * np.conj(fft[:, :-1, :])
    return z / (np.abs(z) + _DIV_EPS)

def delay_slope(fft: np.ndarray, df: float) -> float:
    """Per-sample delay slope b (rad/Hz) by magnitude-weighted WRAPPED differencing.

    Deliberately not np.unwrap + least squares (nb57's method): unwrap makes
    irreversible 2*pi decisions in low-magnitude bins that corrupt everything
    downstream. Wrapped differencing has no such failure -- each step stays inside
    +/-pi as long as tau < 1/(2*df) = 0.65 s here, far above any plausible jitter.

    MEASURED: the two methods genuinely disagree -- per-laser median slope 1.566
    (unwrap+LSQ) vs 2.063 (wrapped), median |difference| 0.557 rad/Hz. So unwrap
    fragility is real and is a plausible contributor to nb57's -0.136.

    A slope IS estimable here -- dphi has circular resultant 0.90 across frequency
    on the top decile -- but see the comment below: subtracting a per-laser slope
    makes across-laser agreement much worse (R 0.400 -> 0.102). Use this for
    diagnostics, not as a normalization.
    """
    fft = _to_LFC(fft)
    d = np.angle(fft[:, 1:, :] * np.conj(fft[:, :-1, :]))
    w = np.abs(fft[:, 1:, :]) * np.abs(fft[:, :-1, :])
    return float((d * w).sum() / w.sum() / df)

# MEASURED.  An earlier version of this comment claimed there was NO linear-in-f
# structure, based on std(dphi - mean) / std(dphi) = 1.00.  That test was invalid:
# arithmetic std on WRAPPED angles reports ~uniform (1.81 rad) for a distribution
# that is actually concentrated near +/-pi, and subtracting a constant from a
# circular variable cannot change a resultant length anyway.  Redone with circular
# (cos/sin) statistics on the top-decile energy bins, magnitude-weighted:
#
#   is dphi constant in f?  Weighted circular resultant of dphi ACROSS frequency,
#   against a phase-randomized null:
#       all bins   R = 0.867   (null 0.173)
#       top 10%    R = 0.901   (null 0.213)
#   dphi is strongly concentrated, i.e. very nearly constant in f.  There IS a large
#   linear-in-f (delay-like) component.  The opposite of the earlier claim.
#
#   but it is not laser-constant.  Across-laser agreement within one sample
#   (top decile, magnitude weighted, mean of 8 samples):
#       raw phase                            R = 0.400
#       after removing a per-laser ramp fit  R = 0.102     <- much WORSE
#       group delay (differenced along f)    R = 0.993     <- much better
#   So estimating and subtracting a per-laser delay destroys real signal, while
#   differencing cancels the ramp without ever estimating it.  This reproduces both
#   nb57's failure (-0.526 -> -0.136) and nb68's success (0.41 -> 0.76), and explains
#   them with one mechanism: the ramp is real and largely shared, so cancel it
#   structurally -- never fit it.
#
#   The fitted constant delay implies tau ~ -600 ms, consistent across all 8
#   speakers.  That is far too large for trigger jitter and is most likely the log
#   chirp's own sweep timing (1.0 s sweep, each frequency emitted at a known moment),
#   which is a genuine laser-shared ramp.
#
#   per-speaker vs per-sample.  Slope variance decomposition over 8 speakers x 12
#   purple-cube samples: within-speaker std 0.466, between-speaker std 0.326,
#   eta^2(speaker) = 0.297.  Treat this as provisional -- it was computed with the
#   same unweighted wrapped-angle slopes that the test above invalidated, and has
#   not been redone under circular statistics.

def as_cos_sin(phasor: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    """Complex (L,F,C) -> real (L,F,2C) as [cos, sin].

    Both components are kept on purpose. cos alone is even -- cos(+p) == cos(-p) --
    so it collapses exactly the antiphase distinction that a nodal line consists
    of, which is the mode-shape information this whole exercise is trying to keep.
    It also does not reduce noise; it just discards half the signal.

    `weight` (broadcastable to the phasor shape) scales the unit vectors, so
    low-energy bins with random phase contribute proportionally less.
    """
    if weight is not None:
        phasor = phasor * weight
    return np.concatenate([phasor.real, phasor.imag], axis=-1)

#***** 2 stage A: magnitude recipes *****
# Signature is uniform -- (fft, refs) -> (L,F,C) real -- so callers can treat them
# interchangeably. `refs` carries the precomputed log-domain references; both are
# subtractions because log turns the multiplicative chain Y = S(f)H(f) additive.

def mag_none(fft, refs=None):
    return log_mag(_to_LFC(fft))

def mag_sub_speaker(fft, refs):
    return log_mag(_to_LFC(fft)) - refs['speaker']

def mag_sub_emptybox(fft, refs):
    return log_mag(_to_LFC(fft)) - refs['empty_box']

def mag_sub_both(fft, refs):
    return log_mag(_to_LFC(fft)) - refs['speaker'] - refs['empty_box']

MAG_RECIPES = {
    'A0_logmag':            mag_none,
    'A1_sub_speaker':       mag_sub_speaker,
    'A2_sub_emptybox':      mag_sub_emptybox,
    'A3_sub_both':          mag_sub_both,
}

#***** 3 stage B: phase recipes *****
# Each returns real (L, F', K) phase channels only -- stage B features are built
# by concatenating one of these onto the winning stage-A magnitude.

def phase_none(fft, mask=None):
    fft = _to_LFC(fft)
    return np.zeros(fft.shape[:2] + (0,), dtype=np.float32)

def phase_group_delay(fft, mask=None):
    p = group_delay_phasor(fft)
    if mask is not None: p = p[:, mask[:-1], :]
    return as_cos_sin(p)

def phase_group_delay_weighted(fft, mask=None):
    fft = _to_LFC(fft)
    p = group_delay_phasor(fft)
    w = np.abs(fft[:, :-1, :])
    if mask is not None: p, w = p[:, mask[:-1], :], w[:, mask[:-1], :]
    return as_cos_sin(p, weight=w / (w.max() + _DIV_EPS))

def phase_relative_laser(fft, mask=None):
    p = relative_laser_phase(fft)
    if mask is not None: p = p[:, mask, :]
    return as_cos_sin(p)

def phase_relative_laser_weighted(fft, mask=None):
    fft = _to_LFC(fft)
    p = relative_laser_phase(fft)
    w = np.abs(fft)
    if mask is not None: p, w = p[:, mask, :], w[:, mask, :]
    return as_cos_sin(p, weight=w / (w.max() + _DIV_EPS))

PHASE_RECIPES = {
    'B0_none':                phase_none,
    'B1_group_delay':         phase_group_delay,
    'B1w_group_delay_w':      phase_group_delay_weighted,
    'B2_rel_laser':           phase_relative_laser,
    'B3_rel_laser_w':         phase_relative_laser_weighted,
}

#***** 4 combined features *****

def build_feature(fft, refs=None, mag='A0_logmag', phase='B0_none',
                  top_frac: float | None = 0.10, normalize: bool = True,
                  phase_weight: float = 1.0) -> np.ndarray:
    """Assemble one ablation arm: (L,F,C) magnitude ++ (L,F',K) phase.

    Only the magnitude block is std-normalized. The phase block is already
    bounded in [-1,1] by construction, and rescaling it would distort the
    circular geometry that makes cos/sin the right encoding in the first place.
    `phase_weight` sets the relative scale of the two blocks -- notebook 68 found
    the magnitude/phase mix matters monotonically, so it is exposed here.
    """
    fft = _to_LFC(fft)
    mask = energy_mask(fft, top_frac) if top_frac is not None else None

    m = MAG_RECIPES[mag](fft, refs)
    if normalize: m = std_normalize(m)

    p = PHASE_RECIPES[phase](fft, mask)
    if p.shape[-1] == 0:
        return m.astype(np.float32)

    # phase lives on the masked bin subset, so the two blocks are concatenated on
    # the channel axis only when their F agrees; otherwise return them separately
    # padded to the magnitude's F grid (zeros elsewhere carry no phase claim).
    if p.shape[1] != m.shape[1]:
        full = np.zeros(m.shape[:2] + (p.shape[-1],), dtype=np.float32)
        idx = np.flatnonzero(mask)[:p.shape[1]]
        full[:, idx, :] = p
        p = full
    return np.concatenate([m, phase_weight * p], axis=-1).astype(np.float32)

#***** 5 references *****

def compute_refs(ffts: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Log-domain references from a list of complex spectra (one group, e.g. all
    train samples for one speaker).

    Averaged AFTER the log-magnitude, never in the complex domain: trigger jitter
    gives every sample its own tau, so a complex mean averages exp(-2*pi*i*f*tau)
    over random tau and collapses toward zero instead of toward the common term.
    """
    return {'speaker': np.mean([log_mag(_to_LFC(f)) for f in ffts], axis=0)}
