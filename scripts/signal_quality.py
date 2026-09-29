"""Training-free signal-quality benchmark for gastronorm input representations.

Every arm maps the raw complex fft (N, L=100, F=1235, C=2) to a feature vector per capture, and every
arm is scored by the SAME probes, so arms are comparable without an 80-minute training run each.

***** what "signal quality" means here *****

Fidelity to the chirp is not the target: a representation can reproduce the stimulus perfectly and
carry no position information (the speaker dominates the spectrum -- nb69: eta2(speaker)=0.90). What
downstream needs is POSITION information that survives to unseen positions. So the headline metrics
are held-out decodability of the downstream target:

  com_skill   purple-cube (560 = 70 positions x 8 speakers, 1 object). Kernel ridge -> COM (x,y),
              GroupKFold over position_id, so every test position is unseen. skill = 1 - err/chance,
              chance = predicting the train-mean COM. 0 = no information, 1 = perfect.
  mask_iou    the four 2-cube grid layouts (2160 captures). Kernel ridge -> the 32x32 mask the model
              is trained on, GroupKFold over (layout, position). Soft IoU (utils.metrics.soft_iou,
              the trained model's metric), with chance = the train-mean mask.
  knn_skill   5-NN (cosine) COM regression -- a nonparametric "fingerprint lookup" probe.
  loso_skill  leave-one-SPEAKER-out COM skill: positions are seen, the speaker is not.
  frac_pos_gt_noise / median_pos_snr_db
              fidelity diagnostic, per feature: variance across positions (purple-cube, within
              speaker) over variance across REPEATS of the identical empty-box scene (the only
              repeated captures in the dataset, 3-4 per speaker). Measurement-noise SNR in the
              units the arm actually delivers.

The probes are LINEAR (or cosine-kNN) in the features, so they are blind to anything the conv
encoder's inductive bias would exploit, and exactly invariant to fixed per-bin rotations (e.g. src
vs raw phase score identically). They are a proxy; scripts/boombox_input_features.sh trains the
same arms (the `trained:*` rows here are process_vibration-identical), so the proxy can be checked
against trained IoU before trusting it on the new arms.

    PYTHONPATH=src .venv/bin/python scripts/signal_quality.py                 # all arms
    PYTHONPATH=src .venv/bin/python scripts/signal_quality.py --arms logmag logmag-eb
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.fft import dct, idct
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from model.dataset import collect_samples, fft_path, process_vibration, source_phasor, target_name  # noqa: E402
from utils.metrics import soft_iou  # noqa: E402

DATA = ROOT / "experiments/31_07_2026_gastronorm_exp1"
EPS = 1e-3            # dataset.LOG_EPS
GRID = (10, 10)       # row-major laser grid (dataset.infer_laser_grid)
FS = 2500.0           # camera fps
N_TIME = 3250         # samples per capture -> rfft grid of 1626 bins, 0.769 Hz spacing
CHUNK = 64

#***** 1 data *****

def load(layouts):
    samples = [(d, m) for d, m in collect_samples(DATA / "samples", 0) if m["layout"] in layouts]
    meta = pd.DataFrame([dict(sample_id=m["sample_id"], layout=m["layout"], position_id=int(m["position_id"]),
                              speaker=int(m["speaker"]), dir=str(d)) for d, m in samples])
    Y = np.stack([np.load(fft_path(d))["fft"][0] for d, _ in samples])                         # (N,L,F,C) complex64
    com = np.stack([np.asarray(m["coms"][0][0], float) if m["n_objects"] == 1 else np.full(2, np.nan) for _, m in samples])
    mask = np.stack([np.load(d / f"image/{target_name(False, 32, 32)}.npy") for d, _ in samples]).astype(np.float32)
    return meta, torch.from_numpy(Y), com, mask

#***** 2 building blocks *****

def logmag(Y): return torch.log(Y.abs() + EPS)
def unit(z): return z / (z.abs() + 1e-20)
def std_norm(x): return x / x.flatten(1).std(dim=1).clamp_min(1e-8).view(-1, *[1] * (x.dim() - 1))
def cos_sin(z): z = unit(z); return torch.cat([z.real, z.imag], dim=-1)

def per_speaker_mean(x, spk):
    """{speaker: mean over its captures}, x (N,...)"""
    return {s: x[spk == s].mean(0) for s in np.unique(spk)}

def sub_ref(x, spk, ref):
    return x - torch.stack([ref[s] for s in spk])

class Ctx:
    """Everything an arm may reference: the empty-box captures (the only repeats), the source
    phasor, and a reference pool of captures DISJOINT from the probe set (for speaker means etc.)."""
    def __init__(self, Y_eb, spk_eb, Y_pool, spk_pool, freqs):
        self.Y_eb, self.spk_eb, self.Y_pool, self.spk_pool = Y_eb, spk_eb, Y_pool, spk_pool
        self.freqs = freqs
        self.S = None  # source phasor (F,), set in main
        self._cache = {}

    def memo(self, key, fn):
        if key not in self._cache: self._cache[key] = fn()
        return self._cache[key]

#***** 3 delay (trigger jitter) estimation -- GCC-PHAT pooled over lasers *****

K0 = None  # index of the first in-band bin on the full rfft grid, set in main

def gcc_delay(Y, T, upsample=16):
    """Per-capture delay tau of Y relative to template T (same L,F,C), in seconds.
    Each laser/axis is compared to ITS OWN template channel, so the box's own phase cancels and only
    the delay common to all channels survives; summing the PHAT-normalized cross-spectra over all 200
    channels then adds coherently. Y ~ T.exp(-i2pi f tau)  ->  irfft peaks at t=tau."""
    P = unit(Y * T.conj()).sum(dim=(1, 3))                                     # (N,F)
    n_full = N_TIME // 2 + 1
    full = torch.zeros(P.shape[0], n_full, dtype=torch.complex128)
    full[:, K0:K0 + P.shape[1]] = P.to(torch.complex128)
    n = N_TIME * upsample
    c = torch.fft.irfft(full, n=n)                                             # zero-pad in freq = sinc interp in time
    k = c.argmax(dim=1).double()
    k = torch.where(k > n / 2, k - n, k)
    return (k / (FS * upsample)).float()

def shift(Y, tau, freqs):
    """Undo a delay tau: multiply by exp(+i2pi f tau)."""
    ph = torch.exp(1j * 2 * np.pi * torch.as_tensor(freqs, dtype=torch.float64)[None, :] * tau.double()[:, None]).to(torch.complex64)
    return Y * ph[:, None, :, None]

def eb_templates(ctx, iters=3):
    """{speaker: jitter-aligned complex mean of its empty-box repeats}, built by iterative
    align-to-mean starting from the first repeat."""
    def build():
        T = {}
        for s in np.unique(ctx.spk_eb):
            Ys = ctx.Y_eb[ctx.spk_eb == s]
            t = Ys[:1].clone()
            for _ in range(iters):
                al = shift(Ys, gcc_delay(Ys, t.expand_as(Ys)), ctx.freqs)
                t = al.mean(0, keepdim=True)
            T[s] = t[0]
        return T
    return ctx.memo("eb_templates", build)

def align(Y, spk, ctx):
    T = eb_templates(ctx)
    Tt = torch.stack([T[s] for s in spk])
    return shift(Y, gcc_delay(Y, Tt), ctx.freqs)

#***** 4 impulse response gating *****

def gate(Y, spk, ctx, gate_ms, pre_ms=20.0):
    """Aligned transfer function H = Y.conj(S) -> impulse response -> keep [onset-pre, onset+gate]
    (Tukey taper) -> back to the band. Onset = peak of the speaker template's IR energy."""
    H = align(Y, spk, ctx) * ctx.S.conj()[None, None, :, None]
    n_full = N_TIME // 2 + 1
    def to_ir(Hb):
        full = torch.zeros(Hb.shape[0], Hb.shape[1], n_full, Hb.shape[3], dtype=torch.complex64)
        full[:, :, K0:K0 + Hb.shape[2]] = Hb
        return torch.fft.irfft(full, n=N_TIME, dim=2)                          # (N,L,T,C)
    T = eb_templates(ctx)
    onset = ctx.memo("ir_onset", lambda: {s: int((to_ir((T[s] * ctx.S.conj()[None, :, None])[None]) ** 2).sum(dim=(0, 1, 3)).argmax()) for s in T})
    pre, g = int(pre_ms * FS / 1000), int(gate_ms * FS / 1000)
    w = torch.zeros(N_TIME)
    taper = int(0.2 * (pre + g))
    win = torch.ones(pre + g)
    if taper > 0:
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0, np.pi, taper))
        win[:taper], win[-taper:] = ramp, ramp.flip(0)
    w[:pre + g] = win
    h = to_ir(H)
    out = torch.empty_like(H)
    for s in np.unique(spk):
        i = np.where(spk == s)[0]
        ws = torch.roll(w, onset[s] - pre)                                      # window starts at onset-pre (circular)
        out[i] = torch.fft.rfft(h[i] * ws[None, None, :, None], dim=2)[:, :, K0:K0 + H.shape[2]]
    return out

#***** 5 FDD: frequency-smoothed cross-spectral SVD across the 200 laser-axis channels *****

def fdd(Y, half=3, k=1):
    """Per capture, per frequency: the (2w+1)-bin window of 200-channel vectors, its top-k singular
    subspace (eigh of the tiny Gram), and y(f) projected onto it. Returns (denoised Y, singular
    values (N,F,3), u1 (N,L,F,C))."""
    N, L, F, C = Y.shape
    v = Y.permute(0, 2, 1, 3).reshape(N, F, L * C)                            # (N,F,200)
    pad = torch.nn.functional.pad(v.permute(0, 2, 1), (half, half), mode="replicate").permute(0, 2, 1)  # (N,F+2w,200)
    W = pad.unfold(1, 2 * half + 1, 1)                                         # (N,F,200,2w+1)
    G = W.conj().transpose(-1, -2) @ W                                         # (N,F,2w+1,2w+1)
    lam, V = torch.linalg.eigh(G)                                              # ascending
    lam, V = lam.flip(-1), V.flip(-1)
    U = (W @ V[..., :k]) / lam[..., :k].clamp_min(1e-30).sqrt().unsqueeze(-2)  # (N,F,200,k)
    den = (U @ (U.conj().transpose(-1, -2) @ v.unsqueeze(-1))).squeeze(-1)    # (N,F,200)
    to_LFC = lambda z: z.reshape(N, F, L, C).permute(0, 2, 1, 3)
    return to_LFC(den), lam[..., :3].clamp_min(1e-30), to_LFC(U[..., 0])

#***** 6 arms *****
# arm(Y, spk, ctx) -> (N, D) float32 features. Anything referenced to the empty box is computed by
# running the SAME transform on ctx.Y_eb, so reference and sample always share a pipeline.

def eb_ref_of(name, fn, ctx):
    return ctx.memo(("eb", name), lambda: per_speaker_mean(fn(ctx.Y_eb, ctx.spk_eb), ctx.spk_eb))

def mag_eb(fn, name):
    """std_norm(f(Y) - per-speaker empty-box f)."""
    def arm(Y, spk, ctx):
        return std_norm(sub_ref(fn(Y, spk, ctx), spk, eb_ref_of(name, lambda y, s: fn(y, s, ctx), ctx)))
    return arm

def trained(signal_mode, eb=False, phase_arm=None):
    """Byte-for-byte the features scripts/boombox_input_features.sh trains on (process_vibration)."""
    def arm(Y, spk, ctx):
        f = torch.as_tensor(ctx.freqs)
        eb_ref = ctx.memo("eb_logmag", lambda: per_speaker_mean(logmag(ctx.Y_eb).double(), ctx.spk_eb)) if eb else None
        pref = None
        if phase_arm == "src": pref = {s: ctx.S[None, None, :, None] for s in np.unique(spk)}
        if phase_arm == "src_eb": pref = ctx.memo("src_eb_ref", lambda: {s: ctx.S[None, None, :, None] * unit(unit(ctx.Y_eb[ctx.spk_eb == s] * ctx.S.conj()[None, None, :, None]).sum(0, keepdim=True)) for s in np.unique(ctx.spk_eb)})
        out = []
        for i in range(Y.shape[0]):
            s = spk[i]
            out.append(process_vibration(Y[i:i + 1], f, signal_mode, "std", 32, augment=0.0,
                                         empty_box_ref=eb_ref[s].float() if eb else None, phase_arm=phase_arm,
                                         phase_ref=pref[s] if pref else None).flatten(1))
        return torch.cat(out)
    return arm

def lm(Y, spk, ctx): return logmag(Y)

def xy_principal(Y, spk, ctx):
    """Project each laser's (x,y) onto its principal vibration axis, learned from the reference pool."""
    E = ctx.memo("xy_axes", lambda: torch.linalg.eigh(torch.einsum("nlfc,nlfd->lcd", ctx.Y_pool, ctx.Y_pool.conj()))[1][..., -1])  # (L,2)
    return logmag(torch.einsum("nlfc,lc->nlf", Y, E.conj()))[..., None]

def xy_power(Y, spk, ctx): return (0.5 * torch.log((Y.abs() ** 2).sum(-1, keepdim=True) + EPS ** 2))

def ceps(K):
    def f(Y, spk, ctx):
        c = dct(logmag(Y).numpy(), type=2, norm="ortho", axis=2)
        c[:, :, :K] = 0
        return torch.from_numpy(idct(c, type=2, norm="ortho", axis=2)).float()
    return f

def spk_mean_arm(Y, spk, ctx):
    ref = ctx.memo("pool_spk_mean", lambda: per_speaker_mean(logmag(ctx.Y_pool), ctx.spk_pool))
    return std_norm(sub_ref(logmag(Y), spk, ref))

def eb_spk_arm(Y, spk, ctx):
    eb = eb_ref_of("lm", lambda y, s: logmag(y), ctx)
    ref = ctx.memo("pool_spk_mean_eb", lambda: per_speaker_mean(sub_ref(logmag(ctx.Y_pool), ctx.spk_pool, eb), ctx.spk_pool))
    return std_norm(sub_ref(sub_ref(logmag(Y), spk, eb), spk, ref))

def logmag_eb(Y, spk, ctx): return mag_eb(lm, "lm")(Y, spk, ctx)

def with_phase(mag_arm, phase_fn):
    def arm(Y, spk, ctx): return torch.cat([mag_arm(Y, spk, ctx).flatten(1), phase_fn(Y, spk, ctx).flatten(1)], dim=1)
    return arm

def ph_tau_src(Y, spk, ctx): return cos_sin(align(Y, spk, ctx) * ctx.S.conj()[None, None, :, None])
def ph_tau_eb(Y, spk, ctx):
    T = eb_templates(ctx)
    return cos_sin(align(Y, spk, ctx) * torch.stack([T[s] for s in spk]).conj())
def ph_raw_src(Y, spk, ctx): return cos_sin(Y * ctx.S.conj()[None, None, :, None])

def fdd_den(half, k):
    return lambda Y, spk, ctx: logmag(fdd(Y, half, k)[0])

def fdd_sv(Y, spk, ctx): return torch.log(fdd(Y, 3, 1)[1])

def salsa(Y, spk, ctx):
    """Principal eigenvector of the frequency-smoothed CSD, phase referenced to its own channel sum
    (common phase -- incl. trigger jitter -- cancels exactly), as [cos, sin]."""
    u1 = fdd(Y, 3, 1)[2]
    ref = u1.sum(dim=(1, 3), keepdim=True)
    return cos_sin(u1 * ref.conj())

def curvature(Y, spk, ctx):
    x = sub_ref(logmag(Y), spk, eb_ref_of("lm", lambda y, s: logmag(y), ctx))
    N, L, F, C = x.shape
    g = x.reshape(N, *GRID, F, C)
    p = torch.nn.functional.pad(g.permute(0, 3, 4, 1, 2).reshape(N * F * C, 1, *GRID), (1, 1, 1, 1), mode="replicate")
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=x.dtype)[None, None]
    lap = torch.nn.functional.conv2d(p, k).reshape(N, F, C, *GRID).permute(0, 3, 4, 1, 2).reshape(N, L, F, C)
    return std_norm(lap)

def rapid(n_bands):
    """Signal difference coefficient 1 - pearson_f(logmag, empty-box logmag) per laser, axis, band."""
    def arm(Y, spk, ctx):
        x = logmag(Y)
        r = torch.stack([eb_ref_of("lm", lambda y, s: logmag(y), ctx)[s] for s in spk])
        out = []
        for xb, rb in zip(x.tensor_split(n_bands, dim=2), r.tensor_split(n_bands, dim=2)):
            xc, rc = xb - xb.mean(2, keepdim=True), rb - rb.mean(2, keepdim=True)
            out.append(1 - (xc * rc).sum(2) / (xc.norm(dim=2) * rc.norm(dim=2) + 1e-12))
        return torch.stack(out, -1)
    return arm

def chunked(fn, Y, spk, ctx):
    return torch.cat([fn(Y[i:i + CHUNK], spk[i:i + CHUNK], ctx) for i in range(0, Y.shape[0], CHUNK)])

def snr_weighted(mode, fn=lm, name="lm", keep=0.5, sub=True):
    """fn features (logmag by default) with each (laser,f,axis) scaled by its Wiener gain
    var_s/(var_s+var_n) -- var_n from empty-box repeats, var_s from the reference pool -- or
    hard-masked to the top `keep` fraction by that gain. The gain is computed on fn's own output, so
    a gated or reprojected signal gets its own SNR map. sub=False keeps the reference out of the
    features (for linear magnitude, where subtracting a reference is the wrong operation)."""
    def arm(Y, spk, ctx):
        ref = eb_ref_of(name, lambda y, s: fn(y, s, ctx), ctx)
        def w():
            f_eb = fn(ctx.Y_eb, ctx.spk_eb, ctx)
            var_n = sub_ref(f_eb, ctx.spk_eb, ref).pow(2).mean(0)
            var_s = sub_ref(chunked(fn, ctx.Y_pool, ctx.spk_pool, ctx), ctx.spk_pool, ref).var(0)
            g = var_s / (var_s + var_n)
            return g if mode == "wiener" else (g >= g.flatten().quantile(1 - keep)).float()
        W = ctx.memo(("snr_w", mode, name, keep), w)
        x = fn(Y, spk, ctx)
        return std_norm((sub_ref(x, spk, ref) if sub else x) * W)
    return arm

gated_lm = lambda g: (lambda Y, s, c: logmag(gate(Y, s, c, g)))
gated_mag = lambda g: (lambda Y, s, c: gate(Y, s, c, g).abs())
def gated_xypow(g):
    return lambda Y, s, c: xy_power(gate(Y, s, c, g), s, c)

def ARMS():
    A = {
        # --- the 12 trained arms (scripts/boombox_input_features.sh), process_vibration-identical ---
        "trained:f1-mag":            trained("magnitude"),
        "trained:f2-phase":          trained("none", phase_arm="raw_phase"),
        "trained:f3-logmag":         trained("log_magnitude"),
        "trained:f4-logmag-eb":      trained("log_magnitude", eb=True),
        "trained:f5-logmag+phase":   trained("log_magnitude", phase_arm="raw_phase"),
        "trained:f6-logmag-eb+phase": trained("log_magnitude", eb=True, phase_arm="raw_phase"),
        "trained:f7-phase-src":      trained("none", phase_arm="src"),
        "trained:f8-phase-src-eb":   trained("none", phase_arm="src_eb"),
        "trained:f9-logmag+src":     trained("log_magnitude", phase_arm="src"),
        "trained:f10-logmag+src-eb": trained("log_magnitude", phase_arm="src_eb"),
        "trained:f11-logmag-eb+src": trained("log_magnitude", eb=True, phase_arm="src"),
        "trained:f12-logmag-eb+src-eb": trained("log_magnitude", eb=True, phase_arm="src_eb"),
        # --- references ---
        "logmag-eb":                 logmag_eb,
        "logmag-spk":                spk_mean_arm,
        "logmag-eb-spk":             eb_spk_arm,
        # --- 1 delay removal (GCC-PHAT) -> usable absolute / complex-referenced phase ---
        "T1 phase(raw.S*) all bins": lambda Y, s, c: ph_raw_src(Y, s, c).flatten(1),
        "T1 phase(tau.S*)":          lambda Y, s, c: ph_tau_src(Y, s, c).flatten(1),
        "T1 phase(tau.EB*)":         lambda Y, s, c: ph_tau_eb(Y, s, c).flatten(1),
        "T1 logmag-eb+phase(tau.EB*)": with_phase(logmag_eb, ph_tau_eb),
        # --- 2 impulse-response time gating ---
        **{f"T2 gate {g}ms":         mag_eb(lambda Y, s, c, g=g: logmag(gate(Y, s, c, g)), f"gate{g}") for g in (25, 50, 100, 200, 400)},
        # plain magnitude (F1's domain, which trains best so far) gated, no reference
        **{f"T2 gate {g}ms mag":     (lambda g: (lambda Y, s, c: std_norm(gate(Y, s, c, g).abs())))(g) for g in (100, 200)},
        # --- 3 x/y axis handling ---
        "T3 xy principal axis":      mag_eb(xy_principal, "xyp"),
        "T3 xy power":               mag_eb(xy_power, "xypow"),
        # --- 4 FDD rank-k denoising / modal spectra ---
        **{f"T4 fdd w{h} k{k}":      mag_eb(fdd_den(h, k), f"fdd{h}{k}") for h, k in ((2, 1), (3, 1), (3, 3), (6, 3))},
        "T4 fdd singular values":    mag_eb(fdd_sv, "fddsv"),
        # --- 5 mode-shape curvature of the empty-box difference ---
        "T5 curvature":              curvature,
        "T5 logmag-eb+curvature":    lambda Y, s, c: torch.cat([logmag_eb(Y, s, c), curvature(Y, s, c)], 1),
        # --- 6 SALSA: principal-eigenvector inter-laser phase ---
        "T6 salsa phase":            lambda Y, s, c: salsa(Y, s, c).flatten(1),
        "T6 logmag-eb+salsa":        with_phase(logmag_eb, salsa),
        # --- 7 RAPID signal-difference coefficients ---
        "T7 rapid 1 band":           rapid(1),
        "T7 rapid 20 bands":         rapid(20),
        # --- 9 cepstral liftering (no reference at all) ---
        **{f"T9 lifter K={k}":       (lambda k: (lambda Y, s, c: std_norm(ceps(k)(Y, s, c))))(k) for k in (5, 20, 80)},
        "T9 lifter K=20 -eb":        mag_eb(ceps(20), "ceps20"),
        # --- 10 SNR weighting ---
        "T10 snr wiener":            snr_weighted("wiener"),
        "T10 snr top-half mask":     snr_weighted("mask"),
        "T10 snr top-quarter mask":  snr_weighted("mask", keep=0.25),
        # --- combinations of the winners ---
        "C gate200 + snr half":      snr_weighted("mask", gated_lm(200), "g200lm"),
        "C gate200 + snr quarter":   snr_weighted("mask", gated_lm(200), "g200lm", keep=0.25),
        "C gate200 xy power":        mag_eb(gated_xypow(200), "g200xyp"),
        "C gate200 xy power + snr half": snr_weighted("mask", gated_xypow(200), "g200xyp"),
        "C gate200 mag + snr half":  snr_weighted("mask", gated_mag(200), "g200mag", sub=False),
        "C mag + snr half":          snr_weighted("mask", lambda Y, s, c: Y.abs(), "mag", sub=False),
    }
    return A
# (T8, fingerprint matching, is the knn_skill probe column -- it is scored on every arm.)

def features(arm, Y, spk, ctx):
    return torch.cat([arm(Y[i:i + CHUNK], spk[i:i + CHUNK], ctx).flatten(1).float() for i in range(0, Y.shape[0], CHUNK)])

#***** 7 probes *****

def gram(X):
    K = torch.zeros(X.shape[0], X.shape[0], dtype=torch.float64)
    for j in range(0, X.shape[1], 65536):
        xb = X[:, j:j + 65536].double()
        K += xb @ xb.T
    return K.numpy()

def center(K, tr, te):
    """Center train and test kernels on the TRAIN feature mean."""
    Ktr, Kte = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
    mtr = Ktr.mean(0)
    Ktr_c = Ktr - mtr[None] - mtr[:, None] + mtr.mean()
    Kte_c = Kte - mtr[None] - Kte.mean(1, keepdims=True) + mtr.mean()
    return Ktr_c, Kte_c

ALPHAS = 10.0 ** np.arange(-4, 3)

def ridge_fit_predict(K, y, tr, te, groups, inner=5):
    """Kernel ridge; alpha (relative to mean kernel diagonal) picked by inner GroupKFold on train."""
    def solve(tr_, te_, a):
        Ktr, Kte = center(K, tr_, te_)
        lam, Q = np.linalg.eigh(Ktr)
        ym = y[tr_].mean(0)
        preds = {}
        scale = np.trace(Ktr) / len(tr_)
        for al in np.atleast_1d(a):
            coef = Q @ ((Q.T @ (y[tr_] - ym)) / (lam + al * scale)[:, None])
            preds[al] = Kte @ coef + ym
        return preds
    ig = groups[tr]
    err = {a: 0.0 for a in ALPHAS}
    for itr, ite in GroupKFold(min(inner, len(np.unique(ig)))).split(tr, groups=ig):
        p = solve(tr[itr], tr[ite], ALPHAS)
        for a in ALPHAS: err[a] += ((p[a] - y[tr[ite]]) ** 2).sum()
    best = min(err, key=err.get)
    return solve(tr, te, best)[best], best

def com_probe(K, com, groups, folds=10):
    pred, chance = np.zeros_like(com), np.zeros_like(com)
    for tr, te in GroupKFold(folds).split(com, groups=groups):
        pred[te], _ = ridge_fit_predict(K, com, tr, te, groups)
        chance[te] = com[tr].mean(0)
    e, c = np.linalg.norm(pred - com, axis=1).mean(), np.linalg.norm(chance - com, axis=1).mean()
    return 1 - e / c, e, c

def knn_probe(K, com, groups, k=5, folds=10):
    d = np.sqrt(np.diag(K))
    S = K / d[:, None] / d[None]
    pred, chance = np.zeros_like(com), np.zeros_like(com)
    for tr, te in GroupKFold(folds).split(com, groups=groups):
        nn = np.argsort(-S[np.ix_(te, tr)], axis=1)[:, :k]
        pred[te] = com[tr][nn].mean(1)
        chance[te] = com[tr].mean(0)
    return 1 - np.linalg.norm(pred - com, axis=1).mean() / np.linalg.norm(chance - com, axis=1).mean()

def loso_probe(K, com, spk, pos):
    pred, chance = np.zeros_like(com), np.zeros_like(com)
    for s in np.unique(spk):
        tr, te = np.where(spk != s)[0], np.where(spk == s)[0]
        pred[te], _ = ridge_fit_predict(K, com, tr, te, pos)
        chance[te] = com[tr].mean(0)
    return 1 - np.linalg.norm(pred - com, axis=1).mean() / np.linalg.norm(chance - com, axis=1).mean()

def mask_probe(K, masks, groups, folds=5):
    y = masks.reshape(len(masks), -1)
    pred, chance = np.zeros_like(y), np.zeros_like(y)
    for tr, te in GroupKFold(folds).split(y, groups=groups):
        pred[te], _ = ridge_fit_predict(K, y, tr, te, groups, inner=3)
        chance[te] = y[tr].mean(0)
    iou = lambda p: soft_iou(np.clip(p, 0, 1).reshape(masks.shape), masks).mean()
    return iou(pred), iou(chance)

def pos_snr(X_pc, spk_pc, X_eb, spk_eb):
    """Per feature: within-speaker variance across positions (purple-cube) over within-speaker
    variance across repeats of the identical empty-box scene. Returns (fraction of features whose
    position variance exceeds repeat noise, median ratio in dB). Per-feature rather than summed, so
    the many near-dead bins -- noise-dominated in BOTH sets -- cannot swamp the informative ones."""
    v_pos = torch.stack([X_pc[spk_pc == s].var(0) for s in np.unique(spk_pc)]).mean(0)
    v_rep = torch.stack([X_eb[spk_eb == s].var(0) for s in np.unique(spk_eb) if (spk_eb == s).sum() > 1]).mean(0)
    r = v_pos / v_rep.clamp_min(1e-12)
    return float((r > 1).float().mean()), float(10 * torch.log10(r.median().clamp_min(1e-12)))

#***** 8 main *****

def main():
    global K0
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None, help="substring filter on arm names")
    ap.add_argument("--no-mask", action="store_true", help="skip the (slower) 2-cube mask probe")
    ap.add_argument("--out", default=str(ROOT / "runs/signal_quality/results.csv"))
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    t0 = time.time()
    grids = [f"purple--green-cube-grid{i}" for i in (1, 2, 3, 4)]
    meta, Y, com, masks = load(["empty-box", "purple-cube", *grids])
    freqs = np.load(fft_path(Path(meta.dir[0])))["freqs"]
    K0 = int(round(freqs[0] / (FS / N_TIME)))
    assert abs(K0 * FS / N_TIME - freqs[0]) < 1e-6, "fft bins are not on the rfft grid"
    spk = meta.speaker.to_numpy()
    eb, pc, gr = (meta.layout == "empty-box").to_numpy(), (meta.layout == "purple-cube").to_numpy(), meta.layout.isin(grids).to_numpy()
    print(f"loaded {len(meta)} captures in {time.time() - t0:.0f}s: {eb.sum()} empty-box, {pc.sum()} purple-cube, {gr.sum()} 2-cube grid")

    S = source_phasor(collect_samples(DATA / "samples", 0)[0][1]["audio_dir"], freqs)
    # Two contexts: each probe set draws its reference pool from the OTHER set, so no probe capture
    # ever contributes to a statistic it is then scored with.
    ctx_pc = Ctx(Y[eb], spk[eb], Y[gr], spk[gr], freqs); ctx_pc.S = S
    ctx_gr = Ctx(Y[eb], spk[eb], Y[pc], spk[pc], freqs); ctx_gr.S = S

    # diagnostic: does GCC-PHAT alignment make the empty-box repeats phase-coherent?
    T = eb_templates(ctx_pc)
    Yeb, seb = Y[eb], spk[eb]
    def R(Z): return np.mean([unit(Z[seb == s]).mean(0).abs().mean().item() for s in np.unique(seb)])
    al = shift(Yeb, gcc_delay(Yeb, torch.stack([T[s] for s in seb])), freqs)
    taus = gcc_delay(Yeb, torch.stack([T[s] for s in seb])) * 1000
    print(f"empty-box repeat phase coherence R (1 = identical, ~{1/np.sqrt(3.9):.2f} = random for n~4): raw {R(Yeb):.3f} -> delay-aligned {R(al):.3f};  "
          f"estimated delays ms: {np.round(taus.numpy(), 2).tolist()}")

    arms = ARMS()
    names = [n for n in arms if args.arms is None or any(a in n for a in args.arms)]
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out).to_dict("records") if out.exists() else []
    done = {r["arm"] for r in rows}
    pos_pc, pos_gr = meta.position_id[pc].to_numpy(), (meta.layout[gr] + ":" + meta.position_id[gr].astype(str)).to_numpy()
    for name in names:
        if name in done: print(f"skip {name} (in {out})"); continue
        t = time.time()
        try:
            X_pc = features(arms[name], Y[pc], spk[pc], ctx_pc)
            X_eb = features(arms[name], Y[eb], spk[eb], ctx_pc)
            K = gram(X_pc)
            skill, err, chance = com_probe(K, com[pc], pos_pc)
            row = dict(arm=name, dims=X_pc.shape[1], com_skill=skill, com_err_px=err, com_chance_px=chance,
                       knn_skill=knn_probe(K, com[pc], pos_pc), loso_skill=loso_probe(K, com[pc], spk[pc], pos_pc),
                       )
            row["frac_pos_gt_noise"], row["median_pos_snr_db"] = pos_snr(X_pc, spk[pc], X_eb, spk[eb])
            del X_pc, X_eb, K
            if not args.no_mask:
                X_gr = features(arms[name], Y[gr], spk[gr], ctx_gr)
                row["mask_iou"], row["mask_iou_chance"] = mask_probe(gram(X_gr), masks[gr], pos_gr)
                del X_gr
        except Exception as e:  # one broken arm must not kill a multi-hour sweep
            import traceback; traceback.print_exc()
            row = dict(arm=name, error=repr(e))
        row["secs"] = time.time() - t
        rows.append(row)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    df = pd.DataFrame(rows)
    print(df.sort_values("com_skill", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

if __name__ == "__main__":
    main()
