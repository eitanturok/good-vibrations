"""A physical forward model of the candle: the box as K modes, the candle as a point mass at its centroid c.

Empty box, in modal coordinates q (one per mode k):  (w_k^2 - w^2 + 2i z_k w_k w) q_k = b_k S(w)      G = diag(1 / that bracket)
A rigid mass m at floor point c pushes back on the floor with its inertia, m w^2 u(c), u(c) = v . q, v_k = phi_k(c) (sqrt(m) absorbed
into v): G^-1 q = b S + w^2 v (v . q). Sherman-Morrison solves it exactly:
    q = S [ G b + G v  w^2 (v . G b) / (1 - w^2 v . G v) ]
    Y_l = Phi_l . q                       laser l, axis a: Phi (L,A,K) = the mode shapes on the wall
The measured phase is not consistent between captures (coherence .1-.9 between two empty boxes), and 12 modes can't draw the box's
fine detail, so the model predicts log power as the measured box times the candle's physical RATIO (S cancels in it):
    log|Y(c)|^2 = B + log|Phi q(c)|^2 - log|Phi G b|^2          B (L,A,F) free, initialized at the mean empty-box log power
--contact: the candle is more than a mass. Its push-back w^2 m becomes the impedance Z(w) = w^2 + kappa + i w eta (m in v; kappa a
contact spring, eta contact damping): plastic's resonances moved UP with the cylinders' mass (stiffness), which a pure mass can't do.
--sessions: one B per recording session (layout): sessions differ by .16-.29 in log10 power, as much as the candle, with structure
(peaks moving), not noise. --loss l1: the repeat error is heavy-tailed (kurtosis 12-32: deep notches), squared error lets them dominate.
In words: the speaker shakes the floor at c (v . G b), the candle's inertia turns that into a force (w^2), the force travels to the
lasers through every mode (Phi G v), and the candle feeds back on itself (1 - w^2 v . G v): nonlinear in the mass.
Variants (what each assumption keeps):
    full      the formula above
    linear    first order in the mass: the denominator set to 1 (Born approximation)
    shift     no mode mixing: each mode alone gets heavier, q_k = b_k S / (w_k^2 - w^2 (1 + v_k^2) + 2i z_k w_k w): frequencies move
              down, mode shapes on the wall stay fixed (first-order Rayleigh)
v_k(c) = phi_k on the floor: a smooth field, sum_j W_kj cos(pi p_j r) cos(pi q_j c), p, q <= 5 (a plate's modes look like these).
Fitted to log power by Adam on the train candle captures of one box + speaker (+ that speaker's empty boxes, v = 0). Scored on the
held-out candle positions (the dataset's eval split):
    fve_mean  1 - SSE / SSE of the train mean spectrum: how much of the position-dependent change the model explains
    knn       the 3 train captures nearest in position, averaged: a model-free bar
    loc_err   inverse: the c on a 64 x 64 grid whose predicted spectrum fits the capture best, distance to the true centroid (box
              fractions); kNN in spectrum space as the bar
  .venv/bin/python src8/candle_physics.py [--box green-plastic] [--speaker 1] [--modes 12]   -> runs/candle_physics/<box>_<spk>/"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks, peak_widths

sys.path.insert(0, str(Path(__file__).parent))
import dataset as D
from linear_physics import _meta
from probe import ROOT

F_LO, F_HI = 100.0, 1000.0                     # Hz: the band the model fits (--f-hi to narrow it)
W_REF = 2 * math.pi * 500.0                    # rad/s: frequencies are scaled by this, so every term is O(1)

#*** data

def load(box, speaker, f_hi=F_HI):
    """complex spectra Y (N,L,A,F) in the band, centroid (N,2), train mask (N,), empty mask (N,), frequencies (F,) Hz"""
    df = pd.read_csv(ROOT / "runs" / "linear_physics" / "samples.csv")
    g = df[(df.box == box) & (df.speaker == speaker) & df.object.isin(["candle", "empty"])].reset_index(drop=True)
    m = _meta(Path(g.dir[0]))
    f = np.linspace(m["min_freq"], m["max_freq"], D.load_fft_raw(Path(g.dir[0]), m).shape[1])
    band = (f >= F_LO) & (f <= f_hi)
    Y = np.stack([D.load_fft_raw(Path(d), _meta(Path(d))).numpy()[:, band] for d in g.dir])  # (N,L,F,A)
    Y = torch.from_numpy(Y).permute(0, 1, 3, 2).contiguous()                                  # (N,L,A,F)
    rc = torch.tensor(g[["r", "c"]].fillna(0.5).values, dtype=torch.float32)
    g["session"] = pd.factorize(g.layout)[0]
    return g, Y, rc, torch.tensor(g.train.values), torch.tensor((g.object == "empty").values), torch.tensor(f[band], dtype=torch.float32)

def logpow(Y): return (Y.abs() ** 2 + 1e-12).log()

def init_modes(Y, f, K):
    """the K most prominent peaks of the mean power: frequency (Hz) and damping ratio from the half-power width"""
    P = (Y.abs() ** 2).mean((0, 1, 2)).numpy()
    pk, pr = find_peaks(np.log(P), prominence=0.2)
    pk = pk[np.argsort(-pr["prominences"])[:K]]
    w = peak_widths(P, pk, rel_height=0.5)[0] * float(f[1] - f[0])
    order = np.argsort(pk)
    return f[pk[order]], torch.tensor(np.clip(w[order] / (2 * f[pk[order]].numpy()), 0.002, 0.1), dtype=torch.float32)

#*** model

class Candle(torch.nn.Module):
    def __init__(self, f_modes, zeta, n_lasers, n_axes, n_freqs, variant="full", order=5, contact=False, n_sessions=1):
        super().__init__()
        K = len(f_modes)
        self.variant = variant
        self.log_w = torch.nn.Parameter((2 * math.pi * f_modes / W_REF).log())               # (K,) mode frequencies
        self.log_z = torch.nn.Parameter(zeta.log())                                           # (K,) damping ratios
        self.b = torch.nn.Parameter(torch.randn(K) * 0.1 + 1)                                 # (K,) how hard the speaker drives each mode
        self.Phi = torch.nn.Parameter(torch.randn(n_lasers, n_axes, K) * 0.1)                 # (L,A,K) mode shapes on the wall
        pq = [(p, q) for p in range(order + 1) for q in range(order + 1)]
        self.register_buffer("pq", torch.tensor(pq, dtype=torch.float32))                    # (J,2)
        self.W = torch.nn.Parameter(torch.randn(K, len(pq)) * 0.1)                            # (K,J) mode shapes on the floor
        self.contact = torch.nn.Parameter(torch.zeros(2)) if contact else None                # kappa, eta
        self.B = torch.nn.Parameter(torch.zeros(n_sessions, n_lasers, n_axes, n_freqs))       # the box without the candle, log power, per session

    def floor_shape(self, rc):
        """v (N,K): each mode's motion at floor points rc (N,2) in box fractions"""
        B = torch.cos(math.pi * self.pq[:, 0] * rc[:, :1]) * torch.cos(math.pi * self.pq[:, 1] * rc[:, 1:])  # (N,J)
        return B @ self.W.T

    def forward(self, rc, on, w, sess=None):
        """rc (N,2) candle centroid, on (N,) False for an empty box, w (F,) rad/s / W_REF, sess (N,) session -> log power (N,L,A,F)"""
        wk, zk = self.log_w.exp(), self.log_z.exp()
        v = self.floor_shape(rc) * on[:, None]                                                # (N,K)
        w2 = (w ** 2)[None, :, None]                                                          # (1,F,1)
        Z = w2 if self.contact is None else w2 + self.contact[0] + 1j * self.contact[1] * w[None, :, None]  # the candle's push-back
        den = wk ** 2 - w2 + 2j * zk * wk * w[None, :, None]                                  # (1,F,K)
        if self.variant == "shift":
            q = self.b / (den - Z * v[:, None] ** 2)                                          # (N,F,K) each mode heavier on its own
        else:
            G = 1 / den                                                                       # (1,F,K)
            Gb, Gv = G * self.b, G * v[:, None]                                               # (N,F,K)
            vGb, vGv = (v[:, None] * Gb).sum(-1, keepdim=True), (v[:, None] * Gv).sum(-1, keepdim=True)  # (N,F,1)
            fb = 1 if self.variant == "linear" else 1 - Z * vGv
            q = Gb + Gv * Z * vGb / fb                                                       # (N,F,K)
        Phi = self.Phi.to(q.dtype)
        Yc = torch.einsum("lak,nfk->nlaf", Phi, q)                                            # (N,L,A,F) with the candle
        Y0 = torch.einsum("lak,fk->laf", Phi, (self.b / den)[0])                              # (L,A,F) without
        B = self.B[0] if sess is None else self.B[sess]                                       # (L,A,F) or (N,L,A,F)
        return B + logpow(Yc) - logpow(Y0)

#*** fit and score

def fit(model, Y, rc, on, w, sess, steps, lr=3e-2, batch=48, seed=0, loss="l2"):
    torch.manual_seed(seed)
    target, loss_fn = Y, loss
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        i = torch.randint(len(Y), (batch,))
        r = model(rc[i], on[i], w, sess[i]) - target[i]
        loss = (r.abs() if loss_fn == "l1" else r ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == steps - 1: print(f"  {model.variant} step {step} loss {loss.item():.3f}", flush=True)
    return model

def localize(model, Y, w, sess, n=32):
    """the floor point whose predicted spectrum best fits each capture (grid search, n x n), (N,2) box fractions.
    |P - T|^2 = |P|^2 - 2 P.T + |T|^2 chunk by chunk: the n*n predicted spectra never sit in memory at once"""
    g = (torch.arange(n) + 0.5) / n
    grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).view(-1, 2)                    # (n*n,2)
    T = Y.flatten(1)                                                                          # (N,D) log power
    B = model.B[sess].flatten(1) - model.B[0].flatten()[None]                                 # (N,D) each capture's session vs session 0
    T = T - B                                                                                 # compare on session 0's baseline
    err = []
    with torch.no_grad():
        for i in range(0, len(grid), 128):
            P = model(grid[i:i + 128], torch.ones(len(grid[i:i + 128]), dtype=torch.bool), w).flatten(1)  # (128,D)
            err.append((P ** 2).sum(1)[None] - 2 * T @ P.T)                                   # (N,128); |T|^2 is the same for every point
    return grid[torch.cat(err, 1).argmin(1)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="green-plastic"); ap.add_argument("--speaker", type=int, default=1)
    ap.add_argument("--modes", type=int, default=12); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--variants", default="shift,linear,full")
    ap.add_argument("--order", type=int, default=5); ap.add_argument("--contact", type=int, default=0); ap.add_argument("--tag", default="")
    ap.add_argument("--f-hi", type=float, default=F_HI)
    ap.add_argument("--sessions", type=int, default=0); ap.add_argument("--loss", default="l2", choices=("l2", "l1"))
    a = ap.parse_args(); torch.set_num_threads(4)
    out = ROOT / "runs" / "candle_physics" / f"{a.box}_{a.speaker}{a.tag}"; out.mkdir(parents=True, exist_ok=True)

    g, Y, rc, train, empty, f = load(a.box, a.speaker, a.f_hi)
    on, w = ~empty, 2 * math.pi * f / W_REF
    sess = torch.tensor(g.session.values) if a.sessions else torch.zeros(len(g), dtype=torch.long)
    tr, te = train | empty, ~train & ~empty
    print(f"{a.box} speaker {a.speaker}: {int((train & on).sum())} train candle, {int(te.sum())} held-out, {int(empty.sum())} empty; {len(f)} bins")
    f_modes, zeta = init_modes(Y[tr], f, a.modes)
    print("modes (Hz):", f_modes.round().int().tolist())

    T = logpow(Y)
    mean = T[tr & on].mean(0)
    sse = lambda P, idx: ((P - T[idx]) ** 2).sum().item()
    base = sse(mean.expand(int(te.sum()), *mean.shape), te)
    d = torch.cdist(rc[te], rc[tr & on])
    knn = T[tr & on][d.topk(3, largest=False).indices].mean(1)
    rows = [dict(model="train mean", fve_mean=0.0), dict(model="knn (position)", fve_mean=1 - sse(knn, te) / base)]
    # inverse bar: the train capture nearest in log power -> its centroid
    dist = torch.cdist(T[te].flatten(1), T[tr & on].flatten(1))
    rows[1]["loc_err"] = (rc[tr & on][dist.argmin(1)] - rc[te]).norm(dim=1).mean().item()

    for variant in a.variants.split(","):
        model = Candle(f_modes, zeta, Y.shape[1], Y.shape[2], len(f), variant, a.order, bool(a.contact), int(sess.max()) + 1)
        with torch.no_grad(): model.B.copy_(T[empty].mean(0).expand_as(model.B))
        fit(model, T[tr], rc[tr], on[tr], w, sess[tr], a.steps, loss=a.loss)
        torch.save(model.state_dict(), out / f"{variant}.pt")
        with torch.no_grad(): P = model(rc[te], on[te], w, sess[te])
        loc = localize(model, T[te], w, sess[te])
        rows.append(dict(model=variant, fve_mean=1 - sse(P, te) / base, loc_err=(loc - rc[te]).norm(dim=1).mean().item(),
                         train_fve_mean=1 - ((model(rc[tr & on], on[tr & on], w, sess[tr & on]).detach() - T[tr & on]) ** 2).sum().item() / ((mean - T[tr & on]) ** 2).sum().item()))
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows); res.to_csv(out / "results.csv", index=False)
    json.dump(dict(f_modes=f_modes.tolist(), f=[float(f[0]), float(f[-1]), len(f)], n_train=int((train & on).sum()), n_test=int(te.sum())), open(out / "info.json", "w"))
    print(res.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
