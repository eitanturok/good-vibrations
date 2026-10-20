"""Does the linear modal picture hold on our data? For small objects the physics reduces to dy ~ J rho: the change of the wall lasers'
spectra (dy) is a fixed linear map (J: floor cell -> laser x freq) of the object's mass on the floor (rho). Rayleigh: each mode's
shift is d(w_n^2)/w_n^2 ~ -(1/M_n) int rho phi_n^2, so mass LOWERS frequencies, linearly in the mass. No mode parameters are fitted: J
is learned from data by ridge, and the predictions are tested directly. dy = log power minus that of the nearest-in-time empty box of
the same speaker (nb101: drift is speaker-box geometry), in ~7.7 Hz bands (10 bins). Train / held-out = the wedge dataset's splits.
  mass     cylinders 100/200/500/1000 g (same shape, only mass differs): at the same spot and speaker, dy_m projected on a 1000 g
           capture's dy, a_m = <dy_m, dy_ref> / <dy_ref, dy_ref>; linear => a_m = (m / 1000) a_1000
  shift    per frequency band, the lag (Hz) that best aligns a capture's log spectrum with its empty box (cross-correlation over all
           lasers and axes, parabolic sub-bin peak); mass loading => negative, growing with mass. Control: empty vs empty
  ladder   forward models of dy on held-out positions, per (box, speaker): mean per object -> + smooth function of the centroid ->
           + covariance -> full mask J (shared by all objects, and per object), and kNN on the centroid; fve vs the drift ceiling
  inverse  dy-free: box-normalized log power -> 64x64 mask by kernel ridge, linear vs rbf kernel; held-out iou at a threshold
           picked on a train slice; compare with the trained network per box and object
  J        the shared J's SVD: the floor patterns the lasers see most (data-driven modes)
  .venv/bin/python src8/linear_physics.py [--steps load,mass,shift,ladder,inverse]   -> runs/linear_physics/*.csv, *.npz"""
import argparse, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import dataset as D
from additivity import EXPS, SKIP, mask64
from probe import ROOT

OUT = ROOT / "runs" / "linear_physics"
POOL = 10                                                                   # bins per band in the pooled features (~7.7 Hz)
MASS = {"cylinder-100g": 100, "cylinder-200g": 200, "cylinder-500g": 500, "cylinder-1000g": 1000}

def t(meta): return datetime.fromisoformat(meta["timestamp"]).timestamp()
def logpow(d, meta): return np.log(D.load_fft_raw(d, meta).abs().pow(2).numpy() + 1e-6)          # (L,F,C)
def pool(x): F = x.shape[1] // POOL * POOL; return x[:, :F].reshape(x.shape[0], -1, POOL, x.shape[2]).mean(2)  # (L,F/POOL,C)

def moments64(m):
    """area (fraction), centroid (row, col) and covariance (rr, cc, rc) of a 64x64 mask, in box fractions"""
    r, c = np.nonzero(m)
    if len(r) == 0: return dict(area=0.0, r=np.nan, c=np.nan, rr=np.nan, cc=np.nan, rc=np.nan)
    r, c = (r + 0.5) / 64, (c + 0.5) / 64
    return dict(area=len(r) / 64 ** 2, r=r.mean(), c=c.mean(), rr=r.var(), cc=c.var(), rc=((r - r.mean()) * (c - c.mean())).mean())

#*** load: every single-object and empty capture of the three boxes

def load():
    samples = {d: m for e in EXPS.values() for d, m in D.collect_samples(D.EXPERIMENTS / e).items()}
    dirs, metas = list(samples), list(samples.values())
    idxs = D.build_splits(D.DATASETS["gastronorm_plastic_cardboard_wedge"](0.2, [1, 3, 5, 7]), metas)
    split = {}
    for n, ids in idxs.items():
        for i in ids: split[dirs[i]] = n
    keep = [d for d, m in samples.items() if m["n_objects"] <= 1 and not any(s in m["layout"] for s in SKIP) and d in split]
    rows, X, M = [], [], []
    for d in tqdm(keep, desc="load"):
        m = samples[d]
        mk = mask64(d / "image/03_smask.png") if m["n_objects"] else np.zeros((64, 64), bool)
        obj = next(iter(m["objects"])) if m["n_objects"] else "empty"
        rows.append(dict(dir=str(d), box=m["box"], layout=m["layout"], object=obj, position_id=m["position_id"], speaker=m["speaker"],
                         t=t(m), split=split[d], train=split[d] == "train", mass=MASS.get(obj, np.nan), **moments64(mk)))
        X.append(pool(logpow(d, m)).astype(np.float16)); M.append(np.packbits(mk))
    df = pd.DataFrame(rows)
    X, M = np.stack(X), np.stack(M)
    # dy: minus the nearest-in-time empty box of the same box and speaker (an empty's own: the nearest OTHER empty, the drift control)
    E = np.zeros_like(X); ref = np.full(len(df), -1)
    for (b, s), g in df.groupby(["box", "speaker"]):
        emp = g[g.object == "empty"]
        for i in g.index:
            cand = emp.drop(i, errors="ignore")
            if len(cand): ref[i] = (cand.t - df.t[i]).abs().idxmin()
    df["ref"], df["gap_h"] = ref, np.where(ref >= 0, (df.t - df.t.reindex(ref).values).abs() / 3600, np.nan)
    ok = ref >= 0
    E[ok] = (X[ok].astype(np.float32) - X[ref[ok]].astype(np.float32)).astype(np.float16)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "samples.csv", index=False)
    np.savez(OUT / "features.npz", X=X, dY=E, M=M)
    return df, X, E, M

def cached():
    df = pd.read_csv(OUT / "samples.csv"); z = np.load(OUT / "features.npz")
    return df, z["X"], z["dY"], z["M"]

#*** mass: is dy linear in the mass?

def mass(df, E):
    """per cylinder capture: its dy projected on every 1000 g capture of the same box + speaker within 3% of the box (another capture
    for a 1000 g one), averaged; corr too. The empty-vs-empty dy gives the noise level"""
    cyl = df[df.object.isin(MASS)]
    rows = []
    for i, r in cyl.iterrows():
        refs = cyl[(cyl.box == r.box) & (cyl.speaker == r.speaker) & (cyl.mass == 1000) & (cyl.index != i)]
        refs = refs[np.hypot(refs.r - r.r, refs.c - r.c) < 0.03]
        for j in refs.index:
            a, b = E[i].astype(np.float32).ravel(), E[j].astype(np.float32).ravel()
            rows.append(dict(box=r.box, mass=r.mass, speaker=r.speaker, i=i, j=j, dist=float(np.hypot(df.r[j] - r.r, df.c[j] - r.c)),
                             proj=float(a @ b / (b @ b)), corr=float(np.corrcoef(a, b)[0, 1]), norm=float(np.sqrt(a @ a)), ref_norm=float(np.sqrt(b @ b))))
    m = pd.DataFrame(rows)
    noise = df[df.object == "empty"].index
    nn = pd.DataFrame(dict(box=df.box[noise], norm=[float(np.linalg.norm(E[i].astype(np.float32))) for i in noise]))
    m.to_csv(OUT / "mass.csv", index=False); nn.to_csv(OUT / "noise_norm.csv", index=False)
    print(m.groupby(["box", "mass"])[["proj", "corr", "norm"]].mean().round(3), "\nnoise |dy| (empty vs empty):", nn.groupby("box").norm.mean().round(2).to_dict())

#*** shift: do the resonances move down?

def lag(x, y, K=12):
    """the lag (bins) maximizing corr(x[f+k], y[f]) summed over lasers and axes; x, y (L,F,C) one band; parabolic peak"""
    x, y = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    F = x.shape[1]
    c = np.array([(x[:, max(k, 0):F + min(k, 0)] * y[:, max(-k, 0):F - max(k, 0)]).sum() / (F - abs(k)) for k in range(-K, K + 1)])
    k = int(c.argmax())
    if 0 < k < 2 * K: k += 0.5 * (c[k - 1] - c[k + 1]) / (c[k - 1] - 2 * c[k] + c[k + 1])
    return k - K, c.max() / max(c[K], 1e-9)

def shift(df, n_bands=8):
    """cylinders + empties (control) vs their reference empty, per band, at full resolution: lag in Hz (negative = moved down)"""
    sel = df[(df.object.isin(MASS) | (df.object == "empty")) & (df.ref >= 0)]
    rows = []
    for i, r in tqdm(sel.iterrows(), total=len(sel), desc="shift"):
        d, e = Path(r.dir), Path(df.dir[r.ref])
        x, y = logpow(d, _meta(d)), logpow(e, _meta(e))
        mf = _meta(d)
        f = np.linspace(mf["min_freq"], mf["max_freq"], x.shape[1]); df_hz = f[1] - f[0]
        edges = np.linspace(0, x.shape[1], n_bands + 1).astype(int)
        for b in range(n_bands):
            k, gain = lag(x[:, edges[b]:edges[b + 1]], y[:, edges[b]:edges[b + 1]])
            rows.append(dict(i=i, box=r.box, object=r.object, mass=r.mass if r.object != "empty" else 0, band=b, f_hz=float(f[edges[b]:edges[b + 1]].mean()),
                             lag_hz=k * df_hz, rel=k * df_hz / float(f[edges[b]:edges[b + 1]].mean()), gain=gain, gap_h=r.gap_h))
    s = pd.DataFrame(rows); s.to_csv(OUT / "shift.csv", index=False)
    print(s.groupby(["box", "mass"]).lag_hz.mean().unstack().round(3))

_metas = {}
def _meta(d):
    """collect_samples' metadata of capture d (the raw metadata.jsonl lacks fps / freq range)"""
    if not _metas:
        for e in EXPS.values(): _metas.update(D.collect_samples(D.EXPERIMENTS / e))
    return _metas[d]

#*** ladder: forward models of dy, simplest first

def ridge_fit(P, Y, lam):
    A = P.T @ P; A[np.diag_indices_from(A)] += lam * (np.trace(A) / len(A) + 1e-9)
    return np.linalg.solve(A, P.T @ Y)

def poly(r, c, deg):
    """2-D polynomial basis in the centroid, up to total degree deg (centred on the box)"""
    r, c = r - 0.5, c - 0.5
    return np.stack([r ** i * c ** j for i in range(deg + 1) for j in range(deg + 1 - i)], 1)

def features(g, Mk, kind, objs):
    o = np.stack([(g.object == ob).values for ob in objs], 1).astype(np.float32)              # (n,O) one-hot
    if kind == "mean": return o
    pc = poly(g.r.values, g.c.values, 4)                                                      # 15 terms
    if kind == "centroid": return (o[:, :, None] * pc[:, None]).reshape(len(g), -1)
    cov = np.stack([g.rr, g.cc, g.rc], 1) * 100                                               # box-fraction variances -> O(1)
    if kind == "covariance": return np.concatenate([(o[:, :, None] * pc[:, None]).reshape(len(g), -1), (o[:, :, None] * np.concatenate([cov, cov * (g.r.values[:, None] - .5), cov * (g.c.values[:, None] - .5)], 1)[:, None]).reshape(len(g), -1)], 1)
    m16 = Mk.reshape(len(g), 16, 4, 16, 4).mean((2, 4)).reshape(len(g), 256)                  # the mask on a 16x16 grid
    if kind == "J_shared": return np.concatenate([o, m16], 1)                                 # rho = mask (unit density): one J for every object
    if kind == "J_object": return np.concatenate([o, (o[:, :, None] * m16[:, None]).reshape(len(g), -1)], 1)
    raise ValueError(kind)

KINDS = ["mean", "centroid", "covariance", "J_shared", "J_object"]

def ladder(df, X, M):
    """per (box, speaker): fit on train captures, score fve on held-out (eval-split) captures of objects seen in train. Ridge strength
    picked on 20% of the train positions. Ceiling: 1 - |drift|^2 / |dy|^2 with drift = empty-vs-empty dy (same box)"""
    rows, Js = [], {}
    single = df[(df.object != "empty") & df.r.notna()]
    # dy vs the empty box is mostly session drift (cardboard cylinders: corr .78 far vs .80 same spot): each capture minus its
    # session's mean (box, layout, speaker) instead. The object's mean effect over the session goes with it; what is left is position
    E = np.zeros(X.shape, np.float16)
    for _, g in single.groupby(["box", "layout", "speaker"]):
        E[g.index] = (X[g.index].astype(np.float32) - X[g.index].astype(np.float32).mean(0)).astype(np.float16)
    rng = np.random.default_rng(0)
    for (b, s), g in single.groupby(["box", "speaker"]):
        tr = g[g.train]; objs = sorted(tr.object.value_counts().loc[lambda v: v >= 20].index)
        te = g[~g.train & g.object.isin(objs)]
        if len(te) < 20: continue
        Y = lambda h: E[h.index].astype(np.float32).reshape(len(h), -1)
        Mk = lambda h: np.unpackbits(M[h.index], axis=1)[:, :4096].astype(np.float32)
        tr = tr[tr.object.isin(objs)]
        pos = tr.position_id.unique(); val = set(rng.choice(pos, max(1, len(pos) // 5), replace=False))
        a, v = tr[~tr.position_id.isin(val)], tr[tr.position_id.isin(val)]
        Ytr, Yte, Ya, Yv = Y(tr), Y(te), Y(a), Y(v)
        for kind in KINDS:
            f = lambda h: features(h, Mk(h), kind, objs)
            best = min([1e-4, 1e-3, 1e-2, 1e-1, 1, 10], key=lambda lam: ((f(v) @ ridge_fit(f(a), Ya, lam) - Yv) ** 2).sum())
            W = ridge_fit(f(tr), Ytr, best); P = f(te) @ W
            for o in objs + ["all"]:
                k = (te.object == o).values if o != "all" else np.ones(len(te), bool)
                if k.sum() < 5: continue
                rows.append(dict(box=b, speaker=s, model=kind, object=o, n=int(k.sum()), lam=best,
                                 fve=1 - ((P[k] - Yte[k]) ** 2).sum() / (Yte[k] ** 2).sum(),
                                 corr=float(np.mean([np.corrcoef(P[i], Yte[i])[0, 1] for i in np.nonzero(k)[0]]))))
            if kind == "J_shared": Js[f"{b}_{s}"] = W[len(objs):]                               # (256, L*F'*C)
        # kNN on the centroid (same object): the nonparametric "position only" model
        P = np.zeros_like(Yte)
        for o in objs:
            k, kt = (te.object == o).values, (tr.object == o).values
            dist = np.hypot(te.r.values[k, None] - tr.r.values[None, kt], te.c.values[k, None] - tr.c.values[None, kt])
            P[k] = Ytr[kt][dist.argsort(1)[:, :3]].mean(1)
        rows.append(dict(box=b, speaker=s, model="knn_centroid", object="all", n=len(te), lam=np.nan, fve=1 - ((P - Yte) ** 2).sum() / (Yte ** 2).sum(),
                         corr=float(np.mean([np.corrcoef(P[i], Yte[i])[0, 1] for i in range(len(te))]))))
        print(b, s, {r["model"]: round(r["fve"], 3) for r in rows if r["box"] == b and r["speaker"] == s and r["object"] == "all"})
    lad = pd.DataFrame(rows); lad.to_csv(OUT / "ladder.csv", index=False)
    # the J's SVD per box (speakers stacked along the output): floor patterns U (16x16) and their strengths
    svd = {}
    for b in sorted(single.box.unique()):
        Jb = np.concatenate([Js[k] for k in Js if k.startswith(b + "_")], 1)
        U, S, _ = np.linalg.svd(Jb, full_matrices=False)
        svd[f"{b}_U"], svd[f"{b}_S"] = U[:, :12].T.reshape(12, 16, 16), S
    np.savez(OUT / "J_svd.npz", **svd)

#*** inverse: spectrum -> mask, linear vs nonlinear kernel

def inverse(df, X, M, n_max=6000):
    """per box, on box-normalized pooled log power (every feature to mean 0 / std 1 over the box's train captures, as box_norm): kernel
    ridge to the 64x64 mask; lambda and threshold picked on 20% of the train positions; held-out iou per capture"""
    rows = []
    rng = np.random.default_rng(0)
    for b, g in df[df.ref >= 0].groupby("box"):
        tr, te = g[g.train], g[~g.train & (g.object != "empty")]
        if len(tr) > n_max: tr = tr.sample(n_max, random_state=0)
        Xb = lambda h: X[h.index].astype(np.float32).reshape(len(h), -1)
        Xtr = Xb(tr); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Z = lambda h: (Xb(h) - mu) / sd / np.sqrt(Xtr.shape[1])
        Ztr, Zte = Z(tr), Z(te)
        Ytr = np.unpackbits(M[tr.index], axis=1)[:, :4096].astype(np.float32)
        Yte = np.unpackbits(M[te.index], axis=1)[:, :4096].astype(bool)
        pos = tr.position_id.unique(); val = np.isin(tr.position_id, rng.choice(pos, len(pos) // 5, replace=False))
        sq = (Ztr ** 2).sum(1)
        for kern in ("linear", "rbf"):
            if kern == "linear": K, Kte = Ztr @ Ztr.T, Zte @ Ztr.T
            else:
                D2 = sq[:, None] + sq[None] - 2 * Ztr @ Ztr.T; s2 = np.median(D2)
                K = np.exp(-D2 / s2); Kte = np.exp(-((Zte ** 2).sum(1)[:, None] + sq[None] - 2 * Zte @ Ztr.T) / s2)
            ta, tv = np.nonzero(~val)[0], np.nonzero(val)[0]
            best = (-1, None, None)
            for lam in (1e-3, 1e-2, 1e-1, 1):
                A = np.linalg.solve(K[np.ix_(ta, ta)] + lam * np.eye(len(ta)) * np.trace(K) / len(K), Ytr[ta])
                Pv = K[np.ix_(tv, ta)] @ A
                for th in (0.1, 0.2, 0.3, 0.4, 0.5):
                    has = Ytr[tv].sum(1) > 0
                    sc = np.mean(((Pv > th) & (Ytr[tv] > 0)).sum(1)[has] / ((Pv > th) | (Ytr[tv] > 0)).sum(1).clip(1)[has])
                    if sc > best[0]: best = (sc, lam, th)
            _, lam, th = best
            P = Kte @ np.linalg.solve(K + lam * np.eye(len(K)) * np.trace(K) / len(K), Ytr) > th
            iou = (P & Yte).sum(1) / (P | Yte).sum(1).clip(1)
            for k, (i, r) in enumerate(te.iterrows()):
                rows.append(dict(box=b, kernel=kern, object=r.object, layout=r.layout, split=r.split, position_id=r.position_id, speaker=r.speaker,
                                 iou=float(iou[k]), miss=float(not (P[k] & Yte[k]).any()), lam=lam, th=th))
            print(b, kern, f"lam {lam} th {th}", pd.DataFrame(rows).query("box == @b and kernel == @kern").groupby("split").iou.mean().round(3).to_dict())
    pd.DataFrame(rows).to_csv(OUT / "inverse.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", default="load,mass,shift,ladder,inverse")
    steps = ap.parse_args().steps.split(","); torch.set_num_threads(4)
    df, X, E, M = load() if "load" in steps else cached()
    if "mass" in steps: mass(df, E)
    if "shift" in steps: shift(df)
    if "ladder" in steps: ladder(df, X, M)
    if "inverse" in steps: inverse(df, X, M)
