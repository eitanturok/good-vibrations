# checks on "pair ≈ a·A + b·B + c": noise ceiling, denoised singles (mean of K nearest), no-intercept variant, direction vs magnitude.
# all fits per (box, speaker) on even pair positions, metrics on odd. c is fit separately per pair type (never shared cube+vase ↔ cube+cube).
# -> runs/_src9_cache/nb137/<RUN>/additive_check.csv, notebooks/out/137_additive_check.pdf
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src9")); C = ROOT / "runs/_src9_cache/nb137/decres-rescale-aug10-v1"
from dataset import load_splits
PAIRS = {"red-cube+vase": [("red-cube", "red-cube0"), ("vase", "vase0")], "green-cube+red-cube": [("red-cube", "green-cube0"), ("red-cube", "red-cube0")]}
K = 5  # singles averaged per object (nearest by CoM, same box & speaker)
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
dirs = list(load_splits()[0].keys()); df["dir"] = [dirs[i] for i in np.load(ROOT / "runs/_src9_cache/nb136/base.npz")["i"]]
df = df[~df.split.str.contains("shifted|unseen-speaker|lid")]
def com(f): m = np.load(f) > 0; r, c = np.nonzero(m); return (c.mean() + .5) * 64 / m.shape[1] - .5, (r.mean() + .5) * 64 / m.shape[0] - .5
def near(S, x, y, k, skip=None):  # indices of the k singles nearest to (x, y), optionally excluding one sample
    d = np.hypot(S.x - x, S.y - y); d = d.drop(skip) if skip is not None else d; return d.nsmallest(k).index
def fit(Y, X, tr, intercept=True):  # Y ≈ Σ w_j X_j (+ c): scalar weights, vector intercept; returns predictions for all rows
    mY, mX = (Y[tr].mean(0), [x[tr].mean(0) for x in X]) if intercept else (0, [0] * len(X))
    w = np.linalg.lstsq(np.stack([(x - m)[tr].reshape(-1) for x, m in zip(X, mX)], 1), (Y - mY)[tr].reshape(-1), rcond=None)[0]
    return mY + sum(wj * (x - m) for wj, x, m in zip(w, X, mX)), w, mY
def scores(Y, P, mY, te):  # held-out: R² vs predicting the mean, direction = cos(real − mean, pred − mean), magnitude = gain of real along pred
    y, p = (Y - mY)[te], (P - mY)[te]; cos = np.sum(y * p, 1) / np.linalg.norm(y, axis=1) / np.linalg.norm(p, axis=1)
    D = (p / np.linalg.norm(p, axis=1, keepdims=True)) @ (y / np.linalg.norm(y, axis=1, keepdims=True)).T
    return dict(R2=1 - ((y - p) ** 2).sum() / (y ** 2).sum(), cos=np.median(cos), gain=np.median(np.sum(y * p, 1) / np.sum(p * p, 1)),
                top1=np.mean(np.argmax(D, 1) == np.arange(len(D)))), cos, np.sum(y * p, 1) / np.sum(p * p, 1)
rows, H = [], {}
rng = np.random.default_rng(0)
for pair, parts in PAIRS.items():
    P = df[df.object == pair].copy()
    for j, (_, inst) in enumerate(parts): P[[f"x{j}", f"y{j}"]] = [com(Path(d) / f"image/smasks/{inst}.npy") for d in P.dir]
    for (box, spk), p in P.groupby(["box", "speaker"]):
        S = df[(df.box == box) & (df.speaker == spk)]; Y = Z[p.index]; tr = p.pos.values % 2 == 0; te = ~tr
        X1 = [Z[[near(S[S.object == ob], r[f"x{j}"], r[f"y{j}"], 1)[0] for _, r in p.iterrows()]] for j, (ob, _) in enumerate(parts)]
        XK = [np.stack([Z[near(S[S.object == ob], r[f"x{j}"], r[f"y{j}"], K)].mean(0) for _, r in p.iterrows()]) for j, (ob, _) in enumerate(parts)]
        XS = [x[rng.permutation(len(x))] for x in XK]
        for name, X, ic in (("matched 1", X1, True), (f"matched mean of {K}", XK, True), (f"matched mean of {K}, no c", XK, False), ("shuffled", XS, True)):
            Pr, w, mY = fit(Y, X, tr, ic); s, cs, gn = scores(Y, Pr, Y[tr].mean(0), te)
            rows.append(dict(kind=pair, box=box, speaker=spk, model=name, n=int(te.sum()), a=w[0], b=w[1]) | s); H.setdefault((pair, box, name), []).append((cs, gn))
# noise ceiling: a single predicted by the mean of its K nearest OTHER singles of the same object (same box & speaker), same fit / scores
for ob in ("red-cube", "vase"):
    for (box, spk), S in df[df.object == ob].groupby(["box", "speaker"]):
        if len(S) < 40: continue
        Y = Z[S.index]; tr = S.pos.values % 2 == 0; X = [np.stack([Z[near(S, r.x, r.y, K, skip=i)].mean(0) for i, r in S.iterrows()])]
        Pr, w, mY = fit(Y, X, tr); s, cs, gn = scores(Y, Pr, Y[tr].mean(0), ~tr)
        rows.append(dict(kind=f"ceiling {ob}", box=box, speaker=spk, model=f"other {ob}s, mean of {K}", n=int((~tr).sum()), a=w[0], b=np.nan) | s); H.setdefault((f"ceiling {ob}", box, "ceiling"), []).append((cs, gn))
T = pd.DataFrame(rows); T.to_csv(C / "additive_check.csv", index=False)
print(T.groupby(["kind", "model", "box"])[["a", "b", "R2", "cos", "gain", "top1"]].mean().round(2).to_string())
# figure: per box, histograms of direction (cos) and magnitude (gain) — cube+vase matched vs shuffled vs single-object ceiling
f, ax = plt.subplots(2, 3, figsize=(17, 8))
for j, box in enumerate(["plastic", "gastro", "cardboard"]):
    for key, col, lab in (((f"ceiling vase", box, "ceiling"), "0.5", "ceiling: vase from other vases"), (("red-cube+vase", box, f"matched mean of {K}"), "tab:green", f"cube+vase from singles (mean of {K})"),
                          (("red-cube+vase", box, "shuffled"), "tab:orange", "shuffled singles")):
        if key not in H: continue
        cs, gn = (np.concatenate(v) for v in zip(*H[key]))
        ax[0, j].hist(cs, bins=np.linspace(-1, 1, 41), color=col, alpha=.55, density=True, label=f"{lab} (median {np.median(cs):.2f})")
        ax[1, j].hist(np.clip(gn, -2, 6), bins=np.linspace(-2, 6, 41), color=col, alpha=.55, density=True, label=f"{lab} (median {np.median(gn):.2f})")
    ax[0, j].set_title(f"{box} · direction: cos(real − mean, pred − mean)", fontsize=10); ax[1, j].set_title(f"{box} · magnitude: gain of real along pred (1 = right size)", fontsize=10)
    ax[1, j].axvline(1, c="k", lw=.8, ls="--"); [a.legend(fontsize=7) for a in ax[:, j]]
plt.suptitle("held-out positions · per speaker · direction right ⇔ cos ≫ shuffled; magnitude right ⇔ gain ≈ 1 (>1 = prediction too short)", fontsize=11)
plt.tight_layout(); out = ROOT / "notebooks/out/137_additive_check.pdf"; plt.savefig(out); plt.savefig(sys.argv[1], dpi=60) if len(sys.argv) > 1 else None; print(out)
