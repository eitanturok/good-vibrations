# parameter-free versions of "pair = f(cube, vase)" — nothing is learned except averages (leave-one-out, so a pair never sees itself):
#   sum      pair_i ≈ cube_i + vase_i + c,        c = mean(pair) − mean(cube) − mean(vase)
#   avg      pair_i ≈ ½(cube_i + vase_i) + c,     c = mean(pair) − ½(mean(cube) + mean(vase))
#   analogy  pair_i ≈ pair_r + (cube_i − cube_r) + (vase_i − vase_r)   (word2vec-style, one reference pair r; averaged over r it equals "sum")
#   fit      a, b, c by least squares (previous model; even positions → odd) for comparison
# cube/vase = mean of the K nearest singles (same box & speaker). Metrics per (pair type, box, speaker): R², median cos of (pred − mean pair) vs
# (real − mean pair), top-1 retrieval among that group's pairs (word2vec's metric). -> runs/.../paramfree.csv; best example -> notebooks/out/137_analogy_best.pdf
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src9")); C = ROOT / "runs/_src9_cache/nb137/decres-rescale-aug10-v1"
from dataset import load_splits
PAIRS = {"red-cube+vase": [("red-cube", "red-cube0"), ("vase", "vase0")], "green-cube+red-cube": [("red-cube", "green-cube0"), ("red-cube", "red-cube0")]}
K = 5
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
dirs = list(load_splits()[0].keys()); df["dir"] = [dirs[i] for i in np.load(ROOT / "runs/_src9_cache/nb136/base.npz")["i"]]
df = df[~df.split.str.contains("shifted|unseen-speaker|lid")]
def com(f): m = np.load(f) > 0; r, c = np.nonzero(m); return (c.mean() + .5) * 64 / m.shape[1] - .5, (r.mean() + .5) * 64 / m.shape[0] - .5
loo = lambda X: (X.sum(0) - X) / (len(X) - 1)  # leave-one-out mean for every row
def metrics(Y, Pr):
    m = loo(Y); y, p = Y - m, Pr - m; cos = np.sum(y * p, 1) / np.linalg.norm(y, axis=1) / np.linalg.norm(p, axis=1)
    D = (p / np.linalg.norm(p, axis=1, keepdims=True)) @ ((Y - Y.mean(0)) / np.linalg.norm(Y - Y.mean(0), axis=1, keepdims=True)).T
    return dict(R2=1 - ((Y - Pr) ** 2).sum() / (y ** 2).sum(), cos=np.median(cos), top1=np.mean(D.argmax(1) == np.arange(len(D))))
rows, G = [], {}
for pair, parts in PAIRS.items():
    P = df[df.object == pair].copy()
    for j, (_, inst) in enumerate(parts): P[[f"x{j}", f"y{j}"]] = [com(Path(d) / f"image/smasks/{inst}.npy") for d in P.dir]
    for (box, spk), p in P.groupby(["box", "speaker"]):
        S = df[(df.box == box) & (df.speaker == spk)]
        NB = [[np.hypot(S[S.object == ob].x - r[f"x{j}"], S[S.object == ob].y - r[f"y{j}"]).nsmallest(K).index for _, r in p.iterrows()] for j, (ob, _) in enumerate(parts)]
        Y = Z[p.index]; A, B = (np.stack([Z[ix].mean(0) for ix in nb]) for nb in NB)
        tr = p.pos.values % 2 == 0; mA, mB, mY = A[tr].mean(0), B[tr].mean(0), Y[tr].mean(0)
        a, b = np.linalg.lstsq(np.c_[(A - mA)[tr].reshape(-1), (B - mB)[tr].reshape(-1)], (Y - mY)[tr].reshape(-1), rcond=None)[0]
        preds = {"sum (a=b=1)": A + B + loo(Y) - loo(A) - loo(B), "avg (a=b=½)": .5 * (A + B) + loo(Y) - .5 * (loo(A) + loo(B)),
                 "fit (learned a, b)": a * A + b * B + mY - a * mA - b * mB}
        for name, Pr in preds.items():
            s = metrics(Y[~tr], Pr[~tr]) if name.startswith("fit") else metrics(Y, Pr)
            rows.append(dict(pair=pair, box=box, speaker=spk, n=len(p), model=name, a=a if name.startswith("fit") else None, b=b if name.startswith("fit") else None) | s)
        G[(pair, box, spk)] = (p, NB, Y, A, B)
T = pd.DataFrame(rows); T.to_csv(C / "paramfree.csv", index=False)
print(T.groupby(["model", "pair", "box"])[["R2", "cos", "top1"]].mean().round(2).to_string())
# cherry-picked word2vec-style analogy: best (target i, reference r, singles among the K nearest for both) in the best "sum" group
best = T[T.model == "sum (a=b=1)"].sort_values("R2", ascending=False).iloc[0]; key = (best.pair, best.box, best.speaker); p, NB, Y, A, B = G[key]
sc = lambda v: np.linalg.norm(v)
cands = []
for i in range(len(p)):
    for r in range(len(p)):
        if r == i: continue
        base = Y[r] - A[r] - B[r]  # reference pair minus its own (mean-of-K) singles = that pair's interaction
        for u in NB[0][i]:
            for v in NB[1][i]: cands.append((sc(Y[i] - (base + Z[u] + Z[v])) / sc(Y[i] - Y.mean(0)), i, r, u, v))
e, i, r, u, v = min(cands); print(f"best analogy {key}: target pos {p.pos.iloc[i]}, reference pos {p.pos.iloc[r]}, relative miss {e:.2f}")
cu_r, va_r = NB[0][r][0], NB[1][r][0]  # reference singles shown = nearest single of each (the analogy used their mean of K)
vec = dict(ref=Y[r], dcube=Z[u] - A[r], dvase=Z[v] - B[r], real=Y[i]); vec["pred"] = vec["ref"] + vec["dcube"] + vec["dvase"]
W = np.linalg.svd(np.r_[Y, A, B], full_matrices=False)[2][:2].T; pj = lambda x: x @ W
O, t1, t2 = pj(vec["ref"]), pj(vec["ref"] + vec["dcube"]), pj(vec["pred"])
f = plt.figure(figsize=(17, 10)); gs = f.add_gridspec(2, 4, height_ratios=[2, 1]); ax = f.add_subplot(gs[0, 1:3])
arr = lambda s, e, c, ls="-", lw=2.4: ax.annotate("", e, s, arrowprops=dict(arrowstyle="-|>", color=c, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, mutation_scale=18))
arr((0, 0), O, "0.45"); arr(O, t1, "tab:red"); arr(t1, t2, "tab:blue"); arr((0, 0), pj(vec["real"]), "black", ls=(0, (4, 2)), lw=1.8)
ID = lambda j: Path(df.loc[j, "dir"]).name
for c_, ls, l in (("0.45", "-", f"reference pair (sample {ID(p.index[r])})"), ("tab:red", "-", f"+ (cube − reference cube)   cube {ID(u)}"), ("tab:blue", "-", f"+ (vase − reference vase)   vase {ID(v)}"),
                  ("black", "--", f"real pair (sample {ID(p.index[i])})")): ax.plot([], [], c=c_, ls=ls, lw=2.2, label=l)
ax.scatter(*pj(vec["real"]), s=230, c="k", zorder=5, label="real pair"); ax.scatter(*t2, s=260, marker="*", c="tab:green", zorder=5, label="analogy prediction")
ic = {"red-cube+vase": "cube_and_vase", "green-cube+red-cube": "two_cubes"}[best.pair]
def icon(xy, name, px=70, d=(0, 34), al=1.):
    im = Image.open(ROOT / f"assets/objects/individual/{name}.png"); im.thumbnail((px * 3, px * 3)); ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im), zoom=1 / 3, alpha=al), xy, xybox=d, boxcoords="offset points", frameon=False, zorder=6))
icon(O, ic, d=(-40, -30), al=.6); icon(pj(vec["real"]), ic, px=80, d=(-45, 35)); icon(t2, ic, px=60, d=(45, 35), al=.6)
pts = np.array([(0, 0), O, t1, t2, pj(vec["real"])]); lo, hi = pts.min(0), pts.max(0); rr = (hi - lo).max() * .62; ctr = (lo + hi) / 2
ax.set_xlim(ctr[0] - rr, ctr[0] + rr); ax.set_ylim(ctr[1] - rr, ctr[1] + rr); ax.set_aspect("equal"); ax.scatter(0, 0, s=40, c="k"); ax.annotate("origin", (0, 0), xytext=(6, 6), textcoords="offset points")
ax.legend(fontsize=8, loc="lower right"); ax.set_xlabel("PC1 (linear, uncentred)"); ax.set_ylabel("PC2")
ax.set_title(f"word2vec-style analogy, nothing learned · {best.pair} · {best.box} · speaker {best.speaker}\nreal ≈ reference pair + (cube − ref cube) + (vase − ref vase) · relative miss {e:.2f} (cherry-picked target, reference and singles)", fontsize=10)
for j, (s_, t) in enumerate([(p.index[r], "reference pair"), (cu_r, "reference cube (nearest)"), (va_r, "reference vase (nearest)"), (p.index[i], "real pair (target)")]):
    b_ = f.add_subplot(gs[1, j]); b_.imshow(Image.open(Path(df.loc[s_, "dir"]) / "image/02_cropped_overhead.png")); b_.set_axis_off(); b_.set_title(f"{t} (sample {ID(s_)})", fontsize=9)
for j, (s_, t) in enumerate([(u, "cube used"), (v, "vase used")]):
    b_ = f.add_subplot(gs[0, 0 if j == 0 else 3]); b_.imshow(Image.open(Path(df.loc[s_, "dir"]) / "image/02_cropped_overhead.png")); b_.set_axis_off(); b_.set_title(f"{t} (sample {ID(s_)})", fontsize=9)
plt.tight_layout(); out = ROOT / "notebooks/out/137_analogy_best.pdf"; plt.savefig(out); plt.savefig(sys.argv[1], dpi=60) if len(sys.argv) > 1 else None; print(out)
