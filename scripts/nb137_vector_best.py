# best additive example as vectors: pair ≈ a·cube + b·vase + c (cube / vase = mean of the K nearest singles, a, b, c fit on even positions of
# this box & speaker); held-out sample whose prediction is closest to its real pair. 2-D projection = top-2 PCs of this group's latents applied
# WITHOUT subtracting the mean (linear), so tip-to-tail sums in the plot are exactly the sums in latent space.
# usage: nb137_vector_best.py [pair] [box] [speaker] [png]  -> notebooks/out/137_vector_best_<pair>_<box>_<spk>.pdf
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src9")); C = ROOT / "runs/_src9_cache/nb137/decres-rescale-aug10-v1"
from dataset import load_splits
PAIR, BOX, SPK = (sys.argv[1:4] + ["red-cube+vase", "gastro", "1"][len(sys.argv[1:4]):]) if len(sys.argv) > 1 else ("red-cube+vase", "gastro", "1")
PARTS = {"red-cube+vase": [("red-cube", "red-cube0"), ("vase", "vase0")], "green-cube+red-cube": [("red-cube", "green-cube0"), ("red-cube", "red-cube0")]}[PAIR]
K = 5
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
dirs = list(load_splits()[0].keys()); df["dir"] = [dirs[i] for i in np.load(ROOT / "runs/_src9_cache/nb136/base.npz")["i"]]
df = df[~df.split.str.contains("shifted|unseen-speaker|lid") & (df.box == BOX) & (df.speaker == SPK)]
def com(f): m = np.load(f) > 0; r, c = np.nonzero(m); return (c.mean() + .5) * 64 / m.shape[1] - .5, (r.mean() + .5) * 64 / m.shape[0] - .5
P = df[df.object == PAIR].copy()
for j, (_, inst) in enumerate(PARTS): P[[f"x{j}", f"y{j}"]] = [com(Path(d) / f"image/smasks/{inst}.npy") for d in P.dir]
NB = [[np.hypot(df[df.object == ob].x - r[f"x{j}"], df[df.object == ob].y - r[f"y{j}"]).nsmallest(K).index for _, r in P.iterrows()] for j, (ob, _) in enumerate(PARTS)]
Y = Z[P.index]; A, B = (np.stack([Z[ix].mean(0) for ix in nb]) for nb in NB); tr = P.pos.values % 2 == 0
mA, mB, mY = A[tr].mean(0), B[tr].mean(0), Y[tr].mean(0)
a, b = np.linalg.lstsq(np.c_[(A - mA)[tr].reshape(-1), (B - mB)[tr].reshape(-1)], (Y - mY)[tr].reshape(-1), rcond=None)[0]
c = mY - a * mA - b * mB; pred = a * A + b * B + c                      # c = average pair interaction (one vector)
err = np.linalg.norm(Y - pred, axis=1) / np.linalg.norm(Y - mY, axis=1)  # miss relative to how far the pair is from the average pair
te = np.where(~tr)[0]; k = te[np.argmin(err[te])]
PICK = True  # cherry-pick for illustration: a, b, c stay from the mean-of-K fit; the drawn cube & vase are the ONE single of each (among the K nearest)
if PICK:     # whose a·cube + b·vase + c lands closest to the real pair, and the example is the held-out pair where that works best
    best = {i: min(((np.linalg.norm(Y[i] - a * Z[u] - b * Z[v] - c) / np.linalg.norm(Y[i] - mY), u, v) for u in NB[0][i] for v in NB[1][i])) for i in te}
    k = min(best, key=lambda i: best[i][0]); _, cu, va = best[k]
    A[k], B[k] = Z[cu], Z[va]; pred[k] = a * A[k] + b * B[k] + c; err[k] = best[k][0]
else: cu, va = NB[0][k][0], NB[1][k][0]
SRC = f"single, best of {K} nearest (cherry-picked)" if PICK else f"single, mean of {K}"
print(f"{PAIR} · {BOX} · speaker {SPK}: a {a:.2f} b {b:.2f}; held-out R² {1 - ((Y - pred)[te] ** 2).sum() / ((Y - mY)[te] ** 2).sum():.2f}; best example pos {P.pos.iloc[k]} relative miss {err[k]:.2f}")
W = np.linalg.svd(np.r_[A, B, Y], full_matrices=False)[2][:2].T; p = lambda v: v @ W   # linear 2-D projection (no centring)
V = dict(cube=p(A[k]), vase=p(B[k]), inter=p(c), real=p(Y[k]), pred=p(pred[k]))
CL = dict(cube="tab:red", vase="tab:blue", inter="tab:purple", real="black", pred="tab:green")
f = plt.figure(figsize=(21, 11)); gs = f.add_gridspec(2, 3, height_ratios=[2, 1])
arrow = lambda ax, s, e, col, ls="-", lw=2.4: ax.annotate("", e, s, arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, mutation_scale=18))
def fitlim(ax, pts, m=.12):  # annotate() arrows don't expand the axes, so set limits from every endpoint (equal scale)
    pts = np.array(pts); lo, hi = pts.min(0), pts.max(0); r = (hi - lo).max() * (.5 + m); ctr = (lo + hi) / 2
    ax.set_xlim(ctr[0] - r, ctr[0] + r); ax.set_ylim(ctr[1] - r, ctr[1] + r); ax.set_aspect("equal")
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # object pictures (assets/objects/individual) at the arrow tips
ICON = {"red-cube": "cube", "vase": "vase", "red-cube+vase": "cube_and_vase", "green-cube+red-cube": "two_cubes"}
def icon(ax, xy, name, px=70, dxy=(0, 34), alpha=1.):
    im = Image.open(ROOT / f"assets/objects/individual/{ICON[name]}.png"); im.thumbnail((px * 3, px * 3))
    ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im), zoom=1 / 3, alpha=alpha), xy, xybox=dxy, boxcoords="offset points", frameon=False, zorder=6))
SID = {key: Path(df.loc[v, "dir"]).name for key, v in (("cube", cu), ("vase", va), ("real", P.index[k]))}; print("sample ids", SID)
t1, t2 = p(a * A[k]), p(a * A[k] + b * B[k]); lim = [(0, 0), t1, t2, *V.values()]
ax1, axS, ax2 = f.add_subplot(gs[0, 0]), f.add_subplot(gs[0, 1]), f.add_subplot(gs[0, 2])
# (1) the vectors, each from the origin
for key, lab in (("cube", f"cube ({SRC}) (sample {SID['cube']})"), ("vase", f"vase ({SRC}) (sample {SID['vase']})"), ("inter", "c = average pair interaction"), ("real", f"real {PAIR} (sample {SID['real']})")):
    arrow(ax1, (0, 0), V[key], CL[key]); ax1.plot([], [], c=CL[key], lw=2.4, label=lab)
icon(ax1, V["cube"], PARTS[0][0], dxy=(10, -40)); icon(ax1, V["vase"], PARTS[1][0], dxy=(-10, 38)); icon(ax1, V["real"], PAIR, px=80, dxy=(-45, 30))
ax1.set_title("1 · the vectors (each from the origin)", fontsize=11)
# (2) the sum, tip-to-tail, vs the real pair
arrow(axS, (0, 0), V["real"], CL["real"], lw=1.8, ls=(0, (4, 2))); axS.plot([], [], c=CL["real"], lw=1.8, ls="--", label=f"real {PAIR} (sample {SID['real']})")
arrow(axS, (0, 0), t1, CL["cube"]); arrow(axS, t1, t2, CL["vase"]); arrow(axS, t2, V["pred"], CL["inter"])
for c_, l in ((CL["cube"], f"a·cube (a = {a:.2f})"), (CL["vase"], f"+ b·vase (b = {b:.2f})"), (CL["inter"], "+ c")): axS.plot([], [], c=c_, lw=2.4, label=l)
for key, mk, lab in (("real", "o", "real pair"), ("pred", "*", "predicted pair = a·cube + b·vase + c")): axS.scatter(*V[key], s=230, marker=mk, c=CL[key], zorder=5, label=lab)
icon(axS, V["real"], PAIR, px=80, dxy=(-45, 30)); icon(axS, V["pred"], PAIR, px=60, dxy=(40, 30), alpha=.55)
axS.set_title("2 · tip-to-tail: a·cube + b·vase + c ≈ real pair", fontsize=11)
for ax in (ax1, axS):
    ax.scatter(0, 0, s=40, c="k"); ax.annotate("origin", (0, 0), xytext=(6, 6), textcoords="offset points", fontsize=9)
    fitlim(ax, lim); ax.axhline(0, c="0.9", lw=.6); ax.axvline(0, c="0.9", lw=.6); ax.legend(fontsize=7.5, loc="lower right"); ax.set_xlabel("PC1 (linear, uncentred)"); ax.set_ylabel("PC2")
f.suptitle(f"{PAIR} · {BOX} · speaker {SPK} · held-out position {P.pos.iloc[k]} · relative miss {err[k]:.2f}{' (cherry-picked singles)' if PICK else ''} (0 = exact, 1 = no better than the average pair)", fontsize=12)
# zoom: deviations from the average pair — real vs predicted (this is where position lives)
for key, lab in (("real", "real − average pair"), ("pred", "predicted − average pair")): arrow(ax2, (0, 0), p(Y[k] if key == "real" else pred[k]) - p(mY), CL[key]); ax2.plot([], [], c=CL[key], lw=2.4, label=lab)
others = np.array([p(Y[i]) - p(mY) for i in te]); ax2.scatter(*others.T, s=10, c="0.75", label="other held-out pairs (real − avg)")
fitlim(ax2, [(0, 0), p(Y[k]) - p(mY), p(pred[k]) - p(mY), *others]); ax2.axhline(0, c="0.9", lw=.6); ax2.axvline(0, c="0.9", lw=.6); ax2.legend(fontsize=8)
cosk = np.dot(Y[k] - mY, pred[k] - mY) / np.linalg.norm(Y[k] - mY) / np.linalg.norm(pred[k] - mY)
ax2.set_title(f"3 · zoom: deviation from the average pair\ncos(real, pred) in full 512-d = {cosk:.2f}", fontsize=10)
NBk = [cu, va, P.index[k]]
for j, (v, t) in enumerate(zip(NBk, [f"single {PARTS[0][0]} ({'drawn' if PICK else f'nearest of {K}'}) (sample {SID['cube']})", f"single {PARTS[1][0]} ({'drawn' if PICK else f'nearest of {K}'}) (sample {SID['vase']})", f"real {PAIR} (sample {SID['real']})"])):
    b_ = f.add_subplot(gs[1, j]); b_.imshow(Image.open(Path(df.loc[v, "dir"]) / "image/02_cropped_overhead.png")); b_.set_axis_off(); b_.set_title(t, fontsize=9)
plt.tight_layout(); out = ROOT / f"notebooks/out/137_vector_best_{PAIR}_{BOX}_{SPK}.pdf"; plt.savefig(out); plt.savefig(sys.argv[4], dpi=60) if len(sys.argv) > 4 else None; print(out)
