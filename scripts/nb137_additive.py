# is a two-object latent = pair offset + linear combination of the two single-object latents at the same positions?
# per pair sample: each object's CoM from its instance mask; nearest clean single of that object (same box & speaker) by CoM.
# model per (box, speaker), everything centred on its group mean: Y − Ȳ ≈ α(A − Ā) + β(B − B̄); α, β fit on even positions, scored on odd.
# usage: nb137_additive.py <pair object> (red-cube+vase | green-cube+red-cube) [png for a quick look]
#   -> runs/_src9_cache/nb137/<RUN>/additive_<pair>.csv, notebooks/out/137_additive_<pair>.pdf (UMAP + best single example with photos)
import sys, numpy as np, pandas as pd, umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src9")); C = ROOT / "runs/_src9_cache/nb137/decres-rescale-aug10-v1"
from dataset import load_splits
PAIR = sys.argv[1] if len(sys.argv) > 1 else "red-cube+vase"
PARTS = {"red-cube+vase": [("red-cube", "red-cube0"), ("vase", "vase0")],
         "green-cube+red-cube": [("red-cube", "green-cube0"), ("red-cube", "red-cube0")]}[PAIR]  # (single object to match, instance mask); no single green cube exists → red cube stands in
BAD = "shifted|unseen-speaker|lid"  # splits whose latents are not comparable (moved lasers, unseen speaker, lids)
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
dirs = list(load_splits()[0].keys()); df["dir"] = [dirs[i] for i in np.load(ROOT / "runs/_src9_cache/nb136/base.npz")["i"]]  # row ↔ dir alignment verified (speaker + position match 100%)
df = df[~df.split.str.contains(BAD)]
def com(f):  # CoM of a full-res instance mask in the 64×64 mask frame used for df.x / df.y (pixel centres)
    m = np.load(f) > 0; r, c = np.nonzero(m); return (c.mean() + .5) * 64 / m.shape[1] - .5, (r.mean() + .5) * 64 / m.shape[0] - .5
P = df[df.object == PAIR].copy()
for j, (_, inst) in enumerate(PARTS): P[[f"x{j}", f"y{j}"]] = [com(Path(d) / f"image/smasks/{inst}.npy") for d in P.dir]
rows, rng = [], np.random.default_rng(0)
for (box, spk), p in P.groupby(["box", "speaker"]):
    S = df[(df.box == box) & (df.speaker == spk)]
    for i, r in p.iterrows():
        row = dict(i=i, box=box, speaker=spk, pos=r.pos)
        for j, (ob, _) in enumerate(PARTS):
            s = S[S.object == ob]; dist = np.hypot(s.x - r[f"x{j}"], s.y - r[f"y{j}"])
            row |= {f"m{j}": dist.idxmin(), f"d{j}": dist.min(), f"s{j}": rng.choice(s.index)}  # matched single, its CoM distance, random partner
        rows.append(row)
R = pd.DataFrame(rows)
cent = lambda X: X - X.mean(0)
def r2(Y, Yh): return 1 - ((Y - Yh) ** 2).sum() / (Y ** 2).sum()  # Y already centred → baseline = predicting the group mean
def rank(Y, Q):  # within a group: rank percentile of each prediction's own pair among all pairs (cosine; 0 = own pair closest, .5 = chance)
    A, B = Y / np.linalg.norm(Y, axis=1, keepdims=True), Q / np.linalg.norm(Q, axis=1, keepdims=True); D = B @ A.T
    return (D > np.diag(D)[:, None]).sum(1) / (len(D) - 1)
out = []
for box, g in R.groupby("box"):
    Y, A, B, As, Bs, tr = [], [], [], [], [], []
    for _, h in g.groupby("speaker"):  # centre per speaker group (removes the speaker AND the pair/single offsets)
        Y.append(cent(Z[h.i])); A.append(cent(Z[h.m0])); B.append(cent(Z[h.m1])); As.append(cent(Z[h.s0])); Bs.append(cent(Z[h.s1])); tr.append(h.pos.values % 2 == 0)
    Y, A, B, As, Bs, tr = map(np.concatenate, (Y, A, B, As, Bs, tr))
    w = np.linalg.lstsq(np.c_[A[tr].reshape(-1), B[tr].reshape(-1)], Y[tr].reshape(-1), rcond=None)[0]; te = ~tr
    one = lambda X: np.linalg.lstsq(X[tr].reshape(-1, 1), Y[tr].reshape(-1), rcond=None)[0][0]  # best single scale for one object
    preds = {"avg ½A+½B": .5 * (A + B), "fit αA+βB": w[0] * A + w[1] * B, "A only": one(A) * A, "B only": one(B) * B, "shuffled ½A+½B": .5 * (As + Bs)}
    o = dict(box=box, n=len(g), test_n=int(te.sum()), med_match_px=np.median(np.r_[g.d0, g.d1]), max_match_px=np.max(np.r_[g.d0, g.d1]), alpha=w[0], beta=w[1])
    for k, Q in preds.items():
        rk = np.concatenate([rank(Y[te & (g.speaker.values == s)], Q[te & (g.speaker.values == s)]) for s in g.speaker.unique()])
        o |= {f"R2 {k}": r2(Y[te], Q[te]), f"rank {k}": np.median(rk), f"top1 {k}": np.mean(rk == 0)}
    out.append(o)
O = pd.DataFrame(out).set_index("box"); O.to_csv(C / f"additive_{PAIR}.csv"); print(PAIR); print(O.round(2).T.to_string())
# figure: best box; UMAP of that box & best speaker (raw latents): singles, pairs, predictions Ŷ = Ȳ + α(A − Ā) + β(B − B̄); one highlighted example + photos
box = O["top1 fit αA+βB"].idxmax(); o = O.loc[box]; g = R[R.box == box]
spk = g.groupby("speaker").size().idxmax(); h = g[g.speaker == spk].reset_index(drop=True)
Y, A, B = Z[h.i], Z[h.m0], Z[h.m1]; Yh = Y.mean(0) + o.alpha * cent(A) + o.beta * cent(B)
cos = np.sum(cent(Y) * (Yh - Y.mean(0)), 1) / np.linalg.norm(cent(Y), axis=1) / np.linalg.norm(Yh - Y.mean(0), axis=1)
k = int(np.argmax(np.where(h.pos.values % 2 == 1, cos, -9)))  # best held-out example
S = df[(df.box == box) & (df.speaker == spk) & df.object.isin([ob for ob, _ in PARTS] + [PAIR])]
um = umap.UMAP(n_neighbors=30, min_dist=.3, random_state=0).fit(Z[S.index]); E = um.embedding_; Ep = um.transform(Yh); at = {v: n for n, v in enumerate(S.index)}
f = plt.figure(figsize=(16, 9)); gs = f.add_gridspec(2, 4, height_ratios=[2.2, 1])
a = f.add_subplot(gs[0, :]); cl = {"red-cube": "tab:red", "vase": "tab:blue"}
for ob in dict.fromkeys(ob for ob, _ in PARTS): s = (S.object == ob).values; a.scatter(*E[s].T, s=6, c=cl[ob], alpha=.3, label=f"single {ob}")
Ey = E[[at[v] for v in h.i]]; a.scatter(*Ey.T, s=14, c="k", label=f"{PAIR} (real)"); a.scatter(*Ep.T, s=18, facecolors="none", edgecolors="tab:green", label="prediction Ȳ + α(A−Ā) + β(B−B̄)")
for p, q in zip(Ey, Ep): a.plot(*np.c_[p, q], c="0.6", lw=.4)
ex = [E[at[h.m0[k]]], E[at[h.m1[k]]], Ey[k], Ep[k]]
for (x, y), lab, mk in zip(ex, ["single A", "single B", "real pair", "prediction"], ["^", "s", "o", "*"]):
    a.scatter(x, y, s=260, marker=mk, facecolors="none", edgecolors="gold", linewidths=2.5); a.annotate(lab, (x, y), xytext=(8, 8), textcoords="offset points", fontsize=10, weight="bold")
a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=9, loc="best")
a.set_title(f"{PAIR} · {box} · speaker {spk} · UMAP of raw latents · α {o.alpha:.2f} β {o.beta:.2f} · held-out R² {o['R2 fit αA+βB']:.2f} · own pair ranked 1st {o['top1 fit αA+βB']:.0%} (shuffled {o['top1 shuffled ½A+½B']:.0%})", fontsize=10)
for j, (v, t) in enumerate([(h.m0[k], f"single A ({PARTS[0][0]})"), (h.m1[k], f"single B ({PARTS[1][0]})"), (h.i[k], f"real {PAIR}")]):
    b = f.add_subplot(gs[1, j]); b.imshow(Image.open(Path(df.loc[v, "dir"]) / "image/02_cropped_overhead.png")); b.set_axis_off(); b.set_title(f"{t} · pos {df.loc[v, 'pos']}", fontsize=9)
b = f.add_subplot(gs[1, 3]); b.set_axis_off()
b.text(0, .5, f"best held-out example\ncos(real − Ȳ, pred − Ȳ) = {cos[k]:.2f}\nCoM match: A {h.d0[k]:.1f} px, B {h.d1[k]:.1f} px", fontsize=11, va="center")
plt.tight_layout(); out = ROOT / f"notebooks/out/137_additive_{PAIR}.pdf"; plt.savefig(out); plt.savefig(sys.argv[2], dpi=60) if len(sys.argv) > 2 else None; print(out)
# linear view: PCA of the centred real pair latents (held-out positions, all speakers of the best box); real vs predicted α(A−Ā)+β(B−B̄) and shuffled partners
Yc, Pc, Sc, te, kk = [], [], [], [], []
for s, hh in g.groupby("speaker"):
    Yc.append(cent(Z[hh.i])); Pc.append(o.alpha * cent(Z[hh.m0]) + o.beta * cent(Z[hh.m1])); Sc.append(o.alpha * cent(Z[hh.s0]) + o.beta * cent(Z[hh.s1]))
    te.append(hh.pos.values % 2 == 1)
Yc, Pc, Sc, te = map(np.concatenate, (Yc, Pc, Sc, te))
ex = np.where(np.concatenate([g[g.speaker == s].i.values for s, _ in g.groupby("speaker")]) == h.i[k])[0][0]  # the highlighted example
from sklearn.decomposition import PCA
pca = PCA(4).fit(Yc[te]); Ry, Rp, Rs = pca.transform(Yc), pca.transform(Pc), pca.transform(Sc)
f, ax = plt.subplots(1, 4, figsize=(20, 5.2))
for a, (Q, name, c) in zip(ax[:2], [(Rp, "prediction α(A−Ā)+β(B−B̄)", "tab:green"), (Rs, "shuffled partners", "tab:orange")]):
    a.scatter(*Ry[te, :2].T, s=12, c="k", label="real pair"); a.scatter(*Q[te, :2].T, s=14, facecolors="none", edgecolors=c, label=name)
    for p, q in zip(Ry[te, :2], Q[te, :2]): a.plot(*np.c_[p, q], c="0.7", lw=.4)
    a.scatter(*Ry[ex, :2], s=250, facecolors="none", edgecolors="gold", lw=2.5); a.scatter(*Q[ex, :2], s=250, marker="*", facecolors="none", edgecolors="gold", lw=2)
    a.set_xlabel(f"PC1 of real pairs ({pca.explained_variance_ratio_[0]:.0%})"); a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})"); a.legend(fontsize=8); a.set_title(name, fontsize=10)
for a, j in zip(ax[2:], (0, 1)):
    for Q, c, lab in ((Rs, "tab:orange", "shuffled"), (Rp, "tab:green", "prediction")):
        r = np.corrcoef(Ry[te, j], Q[te, j])[0, 1]; a.scatter(Ry[te, j], Q[te, j], s=8, c=c, alpha=.6, label=f"{lab} r {r:.2f}")
    a.axline((0, 0), slope=1, c="0.5", lw=.8, ls="--"); a.set_xlabel(f"real PC{j + 1}"); a.set_ylabel(f"predicted PC{j + 1}"); a.legend(fontsize=8); a.set_title(f"PC{j + 1}: real vs predicted", fontsize=10)
plt.suptitle(f"{PAIR} · {box} · all speakers, each centred · held-out positions only · gold = example from the other figure", fontsize=11)
plt.tight_layout(); out = ROOT / f"notebooks/out/137_additive_pc_{PAIR}.pdf"; plt.savefig(out); plt.savefig(sys.argv[2].replace(".png", "_pc.png"), dpi=60) if len(sys.argv) > 2 else None; print(out)
