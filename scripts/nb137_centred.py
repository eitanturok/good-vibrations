# unsupervised check: subtract each speaker's mean latent, then plain PCA and plain UMAP (2-D) on all speakers together (one box · object).
# score = pooled Procrustes R² vs true (x, y) over ALL speakers (one shared map) + per-speaker R² with that same map. -> notebooks/out/137_centred_<box>_<object>.png
import sys, itertools, colorsys, numpy as np, pandas as pd, umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
BOX, OBJ = sys.argv[1], sys.argv[2]
ROOT = Path(__file__).resolve().parents[1]; C = ROOT / "runs/_src9_cache/nb137/decres-rescale-aug10-v1"
HIDE = ["shifted", "unseen-speaker", "cylinder", "portable-charger", "biscoff", "coffee", "lid"]
RENAME = {"green-cube+red-cube": "two objects", "red-cube+vase": "two objects", "soap dispenser": "soap", "red-cube": "cube"}
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
ok = ~(df.split + " " + df.object).str.contains("|".join(HIDE)).values & df.x.notna().values; df["object"] = df.object.replace(RENAME)
m = ok & (df.box == BOX).values & (df.object == OBJ).values; d, X = df[m].reset_index(drop=True), Z[m].copy()
SP = sorted(d.speaker.unique(), key=int); sp = d.speaker.values; G = d[["x", "y"]].values
for k in SP: X[sp == k] -= X[sp == k].mean(0)
def fitmap(G, P):  # Procrustes (rotation/flip/scale/shift) of P onto G → R², mapped P
    Gc, Pc = G - G.mean(0), P - P.mean(0); U, S, Vt = np.linalg.svd(Pc.T @ Gc)
    return S.sum() ** 2 / ((Gc ** 2).sum() * (Pc ** 2).sum()), S.sum() / (Pc ** 2).sum() * Pc @ (U @ Vt) + G.mean(0)
r2 = lambda y, p: 1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()
P10 = PCA(10, random_state=0).fit(X); pcs = P10.transform(X)
sc = {p: fitmap(G, pcs[:, list(p)])[0] for p in itertools.combinations(range(10), 2)}; bp = max(sc, key=sc.get)
emb = {f"PCA PC{bp[0] + 1}×PC{bp[1] + 1}": pcs[:, list(bp)], "PCA PC1×PC2": pcs[:, :2]}
P50 = PCA(50, random_state=0).fit_transform(X)
for nn, md in ((30, .1), (50, .5)): emb[f"UMAP nn{nn} md{md}"] = umap.UMAP(n_neighbors=nn, min_dist=md, random_state=0).fit_transform(P50)
print(f"{BOX} · {OBJ}: {len(d)} samples, speakers {SP}; PCA EV top10 {np.round(P10.explained_variance_ratio_, 3)}")
MARK = {"1": "o", "3": "^", "5": "s", "7": "x"}  # marker per speaker in the all-speakers panel
U = (G - G.min(0)) / (G.max(0) - G.min(0)); col = np.array([colorsys.hls_to_rgb(.8 * a, .8 - .55 * b, .9) for a, b in U])
f, ax = plt.subplots(len(emb), len(SP) + 1, figsize=(3.2 * (len(SP) + 1) + 2, 3.2 * len(emb)), gridspec_kw=dict(width_ratios=[2] + [1] * len(SP)))
for i, (name, E) in enumerate(emb.items()):
    R, M = fitmap(G, E); per = {k: r2(G[sp == k], M[sp == k]) for k in SP}
    print(f"{name:22s} shared-map R² {R:.2f} · per speaker (same map) " + " ".join(f"{k}:{v:.2f}" for k, v in per.items()))
    for k in SP: ax[i, 0].scatter(*E[sp == k].T, c=col[sp == k], s=14, marker=MARK.get(k, "o"), label=f"speaker {k}", linewidths=.6)
    ax[i, 0].legend(fontsize=7, markerscale=1.2, handletextpad=.2, loc="best").legend_handles and [h.set_color("0.4") for h in ax[i, 0].get_legend().legend_handles]
    ax[i, 0].set_title(f"{name}\nall speakers · R² {R:.2f}", fontsize=9)
    for j, k in enumerate(SP):
        s = sp == k; ax[i, j + 1].scatter(*E[~s].T, c="0.88", s=3); ax[i, j + 1].scatter(*E[s].T, c=col[s], s=7); ax[i, j + 1].set_title(f"speaker {k} · {per[k]:.2f}", fontsize=9)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); out = ROOT / f"notebooks/out/137_centred_{BOX}_{OBJ}.pdf"; plt.savefig(out); print(out)
