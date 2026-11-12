# UMAP param sweep for "does the latent look like the 2-D grid": 2-D UMAP of the latent for one speaker per box, Procrustes R² vs true mask CoM.
# grid: pre (PCA-50 vs raw 512-d) × n_neighbors × min_dist × metric × init. Fit on IGNORE-filtered rows, scored on shown (non-HIDE) rows with a mask.
import itertools, os, time, numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
RUN, IGNORE = "decres-rescale-aug10-v1", ["eval/cardboard/vase-laser-shifted-4"]
HIDE = ["shifted", "unseen-speaker", "cylinder", "portable-charger", "biscoff", "coffee"]
C = Path(__file__).resolve().parents[1] / "runs/_src9_cache/nb137" / RUN
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
keep = ~df.split.isin(IGNORE).values; df, Z = df[keep].reset_index(drop=True), Z[keep]
show = ~(df.split + " " + df.object).str.contains("|".join(HIDE)).values & df.x.notna().values
SUBS = [("plastic", "1"), ("gastro", "1"), ("cardboard", "1")]
GRID = list(itertools.product(["pca50", "raw"], [15, 30, 50, 100, 200], [0.1, 0.5], ["euclidean", "cosine"], ["spectral", "pca"]))
def r2(G, P): G, P = G - G.mean(0), P - P.mean(0); return np.linalg.svd(G.T @ P, compute_uv=False).sum() ** 2 / ((G ** 2).sum() * (P ** 2).sum())
def run(box, spk, pre, nn, md, metric, init):
    import umap
    from sklearn.decomposition import PCA
    m = (df.box == box).values & (df.speaker == spk).values; X = Z[m] - Z[m].mean(0)
    if pre == "pca50": X = PCA(50, random_state=0).fit_transform(X)
    t0 = time.time(); e = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=md, metric=metric, init=init, random_state=0).fit_transform(X)
    s = show[m]; return dict(box=box, speaker=spk, pre=pre, n_neighbors=nn, min_dist=md, metric=metric, init=init, r2=r2(df[m][["x", "y"]].values[s], e[s]), sec=time.time() - t0)
R = pd.DataFrame(Parallel(n_jobs=8)(delayed(run)(b, s, *g) for b, s in SUBS for g in GRID))
R.to_csv(C / "umap_sweep.csv", index=False)
for b in R.box.unique(): print(R[R.box == b].sort_values("r2", ascending=False).head(6).round(3).to_string(index=False))
P = ["pre", "n_neighbors", "min_dist", "metric", "init"]
for p in P: print(R.pivot_table("r2", p, "box").round(3))
print(R.groupby(P).r2.mean().sort_values(ascending=False).head(8).round(3))
