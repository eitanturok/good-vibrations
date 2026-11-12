# best (box, speaker, object) for UMAP: same filters / UMAP params / cache files as nb137 (results load instantly there), latent only.
# score = best Procrustes R² over the 3 UMAP pairs vs true mask CoM, on shown (non-HIDE) single-object rows. -> runs/_src9_cache/nb137/<RUN>/umap_combos.csv
import hashlib, itertools, time, numpy as np, pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
RUN = "decres-rescale-aug10-v1"
IGNORE = ["eval/cardboard/vase-laser-shifted-4", "shifted", "unseen-speaker", "cylinder", "portable-charger", "biscoff", "coffee", 'lid']
HIDE = ["shifted", "unseen-speaker", "cylinder", "portable-charger", "biscoff", "coffee", 'lid']
UMAP = dict(n_neighbors=30, min_dist=0.1)
RENAME = {"green-cube+red-cube": "two objects", "red-cube+vase": "two objects", "soap dispenser": "soap", "red-cube": "cube"}
C = Path(__file__).resolve().parents[1] / "runs/_src9_cache/nb137" / RUN
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
keep = ~df.split.isin(IGNORE).values; df, Z = df[keep].reset_index(drop=True), Z[keep]
hidden = (df.split + " " + df.object).str.contains("|".join(HIDE)).values; df["object"] = df.object.replace(RENAME)
score_ok = ~hidden & df.x.notna().values & ~df.object.isin(["empty", "two objects"]).values
def r2(G, P): G, P = G - G.mean(0), P - P.mean(0); return np.linalg.svd(G.T @ P, compute_uv=False).sum() ** 2 / ((G ** 2).sum() * (P ** 2).sum())
def run(box, spk, obj):
    import umap
    from sklearn.decomposition import PCA
    m = (df.box == box).values
    if spk: m = m & (df.speaker == spk).values
    if obj: m = m & (df.object == obj).values
    s = score_ok[m]
    if s.sum() < 100: return None
    key = ("umap", sorted(IGNORE), box, [spk] if spk else None, [obj] if obj else None, "latent") + tuple(sorted(UMAP.items()))
    f = C / f"red_{hashlib.sha1(repr(key).encode()).hexdigest()[:10]}.npz"
    if not f.exists():
        X = Z[m] - Z[m].mean(0)
        r = umap.UMAP(n_components=3, **(UMAP | dict(n_neighbors=min(UMAP["n_neighbors"], len(X) - 1))), random_state=0).fit_transform(PCA(min(50, *X.shape), random_state=0).fit_transform(X))
        np.savez(f, r=r, ev=np.full(3, np.nan))
    r = np.load(f)["r"][s]; G = df[m][["x", "y"]].values[s]
    sc = {p: r2(G, r[:, list(p)]) for p in itertools.combinations(range(3), 2)}; b = max(sc, key=sc.get)
    return dict(box=box, speaker=spk or "all", object=obj or "all", n=int(s.sum()), axes=f"UMAP {b[0] + 1} × UMAP {b[1] + 1}", r2=round(sc[b], 3))
objs = [None, "cube", "soap", "candle", "vase"]
R = pd.DataFrame([x for x in Parallel(n_jobs=8)(delayed(run)(b, s, o) for b in ["gastro", "plastic", "cardboard"] for s in [None, "1", "3", "5", "7"] for o in objs) if x])
R = R.sort_values("r2", ascending=False); R.to_csv(C / "umap_combos.csv", index=False)
for b in ["gastro", "plastic", "cardboard"]: print(R[R.box == b].head(8).to_string(index=False), "\n")
