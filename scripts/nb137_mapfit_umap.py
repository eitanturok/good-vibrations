# UMAP version of the map-fit search over (box, speaker): latent → 50-d PCA → UMAP-3 (same params + cache files as nb137 reduce), then Procrustes R²
# of each of the 3 UMAP pairs vs the true mask CoM; null95 = shuffled positions. -> runs/_src9_cache/nb137/<RUN>/mapfit_umap.csv
import hashlib, itertools, time, numpy as np, pandas as pd, umap
from pathlib import Path
from sklearn.decomposition import PCA
RUN, IGNORE = "decres-rescale-aug10-v1", ["eval/cardboard/vase-laser-shifted-4"]
C = Path(__file__).resolve().parents[1] / "runs/_src9_cache/nb137" / RUN
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
keep = ~df.split.isin(IGNORE).values; df, Z = df[keep].reset_index(drop=True), Z[keep]
def r2(G, P): G, P = G - G.mean(0), P - P.mean(0); return np.linalg.svd(G.T @ P, compute_uv=False).sum() ** 2 / ((G ** 2).sum() * (P ** 2).sum())
rows = []
for box, spk in itertools.product(["gastro", "plastic", "cardboard"], [None, "1", "3", "4", "5", "7"]):
    m = (df.box == box).values & (True if spk is None else (df.speaker == spk).values)
    if m.sum() < 100: continue
    key = ("umap", sorted(IGNORE), box, None if spk is None else [spk], None, "latent")  # identical to nb137 reduce() → shared cache
    f = C / f"red_{hashlib.sha1(repr(key).encode()).hexdigest()[:10]}.npz"
    if not f.exists():
        t0 = time.time(); X = Z[m] - Z[m].mean(0)
        r = umap.UMAP(n_components=3, n_neighbors=min(30, len(X) - 1), min_dist=0.1, random_state=0).fit_transform(PCA(min(50, *X.shape), random_state=0).fit_transform(X))
        np.savez(f, r=r, ev=np.full(3, np.nan)); print(box, spk, f"{time.time() - t0:.0f}s", flush=True)
    r = np.load(f)["r"]; ok = df.x[m].notna().values; G, r = df[m][["x", "y"]].values[ok], r[ok]
    sc = {p: r2(G, r[:, list(p)]) for p in itertools.combinations(range(3), 2)}; b = max(sc, key=sc.get)
    rng = np.random.default_rng(0); null = [max(r2(G[rng.permutation(len(G))], r[:, list(p)]) for p in sc) for _ in range(200)]
    rows.append(dict(box=box, speaker=spk or "all", n=int(ok.sum()), axes=f"umap{b[0] + 1}×umap{b[1] + 1}", r2=sc[b], null95=np.quantile(null, .95)))
R = pd.DataFrame(rows).sort_values("r2", ascending=False); R.to_csv(C / "mapfit_umap.csv", index=False); print(R.round(3).to_string(index=False))
