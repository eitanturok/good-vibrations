# which (box, speaker, object) subset has a latent PC pair that looks like the 2-D position grid?
# per subset: PCA-10 on latent, per-position centroids, Procrustes R² (rotation/scale/flip) of every PC pair vs the true grid (mean mask CoM per position);
# null = same with positions shuffled (max over pairs, like the real score). -> runs/_src9_cache/nb137/<RUN>/mapfit.csv
import sys, itertools, numpy as np, pandas as pd
from pathlib import Path
RUN, IGNORE, NMIN, PMIN, NPERM = "decres-rescale-aug10-v1", ["eval/cardboard/vase-laser-shifted-4"], 100, 20, 200
C = Path(__file__).resolve().parents[1] / "runs/_src9_cache/nb137" / RUN
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
ok = ~df.split.isin(IGNORE).values & df.x.notna().values; df, Z = df[ok].reset_index(drop=True), Z[ok]
PAIRS = np.array(list(itertools.combinations(range(10), 2)))

def r2(G, Cp):  # G: (P,2) centred grid, Cp: (K,P,2) centred centroid pairs -> (K,) Procrustes R² (scale + rotation + reflection)
    s = np.linalg.svd(np.einsum("pi,kpj->kij", G, Cp), compute_uv=False).sum(-1)
    return s ** 2 / ((G ** 2).sum() * (Cp ** 2).sum((1, 2)))

def fit(m):
    X = Z[m] - Z[m].mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False); P = U[:, :10] * S[:10]; ev = S[:10] ** 2 / (S ** 2).sum()
    d = df[m].assign(**{f"pc{i}": P[:, i] for i in range(10)}).groupby("pos").mean(numeric_only=True)
    G = d[["x", "y"]].values; G = G - G.mean(0); Cn = d[[f"pc{i}" for i in range(10)]].values; Cn = Cn - Cn.mean(0)
    sc = r2(G, Cn[:, PAIRS].transpose(1, 0, 2)); b = sc.argmax()
    rng = np.random.default_rng(0); null = np.array([r2(G[rng.permutation(len(G))], Cn[:, PAIRS].transpose(1, 0, 2)).max() for _ in range(NPERM)])
    return dict(pc_a=PAIRS[b, 0] + 1, pc_b=PAIRS[b, 1] + 1, r2=sc[b], r2_pc12=sc[0], null_mean=null.mean(), null95=np.quantile(null, .95),
                ev_a=ev[PAIRS[b, 0]], ev_b=ev[PAIRS[b, 1]], npos=len(G))

rows = []
for box, spk, obj in itertools.product([None, *df.box.unique()], [None, *sorted(df.speaker.unique(), key=int)], [None, *df.object.unique()]):
    m = np.ones(len(df), bool)
    for c, v in (("box", box), ("speaker", spk), ("object", obj)):
        if v is not None: m &= (df[c] == v).values
    if m.sum() < NMIN or df.pos[m].nunique() < PMIN: continue
    rows.append(dict(box=box, speaker=spk, object=obj, n=int(m.sum())) | fit(m))
R = pd.DataFrame(rows); R["excess"] = R.r2 - R.null95
R.sort_values("excess", ascending=False).to_csv(C / "mapfit.csv", index=False)
print(len(R), "subsets"); print(R.sort_values("excess", ascending=False).head(15).round(3).to_string(index=False))
