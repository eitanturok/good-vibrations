# stacked-grid figure: speaker-centred latent → one joint 2-D UMAP (nn50, md0.5), each speaker drawn as its own layer at height z = speaker.
# colour = true position (hue = x, light = y), marker = speaker; grey verticals join the SAME physical position across speakers (a few).
import sys, colorsys, numpy as np, pandas as pd, umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
BOX, OBJ = sys.argv[1], sys.argv[2]
ROOT = Path(__file__).resolve().parents[1]; C = ROOT / "runs/_src9_cache/nb137/decres-rescale-aug10-v1"
HIDE = ["shifted", "unseen-speaker", "cylinder", "portable-charger", "biscoff", "coffee", "lid"]
RENAME = {"green-cube+red-cube": "two objects", "red-cube+vase": "two objects", "soap dispenser": "soap", "red-cube": "cube"}
MARK = {"1": "o", "3": "^", "5": "s", "7": "D"}
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
ok = ~(df.split + " " + df.object).str.contains("|".join(HIDE)).values & df.x.notna().values; df["object"] = df.object.replace(RENAME)
m = ok & (df.box == BOX).values & (df.object == OBJ).values; d, X = df[m].reset_index(drop=True), Z[m].copy()
SP = sorted(d.speaker.unique(), key=int); sp = d.speaker.values; G = d[["x", "y"]].values
for k in SP: X[sp == k] -= X[sp == k].mean(0)
E = umap.UMAP(n_neighbors=50, min_dist=.5, random_state=0).fit_transform(PCA(50, random_state=0).fit_transform(X))
Gc, Ec = G - G.mean(0), E - E.mean(0); U_, S, Vt = np.linalg.svd(Ec.T @ Gc); E = Ec @ (U_ @ Vt)  # rotate/flip so layer axes line up with the box (x right, y down)
U = (G - G.min(0)) / (G.max(0) - G.min(0)); col = np.array([colorsys.hls_to_rgb(.8 * a, .8 - .55 * b, .9) for a, b in U])
fig = plt.figure(figsize=(9, 8)); ax = fig.add_subplot(projection="3d")
lo, hi = E.min(0), E.max(0)
for z, k in enumerate(SP):
    s = sp == k; xx, yy = np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]])
    ax.plot_surface(xx, yy, np.full_like(xx, z), color="0.85", alpha=.25, shade=False)
    ax.scatter(E[s, 0], E[s, 1], z, c=col[s], s=10, marker=MARK.get(k, "o"), depthshade=False, label=f"speaker {k}")
pos = d.pos.value_counts(); pos = pos[pos == len(SP)].index.values; rng = np.random.default_rng(0)
for p in rng.choice(pos, min(12, len(pos)), replace=False):  # same physical position across speakers
    i = [np.where((d.pos.values == p) & (sp == k))[0][0] for k in SP]
    ax.plot(E[i, 0], E[i, 1], range(len(SP)), color="0.35", lw=.7)
ax.set_zticks(range(len(SP))); ax.set_zticklabels([f"speaker {k}" for k in SP]); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP (aligned to box x)"); ax.set_ylabel("UMAP (aligned to box y)"); ax.invert_yaxis(); ax.view_init(elev=22, azim=-60)
leg = ax.legend(loc="upper left", fontsize=8); [h.set_color("0.4") for h in leg.legend_handles]
ax.set_title(f"{BOX} · {OBJ}: speaker-centred latent, joint UMAP, one layer per speaker\ncolour = true position (hue = x, light = y) · lines = same position across speakers", fontsize=10)
out = ROOT / f"notebooks/out/137_layers_{BOX}_{OBJ}.pdf"; plt.savefig(out, bbox_inches="tight"); plt.savefig(sys.argv[3], dpi=70, bbox_inches="tight") if len(sys.argv) > 3 else None; print(out)
