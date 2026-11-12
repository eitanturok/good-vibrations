# stacked-grid test for one (box, object), all speakers, latent.
# 1) transfer: ridge (x, y) readout trained on speaker A, tested on speaker B (raw vs speaker-centred latents); 2) linear stacked view: speaker-centred
# latents → PLS-2 to (x, y) fit on half the positions, z = speaker; held-out R² per speaker. -> notebooks/out/137_stacked_<box>_<object>.{html,png}
import sys, colorsys, numpy as np, pandas as pd, plotly.graph_objects as go, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
BOX, OBJ = (sys.argv[1:3] + ["plastic", "candle"][len(sys.argv[1:3]):]) if len(sys.argv) > 1 else ("plastic", "candle")
ROOT = Path(__file__).resolve().parents[1]; C = ROOT / "runs/_src9_cache/nb137/decres-rescale-aug10-v1"
HIDE = ["shifted", "unseen-speaker", "cylinder", "portable-charger", "biscoff", "coffee", "lid"]
RENAME = {"green-cube+red-cube": "two objects", "red-cube+vase": "two objects", "soap dispenser": "soap", "red-cube": "cube"}
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
ok = ~(df.split + " " + df.object).str.contains("|".join(HIDE)).values & df.x.notna().values; df["object"] = df.object.replace(RENAME)
m = ok & (df.box == BOX).values & (df.object == OBJ).values; d, X = df[m].reset_index(drop=True), Z[m]
SP = sorted(d.speaker.unique(), key=int); G = d[["x", "y"]].values; sp = d.speaker.values
Xc = X.copy()
for k in SP: Xc[sp == k] -= X[sp == k].mean(0)  # remove each speaker's offset
r2 = lambda y, p: 1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()
print(f"{BOX} · {OBJ}: {len(d)} samples, speakers {SP}, {d.pos.nunique()} positions")
for name, A in (("raw", X), ("speaker-centred", Xc)):
    P = PCA(50, random_state=0).fit_transform(A - A.mean(0)); T = pd.DataFrame(index=SP, columns=SP, dtype=float)
    for a in SP:
        for b in SP:
            tr, te = sp == a, sp == b
            if a == b:  # within speaker: 2-fold split by position
                h = d.pos.values % 2 == 0; T.loc[a, b] = np.mean([r2(G[tr & ~hh], RidgeCV(np.logspace(-2, 4, 13)).fit(P[tr & hh], G[tr & hh]).predict(P[tr & ~hh])) for hh in (h, ~h)])
            else: T.loc[a, b] = r2(G[te], RidgeCV(np.logspace(-2, 4, 13)).fit(P[tr], G[tr]).predict(P[te]))
    off = T.values[~np.eye(len(SP), dtype=bool)].mean()
    print(f"\n{name}: (x, y) R², train speaker (row) → test speaker (col); within {np.diag(T.values).mean():.2f}, across {off:.2f}\n{T.round(2)}")
# linear stacked view: PLS on speaker-centred PCA-50, fit on even positions, all speakers pooled
P = PCA(50, random_state=0).fit_transform(Xc); fit = d.pos.values % 2 == 0
pls = PLSRegression(2).fit(P[fit], G[fit]); E = pls.transform(P)
Q = np.zeros_like(G, dtype=float)  # held-out map quality per speaker: Procrustes-free, just linear fit of (x, y) from the 2 PLS axes on train positions
for k in SP:
    s = sp == k; L = np.c_[E, np.ones(len(E))]; w = np.linalg.lstsq(L[fit & s], G[fit & s], rcond=None)[0]
    print(f"speaker {k}: held-out (x, y) R² from shared PLS plane {r2(G[~fit & s], L[~fit & s] @ w):.2f}")
lo, hi = G.min(0), G.max(0); U = (G - lo) / (hi - lo)
col = np.array([colorsys.hls_to_rgb(.8 * a, .8 - .55 * b, .9) for a, b in U])
z = np.array([SP.index(k) for k in sp], float)
out = ROOT / f"notebooks/out/137_stacked_{BOX}_{OBJ}"
fig = go.Figure([go.Scatter3d(x=E[s, 0], y=E[s, 1], z=z[s], mode="markers", name=f"speaker {k}" + (" (held-out pos)" if h == 0 else ""),
                              marker=dict(size=4, color=[f"rgb({r*255:.0f},{g*255:.0f},{b*255:.0f})" for r, g, b in col[s]], symbol="circle" if h else "diamond-open"))
                 for k in SP for h in (1, 0) for s in [(sp == k) & (fit == bool(h))]])
fig.update_layout(width=1000, height=800, title=f"{BOX} · {OBJ} · speaker-centred latent → shared PLS plane (fit on even positions) · z = speaker · colour = true position (hue = x, light = y)",
                  scene=dict(xaxis_title="PLS 1", yaxis_title="PLS 2", zaxis=dict(title="speaker", tickvals=list(range(len(SP))), ticktext=SP)))
fig.write_html(f"{out}.html", include_plotlyjs="cdn")
f, ax = plt.subplots(1, len(SP), figsize=(4 * len(SP), 4))
for j, k in enumerate(SP):
    s = sp == k; ax[j].scatter(E[s, 0], E[s, 1], c=col[s], s=14); ax[j].set_title(f"speaker {k}"); ax[j].set_xticks([]); ax[j].set_yticks([])
plt.tight_layout(); plt.savefig(f"{out}.png", dpi=80); print(out)
