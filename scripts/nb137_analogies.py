# two parameter-free latent analogies, searched over box × speaker × objects, best (cherry-picked) example of each plotted.
#  1) pair:  ½(cube + vase) + c = cube-vase pair,  c = mean(pair) − ½(mean(cube) + mean(vase))  (mean interaction; leave-one-out), also without c
#  2) swap:  X@p − mean(X) + mean(Y) = Y@p           (e.g. cube at p1 − avg cube + avg candle = candle at p1)
# retrieval like word2vec: cosine of the prediction against every candidate (centred on the candidates' mean); hit = the true one is the most similar;
# margin = cos(pred, true) − cos(pred, best wrong candidate). -> runs/.../analogies_{pair,swap}.csv, notebooks/out/137_analogy_{pair,swap}.pdf
import sys, itertools, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src9")); C = ROOT / "runs/_src9_cache/nb137/decres-rescale-aug10-v1"
from dataset import load_splits
PAIRS = {"red-cube+vase": [("red-cube", "red-cube0"), ("vase", "vase0")], "green-cube+red-cube": [("red-cube", "green-cube0"), ("red-cube", "red-cube0")]}
OBJS, K, R_PX = ["red-cube", "soap dispenser", "candle", "vase"], 5, 2.0  # singles searched per object part; swap: "same position" = within R_PX
ICON = {"red-cube": "cube", "vase": "vase", "candle": "candle", "soap dispenser": "soap", "red-cube+vase": "cube_and_vase", "green-cube+red-cube": "two_cubes"}
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
dirs = list(load_splits()[0].keys()); df["dir"] = [dirs[i] for i in np.load(ROOT / "runs/_src9_cache/nb136/base.npz")["i"]]
df = df[~df.split.str.contains("shifted|unseen-speaker|lid")]
ID = lambda j: Path(df.loc[j, "dir"]).name
def com(f): m = np.load(f) > 0; r, c = np.nonzero(m); return (c.mean() + .5) * 64 / m.shape[1] - .5, (r.mean() + .5) * 64 / m.shape[0] - .5
unit = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
# ---------- 1) pair = ½(cube + vase) + c ----------
rows, best1 = [], None
for pair, parts in PAIRS.items():
    P = df[df.object == pair].copy()
    for j, (_, inst) in enumerate(parts): P[[f"x{j}", f"y{j}"]] = [com(Path(d) / f"image/smasks/{inst}.npy") for d in P.dir]
    for (box, spk), p in P.groupby(["box", "speaker"]):
        S = df[(df.box == box) & (df.speaker == spk)]; Y = Z[p.index]; n = len(p)
        mo = [Z[S.index[S.object == ob]].mean(0) for ob, _ in parts]  # mean single cube / vase (all positions)
        NB = [[np.hypot(S[S.object == ob].x - r[f"x{j}"], S[S.object == ob].y - r[f"y{j}"]).nsmallest(K).index for _, r in p.iterrows()] for j, (ob, _) in enumerate(parts)]
        hit = {"half + c": [], "half, no c": []}
        for i in range(n):
            m_i = (Y.sum(0) - Y[i]) / (n - 1); c = m_i - .5 * (mo[0] + mo[1])  # leave-one-out mean pair → mean interaction
            Yc = unit(Y - m_i)
            for u, v in itertools.product(NB[0][i], NB[1][i]):
                half = .5 * (Z[u] + Z[v])
                for name, pred, cand in (("half + c", unit(half + c - m_i), Yc), ("half, no c", unit(half), unit(Y))):  # no c: raw cosine
                    cs = cand @ pred; true = cs[i]; other = np.max(np.delete(cs, i)); first = (u == NB[0][i][0]) and (v == NB[1][i][0])
                    if first: hit[name].append(true > other)  # honest: nearest single of each
                    if name == "half + c" and (best1 is None or true - other > best1[0]): best1 = (true - other, pair, box, spk, i, u, v, cs, c)
        rows.append(dict(pair=pair, box=box, speaker=spk, n=n) | {f"top1 {k}": np.mean(v) for k, v in hit.items()})
T1 = pd.DataFrame(rows).sort_values("top1 half + c", ascending=False); T1.to_csv(C / "analogies_pair.csv", index=False); print(T1.round(2).to_string(index=False))
# ---------- 2) swap: X@p − mean X + mean Y = Y@p ----------
rows, best2 = [], None
for (box, spk), S in df.groupby(["box", "speaker"]):
    for X, Yo in itertools.permutations(OBJS, 2):
        xs, ys = S[S.object == X], S[S.object == Yo]
        if len(xs) < 30 or len(ys) < 30: continue
        mX, mY = Z[xs.index].mean(0), Z[ys.index].mean(0); Yc = unit(Z[ys.index] - mY); hits, chance, margins = [], [], []
        for j, r in xs.iterrows():
            d = np.hypot(ys.x.values - r.x, ys.y.values - r.y); ok = d <= R_PX
            if not ok.any() or ok.all(): continue
            cs = Yc @ unit(Z[j] - mX)  # (X@p − mX + mY) − mY = X@p − mX
            t, o = cs[ok].max(), cs[~ok].max(); hits.append(t > o); chance.append(ok.mean()); margins.append(t - o)
            if best2 is None or t - o > best2[0]: best2 = (t - o, box, spk, X, Yo, j, ys.index[ok][np.argmax(cs[ok])], ys.index[~ok][np.argmax(cs[~ok])], cs, ok, mX, mY)
        if hits: rows.append(dict(box=box, speaker=spk, X=X, Y=Yo, n=len(hits), top1=np.mean(hits), chance=np.mean(chance), median_margin=np.median(margins)))
T2 = pd.DataFrame(rows).sort_values("top1", ascending=False); T2.to_csv(C / "analogies_swap.csv", index=False); print(T2.round(3).head(15).to_string(index=False))
print(T2.groupby(["X", "Y"]).top1.mean().unstack().round(2))
# ---------- figures ----------
def arrows(ax, segs, pts):
    for s, e, col, ls in segs: ax.annotate("", e, s, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, linestyle=ls, shrinkA=0, shrinkB=0, mutation_scale=18))
    pts = np.array(pts); lo, hi = pts.min(0), pts.max(0); r = (hi - lo).max() * .62; c = (lo + hi) / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_aspect("equal"); ax.scatter(0, 0, s=30, c="k"); ax.annotate("origin", (0, 0), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("PC 0"); ax.set_ylabel("PC 1"); ax.grid(True, color="0.93", lw=.6); ax.set_axisbelow(True)  # faint Euclidean grid
    ax.axhline(0, c="0.6", lw=.9, zorder=0); ax.axvline(0, c="0.6", lw=.9, zorder=0)  # crosshair through the origin
def icon(ax, xy, name, d=(0, 34), al=1., px=65):
    im = Image.open(ROOT / f"assets/objects/individual/{ICON[name]}.png"); im.thumbnail((px * 3, px * 3)); ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(im), zoom=1 / 3, alpha=al), xy, xybox=d, boxcoords="offset points", frameon=False, zorder=6))
def ranking(ax, cs, true_mask, title):
    bt = np.where(true_mask)[0][np.argmax(cs[true_mask])]; keep = ~true_mask; keep[bt] = True  # one green bar: drop the other true candidates
    cs, true_mask = cs[keep], (np.arange(len(keep)) == bt)[keep]
    o = np.argsort(-cs); ax.bar(range(len(cs)), cs[o], color=np.where(true_mask[o], "tab:green", "0.7"), width=1.)
    t, w = cs[true_mask].max(), cs[~true_mask].max(); ax.axhline(w, c="tab:red", lw=.8, ls="--"); ax.set_xlabel("candidates, most similar first"); ax.set_ylabel("cosine to prediction")
    ax.set_title(f"{title}\ntrue (green) {t:.2f} vs best wrong {w:.2f} · margin {t - w:.2f}", fontsize=10)
def equation(f, cell, items):  # bottom row as an equation: ("photo", sample, caption) | ("icon", object name) | ("text", str)
    W = [3 if k == "photo" else 1.6 if k == "icon" else .9 if len(it[1]) <= 2 else 1.4 for it in items for k in [it[0]]]
    sub = cell.subgridspec(1, len(items), width_ratios=W, wspace=.05)
    for g, it in zip(sub, items):
        ax = f.add_subplot(g); ax.set_axis_off()
        if it[0] == "photo": ax.imshow(Image.open(Path(df.loc[it[1], "dir"]) / "image/02_cropped_overhead.png")); ax.set_title(f"{it[2]}\n(sample {ID(it[1])})", fontsize=9)
        elif it[0] == "icon": ax.imshow(Image.open(ROOT / f"assets/objects/individual/{ICON[it[1]]}.png")); ax.set_title(f"{'mean ' if len(it) < 3 else ''}{it[1]}", fontsize=9)
        else: ax.text(.5, .5, it[1], ha="center", va="center", fontsize=34, weight="bold"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
def photos(f, gs_row, items):
    for j, (s_, t) in enumerate(items):
        b_ = f.add_subplot(gs_row[j]); b_.imshow(Image.open(Path(df.loc[s_, "dir"]) / "image/02_cropped_overhead.png")); b_.set_axis_off(); b_.set_title(f"{t} (sample {ID(s_)})", fontsize=9)
# figure 1: pair
mg, pair, box, spk, i, u, v, cs, c = best1; parts = PAIRS[pair]; p = df[(df.object == pair) & (df.box == box) & (df.speaker == spk)]; Y = Z[p.index]
W = np.linalg.svd(np.r_[Y, Z[[u, v]]], full_matrices=False)[2][:2].T; pj = lambda x: x @ W
cu, va, real, h1, h2, pred = pj(Z[u]), pj(Z[v]), pj(Y[i]), pj(.5 * Z[u]), pj(.5 * (Z[u] + Z[v])), pj(.5 * (Z[u] + Z[v]) + c)
f = plt.figure(figsize=(17, 10)); gs = f.add_gridspec(2, 4, height_ratios=[2, 1]); a1, a2 = f.add_subplot(gs[0, :2]), f.add_subplot(gs[0, 2:])
arrows(a1, [((0, 0), cu, "tab:red", (0, (3, 2))), ((0, 0), va, "tab:blue", (0, (3, 2))), ((0, 0), real, "k", (0, (5, 2))), ((0, 0), h1, "tab:red", "-"), (h1, h2, "tab:blue", "-"), (h2, pred, "tab:purple", "-")], [(0, 0), cu, va, real, h1, h2, pred])
for col, ls, l in (("tab:red", ":", f"cube (sample {ID(u)})"), ("tab:blue", ":", f"{parts[1][0]} (sample {ID(v)})"), ("k", "--", f"real pair (sample {ID(p.index[i])})"), ("tab:red", "-", "½·cube"), ("tab:blue", "-", f"+ ½·{parts[1][0]}"), ("tab:purple", "-", "+ c (mean interaction)")): a1.plot([], [], c=col, ls=ls, lw=2, label=l)
a1.scatter(*real, s=200, c="k", zorder=5); a1.scatter(*pred, s=240, marker="*", c="tab:green", zorder=5, label="½(cube + vase) + c"); a1.legend(fontsize=8, loc="lower right")
icon(a1, cu, parts[0][0], d=(10, -38)); icon(a1, va, parts[1][0], d=(-10, 38)); icon(a1, real, pair, d=(-45, 30), px=75); icon(a1, pred, pair, d=(45, 30), al=.55, px=55)
a1.set_title(f"{pair} · {box} · speaker {spk} · nothing learned: ½(cube + vase) + mean interaction (cherry-picked)", fontsize=10)
tm = np.zeros(len(cs), bool); tm[i] = True; ranking(a2, cs, tm, f"which real pair is closest? ({len(cs)} pairs, centred on the mean pair)")
equation(f, gs[1, :], [("text", "½ ×"), ("text", "("), ("photo", u, "cube"), ("text", "+"), ("photo", v, parts[1][0]), ("text", ")"), ("text", "+"), ("text", "c"), ("text", "="), ("photo", p.index[i], f"real {pair}")])
plt.tight_layout(); plt.savefig(ROOT / "notebooks/out/137_analogy_pair.pdf"); plt.savefig(sys.argv[1] + "_pair.png", dpi=55) if len(sys.argv) > 1 else None
# figure 2: swap
mg, box, spk, X, Yo, j, jt, jw, cs, ok, mX, mY = best2
Q = Z[df.index[(df.box == box) & (df.speaker == spk) & df.object.isin([X, Yo])]]  # axes = top-2 PCs of the CENTRED latents (position/object variation),
W = np.linalg.svd(np.r_[Q - Q.mean(0), [mX - mY]], full_matrices=False)[2][:2].T; pj = lambda x: x @ W   # applied linearly (no centring) so sums stay sums
xp, t1, pred, real = pj(Z[j]), pj(Z[j] - mX), pj(Z[j] - mX + mY), pj(Z[jt])
f = plt.figure(figsize=(17, 10)); gs = f.add_gridspec(2, 3, height_ratios=[2, 1]); a1, a2 = f.add_subplot(gs[0, :2]), f.add_subplot(gs[0, 2])
arrows(a1, [((0, 0), xp, "tab:red", "-"), (xp, t1, "0.45", "-"), (t1, pred, "tab:orange", "-"), ((0, 0), real, "k", (0, (5, 2)))], [(0, 0), xp, t1, pred, real])
for col, ls, l in (("tab:red", "-", f"{X} at p (sample {ID(j)})"), ("0.45", "-", f"− mean {X}"), ("tab:orange", "-", f"+ mean {Yo}"), ("k", "--", f"real {Yo} at p (sample {ID(jt)})")): a1.plot([], [], c=col, ls=ls, lw=2, label=l)
a1.scatter(*real, s=200, c="k", zorder=5); a1.scatter(*pred, s=240, marker="*", c="tab:green", zorder=5, label=f"{X}@p − mean {X} + mean {Yo}"); a1.legend(fontsize=8, loc="lower right")
lo_, hi_ = a1.get_ylim(); a1.set_ylim(lo_, hi_ + .25 * (hi_ - lo_)); lo_, hi_ = a1.get_xlim(); a1.set_xlim(lo_ - .12 * (hi_ - lo_), hi_ + .12 * (hi_ - lo_))  # room for the pictures
icon(a1, xp, X, d=(0, 42)); icon(a1, real, Yo, d=(58, 0), px=70); icon(a1, pred, Yo, d=(-58, 0), al=.6, px=60)  # cube above the red arrow tip, real vase right of the black dot, prediction left of the star
a1.set_title(f"{box} · speaker {spk} · nothing learned: {X} at p − mean {X} + mean {Yo} ≈ {Yo} at p (cherry-picked)", fontsize=10)
ranking(a2, cs, ok, f"which {Yo} is closest? ({(~ok).sum() + 1} candidates; green = closest {Yo} within {R_PX:.0f} px of p)")
equation(f, gs[1, :], [("photo", j, f"{X} at p"), ("text", "−"), ("icon", X), ("text", "+"), ("icon", Yo), ("text", "="), ("photo", jt, f"{Yo} at p")])
plt.tight_layout(); plt.savefig(ROOT / "notebooks/out/137_analogy_swap.pdf"); plt.savefig(sys.argv[1] + "_swap.png", dpi=55) if len(sys.argv) > 1 else None
print("best pair example", best1[:5], "margin", round(best1[0], 3)); print("best swap example", best2[1:5], "margin", round(best2[0], 3))
