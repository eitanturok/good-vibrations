"""nb139: decode every latent analogy (top-3 configs of each of the 5 analogies) with the trained decoder of decres-rescale-aug10-v1.

  .venv/bin/python scripts/nb139_decode.py            # all 15 figures + summary
  .venv/bin/python scripts/nb139_decode.py swap 1     # one figure

Analogies (same math, instances, candidates and top-3 ranking as the nb139 cells 3-7; names as in the paper):
  combine     combine objects   ½·X + ½·Y + c = pair                      (nb139 "pair")
  swap        object swap       X at p − mean X + mean Y = Y at p
  boxswap     box swap          X at p in A − mean A + mean B = X at p in B
  compose     composition       mean X in A − mean A + mean at p in B = X at p in B
  compose_simple  simple composition  mean X in box + mean at p − mean box = X at p
Notation: ẑ = predicted latent (the analogy), z = ground-truth latent (the target recording), Î = decoder(ẑ) binarised at 0.08, I = the target's label.
Target recording = the notebook's ground truth: the best-cosine true candidate (combine objects: the held-out pair itself).
Candidates = the notebook's candidates for that analogy; the other true (same-position) candidates are dropped. Rank = 1 + wrong candidates better than the true one:
latent L1 / cos (centred like score()), layout IoU / CoM of Î vs every candidate's label. Plotted example = the notebook figure's (largest margin).
Ceiling = decoder(z) vs I; baseline = decoder(z of the source recording) vs I (object swap / box swap only; the others have no single source).
Cache: runs/_src9_cache/nb139/<run>/decode/<analogy>_<k>.{npz,csv} + summary.csv. Figures: notebooks/out/139_decode_<analogy>_<k>.pdf."""
import itertools, json, sys, numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src9"))
from dataset import load_splits, downsample

RUN, K, R_PX, TH, TOP, BAD = "decres-rescale-aug10-v1", 5, 2.0, .08, 3, "shifted|unseen-speaker|lid"
OBJS = ["red-cube", "soap dispenser", "candle", "vase"]
PAIRS = {"red-cube+vase": [("red-cube", "red-cube0"), ("vase", "vase0")], "green-cube+red-cube": [("red-cube", "green-cube0"), ("red-cube", "red-cube0")]}
ICON = {"red-cube": "cube", "vase": "vase", "candle": "candle", "soap dispenser": "soap"}
NAME = {"combine": "combine objects", "swap": "object swap", "boxswap": "box swap", "compose": "composition", "compose_simple": "add position", "addpos_pooled": "add position", "addpos_objpos": "add position", "addpos_single": "add position"}  # simple composition = "add position" (paper framing)
C, DC = ROOT / "runs/_src9_cache/nb137" / RUN, ROOT / "runs/_src9_cache/nb139" / RUN / "decode"; DC.mkdir(parents=True, exist_ok=True)

# ---- nb139 cells 1-2
df, Z = pd.read_csv(C / "meta.csv", dtype={"speaker": str}), np.load(C / "feats.npz")["latent"]
dirs = list(load_splits()[0].keys()); df["dir"] = [dirs[i] for i in np.load(ROOT / "runs/_src9_cache/nb136/base.npz")["i"]]
df = df[~df.split.str.contains(BAD)]
def com(f): m = np.load(f) > 0; r, c = np.nonzero(m); return (c.mean() + .5) * 64 / m.shape[1] - .5, (r.mean() + .5) * 64 / m.shape[0] - .5
for pair, parts in PAIRS.items():
    ix = df.index[df.object == pair]
    for j, (_, inst) in enumerate(parts): df.loc[ix, [f"{pair}_x{j}", f"{pair}_y{j}"]] = [com(Path(d) / f"image/smasks/{inst}.npy") for d in df.dir[ix]]
EMPTY = {b: df.index[(df.object == "empty") & (df.box == b)][0] for b in df.box.unique()}
unit = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)
def score(pred, cand, ok, ctr):  # notebook's score(): cos rank of the closest true candidate
    cs = unit(cand - ctr) @ unit(pred - ctr); return cs, 1 + int((cs[~ok] > cs[ok].max()).sum())
PS = lambda j: f"({df.loc[j, 'pos']}-{df.loc[j, 'speaker']})"  # position_id-speaker
inst = lambda z, t, cand, ok, ctr, src=None, mg=np.nan: dict(z=z, t=t, cand=cand, ok=ok, ctr=ctr, src=src, mg=mg)
# a config = dict(analogy, key, sub, top1, cand, Q, inst=[...honest instances], ex=example instance + terms); terms = [(sign, latent, img, label)],
# img = ("photo", row) | ("icon", object) | ("box", box, p or None) | ("tile", text)

# ---- 1) combine objects (nb139 "pair"): ½(X + Y) + c = pair; honest instance = nearest singles, example = best margin over the K×K neighbours
def combine():
    for pair, parts in PAIRS.items():
        for (box, spk), p in df[df.object == pair].groupby(["box", "speaker"]):
            S = df[(df.box == box) & (df.speaker == spk)]; Y = Z[p.index]; n = len(p); mo = [Z[S.index[S.object == ob]].mean(0) for ob, _ in parts]
            NB = [[np.hypot(S[S.object == ob].x - r[f"{pair}_x{j}"], S[S.object == ob].y - r[f"{pair}_y{j}"]).nsmallest(K).index for _, r in p.iterrows()] for j, (ob, _) in enumerate(parts)]
            rc, I, best = [], [], None
            for i in range(n):
                m = (Y.sum(0) - Y[i]) / (n - 1); c = m - .5 * (mo[0] + mo[1]); ok = np.arange(n) == i
                for u, v in itertools.product(NB[0][i], NB[1][i]):
                    zh = .5 * (Z[u] + Z[v]) + c; cs, a = score(zh, Y, ok, m); mg = cs[i] - np.delete(cs, i).max()
                    if u == NB[0][i][0] and v == NB[1][i][0]: rc.append(a); I.append(inst(zh, p.index[i], p.index, ok, m) | dict(u=u, v=v, c=c))
                    if best is None or mg > best[0]: best = (mg, i, u, v, zh, c, ok, m)
            mg, i, u, v, zh, c, ok, m = best
            ex = inst(zh, p.index[i], p.index, ok, m, mg=mg) | dict(terms=[(1, .5 * Z[u], ("photo", u), f"½ · {parts[0][0]} {PS(u)}"), (1, .5 * Z[v], ("photo", v), f"½ · {parts[1][0]} {PS(v)}"),
                                                                    (1, c, ("tile", "c"), "c (mean interaction)")], gt=f"ground truth\n({pair}) {PS(p.index[i])}")
            yield dict(analogy="combine", key=(pair, box, spk), sub=f"{pair} · {box} · spk {spk}", top1=np.mean(np.array(rc) == 1), inst=I, ex=ex,
                       Q=np.sort(np.r_[p.index, S.index[S.object.isin([ob for ob, _ in parts])]]))

# ---- 2) object swap / 3) box swap: Z[x] + base = target; candidates ys, true = within R_PX of x (best cos = the target), cos centred on ctr
def swap_search(xs, ys, base, ctr):
    rc, I = [], []
    for j, r in xs.iterrows():
        ok = np.hypot(ys.x.values - r.x, ys.y.values - r.y) <= R_PX
        if not ok.any() or ok.all(): continue
        cs, a = score(Z[j] + base, Z[ys.index], ok, ctr); rc.append(a); I.append(inst(Z[j] + base, ys.index[ok][np.argmax(cs[ok])], ys.index, ok, ctr, src=j, mg=cs[ok].max() - cs[~ok].max()))
    return rc, I
def best_ex(I): return dict(I[int(np.argmax([q["mg"] for q in I]))])  # first max margin, as the notebook's strict ">"
def swap():
    for (box, spk), S in df.groupby(["box", "speaker"]):
        for X, Yo in itertools.permutations(OBJS, 2):
            xs, ys = S[S.object == X], S[S.object == Yo]
            if len(xs) < 30 or len(ys) < 30: continue
            mX, mY = Z[xs.index].mean(0), Z[ys.index].mean(0); rc, I = swap_search(xs, ys, mY - mX, mY); ex = best_ex(I)
            ex |= dict(terms=[(1, Z[ex["src"]], ("photo", ex["src"]), f"{X} at p {PS(ex['src'])}"), (-1, mX, ("icon", X), f"mean {X}"), (1, mY, ("icon", Yo), f"mean {Yo}")],
                       gt=f"ground truth\n({Yo} at p) {PS(ex['t'])}")
            yield dict(analogy="swap", key=(box, spk, X, Yo), sub=f"{X} → {Yo} · {box} · spk {spk}", top1=np.mean(np.array(rc) == 1), inst=I, ex=ex, Q=np.sort(np.r_[xs.index, ys.index]), mX=mX, mY=mY)
def boxswap():
    for spk, S in df.groupby("speaker"):
        for X in OBJS:
            for A, B in itertools.permutations(sorted(S.box.unique()), 2):
                xs, ys = S[(S.object == X) & (S.box == A)], S[(S.object == X) & (S.box == B)]
                if len(xs) < 30 or len(ys) < 30: continue
                mA, mB = Z[S.index[S.box == A]].mean(0), Z[S.index[S.box == B]].mean(0); rc, I = swap_search(xs, ys, mB - mA, Z[ys.index].mean(0)); ex = best_ex(I)
                ex |= dict(terms=[(1, Z[ex["src"]], ("photo", ex["src"]), f"{X} at p in {A} {PS(ex['src'])}"), (-1, mA, ("box", A, None), f"mean {A}"), (1, mB, ("box", B, None), f"mean {B}")],
                           gt=f"ground truth\n({X} at p in {B}) {PS(ex['t'])}")
                yield dict(analogy="boxswap", key=(spk, X, A, B), sub=f"{X} · {A} → {B} · spk {spk}", top1=np.mean(np.array(rc) == 1), inst=I, ex=ex, Q=np.sort(np.r_[xs.index, ys.index]))

# ---- 4) composition (strict) and 4b) simple composition: candidates = X in the target box, true = within R_PX of the query position (best cos = the target)
def compose():
    for spk, S in df.groupby("speaker"):
        for X in OBJS:
            for A, B in itertools.permutations(sorted(S.box.unique()), 2):
                tg, src_, restB = S[(S.object == X) & (S.box == B)], S[(S.object == X) & (S.box == A)], S[(S.box == B) & (S.object != X)]
                if len(tg) < 30 or len(src_) < 30: continue
                mXA, mA, ctr = Z[src_.index].mean(0), Z[S.index[S.box == A]].mean(0), Z[tg.index].mean(0); rc, I = [], []
                for j, r in tg.iterrows():
                    ok = np.hypot(tg.x.values - r.x, tg.y.values - r.y) <= R_PX; nb_ = restB.index[np.hypot(restB.x - r.x, restB.y - r.y) <= R_PX]
                    if ok.all() or len(nb_) < 3: continue
                    mPos = Z[nb_].mean(0); zh = mXA - mA + mPos; cs, a = score(zh, Z[tg.index], ok, ctr); rc.append(a)
                    I.append(inst(zh, tg.index[ok][np.argmax(cs[ok])], tg.index, ok, ctr, mg=cs[ok].max() - cs[~ok].max()) | dict(mPos=mPos, p=(r.x, r.y)))
                if not rc: continue
                ex = best_ex(I); ex |= dict(terms=[(1, mXA, ("icon", X), f"mean {X} in {A}"), (-1, mA, ("box", A, None), f"mean {A}"), (1, ex["mPos"], ("box", B, ex["p"]), f"mean at p in {B}")],
                                            gt=f"ground truth\n({X} at p in {B}) {PS(ex['t'])}")
                yield dict(analogy="compose", key=(spk, X, A, B), sub=f"{X} · {A} → {B} · spk {spk}", top1=np.mean(np.array(rc) == 1), inst=I, ex=ex, Q=np.sort(S.index[S.box.isin([A, B])]))
def compose_simple():
    for (spk, box), S in df.groupby(["speaker", "box"]):
        for X in OBJS:
            tg, rest = S[S.object == X], S[S.object != X]
            if len(tg) < 30: continue
            mRest, ctr = Z[rest.index].mean(0), Z[tg.index].mean(0); rc, I = [], []
            for j, r in tg.iterrows():
                ok = np.hypot(tg.x.values - r.x, tg.y.values - r.y) <= R_PX; nb = rest.index[np.hypot(rest.x - r.x, rest.y - r.y) <= R_PX]
                if ok.all() or len(nb) < 3: continue
                mX, mPos = (Z[tg.index].sum(0) - Z[j]) / (len(tg) - 1), Z[nb].mean(0); zh = mX + mPos - mRest; cs, a = score(zh, Z[tg.index], ok, ctr); rc.append(a)
                I.append(inst(zh, tg.index[ok][np.argmax(cs[ok])], tg.index, ok, ctr, mg=cs[ok].max() - cs[~ok].max()) | dict(mX=mX, mPos=mPos, p=(r.x, r.y)))
            if not rc: continue
            ex = best_ex(I); ex |= dict(terms=[(1, ex["mX"], ("icon", X), f"mean {X} in {box}"), (1, ex["mPos"], ("box", box, ex["p"]), "mean at p"), (-1, mRest, ("box", box, None), f"mean {box}")],
                                        gt=f"ground truth\n({X} at p) {PS(ex['t'])}")
            yield dict(analogy="compose_simple", key=(spk, box, X), sub=f"{X} in {box} · spk {spk}", top1=np.mean(np.array(rc) == 1), inst=I, ex=ex, Q=np.sort(S.index), mRest=mRest)
GEN = dict(combine=combine, swap=swap, boxswap=boxswap, compose=compose, compose_simple=compose_simple)

# ---- decoder (latest checkpoint, fp32, tf32 off, eager); res decoder: no xattn, no adapter → decoder(u) alone. Loaded lazily, once.
_D = []
def D(U):
    if not _D:
        from arch.transformer4 import Decoder
        a = json.load(open(ROOT / "runs" / RUN / "args.json")); assert a["model"] == "transformer4" and a["decoder"] == "res" and not a.get("adapter_dim")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        sd = torch.load(ROOT / "runs" / RUN / "checkpoints/latest-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"]
        dec = Decoder(a["decoder"]).cuda().eval(); print(dec.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("decoder.")})); _D.append(dec)
        # sanity: cached Z → decoder → 4×4 pool = cached output, to < 1 byte step (the nb137 cache stores the latent as fp16 and the mask as round(255·p) bytes)
        chk = np.random.default_rng(0).choice(len(Z), 512, replace=False)
        err = np.abs((D(Z[chk]) * 255).round().reshape(-1, 16, 4, 16, 4).mean((2, 4)).reshape(len(chk), -1) / 255 - np.load(C / "feats.npz")["output"][chk]).max()
        print(f"decode check: max |Δ| vs cached output = {err:.2e} (1 byte step = {1 / 255:.2e})"); assert err < 1 / 255
    with torch.no_grad(): return np.concatenate([_D[0](torch.from_numpy(np.ascontiguousarray(U[s:s + 256], np.float32)).cuda()).sigmoid().cpu().numpy() for s in range(0, len(U), 256)])
label = lambda j: downsample(np.asarray(Image.open(Path(df.loc[j, "dir"]) / "image/03_smask.png").convert("F")) / 255.0, 64, 64) > .5  # mask_true, as in dataset.process_y
rr, cc = np.mgrid[:64, :64]
def cm(b): n = b.sum((1, 2)); return np.where(n[:, None] > 0, np.stack([(b * cc).sum((1, 2)), (b * rr).sum((1, 2))], 1) / np.maximum(n, 1)[:, None], np.nan)  # (x, y) = (col, row)
iou = lambda a, b: (a & b).sum((-2, -1)) / np.maximum((a | b).sum((-2, -1)), 1)

# ---- decode + metrics for one config: every honest instance and the plotted example (last row, example=True)
def run(cfg, k):
    F = DC / f"{cfg['analogy']}_{k}.npz"; I = cfg["inst"] + [cfg["ex"]]; cand = cfg["cand"] = I[0]["cand"]
    assert all(q["cand"].equals(cand) for q in I)  # one candidate set per config
    zh, src = np.stack([q["z"] for q in I]).astype(np.float32), np.array([-1 if q["src"] is None else q["src"] for q in I])
    if not F.exists():
        np.savez(F, cand=cand.values, src=src, t=[q["t"] for q in I], z_hat=zh, I_hat=D(zh).astype(np.float16), cand_decoded=D(Z[cand]).astype(np.float16),
                 cand_label=np.stack([label(j) for j in cand]).astype(np.uint8), src_decoded=D(Z[src[src >= 0]]).astype(np.float16) if (src >= 0).any() else np.zeros((0, 64, 64), np.float16))
    c = np.load(F); assert np.array_equal(c["cand"], cand.values) and np.array_equal(c["t"], [q["t"] for q in I]) and np.allclose(c["z_hat"], zh, atol=1e-5)
    GC, PCD = c["cand_label"] > 0, c["cand_decoded"].astype(np.float32) > TH; CC = cm(GC); assert np.allclose(CC, df.loc[cand, ["x", "y"]].values, equal_nan=True)  # label CoM = meta
    Ih = c["I_hat"].astype(np.float32) > TH; CH = cm(Ih); SB = np.zeros((len(I), 64, 64), bool); SB[src >= 0] = c["src_decoded"].astype(np.float32) > TH; CS = cm(SB)
    rows = []
    for i, q in enumerate(I):
        t = cand.get_loc(q["t"]); keep = ~q["ok"]; keep[t] = True; w = keep.copy(); w[t] = False
        l1, cs = np.abs(Z[cand] - q["z"]).sum(1), unit(Z[cand] - q["ctr"]) @ unit(q["z"] - q["ctr"])
        iu, dc = iou(GC, Ih[i]), np.hypot(*(CC - CH[i]).T)
        rows.append(dict(source_row=q["src"], target_row=q["t"], margin=q["mg"], example=i == len(I) - 1, n_candidates=int(keep.sum()),
                         l1_latent=l1[t], rank_l1_latent=1 + int((l1[w] < l1[t]).sum()), cos_latent=cs[t], rank_cos_latent=1 + int((cs[w] > cs[t]).sum()),
                         iou_analogy_decoded_vs_ground_truth=iu[t], rank_iou_layout=1 + int((iu[w] > iu[t]).sum()), com_analogy_decoded_vs_ground_truth=dc[t], rank_com_layout=1 + int((dc[w] < dc[t]).sum()),
                         iou_analogy_decoded_vs_analogy_expected=iou(PCD[t], Ih[i]), com_analogy_decoded_vs_analogy_expected=np.hypot(*(cm(PCD[t:t + 1])[0] - CH[i])),
                         iou_analogy_expected_vs_ground_truth=iou(PCD[t], GC[t]), com_analogy_expected_vs_ground_truth=np.hypot(*(cm(PCD[t:t + 1])[0] - CC[t])),
                         iou_source_decoded_vs_ground_truth=iou(SB[i], GC[t]) if q["src"] is not None else np.nan, com_source_decoded_vs_ground_truth=np.hypot(*(CS[i] - CC[t])) if q["src"] is not None else np.nan,
                         iou_source_decoded_vs_analogy_expected=iou(SB[i], PCD[t]) if q["src"] is not None else np.nan, com_source_decoded_vs_analogy_expected=np.hypot(*(CS[i] - cm(PCD[t:t + 1])[0])) if q["src"] is not None else np.nan,
                         area_analogy_decoded=c["I_hat"][i].astype(np.float32).sum(), area_analogy_expected=c["cand_decoded"][t].astype(np.float32).sum(), area_ground_truth=GC[t].sum(), empty_analogy_decoded=not Ih[i].any()))
    M = pd.DataFrame(rows); M.to_csv(DC / f"{cfg['analogy']}_{k}.csv", index=False)
    return M, Ih, GC

# ---- figure (same design as the object-swap figure): rows image | latent vector (2-D PCA, one small plot per term) | layout (Î ≈ I)
PRED = "#E6A700"  # the prediction's single identity: gold, gold ★ with a dark edge
photo = lambda j: Image.open(Path(df.loc[j, "dir"]) / "image/02_cropped_overhead.png"); icon = lambda o: Image.open(ROOT / f"assets/objects/individual/{ICON[o]}.png")
def picture(img): return photo(img[1]) if img[0] == "photo" else icon(img[1]) if img[0] == "icon" else photo(EMPTY[img[1]]) if img[0] == "box" else None
def tile(ax, txt): ax.add_patch(Rectangle((.08, .08), .84, .84, fc="0.95", ec="0.5", lw=1.6)); ax.text(.5, .5, txt, ha="center", va="center", fontsize=60, style="italic", color="0.25"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
def placeholder(ax): ax.add_patch(Rectangle((.08, .08), .84, .84, fill=False, ec=PRED, lw=1.8, ls="--")); ax.scatter(.5, .5, marker="*", s=4000, c=PRED, edgecolors="0.15", linewidths=1.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
def arrow(ax, a, b, col, ls="-", lw=2.2, al=1.): ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, ls=ls, alpha=al, shrinkA=0, shrinkB=0, mutation_scale=24))
def thumb(ax, xy, img, px=52, off=(0, 34), al=.25):
    im = picture(img)
    if im is None: ax.add_artist(AnnotationBbox(TextArea(img[1], textprops=dict(fontsize=15, color="0.3", style="italic")), xy, xybox=off, boxcoords="offset points", bboxprops=dict(ec="0.5", fc="0.95", boxstyle="square,pad=0.4", alpha=max(al, .6))))
    else: t = im.copy(); t.thumbnail((px * 3, px * 3)); ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(t), zoom=1 / 3, alpha=al), xy, xybox=off, boxcoords="offset points", frameon=False))
OWN, CHAIN = [(20, 30), (0, 36), (-38, 22), (0, 36)], [(-6, -28), (16, 36), (-14, 28), (16, 36)]  # thumbnail offsets per term (own panel / chain in the prediction panel)
def figure(cfg, e, Ih, gt, out, fit=False):  # e = the example's metrics row, Ih / gt = its Î / I; fit: latent panels stretch (no equal aspect), thumbnails pushed off the arrow tips
    ex = cfg["ex"]; T = ex["terms"]; nt = len(T); za, zgt = ex["z"], Z[ex["t"]]
    assert np.allclose(sum(s * z for s, z, _, _ in T), za, atol=1e-3)  # the terms add up to ẑ
    OPS = [("−" if s < 0 else "+") for s, _, _, _ in T[1:]] + ["=", "≈"]; cols = [1 + 2 * i for i in range(nt + 2)]; pc, gc = cols[-2], cols[-1]
    f = plt.figure(figsize=(38, 13.5))
    gs = f.add_gridspec(4, 2 * nt + 4, width_ratios=[.6] + [1, .2] * (nt + 1) + [1], height_ratios=[1, .6, .2, .9], wspace=.06, hspace=.1)
    for i, cI in enumerate(cols):  # row 1: image
        ax = f.add_subplot(gs[0, cI]); ax.set_axis_off()
        if i < nt:
            img = T[i][2]; ax.set_title(T[i][3], fontsize=14)
            if img[0] == "tile": tile(ax, img[1])
            else:
                im = picture(img); ax.imshow(im)
                if img[0] == "box" and img[2] is not None: ax.scatter((img[2][0] + .5) * im.size[0] / 64, (img[2][1] + .5) * im.size[1] / 64, s=160, marker="x", c="red", lw=3)  # p
        elif i == nt: ax.set_title("prediction", fontsize=14); placeholder(ax)
        else: ax.set_title(ex["gt"], fontsize=14); ax.imshow(photo(ex["t"]))
        if i < nt + 1: ao = f.add_subplot(gs[0, cI + 1]); ao.set_axis_off(); ao.text(.5, .5, OPS[i], ha="center", va="center", fontsize=34, weight="bold")
    # row 2: latent — one small plot per column, same linear projection W (fitted on the config's latents, centred for the fit only, as figure() in nb139) and limits
    Q = Z[cfg["Q"]]; W = np.linalg.svd(Q - Q.mean(0), full_matrices=False)[2][:2].T; pj = lambda x: x @ W
    V = [(s * pj(z), col) for (s, z, _, _), col in zip(T, ["tab:red", "0.45", "tab:orange", "tab:purple"])]; pa, pg = pj(za), pj(zgt)
    chain = np.cumsum([np.zeros(2)] + [v for v, _ in V], 0); assert np.allclose(chain[-1], pa, atol=1e-3)
    pts = np.r_[chain, [v for v, _ in V], [pa, pg], np.zeros((1, 2))]; lo, hi = pts.min(0), pts.max(0); pad = (hi - lo) * .15 if fit else (hi - lo).max() * .06
    span = np.maximum(hi, 0) - np.minimum(lo, 0) + 2 * pad
    x0 = np.array([min(lo[0], 0), min(lo[1], 0)]) - (pad if fit else np.array([pad, pad]))
    def away(d, r=50):  # offset (points) beside the tip of the arrow 0 → d: perpendicular to it (off the arrow), on the side facing the panel centre (inside, off the edges)
        u = np.asarray(d, float) / span; n = np.linalg.norm(u)
        if n == 0: return (0, r)
        q = np.array([-u[1], u[0]]) / n; tip = (np.asarray(d, float) - x0) / span; q = q if q @ (.5 - tip) >= 0 else -q; return tuple(r * q)
    for i, cI in enumerate(cols):
        ax = f.add_subplot(gs[1, cI])
        pd_ = pad if fit else np.array([pad, pad]); ax.set_xlim(min(lo[0], 0) - pd_[0], max(hi[0], 0) + pd_[0]); ax.set_ylim(min(lo[1], 0) - pd_[1], max(hi[1], 0) + pd_[1])
        if not fit: ax.set_aspect("equal")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_position("zero"); ax.spines[sp].set_color("0.6")
        ax.set_xticks([]); ax.set_yticks([]); ax.text(ax.get_xlim()[1], 0, " PC1", va="center", ha="left", fontsize=9); ax.text(0, ax.get_ylim()[1], "PC2", va="bottom", ha="center", fontsize=9)
        if i < nt:
            arrow(ax, (0, 0), V[i][0], V[i][1]); thumb(ax, V[i][0], T[i][2], off=away(V[i][0]) if fit else OWN[i], al=1.)
            if T[i][0] < 0: ax.annotate(f"− {T[i][3]}", V[i][0] * .5, xytext=(0, -8), textcoords="offset points", color="0.35", fontsize=13, ha="center", va="top")
        elif i == nt:
            for (v, col), a0, a1_, (_, _, img, _), off in zip(V, chain[:-1], chain[1:], T, CHAIN): arrow(ax, a0, a1_, col, al=.65, lw=2.2); thumb(ax, (a0 + a1_) / 2, img, off=away(v, 40) if fit else off, al=.6)  # fit: perpendicular to the segment
            arrow(ax, (0, 0), pa, PRED, lw=3.6); ax.scatter(*pa, marker="*", s=400, c=PRED, edgecolors="0.15", linewidths=1.2, zorder=10, clip_on=False)  # the prediction
            ax.annotate(r"$\hat{z}$", pa, xytext=(10, 6), textcoords="offset points", fontsize=16, color="0.1")
        else:
            arrow(ax, (0, 0), pa, PRED, al=.4, lw=2.2); ax.scatter(*pa, marker="*", s=400, c=PRED, edgecolors="0.15", linewidths=1.2, zorder=10, clip_on=False)
            arrow(ax, (0, 0), pg, "k", ls=(0, (4, 2.5)), lw=2.0); ax.scatter(*pg, s=70, facecolors="none", edgecolors="k", linewidths=1.5, zorder=9, clip_on=False)  # ground-truth tip
            ax.annotate(r"$z$", pg, xytext=(8, 4), textcoords="offset points", fontsize=16, color="0.1"); thumb(ax, pg, ("photo", ex["t"]), off=away(pg) if fit else (-44, 24), al=1.)
        if i < nt: ao = f.add_subplot(gs[1, cI + 1]); ao.set_axis_off(); ao.text(.5, .5, OPS[i], ha="center", va="center", fontsize=34, weight="bold")
    # row 3: layout — Î ≈ I, with decoder / label arrows from the latent row
    for cI, m, how in ((pc, Ih.astype(float), "decoder"), (gc, gt.astype(float), "label")):
        col = "tab:orange" if how == "label" else "#1F77B4"
        ar = f.add_subplot(gs[2, cI]); ar.set_axis_off(); ar.set_xlim(0, 1); ar.set_ylim(0, 1)
        ar.annotate("", (.5, .02), (.5, .52), arrowprops=dict(arrowstyle="-|>", lw=1.8, color=col, mutation_scale=18)); ar.text(.56, .27, how, va="center", fontsize=11, color=col)
        a2 = f.add_subplot(gs[3, cI]); a2.set_axis_off(); a2.imshow(m, cmap="gray", vmin=0, vmax=1); a2.add_patch(Rectangle((-.5, -.5), 64, 64, fill=False, ec="0.55", lw=1.2, clip_on=False))
        if how == "decoder": a2.add_patch(Rectangle((-2.5, -2.5), 68, 68, fill=False, ec=PRED, lw=5, clip_on=False))  # highlight the prediction
        a2.set_title(r"$\hat{I}$ (prediction)" if how == "decoder" else r"$I$ (ground truth)", fontsize=14, y=-.15)
    ao = f.add_subplot(gs[3, pc + 1]); ao.set_axis_off(); ao.text(.5, .5, "≈", ha="center", va="center", fontsize=34, weight="bold")
    for j, h in ((0, "image"), (1, "latent vector"), (3, "layout")): a = f.add_subplot(gs[j, 0]); a.set_axis_off(); a.text(.95, .5, h, ha="right", va="center", fontsize=16, weight="bold")
    f.axes[-2].text(.95, .36, "2D PCA", ha="right", va="center", fontsize=12, color="0.35")  # second line of the latent header
    ae = f.add_subplot(gs[1, pc + 1]); ae.set_axis_off(); ae.text(.5, .5, "≈", ha="center", va="center", fontsize=34, weight="bold")
    am = f.add_subplot(gs[3, 1:2 * nt]); am.set_axis_off()  # all metrics in one box, in the empty layout-row space under the term columns
    am.text(.5, .5, "latent\n" + rf"    L1$(z, \hat{{z}})$ = {e.l1_latent:.0f}  (rank {e.rank_l1_latent} of {e.n_candidates})" + "\n" + rf"    cos$(z, \hat{{z}})$ = {e.cos_latent:.2f}  (rank {e.rank_cos_latent} of {e.n_candidates})"
            + "\n\nlayout\n" + rf"    IoU$(I, \hat{{I}})$ = {e.iou_analogy_decoded_vs_ground_truth:.2f}  (rank {e.rank_iou_layout} of {e.n_candidates})" + "\n" + rf"    CoM$(I, \hat{{I}})$ = {e.com_analogy_decoded_vs_ground_truth:.1f} px  (rank {e.rank_com_layout} of {e.n_candidates})",
            ha="center", va="center", fontsize=15, linespacing=1.6, transform=am.transAxes, bbox=dict(boxstyle="round,pad=0.8", fc="0.97", ec="0.6", lw=1))
    f.suptitle(NAME[cfg["analogy"]], fontsize=34, weight="bold", y=.99); f.text(.5, .935, cfg["sub"], ha="center", va="center", fontsize=16, color="0.3")
    plt.savefig(out, bbox_inches="tight"); plt.close(f); print("saved", out.relative_to(ROOT))

# ---- run: rank configs as the notebook (pandas sort of top1, descending), decode + figure + summary for the top 3
only = sys.argv[1:]; SUM = []
for an, gen in GEN.items():
    if only and an != only[0] or only[:1] in (["cube_best"], ["goals"], ["pooled"]): continue
    CF = list(gen()); T = pd.DataFrame([dict(top1=c_["top1"]) for c_ in CF]).sort_values("top1", ascending=False)
    for k, ix in enumerate(T.index[:TOP], 1):
        if len(only) > 1 and k != int(only[1]): continue
        cfg = CF[ix]; M, Ih, GC = run(cfg, k); H = M[~M.example]
        print(f"{NAME[an]} #{k} {cfg['sub']}: n {len(H)} · nb top1 {cfg['top1']:.3f} · IoU {H.iou_analogy_decoded_vs_ground_truth.mean():.3f} · rank@1 cos {np.mean(H.rank_cos_latent == 1):.3f} IoU {np.mean(H.rank_iou_layout == 1):.3f}")
        SUM.append(dict(analogy=NAME[an], k=k, config=cfg["sub"], n=len(H), n_candidates=H.n_candidates.median(), nb_top1_cos=cfg["top1"], latent_cos_rank1=np.mean(H.rank_cos_latent == 1),
                        mean_iou=H.iou_analogy_decoded_vs_ground_truth.mean(), iou_rank1=np.mean(H.rank_iou_layout == 1), mean_iou_rank=H.rank_iou_layout.mean(),
                        median_com_px=H.com_analogy_decoded_vs_ground_truth.median(), com_rank1=np.mean(H.rank_com_layout == 1), mean_com_rank=H.rank_com_layout.mean(),
                        ceiling_iou=H.iou_analogy_expected_vs_ground_truth.mean(), baseline_iou=H.iou_source_decoded_vs_ground_truth.mean(), empty=int(H.empty_analogy_decoded.sum())))
        figure(cfg, M.iloc[-1], Ih[-1], GC[cfg["cand"].get_loc(cfg["ex"]["t"])], ROOT / f"notebooks/out/139_decode_{an}_{k}.pdf")
if not only:
    S = pd.DataFrame(SUM); S.to_csv(DC / "summary.csv", index=False); print(S.round(3).to_string())

# ---- best samples of "mean red-cube in box + box at p (from the box's OTHER objects) = red-cube at p" (simple composition, X = red-cube), every box × speaker:
#      keep instances with the true recording rank 1 under latent cos AND layout IoU, sort by IoU(I, Î) (CoM tie-break); top 3 overall (distinct boxes first) + top 1 per box
if only[:1] == ["cube_best"]:
    X, R, PK = "red-cube", [], []
    for cfg in compose_simple():
        spk, box, ob = cfg["key"]
        if ob != X: continue
        M, Ih, GC = run(cfg, f"cube_{box}_{spk}"); H = M[~M.example].reset_index(drop=True)
        ok = H[(H.rank_cos_latent == 1) & (H.rank_iou_layout == 1)].sort_values(["iou_analogy_decoded_vs_ground_truth", "com_analogy_decoded_vs_ground_truth"], ascending=[False, True])
        R.append(dict(box=box, speaker=spk, n=len(H), mean_iou=H.iou_analogy_decoded_vs_ground_truth.mean(), iou_rank1=np.mean(H.rank_iou_layout == 1), cos_rank1=np.mean(H.rank_cos_latent == 1),
                      median_com_px=H.com_analogy_decoded_vs_ground_truth.median(), n_rank1_both=len(ok), best_iou=ok.iou_analogy_decoded_vs_ground_truth.max() if len(ok) else np.nan))
        for i in ok.index: PK.append((cfg, i, H.loc[i], Ih[i], GC[cfg["cand"].get_loc(H.target_row[i])]))
    T = pd.DataFrame(R); T.to_csv(DC / "cube_best_table.csv", index=False); print(T.round(3).to_string())
    PK.sort(key=lambda t: (-t[2].iou_analogy_decoded_vs_ground_truth, t[2].com_analogy_decoded_vs_ground_truth))
    top = []
    for t in PK:  # distinct boxes first, then fill
        if len(top) < 3 and t[0]["key"][1] not in {u[0]["key"][1] for u in top}: top.append(t)
    top += [t for t in PK if t not in top][:3 - len(top)]
    per_box = {}
    for t in PK: per_box.setdefault(t[0]["key"][1], t)
    picks = [(f"best_{j + 1}", t) for j, t in enumerate(top)] + [(f"box_{b}", t) for b, t in per_box.items()]
    rows = []
    for tag, (cfg, i, e, Ih_, gt) in picks:
        spk, box, _ = cfg["key"]; q = cfg["inst"][i]; assert q["t"] == e.target_row
        ex = dict(q) | dict(terms=[(1, q["mX"], ("icon", X), f"mean {X} in {box}"), (1, q["mPos"], ("box", box, q["p"]), f"position p in {box}"), (-1, cfg["mRest"], ("box", box, None), f"mean {box}")],
                            gt=f"ground truth\n({X} at p) {PS(q['t'])}")
        figure(dict(cfg, ex=ex, sub=f"{X} · {box} · spk {spk} · best sample"), e, Ih_, gt, ROOT / f"notebooks/out/139_goal_addpos_cube_{tag}.pdf")
        rows.append(dict(pick=tag, box=box, speaker=spk, target=PS(q["t"]), iou=e.iou_analogy_decoded_vs_ground_truth, com_px=e.com_analogy_decoded_vs_ground_truth, rank_cos=e.rank_cos_latent,
                         rank_l1=e.rank_l1_latent, rank_iou=e.rank_iou_layout, rank_com=e.rank_com_layout, n_candidates=e.n_candidates))
    P = pd.DataFrame(rows); P.to_csv(DC / "cube_best_picks.csv", index=False); print(P.round(3).to_string())

# ---- three goals (paper framing): add position (= simple composition), swap object (= object swap), combine objects (= pair). In the #1 config of each:
#      the 3 best instances (rank 1 under latent cos AND layout IoU, sorted by IoU(I, Î) then CoM), at clearly different positions / arrangements (≥ 8 px apart)
def terms_for(an, cfg, q):
    if an == "combine":
        pair = cfg["key"][0]; (o0, _), (o1, _) = PAIRS[pair]
        return [(1, .5 * Z[q["u"]], ("photo", q["u"]), f"½ · {o0} {PS(q['u'])}"), (1, .5 * Z[q["v"]], ("photo", q["v"]), f"½ · {o1} {PS(q['v'])}"), (1, q["c"], ("tile", "c"), "c (mean interaction)")], f"ground truth\n({pair}) {PS(q['t'])}"
    if an == "swap":
        _, _, X, Yo = cfg["key"]
        return [(1, Z[q["src"]], ("photo", q["src"]), f"{X} at p {PS(q['src'])}"), (-1, cfg["mX"], ("icon", X), f"mean {X}"), (1, cfg["mY"], ("icon", Yo), f"mean {Yo}")], f"ground truth\n({Yo} at p) {PS(q['t'])}"
    _, box, X = cfg["key"]
    return [(1, q["mX"], ("icon", X), f"mean {X} in {box}"), (1, q["mPos"], ("box", box, q["p"]), f"position p in {box}"), (-1, cfg["mRest"], ("box", box, None), f"mean {box}")], f"ground truth\n({X} at p) {PS(q['t'])}"
if only[:1] == ["goals"]:
    GOALS = [("addpos", "compose_simple", ("7", "plastic", "vase")), ("swap", "swap", ("plastic", "7", "red-cube", "vase")), ("combine", "combine", ("red-cube+vase", "plastic", "3"))]
    rows, lines = [], []
    for goal, an, key in GOALS:
        cfg = next(c_ for c_ in GEN[an]() if c_["key"] == key)
        k = {"compose_simple": 1, "swap": 1, "combine": 1}[an]; M, Ih, GC = run(cfg, k); H = M[~M.example].reset_index(drop=True)
        ok = H[(H.rank_cos_latent == 1) & (H.rank_iou_layout == 1)].sort_values(["iou_analogy_decoded_vs_ground_truth", "com_analogy_decoded_vs_ground_truth"], ascending=[False, True])
        where = lambda j: df.loc[j, [f"{key[0]}_x0", f"{key[0]}_y0", f"{key[0]}_x1", f"{key[0]}_y1"]].values.astype(float) if an == "combine" else df.loc[j, ["x", "y"]].values.astype(float)
        pick = []
        for i in ok.index:
            if all(np.linalg.norm(where(H.target_row[i]) - where(H.target_row[j])) >= 8 for j in pick): pick.append(i)
            if len(pick) == 3: break
        for b, i in enumerate(pick, 1):
            q, e = cfg["inst"][i], H.loc[i]; assert q["t"] == e.target_row; T_, gtl = terms_for(an, cfg, q)
            figure(dict(cfg, ex=dict(q) | dict(terms=T_, gt=gtl), sub=f"{cfg['sub']} · best {b} of {len(H)}", analogy=an), e, Ih[i], GC[cfg["cand"].get_loc(q["t"])], ROOT / f"notebooks/out/139_goal_{goal}_best_{b}.pdf")
            rows.append(dict(goal=goal, config=cfg["sub"], best=b, target=PS(q["t"]), source=PS(q["src"]) if q["src"] is not None else "", iou=e.iou_analogy_decoded_vs_ground_truth, com_px=e.com_analogy_decoded_vs_ground_truth,
                             rank_cos=e.rank_cos_latent, rank_l1=e.rank_l1_latent, rank_iou=e.rank_iou_layout, rank_com=e.rank_com_layout, n_candidates=e.n_candidates))
        lines.append(dict(goal=goal, config=cfg["sub"], n=len(H), mean_iou=H.iou_analogy_decoded_vs_ground_truth.mean(), n_rank1_both=len(ok),
                          pick_ious=" / ".join(f"{H.iou_analogy_decoded_vs_ground_truth[i]:.2f}" for i in pick)))
    P, L_ = pd.DataFrame(rows), pd.DataFrame(lines); P.to_csv(DC / "goals_picks.csv", index=False); L_.to_csv(DC / "goals_table.csv", index=False)
    print(P.round(3).to_string()); print(L_.round(3).to_string())

# ---- add position, pooled variants (single-object recordings only: no pairs, no empty, not X; K = 10 nearest to p in the box's 64×64 frame; one speaker;
#      every recording at the target position (within R_PX, the candidates counted as true) is left out of every mean). Per box b:
#      mean_b = single-object mean, mX_b = X mean, pos_b(p) = mean of the K nearest non-X single-object samples to p − mean_b
#      pooled: ẑ = mean_tbox + (mean_b mX_b − mean_b mean_b) + mean_b pos_b(p)     objpos: ẑ = (mean_b mX_b − mean_b mean_b) + Kmean_tbox(p)
#      single: ẑ = mX_tbox + pos_tbox(p)                                                  (mean_b = average over the boxes this speaker has)
KP = 10
def addpos_variants(s, tbox, X):
    SO = df[(df.speaker == s) & ~df.object.str.contains("+", regex=False) & (df.object != "empty")]
    tg = SO[(SO.box == tbox) & (SO.object == X)]
    if len(tg) < 30: return None
    BX = sorted(SO.box.unique()); sumA = {b: Z[SO.index[SO.box == b]].sum(0) for b in BX}; nA = {b: (SO.box == b).sum() for b in BX}
    sumX = {b: Z[SO.index[(SO.box == b) & (SO.object == X)]].sum(0) for b in BX}; nX = {b: ((SO.box == b) & (SO.object == X)).sum() for b in BX}
    NX = {b: SO[(SO.box == b) & (SO.object != X)] for b in BX}; ctr = Z[tg.index].mean(0); out = {v: [] for v in ("pooled", "objpos", "single")}
    for j, r in tg.iterrows():
        ok = np.hypot(tg.x.values - r.x, tg.y.values - r.y) <= R_PX
        if ok.all(): continue
        drop = Z[tg.index[ok]].sum(0); nd = ok.sum()
        mb = {b: (sumA[b] - (drop if b == tbox else 0)) / (nA[b] - (nd if b == tbox else 0)) for b in BX}
        mx = {b: (sumX[b] - (drop if b == tbox else 0)) / (nX[b] - (nd if b == tbox else 0)) for b in BX if nX[b] - (nd if b == tbox else 0) > 0}
        km = {}
        for b in BX:
            d = np.hypot(NX[b].x.values - r.x, NX[b].y.values - r.y); nn = NX[b].index[np.argsort(d)[:KP]]; km[b] = (Z[nn].mean(0), len(nn), d[np.argsort(d)[:KP]].max())
        dX = np.mean([mx[b] for b in mx], 0) - np.mean([mb[b] for b in BX], 0); pos_all = np.mean([km[b][0] - mb[b] for b in BX], 0)
        Z3 = dict(pooled=mb[tbox] + dX + pos_all, objpos=dX + km[tbox][0], single=mx[tbox] + km[tbox][0] - mb[tbox])
        for v, zh in Z3.items():
            cs, a = score(zh, Z[tg.index], ok, ctr)
            out[v].append(inst(zh, tg.index[ok][np.argmax(cs[ok])], tg.index, ok, ctr, mg=cs[ok].max() - cs[~ok].max()) | dict(a=a, p=(r.x, r.y), nK=km[tbox][1], rK=km[tbox][2],
                          parts=dict(mt=mb[tbox], dX=dX, pos_all=pos_all, km=km[tbox][0], mxt=mx[tbox], pt=km[tbox][0] - mb[tbox]), nbox=len(BX)))
    Q = np.sort(SO.index)
    return {v: dict(analogy=f"addpos_{v}", key=(v, tbox, X, s), sub=f"{X} in {tbox} · spk {s}", top1=np.mean([q["a"] == 1 for q in I]), inst=I, ex=best_ex(I), Q=Q) for v, I in out.items()}
def terms_addpos(v, tbox, X, q):
    P_ = q["parts"]
    if v == "pooled": return [(1, P_["mt"], ("box", tbox, None), f"mean {tbox}"), (1, P_["dX"], ("icon", X), f"{X} (all boxes)"), (1, P_["pos_all"], ("box", tbox, q["p"]), "position p (all boxes)")]
    if v == "objpos": return [(1, P_["dX"], ("icon", X), f"{X} (all boxes)"), (1, P_["km"], ("box", tbox, q["p"]), f"position p in {tbox}")]
    return [(1, P_["mxt"], ("icon", X), f"mean {X} in {tbox}"), (1, P_["pt"], ("box", tbox, q["p"]), f"position p in {tbox}")]
if only[:1] == ["pooled"]:
    R, BEST = [], {}
    for X in ("vase", "red-cube"):
        for tbox in ("plastic", "gastro", "cardboard"):
            for s in ("1", "3", "5", "7"):
                V3 = addpos_variants(s, tbox, X)
                if V3 is None: continue
                old = next((c_ for c_ in compose_simple() if c_["key"] == (s, tbox, X)), None)
                for v, cfg in list(V3.items()) + [("old", old)]:
                    if cfg is None: continue
                    tag = f"{tbox}_{X.replace(' ', '_')}_{s}"; M, Ih, GC = run(cfg, f"cube_{tbox}_{s}" if v == "old" and X == "red-cube" else tag); H = M[~M.example].reset_index(drop=True)
                    ok = H[(H.rank_cos_latent == 1) & (H.rank_iou_layout == 1)].sort_values(["iou_analogy_decoded_vs_ground_truth", "com_analogy_decoded_vs_ground_truth"], ascending=[False, True])
                    R.append(dict(variant=v, box=tbox, object=X, speaker=s, n=len(H), mean_iou=H.iou_analogy_decoded_vs_ground_truth.mean(), iou_rank1=np.mean(H.rank_iou_layout == 1),
                                  cos_rank1=np.mean(H.rank_cos_latent == 1), median_com_px=H.com_analogy_decoded_vs_ground_truth.median(), best_iou=ok.iou_analogy_decoded_vs_ground_truth.max() if len(ok) else np.nan,
                                  nK=np.mean([q.get("nK", np.nan) for q in cfg["inst"]]), rK_px=np.mean([q.get("rK", np.nan) for q in cfg["inst"]]), n_boxes=cfg["inst"][0].get("nbox", 1)))
                    if (v, tbox, X) in (("pooled", "plastic", "vase"), ("objpos", "cardboard", "red-cube")): BEST.setdefault((v, tbox, X), []).append((H.iou_analogy_decoded_vs_ground_truth.mean(), cfg, H, ok, Ih, GC))
    T = pd.DataFrame(R); T.to_csv(DC / "addpos_variants_table.csv", index=False); print(T.round(3).to_string())
    rows = []
    for (v, tbox, X), L in BEST.items():
        _, cfg, H, ok, Ih, GC = max(L, key=lambda t: t[0])  # the best speaker = highest mean IoU
        pick = []
        for i in ok.index:
            if all(np.hypot(*(df.loc[H.target_row[i], ["x", "y"]].values - df.loc[H.target_row[j], ["x", "y"]].values).astype(float)) >= 8 for j in pick): pick.append(i)
            if len(pick) == 3: break
        for b, i in enumerate(pick, 1):
            q, e = cfg["inst"][i], H.loc[i]; assert q["t"] == e.target_row
            ex = dict(q) | dict(terms=terms_addpos(v, tbox, X, q), gt=f"ground truth\n({X} at p) {PS(q['t'])}")
            figure(dict(cfg, ex=ex, sub=f"{cfg['sub']} · {'pooled over boxes' if v == 'pooled' else 'object + position in a box'} · best {b} of {len(H)}"), e, Ih[i], GC[cfg["cand"].get_loc(q["t"])],
                   ROOT / f"notebooks/out/139_goal_addpos_{v}_{tbox}_{X.replace(' ', '_')}_best_{b}.pdf", fit=True)
            rows.append(dict(variant=v, box=tbox, object=X, speaker=cfg["key"][3], best=b, target=PS(q["t"]), iou=e.iou_analogy_decoded_vs_ground_truth, com_px=e.com_analogy_decoded_vs_ground_truth,
                             rank_cos=e.rank_cos_latent, rank_l1=e.rank_l1_latent, rank_iou=e.rank_iou_layout, rank_com=e.rank_com_layout, n_candidates=e.n_candidates, nK=q["nK"], config_mean_iou=H.iou_analogy_decoded_vs_ground_truth.mean()))
    P = pd.DataFrame(rows); P.to_csv(DC / "addpos_variants_picks.csv", index=False); print(P.round(3).to_string())
