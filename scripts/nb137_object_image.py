"""nb137 object-image explorer (server): PCA/UMAP of input/latent/output, each sample drawn as its object's picture
(assets/objects/individual) with a contour in the point colour; 3-D (three.js, plotly-style walls/ticks) or 2-D.
Everything except run/epoch/port is a dropdown in the page (scripts/nb137_object_image.html).

  .venv/bin/python scripts/nb137_object_image.py --run decres-rescale-aug10-v1 [--epoch 200] [--port 8610]
  .venv/bin/python scripts/nb137_object_image.py --run decres-rescale-aug10-v1 [--epoch 200] --precompute   # cache every dropdown subset, then exit

Caches (runs/_src9_cache/nb137/<run>[/ep<N>]): feats.npz + meta.csv (one eager fp32 pass of the checkpoint, transformer4 runs only;
the default run at the latest epoch reuses nb137's existing cache) and red_*.npz (one per PCA/UMAP fit, same keys as nb137)."""
import argparse, base64, colorsys, functools, hashlib, html, http.server, io, json, socket, sys, time, urllib.parse, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
# same settings as nb137 cell 1 (IGNORE / UMAP are part of the reduction cache key)
IGNORE = ["eval/cardboard/vase-laser-shifted-4", "shifted", "unseen-speaker", "cylinder", "portable-charger", "biscoff", "coffee", "lid"]
DISPLAY_OFF = [  # display-picker entries containing any of these start unticked
    "shifted", "unseen-speaker", "cylinder", "portable-charger", "biscoff", "coffee", "lid"]
UMAP = dict(n_neighbors=30, min_dist=0.1)
RENAME = {"green-cube+red-cube": "two objects", "red-cube+vase": "two objects", "soap dispenser": "soap", "red-cube": "cube"}
COLOURS = {"empty": "black", "two objects": "black", "gastro": "#2ca02c", "plastic": "#1f77b4", "cardboard": "#ff7f0e"}  # fixed colours (boxes as in nb136)
IMAGES = {"candle": "candle", "red-cube": "cube", "soap dispenser": "soap", "vase": "vase", "green-cube+red-cube": "two_cubes",
          "red-cube+vase": "cube_and_vase", "coffee pot": "coffee_pot", "tomato sauce": "tomato_sauce"}
CAT = ["split", "set", "box", "object", "speaker"]
NUM = ["x", "y", "iou", "area", "n_objects"]
BEST = ["none", "xy", "split", "set", "box", "object", "speaker", "x", "y", "iou"]  # pick components by this feature (see best())
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]  # tab10 (more distinct than plotly's blue/violet pair)
BOXES = {"2026_09_27_gastronorm_four_objs": "gastro", "2026_10_05_green_plastic_four_objs": "plastic", "2026_10_11_cardboard_four_objs": "cardboard"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", default="decres-rescale-aug10-v1"); ap.add_argument("--epoch", type=int, default=None, help="checkpoint epoch (default: latest)")
ap.add_argument("--port", type=int, default=8610)
ap.add_argument("--precompute", action="store_true", help="fit + cache every dropdown subset (box × speaker × object × vector × method), then exit")
A = ap.parse_args()

# ---- features for (run, epoch): cached, else one eager fp32 pass (same as scripts/nb136_base.py)
CK = ROOT / "runs" / A.run / "checkpoints"
latest = (CK / "latest-rank0.pt").resolve().name
ckpt = latest if A.epoch is None else next((p.name for p in CK.glob(f"ep{A.epoch}-*rank0.pt")), None)
assert ckpt, f"no checkpoint for epoch {A.epoch} in {CK}: {sorted(p.name for p in CK.glob('ep*'))}"
C = ROOT / "runs/_src9_cache/nb137" / A.run / ("" if ckpt == latest else f"ep{A.epoch}"); C.mkdir(parents=True, exist_ok=True)
if not (C / "feats.npz").exists():
    import torch
    sys.path.insert(0, str(ROOT / "src9"))
    from dataset import build_dataset, load_splits
    from arch.transformer4 import Transformer4
    a = json.load(open(ROOT / "runs" / A.run / "args.json")); assert a["model"] == "transformer4", f"only transformer4 runs supported, got {a['model']}"
    print(f"extracting features from {ckpt} (one pass over every sample, a few minutes)...", flush=True); t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    _, ev, tr_ev, info = build_dataset(a.get("window", "tukey"), num_workers=0)
    model = Transformer4(info, decoder=a.get("decoder", "two"), latent_norm=a.get("latent_norm", "ln"), freq_rope=a.get("freq_rope", "linear")).cuda().eval()
    print(model.load_state_dict(torch.load(CK / ckpt, map_location="cpu", weights_only=False)["state"]["model"]))
    jobs = [("train", tr_ev.dataloader.dataloader.dataset)] + [(e.label, e.dataloader.dataloader.dataset) for e in ev]
    DS, idx = {"train": jobs[0][1].dataset, "eval": jobs[1][1].dataset}, [(n, i) for n, sub in jobs for i in sub.indices]
    B = {k: [] for k in "u p y lb".split()}
    with torch.no_grad():
        for s in range(0, len(idx), 128):
            rows = [DS["train" if n == "train" else "eval"][i] for n, i in idx[s:s + 128]]
            x = torch.stack([r["fft"] for r in rows]).cuda(); o = model(dict(fft=x))
            B["u"].append(o["latent"].float().cpu()); B["p"].append(o["mask_pred"].float().cpu())
            B["y"].append(torch.stack([r["mask_true"] > 0.5 for r in rows]))
            B["lb"].append((x + 1e-3).log().view(len(x), 100, 18, 65, 2).mean((3, 4)).float().cpu())
            if s % 2560 == 0: print(f"  {s}/{len(idx)} {time.time() - t0:.0f}s", flush=True)
    B = {k: torch.cat(v).numpy() for k, v in B.items()}
    P, Y = B["p"], B["y"]; Pb = P > 0.08; rr, cc = np.mgrid[:64, :64]; ar = Y.sum((1, 2))
    metas = list(load_splits()[0].values()); M = [metas[i] for _, i in idx]
    df = pd.DataFrame(dict(split=[n for n, _ in idx], box=[BOXES[m["experiment"]] for m in M], object=["+".join(sorted(m["objects"])) or "empty" for m in M],
                           speaker=[str(m["speaker"]) for m in M], pos=[m["position_id"] for m in M],
                           y=np.where(ar > 0, (Y * rr).sum((1, 2)) / np.maximum(ar, 1), np.nan), x=np.where(ar > 0, (Y * cc).sum((1, 2)) / np.maximum(ar, 1), np.nan),
                           iou=np.where(ar > 0, (Pb & Y).sum((1, 2)) / np.maximum((Pb | Y).sum((1, 2)), 1), np.nan), area=ar))
    df["set"] = np.where(df.split == "train", "train", "eval")
    np.savez(C / "feats.npz", input=B["lb"].reshape(len(df), -1), latent=B["u"], output=P.reshape(-1, 16, 4, 16, 4).mean((2, 4)).reshape(len(df), -1))
    df.to_csv(C / "meta.csv", index=False); print(f"features cached in {C} ({time.time() - t0:.0f}s)")
    del model, B, P, Y; torch.cuda.empty_cache()

# ---- data (same filtering as nb137 cell 2)
F, df = dict(np.load(C / "feats.npz")), pd.read_csv(C / "meta.csv", dtype={"speaker": str})
if "area" not in df: df["area"] = np.load(ROOT / "runs/_src9_cache/nb136/base.npz")["y"].sum((1, 2))  # old nb137 cache (default run only)
keep = ~df.split.isin(IGNORE).values
df["disp"] = np.where(df.split == "train", "train/" + df.object, df.split)  # display-picker entry: eval split, or train/<object>
df["n_objects"] = np.where(df.object == "empty", 0, df.object.str.count(r"\+") + 1); df["object_raw"] = df.object; df["object"] = df.object.replace(RENAME)
df = df[keep].reset_index(drop=True); F = {k: v[keep] for k, v in F.items()}

# ---- sprites: object picture + contour ring (64 px PNG data URIs), one per raw object, sent once
def png(a): b = io.BytesIO(); Image.fromarray(a).save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
def sprite(o):
    p = ROOT / "assets/objects/individual" / f"{IMAGES.get(o, '-')}.png"
    if not p.is_file(): return None  # drawn as a cross (e.g. empty box)
    im = Image.open(p).convert("RGBA"); im.thumbnail((54, 54)); w, h = im.size; a = np.zeros((64, 64, 4), np.uint8)
    a[(64 - h) // 2:(64 - h) // 2 + h, (64 - w) // 2:(64 - w) // 2 + w] = np.asarray(im)
    al = a[..., 3] > 128; ring = np.zeros_like(a); ring[binary_dilation(al, iterations=4) & ~al] = 255
    return [png(a), png(ring)]
OBJS = sorted(df.object_raw.unique()); IX = {o: i for i, o in enumerate(OBJS)}
META = dict(run=A.run, ckpt=ckpt, sprites=[sprite(o) for o in OBJS], box=["all", *sorted(df.box.unique())], speaker=["all", *sorted(df.speaker.unique(), key=int)],
            object=["all", *df.object.value_counts().index], vector=["latent", "input", "output"], split=["both", "train", "test"], method=["pca", "umap"], colour=["xy", *CAT, *NUM], size=["none", *CAT, *NUM], best=BEST,
            splits=[[s, int(n), not any(h in s for h in DISPLAY_OFF)] for s, n in df.disp.value_counts().items()])  # display picker: [split, n, ticked by default (no DISPLAY_OFF match)]

# ---- reduction (same cache key as nb137 reduce(); disk + memory cache)
@functools.lru_cache(maxsize=64)
def reduce(box, spk, obj, data, method):
    m = np.ones(len(df), bool) if box is None else (df.box == box).values.copy()
    if spk: m &= df.speaker.isin(spk).values
    if obj: m &= df.object.isin(obj).values
    if m.sum() < 10: return m, None, None
    tr = (df.set == "train").values[m]  # fit on the train rows of the subset only, then project every row of the subset (train + eval)
    if tr.sum() < 10: return m, None, None
    key = (method, sorted(IGNORE), box, list(spk) if spk else None, list(obj) if obj else None, data) + (("k10",) if method == "pca" else tuple(sorted(UMAP.items()))) + ("fit=train",)
    f = C / f"red_{hashlib.sha1(repr(key).encode()).hexdigest()[:10]}.npz"
    if not f.exists() or len(np.load(f)["r"]) != m.sum():
        from sklearn.decomposition import PCA
        t0 = time.time(); X = F[data][m]; X = X - X[tr].mean(0)
        if method == "pca": p = PCA(10, random_state=0).fit(X[tr]); r = p.transform(X); ev = p.explained_variance_ratio_
        else:
            import umap
            pc = PCA(min(50, *X[tr].shape), random_state=0).fit(X[tr]); Z = pc.transform(X)
            u = umap.UMAP(n_components=3, **(UMAP | dict(n_neighbors=min(UMAP["n_neighbors"], int(tr.sum()) - 1))), random_state=0).fit(Z[tr])
            r = np.zeros((len(X), 3), np.float32); r[tr] = u.embedding_; r[~tr] = u.transform(Z[~tr]) if (~tr).any() else r[~tr]; ev = np.full(3, np.nan)
        tmp = f.with_suffix(".tmp.npz"); np.savez(tmp, r=r, ev=ev); tmp.replace(f)  # atomic: a running server never sees a half-written file
        print(f"fitted {method} {box or 'all'} {data} on {tr.sum()} train, projected {len(X)} in {time.time() - t0:.0f}s", flush=True)
    z = np.load(f); return m, z["r"], z["ev"] if "ev" in z else np.full(3, np.nan)

# ---- colours (quantised to 16 levels/channel → few distinct tinted sprites)
def hexq(rgb01): return "#" + "".join(f"{int(round(round(c * 15) * 17)):02x}" for c in rgb01)
def named(s): return {"black": (0, 0, 0), "lightgray": (.83, .83, .83)}.get(s) or tuple(int(s[i:i + 2], 16) / 255 for i in (1, 3, 5))
def colours(d, c):
    if c == "xy":  # x = hue, y = lightness of the true mask CoM (same as nb137 rgb())
        lo, hi = d[["x", "y"]].min().values, d[["x", "y"]].max().values; u = np.clip((d[["x", "y"]].values - lo) / (hi - lo), 0, 1)
        return [hexq(named(COLOURS[o])) if o in COLOURS else hexq(colorsys.hls_to_rgb(.8 * a, .8 - .55 * b, .9)) if a == a else hexq(named("lightgray")) for o, (a, b) in zip(d.object, u)], {}
    if c in CAT:
        keys = sorted(df[c].astype(str).unique(), key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0, k))  # ALL values in the data → a value keeps its colour in every view
        cm = {k: hexq(named(COLOURS.get(k, PALETTE[i % len(PALETTE)]))) for i, k in enumerate(keys)}; shown = set(d[c].astype(str))
        return [cm[k] for k in d[c].astype(str)], {k: v for k, v in cm.items() if k in shown}
    import matplotlib; v = d[c].astype(float); lo, hi = v.min(), v.max(); t = ((v - lo) / (hi - lo + 1e-12)).fillna(0)
    return [hexq(matplotlib.colormaps["viridis"](x)[:3]) for x in t], {f"{lo:.3g}": hexq(matplotlib.colormaps["viridis"](0.)[:3]), f"{hi:.3g}": hexq(matplotlib.colormaps["viridis"](1.)[:3])}

def sizes(d, c):  # per-point image scale (0.5..1.5, 8 levels): categorical → codes 1..k of the sorted labels, numeric → min..max (NaN = smallest)
    if c == "none": return np.ones(len(d)), {}
    if c in CAT: k = sorted(d[c].astype(str).unique()); code = d[c].astype(str).map({v: i for i, v in enumerate(k)}).values; t = code / max(len(k) - 1, 1); leg = {v: round(.5 + i / max(len(k) - 1, 1), 2) for i, v in enumerate(k)}
    else: v = d[c].astype(float); t = ((v - v.min()) / (v.max() - v.min() + 1e-12)).fillna(0).values; leg = {f"{v.min():.3g}": .5, f"{v.max():.3g}": 1.5}
    return .5 + (t if c in CAT else np.round(t * 7) / 7), leg  # categories: exact levels; numeric: 8 bins

def ticks(vals):
    from matplotlib.ticker import MaxNLocator
    lo, hi = float(vals.min()), float(vals.max()); return [[2 * (t - lo) / (hi - lo or 1) - 1, f"{t:.3g}"] for t in MaxNLocator(6).tick_values(lo, hi) if lo <= t <= hi]

# ---- "best" components (same scores as nb137): xy = Procrustes R² of per-position centroids on a component pair vs the mask-CoM grid
# (+ 3rd = best linear R² on (x, y)); categories = η² (between-group / total variance); x / y / iou = Spearman ρ²
def eta2(v, g): return sum((g == c).sum() * (v[g == c].mean() - v.mean()) ** 2 for c in np.unique(g)) / ((v - v.mean()) ** 2).sum()
def rho2(v, c):
    from scipy.stats import spearmanr
    ok = ~np.isnan(c); return spearmanr(v[ok], c[ok])[0] ** 2
def best(r, d, f, n):  # → (0-based component indices, score text)
    import itertools
    if f == "xy":
        g = d.assign(**{f"c{i}": r[:, i] for i in range(r.shape[1])}).groupby("pos").mean(numeric_only=True).dropna(subset=["x"])
        G = g[["x", "y"]].values - g[["x", "y"]].values.mean(0); Cn = g[[f"c{i}" for i in range(r.shape[1])]].values; Cn = Cn - Cn.mean(0)
        fit = {(a, b): np.linalg.svd(G.T @ Cn[:, [a, b]], compute_uv=False).sum() ** 2 / ((G ** 2).sum() * (Cn[:, [a, b]] ** 2).sum()) for a, b in itertools.combinations(range(r.shape[1]), 2)}
        p = list(max(fit, key=fit.get)); ok = d.x.notna().values; A = np.c_[d.x.values[ok], d.y.values[ok], np.ones(ok.sum())]
        e = r[ok] - A @ np.linalg.lstsq(A, r[ok], rcond=None)[0]; r2 = 1 - e.var(0) / r[ok].var(0)
        p += [int(i) for i in np.argsort(-r2) if i not in p][:n - 2]
        return p, f"map-fit R² {fit[tuple(sorted(p[:2]))]:.2f}" + (f" · 3rd: xy R² {r2[p[2]]:.2f}" if n == 3 else "")
    if d[f].nunique() < 2: return list(range(n)), f"{f} is constant here"
    sc = np.array([(rho2 if f in ("x", "y", "iou") else eta2)(r[:, i], d[f].astype(float).values if f in ("x", "y", "iou") else d[f].astype(str).values) for i in range(r.shape[1])])
    p = [int(i) for i in np.argsort(-sc)[:n]]; return p, " ".join(f"{i + 1}:{sc[i]:.2f}" for i in p) + (" (ρ²)" if f in ("x", "y", "iou") else " (η²)")

def plot(q):
    g = lambda k, dv=None: q.get(k, [dv])[0]
    box, spk, obj = [None if g(k, "all") == "all" else g(k) for k in ("box", "speaker", "object")]
    data, split, method, colour, size = g("vector", "latent"), g("split", "both"), g("method", "pca"), g("colour", "xy"), g("size", "none")
    m, r, ev = reduce(box, (spk,) if spk else None, (obj,) if obj else None, data, method)
    if r is None: return dict(error=f"only {m.sum()} samples (< 10)")
    display = g("display", "").split(",")  # entries ticked in the display picker (never affects the fit)
    v = df.disp.isin(display).values[m] & (True if split == "both" else (df.set.values[m] == "train") == (split == "train"))
    if not v.any(): return dict(error=f"nothing to display (split {split}, {len(display)} ticked display entries) in this subset")
    d, r = df[m][v].reset_index(drop=True), r[v]
    idx = [int(g(k)) - 1 for k in ("x", "y", "z") if g(k, "none") != "none"][:3]; bf, btxt = g("best", "none"), ""
    if bf != "none": idx, btxt = best(r, d, bf, 3 if g("z", "none") != "none" else 2)  # z = none → best 2-D pair, else best 3
    if len(idx) < 2 or max(idx) >= r.shape[1]: return dict(error=f"pick 2-3 components ≤ {r.shape[1]} for {method}")
    names = [(f"PC{i + 1} ({ev[i]:.0%})" if method == "pca" else f"UMAP {i + 1}") for i in idx]
    nrm = lambda c: 2 * (c - c.min()) / (np.ptp(c) or 1) - 1
    xyz = np.stack([nrm(r[:, i]) for i in idx] + ([np.zeros(len(d))] if len(idx) == 2 else []), 1).round(4)
    col, legend = colours(d, colour); sc, sleg = sizes(d, size)
    hov = [html.escape(f"{s.split('/')[-1]} · {o} · spk {k} · pos {p} · true {x:.0f},{y:.0f} · iou {u:.2f}") for s, o, k, p, x, y, u in zip(d.split, d.object, d.speaker, d.pos, d.x, d.y, d.iou)]
    title = (f"{A.run} · {ckpt.split('-')[0]} · box {box or 'all'} · speaker {spk or 'all'} · object {obj or 'all'} · {data} ({F[data].shape[1]}-d) · {method} fit on train · showing {split} · n {len(d)}"
             f" · colour {colour} · size {size}" + (" (x = hue, y = lightness of the true mask CoM)" if colour == "xy" else "") + (f" · best {bf}: {btxt}" if bf != "none" else ""))
    return dict(title=title, axes=names, ticks=[ticks(r[:, i]) for i in idx], d3=len(idx) == 3, legend=legend, size_legend=sleg, picked=[i + 1 for i in idx], ncomp=int(r.shape[1]),
                pts=[[*xyz[i].tolist(), IX[d.object_raw[i]], col[i], hov[i], round(float(sc[i]), 3)] for i in range(len(d))])

if A.precompute:  # PCA first (fast), then UMAP; within each: no filter → speaker → object → speaker × object
    import itertools
    SP, OB, BX = META["speaker"][1:], META["object"][1:], [None, *META["box"][1:]]
    combos = [(None, None)] + [(s, None) for s in SP] + [(None, o) for o in OB] + list(itertools.product(SP, OB))
    jobs = [(mt, s, o, b, v) for mt in ("pca", "umap") for s, o in combos for b in BX for v in ("latent", "input", "output")]
    t0, done, skip = time.time(), 0, 0
    for n, (mt, s, o, b, v) in enumerate(jobs):
        try: _, r, _ = reduce(b, (s,) if s else None, (o,) if o else None, v, mt); done += r is not None; skip += r is None
        except Exception as e: print(f"FAILED {mt} box={b} spk={s} obj={o} {v}: {e!r}", flush=True)
        if n % 50 == 0: print(f"[{n}/{len(jobs)}] {time.time() - t0:.0f}s · {done} cached · {skip} too small", flush=True)
    print(f"precompute done: {done} fits cached, {skip} subsets too small, {time.time() - t0:.0f}s"); sys.exit()

PAGE = (ROOT / "scripts/nb137_object_image.html")
class H(http.server.BaseHTTPRequestHandler):
    def send(self, body, ctype="application/json"):
        b = body.encode(); self.send_response(200); self.send_header("Content-Type", ctype + "; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        try:
            if u.path == "/": self.send(PAGE.read_text(), "text/html")  # re-read each time: edit the template, reload the page
            elif u.path == "/api/meta": self.send(json.dumps(META))
            elif u.path == "/api/plot": t0 = time.time(); out = plot(urllib.parse.parse_qs(u.query)); self.send(json.dumps(out)); print(f"plot {u.query} {time.time() - t0:.1f}s", flush=True)
            else: self.send_error(404)
        except Exception as e:
            import traceback; traceback.print_exc(); self.send(json.dumps(dict(error=repr(e))))
    def log_message(self, *a): pass

srv = http.server.HTTPServer(("", A.port), H)  # single-threaded: one fit at a time
print(f"\n{A.run} · {ckpt} · {len(df)} samples\n  open → http://localhost:{A.port}/\n  remote? ssh -L {A.port}:localhost:{A.port} {socket.gethostname()}  then open the URL above (Ctrl-C to stop)", flush=True)
srv.serve_forever()
