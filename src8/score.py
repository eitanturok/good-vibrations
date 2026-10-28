"""Thresholded scores of a run's saved eval outputs (runs/<run>/outputs_history/eval), so runs from any src are scored the same way.
mask_pred > thr vs mask_true > 0.5 (the true mask BOX-downsampled to the pred's grid, as dataset.process_y does):
  iou  = hard IoU per sample      miss = 1 if no pixel over thr lands on the object
Each split's mean over samples, then means over splits per category (dataset.CATEGORIES) and over all splits, as EvalSummary does.
  .venv/bin/python src8/score.py RUN [RUN ...] [--epoch 750] [--thr 0.08]"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from dataset import EXPERIMENTS, CATEGORIES, UNSEEN_ANGLE, load_meta

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs/_score_truth"

def sample_dir(exp, sid, pos, spk): return EXPERIMENTS / exp / "samples" / (f"{sid:06d}" if sid >= 0 else f"{pos:06d}-{spk}")

def truth(exp, sid, pos, spk, H, W, gray=False):
    """(H,W) true mask in [0,1]; a with-lid sample takes the mask of its without-lid twin (process_y). Cached per sample.
    gray: the overhead photo in grayscale instead (dataset.process_y's gray target)"""
    f = CACHE / f"{H}x{W}{'_gray' if gray else ''}" / exp / f"{sid}_{pos}_{spk}.npy"
    if f.exists(): return np.load(f)
    d = sample_dir(exp, sid, pos, spk)
    meta = load_meta(d)
    if "without_lid_position_id" in meta: d = sample_dir(exp, -1, meta["without_lid_position_id"], spk)
    if not d.exists():  # numbered dirs (gastro): find the twin by metadata
        d = next(s for s in (EXPERIMENTS / exp / "samples").iterdir() if (m := load_meta(s)).get("position_id") == meta["without_lid_position_id"] and m.get("speaker") == spk)
    img = Image.open(d / "image/02_cropped_overhead.png").convert("L") if gray else Image.open(d / "image/03_smask.png").convert("F")
    y = np.clip(np.asarray(img.resize((W, H), Image.BOX), dtype=np.float32) / 255.0, 0, 1)
    f.parent.mkdir(parents=True, exist_ok=True); np.save(f, y)
    return y

def positions(exp):
    """DataFrame (position_id, com_row, com_col) per position of an experiment: the mean of its objects' coms, box fractions. Cached"""
    f = CACHE / f"positions_{exp}.csv"
    if f.exists(): return pd.read_csv(f)
    rows = {}
    for d in (EXPERIMENTS / exp / "samples").iterdir():
        try: m = load_meta(d)
        except FileNotFoundError: continue
        if m.get("position_id") in rows or not m.get("coms"): continue
        H, W = m["crop_overhead_shape"][:2]
        c = np.array([[k[0][0] / H, k[0][1] / W] for k in m["coms"]]).mean(0)
        rows[m["position_id"]] = (m["position_id"], *c)
    df = pd.DataFrame(list(rows.values()), columns=["position_id", "com_row", "com_col"])
    f.parent.mkdir(parents=True, exist_ok=True); df.to_csv(f, index=False)
    return df

def train_dist(run, df):
    """each eval sample's distance (box fractions) to the nearest TRAINING position's com in the same experiment (any object):
    interpolation vs extrapolation, the signal saturates beyond ~0.15 (EXPERIMENTS.md section 2)"""
    tr = pd.read_csv(ROOT / "runs" / run / "positions/train.txt", sep=" ", names=["experiment", "position_id"])
    out = np.full(len(df), np.nan)
    for exp, g in df.groupby("experiment"):
        pos = positions(exp).set_index("position_id")
        t = pos.loc[pos.index.intersection(tr[tr.experiment == exp].position_id), ["com_row", "com_col"]].to_numpy()
        e = pos.reindex(g.position_id)[["com_row", "com_col"]].to_numpy()
        if len(t): out[g.index] = np.sqrt(((e[:, None] - t[None]) ** 2).sum(-1)).min(1)
    return out

def band(m): return m & ~ndimage.binary_erosion(m, iterations=2)

def axis(m):
    """(H,W) bool -> (principal-axis angle, eccentricity (l1-l2)/(l1+l2)), NaN if empty"""
    r, c = np.nonzero(m)
    if len(r) < 3: return np.nan, np.nan
    rr, cc, rc = r.var(), c.var(), ((r - r.mean()) * (c - c.mean())).mean()
    return 0.5 * np.arctan2(2 * rc, rr - cc), np.sqrt((rr - cc) ** 2 + 4 * rc ** 2) / max(rr + cc, 1e-9)

from functools import lru_cache
@lru_cache(maxsize=None)
def layout(exp, sid, pos, spk): return load_meta(sample_dir(exp, sid, pos, spk)).get("layout", "?")

def last_epoch(run): return max(int(p.name[2:6]) for p in (ROOT / "runs" / run / "outputs_history/eval").rglob("ep*.pt"))

SCORE_V = 4  # bump when per_sample's columns change: cached scores of older versions are recomputed

def per_sample(run, epoch, thrs):
    """DataFrame, one row per eval sample: box, split, ids, and iou@thr / miss@thr / ... for every thr. Cached per (run, epoch, thrs)
    in runs/<run>/scores/, keyed on the newest output file's mtime, so a re-saved epoch is rescored"""
    files = sorted((ROOT / "runs" / run / "outputs_history/eval").glob(f"*/*/ep{epoch:04d}-ba*.pt"))
    if not files: raise FileNotFoundError(f"{run}: no eval outputs at epoch {epoch}")
    stamp = max(f.stat().st_mtime_ns for f in files)
    cache = ROOT / "runs" / run / "scores" / f"ep{epoch:04d}_{'_'.join(map(str, thrs))}_v{SCORE_V}.pkl"
    if cache.exists():
        c = pd.read_pickle(cache)
        if c.attrs.get("stamp") == stamp: return c
    df = _per_sample(run, epoch, thrs)
    df.attrs["stamp"] = stamp
    cache.parent.mkdir(parents=True, exist_ok=True); df.to_pickle(cache)
    return df

def _per_sample(run, epoch, thrs):
    rows = []
    for f in sorted((ROOT / "runs" / run / "outputs_history/eval").glob(f"*/*/ep{epoch:04d}-ba*.pt")):
        box, split = f.parent.parent.name, f.parent.name
        d = torch.load(f, weights_only=False, map_location="cpu")
        p, info = d["mask_pred"].float().numpy(), d["info"]
        H, W = p.shape[-2:]
        for i in range(len(p)):
            y = truth(info["experiment"][i], info["sample_id"][i], int(info["position_id"][i]), int(info["speaker"][i]), H, W)
            obj = y > 0.5
            rr, cc = np.nonzero(obj)
            r = dict(box=box, split=split, experiment=info["experiment"][i], position_id=int(info["position_id"][i]), speaker=int(info["speaker"][i]),
                     soft_iou=np.minimum(p[i], y).sum() / max(np.maximum(p[i], y).sum(), 1e-6), peak=p[i].max(),
                     wall_dist=1 - (rr.mean() + 0.5) / H if len(rr) else np.nan,     # lasers sit on the bottom wall of the crop (nb96)
                     layout=layout(info["experiment"][i], int(info["sample_id"][i]), int(info["position_id"][i]), int(info["speaker"][i])),
                     com_r=(rr.mean() + 0.5) / H if len(rr) else np.nan, com_c=(cc.mean() + 0.5) / W if len(rr) else np.nan, true_area=int(obj.sum()),
                     iou_auc=np.mean([((p[i] > t) & obj).sum() / max(((p[i] > t) | obj).sum(), 1) for t in np.linspace(0.02, 0.5, 13)]))
            for t in thrs:
                b = p[i] > t
                r[f"iou@{t}"] = (b & obj).sum() / max((b | obj).sum(), 1)
                r[f"miss@{t}"] = float(not (b & obj).any())
                r[f"area_ratio@{t}"] = b.sum() / max(obj.sum(), 1)
                lab, n = ndimage.label(b)
                r[f"phantom@{t}"] = float(n > 0 and len(set(range(1, n + 1)) - set(np.unique(lab[obj]))) > 0)  # a blob touching no object
                pb, tb = band(b), band(obj)
                r[f"biou@{t}"] = (pb & tb).sum() / max((pb | tb).sum(), 1)
                r[f"_mask@{t}"] = b
                br, bc = np.nonzero(b)
                r[f"com_err@{t}"] = np.hypot(br.mean() / H - rr.mean() / H, bc.mean() / W - cc.mean() / W) if len(br) and len(rr) else np.nan
                (tt, et), (tp, _) = axis(obj), axis(b)
                d = abs(tt - tp) % np.pi                                              # degrees, elongated true masks only (chance ~45)
                r[f"orient_err@{t}"] = np.degrees(min(d, np.pi - d)) if et > 0.2 and not np.isnan(tp) else np.nan
            rows.append(r)
    df = pd.DataFrame(rows)
    df["train_dist"] = train_dist(run, df)
    for t in thrs:  # cross-speaker consistency: mean pairwise IoU between the speakers' predictions at the same position (no labels)
        cons = {}
        for k, g in df.groupby(["experiment", "position_id"]):
            ms = list(g[f"_mask@{t}"])
            v = [(a & b).sum() / max((a | b).sum(), 1) for j, a in enumerate(ms) for b in ms[j + 1:]]
            cons[k] = np.mean(v) if v else np.nan
        df[f"spk_consistency@{t}"] = [cons[k] for k in zip(df.experiment, df.position_id)]
        df = df.drop(columns=f"_mask@{t}")
    return df

def summarize(df, cols):
    """split means -> {box/category: mean over its splits}, plus box/all and ALL (every split counts once)"""
    sp = df.groupby(["box", "split"])[cols].mean().reset_index()
    cat = {s: c for b, cs in CATEGORIES.items() for c, ss in cs.items() for s in ss}
    sp["category"] = [cat.get(s, "?") for s in sp.split]
    out = {}
    for (b, c), g in sp.groupby(["box", "category"]): out[f"{b}/{c}"] = g[cols].mean()
    nc = sp[~sp.category.isin(["control", "unseen-angle", "unseen-position"])]         # WEDGE and UNSEEN-POS on their own (ALL stays comparable)
    for b, g in nc.groupby("box"): out[f"{b}/all"] = g[cols].mean()
    core = sp[sp.split.isin(["candle", "cube", "soap-dispenser", "vase"])]
    out["CORE"] = core[cols].mean()                                                    # the four base objects x 3 boxes: headline
    out["HARD"] = sp[(sp.box == "cardboard") & (sp.split == "soap-dispenser-rotated")][cols].mean()  # the hardest case
    # the 30-60 deg wedge (dataset.UNSEEN_ANGLE): unseen in a *_wedge run (its own split), its 6 eval positions at trained angles
    # otherwise; HARD-kept = the rotated soap outside the wedge, the same angles in both kinds of run
    rot = (df.box == "cardboard") & (df.split == "soap-dispenser-rotated")
    wedge = df.position_id.astype(int).isin(UNSEEN_ANGLE)
    out["WEDGE"] = df[(df.box == "cardboard") & ((df.split == "soap-dispenser-unseen-angle") | rot & wedge)][cols].mean()
    out["HARD-kept"] = df[rot & ~wedge][cols].mean()
    out["UNSEEN-POS"] = sp[sp.split == "unseen-position"][cols].mean()                # the *_region split: single objects >= half inside dataset.REGION, every box
    out["ALL"] = nc[cols].mean()
    out["ALL-ood"] = nc[nc.category != "id"][cols].mean()
    out["ALL-id"] = nc[nc.category == "id"][cols].mean()
    return pd.DataFrame(out).T, sp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--epoch", type=int, default=None, help="default: each run's last saved eval")
    ap.add_argument("--thr", type=float, nargs="+", default=[0.08])
    a = ap.parse_args()
    res = {}
    for run in a.runs:
        ep = a.epoch or last_epoch(run)
        df = per_sample(run, ep, a.thr)
        t = a.thr[0]
        s, _ = summarize(df, [f"iou@{t}", f"miss@{t}", f"phantom@{t}", f"area_ratio@{t}", f"biou@{t}", f"orient_err@{t}", "iou_auc", f"spk_consistency@{t}"])
        res[f"{run}@{ep}"] = s
        print(f"\n== {run} ep{ep} ({len(df)} samples)\n" + s.loc[["CORE", "HARD", "HARD-kept", "WEDGE", "UNSEEN-POS", "ALL", "ALL-ood"] + [k for k in s.index if k.endswith("/all")]].round(4).to_string())
