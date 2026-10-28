"""thresholded scores of a run's saved eval outputs (runs/<run>/outputs_history/eval): mask_pred > thr vs mask_true > 0.5
  iou = hard IoU per sample, miss = 1 if no predicted pixel lands on the object, orient = principal-axis error (elongated masks), deg
Split means, then CORE / HARD / WEDGE / UNSEEN-POS / ALL / ALL-ood (every split counts once), per run side by side:
  .venv/bin/python src9/score.py RUN [RUN ...] [--epoch 500] [--thr 0.08]"""
import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from dataset import ROOT, EXPERIMENTS, CATEGORIES, NOT_OOD, load_meta

CACHE = ROOT / "runs/_src9_cache/truth"

def sample_dir(exp, sid, pos, spk): return EXPERIMENTS / exp / "samples" / (f"{sid:06d}" if sid >= 0 else f"{pos:06d}-{spk}")

def truth(exp, sid, pos, spk, H, W):
    """(H,W) true mask, BOX-downsampled like the dataset; a with-lid sample takes its without-lid twin's. Cached"""
    f = CACHE / f"{H}x{W}" / exp / f"{sid}_{pos}_{spk}.npy"
    if f.exists(): return np.load(f)
    d = sample_dir(exp, sid, pos, spk)
    if (p := load_meta(d).get("without_lid_position_id")) is not None:
        d = sample_dir(exp, -1, p, spk)
        if not d.exists(): d = next(s for s in (EXPERIMENTS / exp / "samples").iterdir() if (m := load_meta(s)).get("position_id") == p and m.get("speaker") == spk)
    y = np.clip(np.asarray(Image.open(d / "image/03_smask.png").convert("F").resize((W, H), Image.BOX), dtype=np.float32) / 255.0, 0, 1)
    f.parent.mkdir(parents=True, exist_ok=True); np.save(f, y)
    return y

def axis(m):
    """principal-axis angle and eccentricity of a bool mask"""
    r, c = np.nonzero(m)
    if len(r) < 3: return np.nan, np.nan
    rr, cc, rc = r.var(), c.var(), ((r - r.mean()) * (c - c.mean())).mean()
    return 0.5 * np.arctan2(2 * rc, rr - cc), np.sqrt((rr - cc) ** 2 + 4 * rc ** 2) / max(rr + cc, 1e-9)

def last_epoch(run): return max(int(p.name[2:6]) for p in (ROOT / "runs" / run / "outputs_history/eval").rglob("ep*.pt"))

def per_sample(run, epoch, thr):
    rows = []
    for f in sorted((ROOT / "runs" / run / "outputs_history/eval").glob(f"*/*/ep{epoch:04d}-ba*.pt")):
        d = torch.load(f, weights_only=False, map_location="cpu")
        p, info = d["mask_pred"].float().numpy(), d["info"]
        for i in range(len(p)):
            y = truth(info["experiment"][i], info["sample_id"][i], int(info["position_id"][i]), int(info["speaker"][i]), *p.shape[-2:]) > 0.5
            if not y.any(): continue  # empty box: nothing to score
            b = p[i] > thr
            (tt, ecc), (tp, _) = axis(y), axis(b)
            e = abs(tt - tp) % np.pi
            rows.append(dict(box=f.parent.parent.name, split=f.parent.name, iou=(b & y).sum() / max((b | y).sum(), 1), miss=float(not (b & y).any()),
                             orient=np.degrees(min(e, np.pi - e)) if ecc > 0.2 and not np.isnan(tp) else np.nan))
    if not rows: raise FileNotFoundError(f"{run}: no eval outputs at epoch {epoch}")
    return pd.DataFrame(rows)

def summarize(df):
    sp = df.groupby(["box", "split"])[["iou", "miss", "orient"]].mean().reset_index()
    cat = {(b, s): c for b, cs in CATEGORIES.items() for c, ss in cs.items() for s in ss}
    sp["category"] = [cat.get((b, s), "?") for b, s in zip(sp.box, sp.split)]
    nc = sp[~sp.category.isin(["control", "unseen-angle", "unseen-position"])]
    pick = {"CORE": sp.category == "id", "HARD": (sp.box == "cardboard") & (sp.split == "soap-dispenser-rotated"),
            "WEDGE": sp.split == "soap-dispenser-unseen-angle", "UNSEEN-POS": sp.split == "unseen-position"}
    out = {k: sp[m] for k, m in pick.items()} | {"ALL": nc, "ALL-ood": nc[nc.category != "id"]}
    out |= {f"{b}/unseen-pos": sp[(sp.box == b) & (sp.split == "unseen-position")] for b in CATEGORIES}
    return pd.DataFrame({k: g[["iou", "miss", "orient"]].mean() for k, g in out.items()}).T

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--epoch", type=int, default=None, help="default: each run's last saved eval")
    ap.add_argument("--thr", type=float, default=0.08)
    a = ap.parse_args()
    tables = {}
    for run in a.runs:
        ep = a.epoch or last_epoch(run)
        tables[f"{run}@{ep}"] = summarize(per_sample(run, ep, a.thr))
    print(pd.concat(tables, axis=1).round(3).to_string())
