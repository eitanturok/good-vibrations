"""Where does a run's IoU go: position, size, or shape? Per eval sample (pred > 0.08 vs true > 0.5), single-object splits:
  com_err      centroid distance, box fractions
  area_ratio   predicted / true area
  orient_err   principal-axis angle, degrees (elongated true masks)
  iou          as scored
  iou_pos      after shifting the prediction so its centroid sits on the true one (removes the position error)
  iou_pos_size after also scaling it about the centroid to the true area (what is left is shape + orientation)
  .venv/bin/python src8/decompose.py RUN@EPOCH [RUN@EPOCH ...]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from score import ROOT, truth, axis

CORE = ("candle", "cube", "soap-dispenser", "vase")

def moments(m):
    r, c = np.nonzero(m)
    return (r.mean(), c.mean(), len(r)) if len(r) else (np.nan, np.nan, 0)

def warp(m, dr, dc, s):
    """shift a bool mask by (dr, dc) px and scale it by s about its centroid (nearest-pixel, via an inverse map)"""
    r0, c0, _ = moments(m)
    H, W = m.shape
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    src_r, src_c = (rr - dr - r0) / s + r0, (cc - dc - c0) / s + c0
    return ndimage.map_coordinates(m.astype(float), [src_r, src_c], order=0, cval=0) > 0.5

def iou(a, b): return (a & b).sum() / max((a | b).sum(), 1)

def run_rows(run, ep, splits):
    rows = []
    for box, split in splits:
        for f in sorted((ROOT / "runs" / run / "outputs_history/eval" / box / split).glob(f"ep{ep:04d}-*.pt")):
            d = torch.load(f, weights_only=False); info = d["info"]
            for i, p in enumerate(d["mask_pred"].float().numpy()):
                y = truth(info["experiment"][i], info["sample_id"][i], int(info["position_id"][i]), int(info["speaker"][i]), *p.shape) > 0.5
                b = p > 0.08
                (rt, ct, at), (rp, cp, ap) = moments(y), moments(b)
                r = dict(box=box, split=split, iou=iou(b, y), area_ratio=ap / max(at, 1), miss=float(not (b & y).any()))
                if ap:
                    H, W = p.shape
                    r["com_err"] = np.hypot((rp - rt) / H, (cp - ct) / W)
                    moved = warp(b, rt - rp, ct - cp, 1.0)
                    r["iou_pos"] = iou(moved, y)
                    r["iou_pos_size"] = iou(warp(b, rt - rp, ct - cp, np.sqrt(at / ap)), y)
                    (tt, et), (tp, _) = axis(y), axis(b)
                    dd = abs(tt - tp) % np.pi
                    r["orient_err"] = np.degrees(min(dd, np.pi - dd)) if et > 0.2 else np.nan
                rows.append(r)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    splits = [(b, s) for b in ("gastro", "plastic", "cardboard") for s in CORE] + [("cardboard", "soap-dispenser-rotated")]
    out = {}
    for arg in sys.argv[1:]:
        run, ep = arg.split("@")
        df = run_rows(run, int(ep), splits)
        df["group"] = np.where(df.split == "soap-dispenser-rotated", "HARD", "CORE")
        g = df.groupby(["group", "box", "split"]).mean(numeric_only=True)
        out[arg] = g
        print(f"\n== {arg}\n" + df.groupby("group").mean(numeric_only=True).round(3).to_string())
    pd.concat(out, names=["run"]).to_csv(ROOT / "runs/decompose_src8.csv")
