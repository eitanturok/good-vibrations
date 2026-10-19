"""How much does a model tolerate a moved laser grid or a stretched frequency axis? The eval-time counterpart of --laser-jitter and
--freq-jitter: shift every eval sample by a FIXED amount and score it, so runs trained with and without the augmentations can be compared
(a model trained with jitter should be flat over the range it was jittered on, and the price is what it no longer resolves).
  laser   the 10x10 field resampled at the grid shifted by d cells along the rows and the columns (bilinear, border clamp), as jitter_lasers
  freq    every spectrum on the axis f -> f (1 + eps), as jitter_freqs (eps = 1e-3 is 0.1 Hz at 100 Hz, 1 Hz at 1 kHz)
  .venv/bin/python src8/shift_probe.py RUN... -> runs/shift_probe.csv (one row per run, kind, amount, box; appended, re-runs replace)"""
import argparse, sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from probe import ROOT, eval_data, load_model, box_index, run_args
from importance import CORE, METRICS, scores, per_box

LASER_D = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
FREQ_EPS = [0.0, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3]

def shift_lasers(x, d, rows, cols):
    B, L, F_, C = x.shape
    f = x.permute(0, 2, 3, 1).reshape(B, F_ * C, rows, cols).float()
    theta = torch.tensor([[1, 0, d * 2 / cols], [0, 1, d * 2 / rows]], dtype=torch.float32).expand(B, 2, 3)
    grid = F.affine_grid(theta, (B, 1, rows, cols), align_corners=False)
    return F.grid_sample(f, grid, mode="bilinear", padding_mode="border", align_corners=False).reshape(B, F_, C, L).permute(0, 3, 1, 2).to(x.dtype)

def stretch_freqs(x, eps, lo, hi):
    F_ = x.shape[2]
    f = torch.linspace(lo, hi, F_)
    src = ((f / (1 + eps) - lo) / (hi - lo) * (F_ - 1)).clamp(0, F_ - 1)
    i0 = src.floor().long().clamp(max=F_ - 2); w = (src - i0)[None, None, :, None]
    xf = x.float()
    return (xf[:, :, i0] * (1 - w) + xf[:, :, i0 + 1] * w).to(x.dtype)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+"); a = ap.parse_args()
    out = ROOT / "runs" / "shift_probe.csv"
    rows_out = []
    for run in a.runs:
        ra = run_args(run)
        X, y, labels, info = eval_data("gastronorm_plastic_cardboard", ra.signal, ra.window, laser_center=bool(getattr(ra, "laser_center", 0)))
        keep = torch.tensor([l.split("/")[1] in CORE or l == "cardboard/soap-dispenser-rotated" for l in labels])
        labels = [l for l, k in zip(labels, keep) if k]
        X, y, box = X[keep], y[keep], box_index(labels, info)
        nb, (R, Cc), (lo, hi) = len(info["boxes"]), (info["n_laser_rows"], info["n_laser_cols"]), info["freq_range"]
        ck = sorted((ROOT / "runs" / run / "checkpoints").glob("ep*-rank0.pt"), key=lambda p: int(p.name[2:].split("-")[0]))[-1]
        m = load_model("src8", run, ck, info, {}).eval()
        for kind, amounts, fn in (("laser", LASER_D, lambda x, v: shift_lasers(x, v, R, Cc)), ("freq", FREQ_EPS, lambda x, v: stretch_freqs(x, v, lo, hi))):
            for v in amounts:
                s = []
                for i in range(0, len(X), 480):                                                # shift on the CPU in chunks: X is ~2 GB
                    s.append(scores(m, fn(X[i:i + 480], v) if v else X[i:i + 480], box[i:i + 480], y[i:i + 480]))
                pb = per_box(torch.cat(s, 1), box, nb)                                         # (M,nb)
                for b, bn in enumerate(info["boxes"]):
                    rows_out.append(dict(run=run, kind=kind, amount=v, box=bn, **{k: float(pb[j, b]) for j, k in enumerate(METRICS)}))
                print(f"{run} {kind} {v}: iou@0.08 " + " ".join(f"{float(pb[0, b]):.3f}" for b in range(nb)), flush=True)
        del X, y, m
    df = pd.DataFrame(rows_out)
    if out.exists():
        old = pd.read_csv(out); df = pd.concat([old[~old.run.isin(a.runs)], df])
    df.to_csv(out, index=False)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
