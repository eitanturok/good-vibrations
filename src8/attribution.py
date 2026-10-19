"""Faithful importance by occlusion: replace one frequency band (~49 Hz, the ModalFormer's band width) or one laser with its box's mean
signal (over the eval set) and measure the iou@0.08 drop, per box, on the CORE + HARD eval samples. The architecture's own gates /
attention are near uniform (band gates within 1.5% of uniform, laser attention within 10%) and a following linear layer can rescale
anything, so they aren't importance. Any run (src7 or src8).
  .venv/bin/python src8/attribution.py RUN [RUN ...]   -> runs/<RUN>/attribution.pt {band: (n_box, P) drop, laser: (n_box, 100) drop, hz}"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from probe import ROOT, eval_data, load_model, box_index, run_args

CORE = ("candle", "cube", "soap-dispenser", "vase")
BAND = 64  # bins per occluded band (~49 Hz), as the ModalFormer stem's tokens

@torch.no_grad()
def iou(m, X, box, y, bs=64):
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", torch.bfloat16):
            p = m(dict(fft=X[i:i + bs].cuda().float(), box=box[i:i + bs].cuda()))["mask_pred"].float() > 0.08
        o = y[i:i + bs].cuda() > 0.5
        out.append(((p & o).flatten(1).sum(1) / (p | o).flatten(1).sum(1).clamp_min(1)).cpu())
    return torch.cat(out)

def attribute(run):
    ra = run_args(run)
    X, y, labels, info = eval_data("gastronorm_plastic_cardboard", ra.signal if ra else "log_power", ra.window if ra else "tukey",
                                   laser_center=bool(getattr(ra, "laser_center", 0)) if ra else False)
    box = box_index(labels, info)
    keep = torch.tensor([l.split("/")[1] in CORE or l == "cardboard/soap-dispenser-rotated" for l in labels])
    X, y, box = X[keep], y[keep], box[keep]
    mean = torch.stack([X[box == b].float().mean(0) for b in range(len(info["boxes"]))]).half()   # (n_box,L,F,C)
    ck = sorted((ROOT / "runs" / run / "checkpoints").glob("ep*-rank0.pt"), key=lambda p: int(p.name[2:].split("-")[0]))[-1]
    m = load_model("src8" if ra else "src7", run, ck, info, {}).eval()
    base = iou(m, X, box, y)
    F_ = X.shape[2]; P = -(-F_ // BAND)
    band, laser = torch.zeros(len(info["boxes"]), P), torch.zeros(len(info["boxes"]), X.shape[1])
    for j in range(P):
        Z = X.clone(); s = slice(j * BAND, (j + 1) * BAND); Z[:, :, s] = mean[box][:, :, s]
        d = base - iou(m, Z, box, y)
        band[:, j] = torch.stack([d[box == b].mean() for b in range(len(info["boxes"]))])
    for l in range(X.shape[1]):
        Z = X.clone(); Z[:, l] = mean[box][:, l]
        d = base - iou(m, Z, box, y)
        laser[:, l] = torch.stack([d[box == b].mean() for b in range(len(info["boxes"]))])
    lo, hi = info["freq_range"]
    hz = torch.tensor([lo + (j * BAND + BAND / 2) * (hi - lo) / F_ for j in range(P)])
    out = dict(band=band, laser=laser, hz=hz, boxes=info["boxes"], base=torch.stack([base[box == b].mean() for b in range(len(info["boxes"]))]))
    torch.save(out, ROOT / "runs" / run / "attribution.pt")
    print(f"\n{run}: base iou per box {dict(zip(info['boxes'], out['base'].numpy().round(3)))}")
    for b, name in enumerate(info["boxes"]):
        top = band[b].argsort(descending=True)[:4]
        print(f"  {name:13s} top bands (Hz: iou drop) " + ", ".join(f"{hz[j]:.0f}: {band[b, j]:.3f}" for j in top)
              + f" | band drop sum {band[b].sum():.3f} | top lasers " + ", ".join(f"({l // 10},{l % 10}): {laser[b, l]:.3f}" for l in laser[b].argsort(descending=True)[:4]))

if __name__ == "__main__":
    for r in sys.argv[1:]: attribute(r)
