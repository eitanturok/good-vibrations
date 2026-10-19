"""Test-time normalization (AdaBN, Li et al. 2016; the statistics part of TENT, Wang et al. 2021) for unseen speakers / lasers:
re-estimate every BatchNorm's statistics and ModalFormer's per-box input norm on UNLABELED captures of the new setup, then predict.
No labels, no gradient steps. A new speaker scales every capture by its S(f) and a laser shift changes every channel's baseline,
both largely normalization-level shifts.

Domains are groups of eval splits that share a setup (box + speaker or box + laser placement). They contain several objects, so the
adapted statistics don't simply absorb one object's own signature:
  .venv/bin/python src8/adabn.py RUN [--device cpu]   -> runs/<RUN>/adabn.csv (iou@0.08 / miss@0.08 per split: train stats vs adapted)"""
import argparse, copy, sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from probe import ROOT, eval_data, load_model, box_index, run_args, score

DOMAINS = {  # adapt on all of these splits' captures together, score each
    "gastro unseen speaker":    ["gastro/cube-unseen-speaker-unseen"],
    "plastic unseen speaker":   ["plastic/cube-unseen-speaker-unseen"],
    "cardboard unseen speaker": ["cardboard/vase-unseen-speaker-unseen"],
    "plastic shifted lasers":   ["plastic/cube-shifted-lasers-1", "plastic/cube-shifted-lasers-2", "plastic/cube-shifted-lasers-3-backside"],
    "cardboard shifted lasers": ["cardboard/vase-laser-shifted-2", "cardboard/vase-laser-shifted-3", "cardboard/vase-laser-shifted-4"],
    "cardboard core (control)": ["cardboard/candle", "cardboard/cube", "cardboard/soap-dispenser", "cardboard/vase"],
    "plastic core (control)":   ["plastic/candle", "plastic/cube", "plastic/soap-dispenser", "plastic/vase"],
}

@torch.no_grad()
def predict(m, X, box, dev, bs=64):
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast(dev, torch.bfloat16): out.append(m(dict(fft=X[i:i + bs].to(dev).float(), box=box[i:i + bs].to(dev)))["mask_pred"].float().cpu())
    return torch.cat(out)

@torch.no_grad()
def adapt(m, X, box, dev, bs=64):
    """a copy of m whose BatchNorm stats and box-norm stats are the averages over X (train mode, momentum None = cumulative)"""
    a = copy.deepcopy(m)
    for mod in a.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm): mod.reset_running_stats(); mod.momentum = None; mod.train()
    mf = getattr(a, "modal", None)
    if mf is not None and mf.box_norm:
        for b in box.unique(): mf.in_n[b] = 0                                    # the cumulative average restarts for these boxes
        mf.train()                                                              # normalize() updates the box stats in training mode
    for i in range(0, len(X), bs):
        with torch.autocast(dev, torch.bfloat16): a(dict(fft=X[i:i + bs].to(dev).float(), box=box[i:i + bs].to(dev)))
    return a.eval()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(8)
    ra = run_args(a.run)
    X, y, labels, info = eval_data("gastronorm_plastic_cardboard", ra.signal if ra else "log_power", ra.window if ra else "tukey",
                                   laser_center=bool(getattr(ra, "laser_center", 0)) if ra else False)
    ck = sorted((ROOT / "runs" / a.run / "checkpoints").glob("ep*-rank0.pt"), key=lambda p: int(p.name[2:].split("-")[0]))[-1]
    m = load_model("src8" if ra else "src7", a.run, ck, info, {}).to(a.device).eval()
    box = box_index(labels, info)
    rows = []
    for dom, splits in DOMAINS.items():
        idx = torch.tensor([i for i, l in enumerate(labels) if l in splits])
        if not len(idx): continue
        Xd, yd, ld, bd = X[idx], y[idx], [labels[i] for i in idx], box[idx]
        base = score(predict(m, Xd, bd, a.device), yd, ld)
        ad = score(predict(adapt(m, Xd, bd, a.device), Xd, bd, a.device), yd, ld)
        rows.append(dict(domain=dom, n=len(idx), iou=base["iou"], iou_adabn=ad["iou"], miss=base["miss"], miss_adabn=ad["miss"]))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    print(df.round(3).to_string())
    df.to_csv(ROOT / "runs" / a.run / "adabn.csv", index=False)
