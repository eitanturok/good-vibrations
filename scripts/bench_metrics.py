"""Measure the segmentation-metric speedups (docs/metric_speedups.md).

  1. utils.metrics._label -- drop the per-round torch.equal() convergence check (on CUDA
     that is a device sync; it fired every round, twice per localization(), every train
     batch) and run a fixed 2*max(H,W) rounds instead.
  2. model.arch -- train_metrics use CHEAP_SEG_KEYS only (bce/iou/mass: pure GPU
     reductions); localization*/contour move to the eval loaders.

  PYTHONPATH=. python scripts/bench_metrics.py [--full] [--batch N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from utils import metrics                            # noqa: E402
from model import arch                               # noqa: E402
from model.arch import CHEAP_SEG_KEYS, SEG_KEYS      # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H = W = 32
FIXED = 2 * (H + W)                                  # the new _label default round count


def label_converging(x, cap=512):
    """The pre-change _label: propagate labels until torch.equal() reports a fixed point.
    Returns (ids, rounds_used) so the bench can see how many rounds masks really need."""
    _, h, w = x.shape
    xl = x.long()
    ids = torch.arange(1, h * w + 1, device=x.device).view(1, h, w) * xl
    for i in range(cap):
        nxt = F.max_pool2d(ids[:, None].float(), 3, 1, 1)[:, 0].long() * xl
        if torch.equal(nxt, ids):
            return ids, i
        ids = nxt
    return ids, cap


def seg_batch_old(logits, pred, true):
    """The pre-change _seg_batch: whole suite every call, localization + contour always."""
    b = len(pred)
    nan = lambda v: (v[~v.isnan()].sum(), int((~v.isnan()).sum()))
    return {"bce": (F.binary_cross_entropy_with_logits(logits, true, reduction="sum"), true.numel()),
            "iou": (metrics.soft_iou(pred, true).sum(), b),
            "mass": (metrics.mass_error(pred, true).sum(), b),
            "contour": (metrics.contour_f(pred, true).sum(), b),
            **{k: nan(v) for k, v in metrics.localization(pred, true).items()}}


def mask_zoo(cap=6000):
    """CPU batches spanning what _label sees: synthetic rings (the worst connected
    component -- internal path exceeds the bbox diagonal), real 32x32 GT masks on disk,
    and real half-trained model predictions saved in runs/*/outputs_history."""
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    rad = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2).sqrt()
    zoo = {"rings": torch.stack([((rad < s) & (rad > s - 3)).float()
                                 for s in torch.linspace(6, min(H, W) / 2, 64)])}

    gt = [np.load(f) > 0 for f in
          sorted(ROOT.glob(f"experiments/*/samples/*/image/04_downsampled_smask_{H}h_{W}w.npy"))[:cap]]
    if gt:
        zoo["real-gt"] = torch.from_numpy(np.stack(gt).astype("float32"))

    preds = []
    for f in sorted(ROOT.glob("runs/*/outputs_history/*/*.pt")):
        try:
            p = torch.load(f, map_location="cpu", weights_only=False)["mask_pred"]
        except Exception:
            continue                                 # skip a partially-written checkpoint
        if p.shape[-2:] == (H, W):
            preds.append((p > 0.5).float())
        if sum(len(x) for x in preds) >= cap:
            break
    if preds:
        zoo["real-preds"] = torch.cat(preds)[:cap]
    return zoo


def ms(fn, n=50, warm=10):
    for _ in range(warm):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--full", action="store_true", help="also time a full boombox train step")
    ap.add_argument("--box", default="experiments/31_07_2026_gastronorm_exp1")
    ap.add_argument("--d-model", type=int, default=1024)
    a = ap.parse_args()
    torch.manual_seed(0)
    print(f"device={DEV}  grid={H}x{W}  batch={a.batch}  new _label rounds={FIXED}")

    zoo = {k: v.to(DEV) for k, v in mask_zoo().items()}
    allm = torch.cat(list(zoo.values()))

    # 1. is the fixed round count enough, and does it change localization()'s answer?
    print("\n[_label] rounds the old code needed to converge, per mask family:")
    worst = 0
    for name, m in zoo.items():
        r = np.array([label_converging(x[None])[1] for x in m[:300]])
        worst = max(worst, int(r.max()))
        print(f"  {name:<11} n={len(m):<5} max {r.max():>3}  p99 {np.percentile(r, 99):>3.0f}  mean {r.mean():>4.1f}")
    print(f"  -> worst {worst} vs fixed {FIXED}: {'OK' if worst <= FIXED else 'RAISE iters'}")

    pred, true = allm, torch.roll(allm, (2, 3), (-2, -1))
    new = metrics.localization(pred, true)
    orig, metrics._label = metrics._label, lambda x, iters=None: label_converging(x)[0]
    old = metrics.localization(pred, true)
    metrics._label = orig
    diff = max(float((new[k] - old[k]).abs().nan_to_num().max()) for k in new)
    print(f"  localization() new vs old _label: max abs diff = {diff:.2e}")

    # 2. _label timing
    print("\n[_label] ms/call")
    x = metrics._bin(allm[:a.batch])
    old_ms, new_ms = ms(lambda: label_converging(x)), ms(lambda: metrics._label(x))
    print(f"  old (torch.equal early-exit): {old_ms:7.3f}")
    print(f"  new (fixed, no sync):         {new_ms:7.3f}   {old_ms / new_ms:.2f}x")

    # 3. _seg_batch timing: old suite vs new full (eval path) vs new cheap (train path)
    print("\n[_seg_batch] ms/call")
    logits = torch.randn(a.batch, H, W, device=DEV)
    pred, true = logits.sigmoid(), allm[:a.batch]

    def run(keys):
        arch._seg_cache.clear()
        return arch._seg_batch(logits, pred, true, keys)

    old_ms = ms(lambda: seg_batch_old(logits, pred, true), n=30)
    full_ms = ms(lambda: run(SEG_KEYS), n=30)
    cheap_ms = ms(lambda: run(CHEAP_SEG_KEYS), n=30)
    print(f"  old suite:              {old_ms:8.3f}")
    print(f"  new full  (eval path):  {full_ms:8.3f}   {old_ms / full_ms:.2f}x")
    print(f"  new cheap (train path): {cheap_ms:8.3f}   {old_ms / cheap_ms:.2f}x")

    if a.full:
        step_bench(a)


def step_bench(a):
    print("\n[step] boombox forward+loss+backward+metric on a real batch, ms/step")
    from model.dataset import build_dataset
    from model.boombox import BoomboxModel
    from model.arch import create_metrics

    loader, *_ = build_dataset(a.box, batch_size=a.batch, eval_batch_size=108, num_workers=4,
                               split="gastronorm", out_h=H, out_w=W, verbose=0)
    dl = loader.dataloader
    ds = dl.dataset
    base = ds
    while hasattr(base, "dataset"):                  # Subset -> [PairedSpeaker ->] VibrationDataset
        base = base.dataset
    _, p, ps, c = ds[0]["fft"].shape
    rows, cols = base.grid_shape
    info = dict(out_h=H, out_w=W, out_c=1, n_laser_rows=rows, n_laser_cols=cols, patch_size=ps,
               n_freqs=p * ps, n_freqs_real=len(base.pk["freqs"]), n_channels=c)

    raw = next(iter(dl))
    batch = {"fft": raw["fft"].to(DEV), "mask_true": raw["mask_true"].to(DEV),
             "info": {"n_objects": raw["info"]["n_objects"].to(DEV)}}
    model = BoomboxModel(a.d_model, info, loss_fn="ce-pixel").to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, fused=(DEV == "cuda"))

    def step(tms):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        model.loss(out, batch).backward()
        opt.step()
        arch._seg_cache.clear()
        for m in tms.values():
            m.update(out["mask_logits"].detach(), out["mask_pred"].detach(), batch["mask_true"], None)

    for keys, label in [(SEG_KEYS, "full  train metrics (old)"),
                        (CHEAP_SEG_KEYS, "cheap train metrics (new)")]:
        tms = {k: v.to(DEV) for k, v in create_metrics(info, keys).items()}
        t = ms(lambda: step(tms), n=30, warm=10)
        print(f"  {label}: {t:7.2f} ms/step   {a.batch / t * 1e3:7.0f} samples/s")


if __name__ == "__main__":
    main()
