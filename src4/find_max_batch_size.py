"""Binary search for the largest --batch-size that fits on this GPU, by actually
running one train step (forward + backward + optimizer.step, matching
composer_model.py:forward/loss exactly) at each candidate size and watching for OOM.

First-principles budget (16303MiB total on this card):
  fixed (~doesn't scale with batch size):
    - frozen weights: VAE (~83M) + text_encoder (~123M) + CLAP (~85M) + UNet (~859M)
      = ~1.15B params, bf16-cast activations but weights held at fp32 -> ~4.6GB
    - trainable weights (~34.6M, fp32) + AdamW state (2x trainable, fp32) + grads
      (1x trainable, fp32) -> ~34.6M * 4B * 4 =~ 0.55GB
    - CUDA context / allocator overhead -> ~0.3-0.5GB
    ballpark fixed cost: ~5.2GB
  variable (scales ~linearly with batch size):
    - UNet/VAE/CLAP forward+backward activations at 512x512, bf16 autocast
    measured directly below rather than estimated further, since cudnn algo
    selection and the caching allocator make this non-linear in practice.

Usage:
    python src4/find_max_batch_size.py
"""
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from src4.composer_model import SonicDiffusionModel  # noqa: E402

TOTAL_MIB = torch.cuda.get_device_properties(0).total_memory / 1024**2
SAFETY_FRAC = 0.90  # leave headroom for CUDA fragmentation + the eval-viz generate() calls


def try_batch_size(model, optimizer, bs: int) -> tuple[bool, float]:
    """Runs one real train step at batch size `bs`. Returns (fit, peak_mib)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = next(model.parameters()).device
    batch = {
        "pixel_values": torch.randn(bs, 3, 512, 512, device=device),
        "waveform": torch.randn(bs, 441000, device=device),
        "prompt": [""] * bs,
    }
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model.forward(batch)
            loss = model.loss(outputs, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**2
        return True, peak
    except torch.cuda.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, float("nan")


def main():
    print(f"GPU: {torch.cuda.get_device_properties(0).name}, total={TOTAL_MIB:.0f}MiB, "
          f"budget={SAFETY_FRAC * TOTAL_MIB:.0f}MiB ({SAFETY_FRAC:.0%})")
    model = SonicDiffusionModel().cuda()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)

    budget = SAFETY_FRAC * TOTAL_MIB

    # exponential search for an upper bound that doesn't fit
    lo, lo_peak = 1, None
    fit, peak = try_batch_size(model, optimizer, lo)
    if not fit:
        print("Even batch_size=1 doesn't fit -- something is very wrong.")
        return
    lo_peak = peak
    print(f"bs={lo}: fits, peak={peak:.0f}MiB")

    hi = lo * 2
    while True:
        fit, peak = try_batch_size(model, optimizer, hi)
        print(f"bs={hi}: {'fits' if fit else 'OOM'}, peak={peak:.0f}MiB" if fit else f"bs={hi}: OOM")
        if not fit or peak > budget:
            break
        lo, lo_peak = hi, peak
        hi *= 2

    # binary search in (lo, hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        fit, peak = try_batch_size(model, optimizer, mid)
        if fit and peak <= budget:
            lo, lo_peak = mid, peak
            print(f"bs={mid}: fits, peak={peak:.0f}MiB -> new lo")
        else:
            hi = mid
            print(f"bs={mid}: {'over budget' if fit else 'OOM'}, peak={peak if fit else float('nan'):.0f}MiB -> new hi")

    print(f"\nMax batch size within {SAFETY_FRAC:.0%} budget ({budget:.0f}MiB): {lo} (peak {lo_peak:.0f}MiB)")


if __name__ == "__main__":
    main()
