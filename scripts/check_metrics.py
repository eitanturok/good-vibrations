"""Smoke test for the segmentation metric suite (utils/metrics.py).

Synthetic mask pairs with known answers -> run `python scripts/check_metrics.py`.
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent))
from utils.metrics import contour_f, localization, mass_error, soft_iou

H, W = 21, 30


def rect(r0, r1, c0, c1, b=1):
    m = torch.zeros(b, H, W)
    m[:, r0:r1, c0:c1] = 1
    return m


def close(a, b, tol=1e-3):
    assert abs(float(a) - float(b)) <= tol, f"{float(a)} != {float(b)}"


t = rect(8, 12, 10, 15)

# identical
close(soft_iou(t.clone(), t), 1)
close(contour_f(t.clone(), t), 1)
close(mass_error(t.clone(), t), 0)
close(localization(t.clone(), t)[0][0], 0)

# pure translation
p = torch.roll(t, shifts=(2, 3), dims=(1, 2))
err, ex, ey = localization(p, t)
close(ex[0], 2)
close(ey[0], 3)
close(err[0], math.hypot(2, 3))

# mass under / over
close(mass_error(0.5 * t, t), -1 / 3)
close(mass_error(torch.zeros_like(t), t), -1)
close(mass_error(rect(8, 12, 10, 20), t), (40 - 20) / (40 + 20))  # 2x the mass

# two objects, one missed -> per-sample mean of (0, grid diagonal)
t2 = torch.clamp(rect(2, 5, 2, 6) + rect(15, 19, 20, 26), 0, 1)
close(localization(rect(2, 5, 2, 6), t2)[0][0], math.hypot(H, W) / 2)

# empty box
z = torch.zeros(1, H, W)
close(soft_iou(z, z), 1)
close(contour_f(z, z), 1)
close(mass_error(z, z), 0)
assert torch.isnan(localization(z, z)[0][0])
close(contour_f(rect(2, 5, 2, 6), z), 0)     # hallucination
close(mass_error(rect(2, 5, 2, 6), z), 1)

print("ok")
