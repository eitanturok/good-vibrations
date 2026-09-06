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
close(localization(t.clone(), t)['localization_rel'][0], 0)

# pure translation: 2 rows down, 3 cols right. _raw is in cells, _rel is that over (H, W).
p = torch.roll(t, shifts=(2, 3), dims=(1, 2))
loc = localization(p, t)
close(loc['localization_raw_h'][0], 2)
close(loc['localization_raw_w'][0], 3)
close(loc['localization_raw'][0], math.hypot(2, 3))
close(loc['localization_rel_h'][0], 2 / H)
close(loc['localization_rel_w'][0], 3 / W)
close(loc['localization_rel'][0], math.hypot(2 / H, 3 / W))

# mass under / over
close(mass_error(0.5 * t, t), -1 / 3)
close(mass_error(torch.zeros_like(t), t), -1)
close(mass_error(rect(8, 12, 10, 20), t), (40 - 20) / (40 + 20))  # 2x the mass

# two objects, one missed -> per-sample mean of (0, worst case)
t2 = torch.clamp(rect(2, 5, 2, 6) + rect(15, 19, 20, 26), 0, 1)
loc2 = localization(rect(2, 5, 2, 6), t2)
close(loc2['localization_raw'][0], math.hypot(H, W) / 2)   # miss = grid diagonal, in cells
close(loc2['localization_rel'][0], math.sqrt(2) / 2)       # miss = unit-square diagonal

# empty box
z = torch.zeros(1, H, W)
close(soft_iou(z, z), 1)
close(contour_f(z, z), 1)
close(mass_error(z, z), 0)
assert all(torch.isnan(v[0]) for v in localization(z, z).values())
close(contour_f(rect(2, 5, 2, 6), z), 0)     # hallucination
close(mass_error(rect(2, 5, 2, 6), z), 1)

print("ok")
