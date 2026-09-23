"""Thin glue around utils/metrics.py: IoU, mass error, contour F-score, and
localization, computed on generated-vs-ground-truth MASK images. --target mask
runs only -- see plan's note on why --target photo skips these (no trustworthy
thresholding heuristic turns a natural photo into a comparable binary mask).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from utils.metrics import soft_iou, mass_error, contour_f, localization, LOC_KEYS  # noqa: E402


def image_to_mask(img: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """RGB/L PIL image -> (H,W) float tensor in [0,1] (grayscale luminance)."""
    arr = np.asarray(img.convert("L").resize(size, Image.BILINEAR)).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def compute_metrics(pred_images: list[Image.Image], gt_images: list[Image.Image]) -> dict[str, float]:
    """pred/gt: same-length lists of PIL images, any size -- resized to the GT's own
    size before scoring. Returns a flat dict of mean scalar metrics."""
    w, h = gt_images[0].size
    pred = torch.stack([image_to_mask(im, size=(w, h)) for im in pred_images])  # (B,H,W)
    true = torch.stack([image_to_mask(im, size=(w, h)) for im in gt_images])

    out = {
        "iou": soft_iou(pred, true).mean().item(),
        "mass_error": mass_error(pred, true).mean().item(),
        "contour_f": contour_f(pred, true).mean().item(),
    }
    loc = localization(pred, true)
    for k in LOC_KEYS:
        vals = loc[k] if isinstance(loc[k], torch.Tensor) else torch.tensor(loc[k])
        finite = vals[torch.isfinite(vals)]
        out[k] = finite.mean().item() if len(finite) else float("nan")
    return out
