"""Metrics shared across the numpy (local/CPU) and torch (model training) pipelines."""

import numpy as np
from PIL import Image

#***** center of mass *****

def _center_of_mass_numpy(mask: np.ndarray, normalize: bool):
    """Unbatched (H,W) mask -> (row, col) tuple of floats. Batched (...,H,W) mask
    -> (...,2) array. Empty masks (total == 0) get a (-1, -1) sentinel."""
    mask = mask.astype(np.float64)
    if mask.max() > 1.0: mask = mask / 255.0
    H, W = mask.shape[-2:]
    rows, cols = np.arange(H), np.arange(W)
    total = mask.sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        row = (mask * rows[:, None]).sum(axis=(-2, -1)) / total
        col = (mask * cols[None, :]).sum(axis=(-2, -1)) / total
    if normalize: row, col = row / (H - 1), col / (W - 1)
    com = np.where(total[..., None] == 0, -1.0, np.stack([row, col], axis=-1))
    return (com[..., 0].item(), com[..., 1].item()) if mask.ndim == 2 else com


def _center_of_mass_torch(mask, normalize: bool, epsilon: float):
    import torch
    mask = mask.float()
    if mask.max() > 1.0: mask = mask / 255.0
    H, W = mask.shape[-2:]
    rows = mask.new_tensor(range(H))
    cols = mask.new_tensor(range(W))
    total = mask.sum(dim=(-2, -1)).clamp(min=epsilon)
    row = (mask * rows[:, None]).sum(dim=(-2, -1)) / total
    col = (mask * cols[None, :]).sum(dim=(-2, -1)) / total
    if normalize: row, col = row / (H - 1), col / (W - 1)
    return torch.stack([row, col], dim=-1)


def center_of_mass(mask, normalize: bool = False, epsilon: float = 1e-6):
    """(row, col) center of mass of a mask. Accepts a numpy array, PIL Image, or a
    (possibly batched, shape (...,H,W)) torch tensor -- routed automatically to the
    matching implementation. Values are normalized by /255 first if they look like
    0-255. normalize=True further divides by (H-1, W-1) to map into [0, 1]."""
    if isinstance(mask, Image.Image): mask = np.array(mask)
    try:
        import torch
        if isinstance(mask, torch.Tensor): return _center_of_mass_torch(mask, normalize, epsilon)
    except ImportError:
        pass
    return _center_of_mass_numpy(np.asarray(mask), normalize)
