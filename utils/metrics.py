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


#***** soft IoU *****

def _soft_iou_numpy(mask_pred: np.ndarray, mask_true: np.ndarray, epsilon: float):
    intersection = np.minimum(mask_pred, mask_true).sum(axis=(-2, -1))
    union = np.maximum(mask_pred, mask_true).sum(axis=(-2, -1))
    return (intersection + epsilon) / (union + epsilon)

def _soft_iou_torch(mask_pred, mask_true, epsilon: float):
    import torch
    intersection = torch.minimum(mask_pred, mask_true).sum(dim=(-2, -1))
    union = torch.maximum(mask_pred, mask_true).sum(dim=(-2, -1))
    return (intersection + epsilon) / (union + epsilon)

def soft_iou(mask_pred, mask_true, epsilon: float = 1e-6):
    """Soft IoU between two [0,1] masks of shape (...,H,W) -> (...) per-mask scores.
    Accepts numpy arrays or torch tensors -- routed automatically to the matching
    implementation. Uses the min/max (Ruzicka) form rather than the product form:
    both agree on binary masks, but only min/max scores a perfect prediction of a
    *soft* mask as 1.0."""
    try:
        import torch
        if isinstance(mask_pred, torch.Tensor): return _soft_iou_torch(mask_pred, mask_true, epsilon)
    except ImportError:
        pass
    return _soft_iou_numpy(np.asarray(mask_pred), np.asarray(mask_true), epsilon)


def _soft_dice_numpy(mask_pred: np.ndarray, mask_true: np.ndarray, epsilon: float):
    intersection = np.minimum(mask_pred, mask_true).sum(axis=(-2, -1))
    total = mask_pred.sum(axis=(-2, -1)) + mask_true.sum(axis=(-2, -1))
    return (2 * intersection + epsilon) / (total + epsilon)


def _soft_dice_torch(mask_pred, mask_true, epsilon: float):
    import torch
    intersection = torch.minimum(mask_pred, mask_true).sum(dim=(-2, -1))
    total = mask_pred.sum(dim=(-2, -1)) + mask_true.sum(dim=(-2, -1))
    return (2 * intersection + epsilon) / (total + epsilon)


def soft_dice(mask_pred, mask_true, epsilon: float = 1e-6):
    """Soft Dice, 2*Sum(min(p,g)) / (Sum(p) + Sum(g)), between two [0,1] masks of shape
    (...,H,W) -> (...) per-mask scores. Accepts numpy arrays or torch tensors -- routed
    automatically to the matching implementation. Like soft_iou it uses min for the
    intersection rather than the product, so a perfect prediction of a *soft* mask
    scores 1.0; the two relate by the exact identity dice = 2*iou / (1 + iou)."""
    try:
        import torch
        if isinstance(mask_pred, torch.Tensor): return _soft_dice_torch(mask_pred, mask_true, epsilon)
    except ImportError:
        pass
    return _soft_dice_numpy(np.asarray(mask_pred), np.asarray(mask_true), epsilon)
