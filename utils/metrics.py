"""Metrics shared across the numpy (local/CPU) and torch (model training) pipelines."""

import numpy as np
import torch
import torch.nn.functional as F
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


#***** segmentation metrics *****
# All take (B,H,W) probability / soft masks and return a (B,) or (n_objects,) tensor.

THRESH = 0.5

def mass_error(pred, true, eps=1e-6):
    """Symmetric relative mass error in [-1, 1]; negative = under-paint."""
    p, t = pred.sum((-2, -1)), true.sum((-2, -1))
    return (p - t) / (p + t + eps)

def _bin(x): return (x > THRESH).float()

def _boundary(x):
    return (x + F.max_pool2d(-x[:, None], 3, 1, 1)[:, 0]).clamp(min=0)

def _reach(a, b, tol):
    return a * (F.max_pool2d(b[:, None], 2 * tol + 1, 1, tol)[:, 0] > 0)

def contour_f(pred, true, tol=1):
    """Boundary F-score: fraction of each outline within `tol` cells of the other."""
    bp, bt = _boundary(_bin(pred)), _boundary(_bin(true))
    ap, at = bp.sum((-2, -1)), bt.sum((-2, -1))
    prec = _reach(bp, bt, tol).sum((-2, -1)) / ap.clamp(min=1)
    rec = _reach(bt, bp, tol).sum((-2, -1)) / at.clamp(min=1)
    f = 2 * prec * rec / (prec + rec).clamp(min=1e-9)
    return torch.where((ap == 0) & (at == 0), torch.ones_like(f), f)

def _label(x, iters=64):
    """(B,H,W) 0/1 mask -> (B,H,W) long component ids, 0 = background."""
    _, h, w = x.shape
    ids = torch.arange(1, h * w + 1, device=x.device).view(1, h, w) * x.long()
    for _ in range(iters):
        nxt = F.max_pool2d(ids[:, None].float(), 3, 1, 1)[:, 0].long() * x.long()
        if torch.equal(nxt, ids): break
        ids = nxt
    return ids

def _centroids(ids):
    ls = ids.unique()
    ls = ls[ls > 0]
    if not len(ls): return ids.new_zeros((0, 2), dtype=torch.float32)
    return torch.stack([(ids == l).nonzero().float().mean(0) for l in ls])

def localization(pred, true):
    """Centre-of-mass error in grid cells, matched greedily GT->pred, averaged over the
    objects in each sample. Returns (err, err_x, err_y), each (B,); NaN where the sample
    has no GT object. A GT object with no predicted blob scores (hypot(H,W), H, W)."""
    h, w = pred.shape[-2:]
    miss = (h * h + w * w) ** 0.5
    P, T = _label(_bin(pred)).cpu(), _label(_bin(true)).cpu()
    err, ex, ey = [], [], []
    for pi, ti in zip(P, T):
        pc, gc = _centroids(pi), _centroids(ti)
        if not len(gc):
            err.append(float('nan')); ex.append(float('nan')); ey.append(float('nan')); continue
        used, es, xs, ys = set(), [], [], []
        for g in gc:
            free = [k for k in range(len(pc)) if k not in used]
            if free:
                k = min(free, key=lambda k: float((pc[k] - g).norm()))
                used.add(k)
                dr, dc = float(pc[k][0] - g[0]), float(pc[k][1] - g[1])
                es.append((dr * dr + dc * dc) ** 0.5); xs.append(abs(dr)); ys.append(abs(dc))
            else:
                es.append(miss); xs.append(float(h)); ys.append(float(w))
        err.append(sum(es) / len(es)); ex.append(sum(xs) / len(xs)); ey.append(sum(ys) / len(ys))
    t = lambda v: torch.tensor(v, dtype=torch.float32)
    return t(err), t(ex), t(ey)
