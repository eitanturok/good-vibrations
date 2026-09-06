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

def _centroids_batch(ids):
    """(B,H,W) component ids (0 = background) -> [(K_b, 2)] centroids per sample, (row, col).

    One bincount over the whole batch instead of the obvious Python loop. The loop made two
    small tensor calls PER BLOB -- 5918 of them on a 3000-sample run, 0.115s, which was the
    single dominant cost inside `localization` and therefore inside every eval batch during
    training (_label, by contrast, is 0.024s). Nothing about the result changes; this is the
    same mean position of each component's cells.

    Labels come from _label as `arange(1, H*W+1) * mask`, so they already live in [0, H*W].
    Offsetting each sample by (H*W+1) makes one global id space, which is what lets a single
    bincount accumulate counts and row/col sums for every blob in the batch at once.
    """
    B, H, W = ids.shape
    n = H * W + 1
    dev = ids.device
    flat = ids.reshape(B, -1).long()
    gid = (torch.arange(B, device=dev)[:, None] * n + flat).reshape(-1)
    rows = torch.arange(H, device=dev, dtype=torch.float32).repeat_interleave(W).repeat(B)
    cols = torch.arange(W, device=dev, dtype=torch.float32).repeat(H * B)
    size = B * n
    cnt = torch.bincount(gid, minlength=size).reshape(B, n).float()
    sr = torch.bincount(gid, weights=rows, minlength=size).reshape(B, n)
    sc = torch.bincount(gid, weights=cols, minlength=size).reshape(B, n)
    keep = cnt > 0
    keep[:, 0] = False                       # id 0 is background, not an object
    bi, li = keep.nonzero(as_tuple=True)
    c = cnt[bi, li]
    cents = torch.stack([sr[bi, li] / c, sc[bi, li] / c], dim=1)
    # split back per sample in one call; bi is sorted by construction, so the split is aligned
    return list(torch.split(cents, torch.bincount(bi, minlength=B).tolist()))


def _centroids(ids):
    """Single-sample form of _centroids_batch, for callers holding one (H,W) label map."""
    return _centroids_batch(ids[None])[0]

def object_centroids(mask) -> list:
    """Per-OBJECT centroids of a batch of masks: [(K_i, 2) float arrays] in (row, col) grid
    coords, one entry per sample, K_i = however many blobs that sample has.

    This is what `localization` matches on, exposed so callers can DRAW the points rather
    than only score them. Binarize -> connected components -> mean position of each
    component's cells, so it is the geometric centroid of the thresholded blob and carries
    no probability weighting -- deliberately the same definition the metric uses, so a
    crosshair never disagrees with the number printed beside it.

    Distinct from utils.metrics.center_of_mass, which is ONE probability-weighted point for
    the whole mask: on a 3-cube sample that lands between the cubes, on nothing.
    """
    import numpy as np
    x = mask if torch.is_tensor(mask) else torch.as_tensor(mask)
    if x.ndim == 2: x = x[None]
    return [_centroids(ids).numpy().astype(np.float64) for ids in _label(_bin(x.float())).cpu()]


def match_objects(pred_c, gt_c, scale=(1.0, 1.0)):
    """Greedy GT->pred pairing on per-object centroids.

    Returns (pairs, missed_gt, extra_pred): `pairs` is [(pred_idx, gt_idx)], `missed_gt` the
    GT indices left with no predicted blob, `extra_pred` the predicted blobs nothing matched.

    Walks GT objects in order and takes the nearest unused prediction for each -- the exact
    rule `localization` scores with, factored out so a drawn line and the number beside it
    can never disagree. `scale` divides (row, col) before measuring, so callers can match in
    fraction-of-box units (1/h, 1/w) rather than cells.

    Note the asymmetry, which is a property of the metric and not of this helper: iteration
    is over GT, so `extra_pred` blobs cost nothing. They are returned anyway because a
    hallucinated blob is worth SEEING even when it is not scored.
    """
    sr, sc = scale
    used, pairs = set(), []
    for gi, g in enumerate(gt_c):
        free = [k for k in range(len(pred_c)) if k not in used]
        if not free: continue
        k = min(free, key=lambda k: ((pred_c[k][0] - g[0]) * sr) ** 2 + ((pred_c[k][1] - g[1]) * sc) ** 2)
        used.add(k)
        pairs.append((k, gi))
    matched_gt = {gi for _, gi in pairs}
    return pairs, [i for i in range(len(gt_c)) if i not in matched_gt], sorted(set(range(len(pred_c))) - used)


LOC_KEYS = ('localization_rel', 'localization_rel_h', 'localization_rel_w',
            'localization_raw', 'localization_raw_h', 'localization_raw_w')

def localization(pred, true, geometry: bool = False):
    """Centre-of-mass error, matched greedily GT->pred and averaged over the objects in each
    sample. Returns a dict of the six LOC_KEYS, each a (B,) tensor, NaN where the sample has
    no GT object.

    Two flavours of the same errors:

      *_rel  -- FRACTION OF THE BOX. A cell is H_box/h pixels tall, so a row error of `dr`
                cells is dr/h of the box height. Independent of (out_h, out_w), so a 21x30 run
                and a 32x32 run are directly comparable and the miss penalty stops growing with
                the grid. `_h`/`_w` are each in [0,1]; the combined `localization_rel` is their
                Euclidean sum and so ranges over [0, sqrt(2)], the diagonal of the unit square.
                Deliberately NOT rescaled into [0,1] -- left alone, 0.05 means "off by 5% of the
                box", a statement you can make out loud. Miss scores (sqrt(2), 1, 1).

      *_raw  -- GRID CELLS, i.e. exactly what this function returned before the _rel variant
                existed. Kept so runs logged earlier stay comparable. Miss scores (hypot(h,w),
                h, w). Do not compare _raw across different grid sizes; that is what _rel is for.

    `_h` is the vertical (row) error and `_w` the horizontal (column) error. The old names were
    `_x` for the row delta and `_y` for the column delta, which read backwards.

    The greedy match runs once, in normalized units, and both flavours report that same
    pairing -- otherwise the two could silently disagree about which blob is which.

    `geometry=True` additionally returns the drawing geometry of that same match, as
    (metrics, geom) -- geom[i] = {"pairs": [[pr,pc,gr,gc],...], "missed": [[r,c],...],
    "extra": [[r,c],...]} in normalized [0,1] cell centres. It is returned from HERE rather
    than recomputed by the caller because labelling and centroiding the masks a second time
    to get the same answer doubled the cost of scoring a run, and left open the possibility
    of a drawn line disagreeing with the number printed beside it.
    """
    h, w = pred.shape[-2:]
    miss_raw = (h * h + w * w) ** 0.5
    # Batched: one labelling pass and one centroid pass for the whole batch. This runs on
    # every eval batch during training, so the per-blob Python loop it replaced was not free.
    P = _centroids_batch(_label(_bin(pred)).cpu())
    T = _centroids_batch(_label(_bin(true)).cpu())
    out = {k: [] for k in LOC_KEYS}
    geom = []
    # Normalized cell CENTRES, for drawing: (r+0.5)/h puts a mark in the middle of the cell
    # the blob occupies, and the fraction is grid-independent so a 21x30 and a 32x32 column
    # can be drawn in identically sized boxes.
    nz = lambda c: [round(float(c[0] + 0.5) / h, 4), round(float(c[1] + 0.5) / w, 4)]
    for pc, gc in zip(P, T):
        if not len(gc):
            for k in LOC_KEYS: out[k].append(float('nan'))
            # No target to match, but a blob predicted here is still worth SEEING.
            if geometry:
                geom.append({"pairs": [], "missed": [], "extra": [nz(c) for c in pc]})
            continue
        acc = {k: [] for k in LOC_KEYS}
        # Matched on the NORMALIZED distance -- the same number _rel reports -- through the
        # shared helper, so viz can draw these exact pairings without re-deriving the rule.
        pairs, missed, extra = match_objects(pc, gc, scale=(1.0 / h, 1.0 / w))
        if geometry:
            geom.append({"pairs": [nz(pc[k]) + nz(gc[g]) for k, g in pairs],
                         "missed": [nz(gc[g]) for g in missed],
                         "extra": [nz(pc[k]) for k in extra]})
        for k, _gi in pairs:
            g = gc[_gi]
            dr, dc = float(pc[k][0] - g[0]), float(pc[k][1] - g[1])
            rr, rc = dr / h, dc / w
            acc['localization_raw'].append((dr * dr + dc * dc) ** 0.5)
            acc['localization_raw_h'].append(abs(dr))
            acc['localization_raw_w'].append(abs(dc))
            acc['localization_rel'].append((rr * rr + rc * rc) ** 0.5)
            acc['localization_rel_h'].append(abs(rr))
            acc['localization_rel_w'].append(abs(rc))
        for _gi in missed:
            for k, v in (('localization_raw', miss_raw), ('localization_raw_h', float(h)),
                         ('localization_raw_w', float(w)), ('localization_rel', 2 ** 0.5),
                         ('localization_rel_h', 1.0), ('localization_rel_w', 1.0)):
                acc[k].append(v)
        for k in LOC_KEYS: out[k].append(sum(acc[k]) / len(acc[k]))
    metrics = {k: torch.tensor(v, dtype=torch.float32) for k, v in out.items()}
    return (metrics, geom) if geometry else metrics
