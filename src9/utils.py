"""what every src9 model shares that isn't a layer: augmentations, mask geometry, losses, metrics. Layers live in each arch/ file."""
import itertools, math

import torch
import torch.nn.functional as F

#*** 1 augment (train time, on the (B,L,F,C) spectrum)

def speaker_gain(x, freq_hz, sd, knot_hz, p=1.0):
    """an unseen speaker: x * exp(g(f)), g a random smooth log-gain over frequency (the speaker's response), shared by every laser and axis.
    knots every knot_hz Hz ~ N(0, sd), linearly interpolated; p < 1: only a random fraction p of the samples get one"""
    n = max(2, int((freq_hz[1] - freq_hz[0]) / knot_hz) + 2)
    g = F.interpolate(torch.randn(len(x), 1, n, device=x.device) * sd, size=x.shape[2], mode="linear", align_corners=True)[:, 0]  # (B,F)
    if p < 1: g = g * (torch.rand(len(x), 1, device=x.device) < p)
    return x * g[:, None, :, None].exp()

def laser_jitter(x, rows, cols, s, p=1.0):
    """resample the wall's laser field at the grid shifted by U(-s, s) cells per axis, per sample (bilinear, border clamp); p < 1: only a random
    fraction p of the samples are shifted"""
    B, L, F_, C = x.shape
    f = x.permute(0, 2, 3, 1).reshape(B, F_ * C, rows, cols).float()                # (B,F*C,R,Cc)
    sh = (torch.rand(B, 2, device=x.device) * 2 - 1) * s                            # (B,2) cells, (row, col)
    if p < 1: sh = sh * (torch.rand(B, 1, device=x.device) < p)
    theta = torch.zeros(B, 2, 3, device=x.device); theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, 0, 2], theta[:, 1, 2] = sh[:, 1] * 2 / cols, sh[:, 0] * 2 / rows       # normalized: x = col, y = row
    f = F.grid_sample(f, F.affine_grid(theta, (B, 1, rows, cols), align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)
    return f.reshape(B, F_, C, L).permute(0, 3, 1, 2).to(x.dtype)

#*** 2 geometry

def grid_coords(h, w, device=None):
    """(h*w,2) cell-centre (row,col) in [0,1]"""
    r, c = torch.meshgrid((torch.arange(h, device=device) + 0.5) / h, (torch.arange(w, device=device) + 0.5) / w, indexing="ij")
    return torch.stack([r, c], -1).view(-1, 2)

def moments(m):
    """(B,H,W) soft mask -> (B,) area px, (B,2) centroid (row,col), (B,3) covariance (rr, cc, rc), in box fractions"""
    m = m.float()
    H, W = m.shape[-2:]
    r, c = (torch.arange(H, device=m.device) + 0.5) / H, (torch.arange(W, device=m.device) + 0.5) / W
    a = m.flatten(1).sum(1).clamp_min(1e-3)
    mr, mc = (m.sum(2) * r).sum(1) / a, (m.sum(1) * c).sum(1) / a
    dr, dc = r[None] - mr[:, None], c[None] - mc[:, None]
    cov = torch.stack([(m.sum(2) * dr ** 2).sum(1), (m.sum(1) * dc ** 2).sum(1), (m * dr[:, :, None] * dc[:, None]).flatten(1).sum(1)], -1) / a[:, None]
    return a, torch.stack([mr, mc], -1), cov

def orientation(cov): return 0.5 * torch.atan2(2 * cov[:, 2], cov[:, 0] - cov[:, 1])  # (B,3) -> (B,) principal axis angle
def eccentricity(cov):
    tr, det = cov[:, 0] + cov[:, 1], cov[:, 0] * cov[:, 1] - cov[:, 2] ** 2
    return (tr ** 2 - 4 * det).clamp_min(0).sqrt() / tr.clamp_min(1e-9)               # (l1 - l2) / (l1 + l2), 0 = round

def com_err(pred, true, n_objects):
    """(B,K,2) predicted and true coms -> (n_real,) distance of each real object's com, under the best order of the first n slots"""
    B, K, _ = pred.shape
    orders = torch.tensor(list(itertools.permutations(range(K))), device=pred.device)  # (P,K)
    real = torch.arange(K, device=pred.device) < n_objects[:, None]                     # (B,K)
    allowed = ((orders < n_objects[:, None, None]) | ~real[:, None]).all(-1)            # (B,P) real objects only take real slots
    d = (pred[:, orders] - true[:, None]).norm(dim=-1)                                  # (B,P,K)
    best = (d * real[:, None]).sum(-1).masked_fill(~allowed, float("inf")).argmin(1)    # (B,)
    return d[torch.arange(B), best][real]

def outline(m):
    """(B,H,W) -> (B,H,W) 1 px each side of the edge: 3x3 dilation minus 3x3 erosion"""
    m = m[:, None]
    return (F.max_pool2d(m, 3, 1, 1) + F.max_pool2d(-m, 3, 1, 1))[:, 0]

#*** 3 losses

def aux_losses(out, batch, n_known):
    """the aux heads: which objects (unseen ones ignored), how many, each one's com, log area, which box"""
    return dict(object=F.cross_entropy(out["object_logits"], batch["object"], ignore_index=n_known),
                n_objects=F.cross_entropy(out["n_objects_logits"], batch["n_objects"]),
                com=com_err(out["com"], batch["coms"], batch["n_objects"]).sum() / batch["n_objects"].sum().clamp_min(1),
                area=F.l1_loss(out["log_area"], batch["mask_true"].flatten(1).sum(1).log1p()),
                box=F.cross_entropy(out["box_logits"], batch["box"]))

def bce_dice(logits, true, dice_w=0.5):
    p = logits.sigmoid()
    dice = 1 - (2 * (p * true).flatten(1).sum(1) + 1e-6).div((p + true).flatten(1).sum(1) + 1e-6).mean()
    return (1 - dice_w) * F.binary_cross_entropy_with_logits(logits, true) + dice_w * dice

def lovasz_hinge(logits, true):
    """Berman et al. 2018, binary, per image: hinge errors 1 - logit * sign sorted descending, dotted with the gradient of the Lovasz
    extension of the Jaccard loss at that order; mean over images"""
    logits, gt = logits.float().flatten(1), (true.flatten(1) > 0.5).float()
    err, order = (1 - logits * (2 * gt - 1)).sort(1, descending=True)
    gt = gt.gather(1, order)
    n = gt.sum(1, keepdim=True)
    jac = 1 - (n - gt.cumsum(1)) / (n + (1 - gt).cumsum(1))                         # Jaccard loss of the top-k pixels predicted positive
    grad = torch.cat([jac[:, :1], jac[:, 1:] - jac[:, :-1]], 1)
    return (F.relu(err) * grad).sum(1).mean()

def instance_moment_loss(pred, inst, scale=False):
    """per object: each pixel goes to the true object nearest in Mahalanobis distance (+ log det), and the predicted mask in each
    object's cell is matched to that object's log area / 10, centroid and 100 x covariance (L1). One object: the whole image.
    scale: scale-invariant, no constants: |log area ratio| + L1 centroid / s + L1 covariance / s^2, s = sqrt(true area) (box fractions, as moments)"""
    B, N, H, W = inst.shape
    real = inst.flatten(2).sum(-1) > 0                                                # (B,N)
    if not real.any(): return pred.sum() * 0
    _, mt, ct = moments(inst.flatten(0, 1))
    mt, ct = mt.view(B, N, 2), ct.view(B, N, 3)
    S = torch.stack([torch.stack([ct[..., 0] + 1e-3, ct[..., 2]], -1), torch.stack([ct[..., 2], ct[..., 1] + 1e-3], -1)], -2)  # (B,N,2,2)
    d = grid_coords(H, W, pred.device)[None, None] - mt[:, :, None]                   # (B,N,HW,2)
    md = torch.einsum("bnpi,bnij,bnpj->bnp", d, torch.linalg.inv(S), d) + torch.logdet(S)[..., None]
    cell = F.one_hot(md.masked_fill(~real[..., None], float("inf")).argmin(1), N).permute(0, 2, 1).float().view(B, N, H, W)
    (ap, mp, cp), (at, mt_, ct_) = moments((pred.float()[:, None] * cell)[real]), moments(inst[real])
    if scale:
        s2 = at / (H * W)                                                             # (object size)^2 in box fractions^2
        return ((ap.log() - at.log()).abs() + (mp - mt_).abs().sum(-1) / s2.sqrt() + (cp - ct_).abs().sum(-1) / s2).mean()
    return (0.1 * (ap.log() - at.log()).abs() + (mp - mt_).abs().sum(-1) + 100 * (cp - ct_).abs().sum(-1)).mean()

def orient_loss(pred, true, n_objects):
    """single objects: pred vs ecc * (cos 2θ, sin 2θ) of the true mask's principal axis (~0 for round objects)"""
    one = n_objects == 1
    if not one.any(): return pred.sum() * 0
    _, _, cov = moments(true[one])
    th = orientation(cov)
    return F.mse_loss(pred[one], torch.stack([(2 * th).cos(), (2 * th).sin()], -1) * eccentricity(cov)[:, None])

#*** 4 metrics (per sample; every model logs the same set)

THR = 0.08  # the threshold the goal is scored at

def iou(out, batch): return torch.minimum(out["mask_pred"], batch["mask_true"]).flatten(1).sum(1) / torch.maximum(out["mask_pred"], batch["mask_true"]).flatten(1).sum(1).clamp_min(1e-6)  # soft
def iou08(out, batch):
    b, o = out["mask_pred"] > THR, batch["mask_true"] > 0.5
    return ((b & o).flatten(1).sum(1) / (b | o).flatten(1).sum(1).clamp_min(1)).float()[o.flatten(1).any(1)]
def miss08(out, batch):
    b, o = out["mask_pred"] > THR, batch["mask_true"] > 0.5
    return (~(b & o).flatten(1).any(1)).float()[o.flatten(1).any(1)]
def obj_recall(out, batch):
    """per sample: the fraction of its true objects (connected components, batch["objs"]) with >= 1 predicted pixel on them"""
    b, k = out["mask_pred"] > THR, batch["objs"] > 0.5                              # (B,H,W), (B,K,H,W)
    real, n = k.flatten(2).any(-1), k.flatten(2).any(-1).sum(1)                     # (B,K), (B,)
    return ((b[:, None] & k).flatten(2).any(-1) & real).sum(1).div(n.clamp_min(1)).float()[n > 0]
def biou(out, batch):
    """boundary iou (Cheng et al. 2021): iou of the 2 px inner bands (mask - 5x5 erosion) of pred and true"""
    band = lambda m: m + F.max_pool2d(-m[:, None], 5, 1, 2)[:, 0]
    b, o = band((out["mask_pred"] > THR).float()), band((batch["mask_true"] > 0.5).float())
    return (torch.minimum(b, o).flatten(1).sum(1) / torch.maximum(b, o).flatten(1).sum(1).clamp_min(1))[(batch["mask_true"] > 0.5).flatten(1).any(1)]
def area_ratio(out, batch):
    """predicted area / true area, pixels at THR vs > 0.5 (1 = right size, < 1 = under-painted)"""
    b, o = (out["mask_pred"] > THR).flatten(1).sum(1), (batch["mask_true"] > 0.5).flatten(1).sum(1)
    return (b / o.clamp_min(1)).float()[o > 0]
def orient_err(out, batch):
    """degrees between predicted and true principal axes, single objects with eccentricity > 0.2 (0..90)"""
    one = batch["n_objects"] == 1
    if not one.any(): return out["mask_pred"].new_zeros(0)
    _, _, cp = moments(out["mask_pred"][one]); _, _, ct = moments(batch["mask_true"][one])
    d = (orientation(cp) - orientation(ct)).abs() % math.pi
    return torch.minimum(d, math.pi - d)[eccentricity(ct) > 0.2] * 180 / math.pi
def object_acc(out, batch): return (out["object_logits"].argmax(-1) == batch["object"])[batch["object"] < out["object_logits"].shape[-1]].float()
def n_objects_err(out, batch): return (out["n_objects_logits"].argmax(-1) - batch["n_objects"]).abs().float()
def box_acc(out, batch): return (out["box_logits"].argmax(-1) == batch["box"]).float()
def speaker_acc(out, batch): return (out["speaker_logits"].argmax(-1) == out["speaker"])[out["speaker"] < out["speaker_logits"].shape[-1]].float()  # transformer4 --spk-grl-w
def com_error(out, batch): return com_err(out["com"], batch["coms"], batch["n_objects"])

METRICS = {"iou08": iou08, "miss08": miss08, "obj_recall": obj_recall, "biou": biou, "area_ratio": area_ratio, "orient_err": orient_err, "iou": iou,
           "object_acc": object_acc, "n_objects_err": n_objects_err, "box_acc": box_acc, "com_err": com_error, "speaker_acc": speaker_acc}
