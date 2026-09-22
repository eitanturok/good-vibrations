"""LDMSeg's Eqn 1 (CE + BCE + Dice with PointRend point sampling), ported from
~/.cache/ldmseg/repo/ldmseg/trainers/losses.py and .../utils/detectron2_utils.py (PointRend,
Kirillov et al., https://arxiv.org/abs/1912.08193) as plain functions -- their SegmentationLosses
class also implements a Hungarian-matcher path (matching predicted "query" channels to instances
by cost) that we don't need: we always know exactly which output channel an instance corresponds
to (its precomputed target_class) and exactly which pixels are its true mask, so there's nothing
to match.
"""
import torch
import torch.nn.functional as F

NUM_POINTS = 12544
OVERSAMPLE_RATIO = 3
IMPORTANCE_SAMPLE_RATIO = 0.75


def point_sample(input: torch.Tensor, point_coords: torch.Tensor, **kwargs) -> torch.Tensor:
    """Bilinear-sample `input` (B,C,H,W) at `point_coords` (B,P,2), normalized to [0,1]x[0,1]."""
    output = F.grid_sample(input, 2.0 * point_coords.unsqueeze(2) - 1.0, **kwargs)
    return output.squeeze(3)


def sample_uncertain_points(logits: torch.Tensor, uncertainty_fn, num_points: int,
                             oversample_ratio: int, importance_sample_ratio: float) -> torch.Tensor:
    """Sample `num_points` locations per image, biased toward wherever `uncertainty_fn` says the
    current prediction is most ambiguous (importance_sample_ratio fraction) plus plain random
    points (the rest) -- PointRend's trick for training a dense predictor without evaluating the
    loss at every pixel."""
    B = logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    coords = torch.rand(B, num_sampled, 2, device=logits.device)
    uncertainty = uncertainty_fn(point_sample(logits, coords, align_corners=False))
    n_uncertain = int(importance_sample_ratio * num_points)
    top_idx = torch.topk(uncertainty[:, 0, :], k=n_uncertain, dim=1)[1]
    shift = num_sampled * torch.arange(B, device=logits.device)
    coords_uncertain = coords.view(-1, 2)[(top_idx + shift[:, None]).view(-1)].view(B, n_uncertain, 2)
    n_random = num_points - n_uncertain
    if n_random == 0:
        return coords_uncertain
    return torch.cat([coords_uncertain, torch.rand(B, n_random, 2, device=logits.device)], dim=1)


def calculate_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    """For a binary (foreground-vs-background) logit map: a value near 0 is the most ambiguous."""
    return -logits.abs()


def calculate_uncertainty_seg(logits: torch.Tensor) -> torch.Tensor:
    """For a multi-class logit map: a small margin between the top-2 classes is most ambiguous."""
    top2 = torch.topk(logits, k=2, dim=1)[0]
    return (top2[:, 1] - top2[:, 0]).unsqueeze(1)


def sigmoid_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = logits.sigmoid()
    numerator = 2 * (probs * targets).sum(-1)
    denominator = probs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).mean()


def sample_ce_points(outputs: torch.Tensor, num_points: int = NUM_POINTS,
                      oversample_ratio: int = OVERSAMPLE_RATIO,
                      importance_sample_ratio: float = IMPORTANCE_SAMPLE_RATIO) -> torch.Tensor:
    """The (B,num_points,2) normalized [0,1]x[0,1] coords loss_ce would sample for `outputs`,
    exposed standalone so a caller (LDMSegVibrationModel.forward) can sample once, reuse the same
    points for the actual loss, and also hand them to a viz callback -- guaranteeing "where we
    plot the loss points" is exactly where the loss was computed, not a fresh resample."""
    with torch.no_grad():
        return sample_uncertain_points(outputs, calculate_uncertainty_seg, num_points,
                                        oversample_ratio, importance_sample_ratio)


def loss_ce_at(outputs: torch.Tensor, target_class_map: torch.Tensor, coords: torch.Tensor,
               ignore_label: int = 0) -> torch.Tensor:
    """loss_ce's actual loss computation, given already-sampled `coords` (see sample_ce_points)."""
    with torch.no_grad():
        labels = point_sample(target_class_map[:, None].float(), coords, mode="nearest",
                               align_corners=False).squeeze(1).long()
    logits = point_sample(outputs, coords, align_corners=False)
    return F.cross_entropy(logits, labels, ignore_index=ignore_label)


def loss_ce(outputs: torch.Tensor, target_class_map: torch.Tensor, ignore_label: int = 0,
            num_points: int = NUM_POINTS, oversample_ratio: int = OVERSAMPLE_RATIO,
            importance_sample_ratio: float = IMPORTANCE_SAMPLE_RATIO) -> torch.Tensor:
    """outputs: (B,128,H,W) logits. target_class_map: (B,H,W) int, each instance's pixels set to
    its own target_class, background/`ignore_label` elsewhere."""
    coords = sample_ce_points(outputs, num_points, oversample_ratio, importance_sample_ratio)
    return loss_ce_at(outputs, target_class_map, coords, ignore_label)


def stack_instances(outputs: torch.Tensor, target_classes: list[torch.Tensor],
                     instance_masks: list[torch.Tensor]
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[None, None, None]:
    """outputs: (B,128,H,W) logits. target_classes/instance_masks: one entry per batch element,
    each a (n_i,)-int tensor of that instance's target_class and a (n_i,H,W)-bool tensor of its
    true mask (n_i varies per sample). For each instance, its predicted logit map is whichever
    output channel its own target_class selects. Returns stacked (N,1,H,W) pred/true across every
    instance in the batch, plus (N,) batch_idx mapping each instance back to its sample index --
    or (None, None, None) if the batch has zero instances."""
    pred, true, batch_idx = [], [], []
    for b, (classes, masks) in enumerate(zip(target_classes, instance_masks)):
        if len(classes) == 0:
            continue
        pred.append(outputs[b, classes])       # (n_i,H,W)
        true.append(masks.float())             # (n_i,H,W)
        batch_idx.append(torch.full((len(classes),), b, dtype=torch.long))
    if not pred:
        return None, None, None
    return torch.cat(pred)[:, None], torch.cat(true)[:, None], torch.cat(batch_idx)


def sample_mask_points(pred: torch.Tensor, num_points: int = NUM_POINTS,
                        oversample_ratio: int = OVERSAMPLE_RATIO,
                        importance_sample_ratio: float = IMPORTANCE_SAMPLE_RATIO) -> torch.Tensor:
    """The (N,num_points,2) coords loss_mask would sample for `pred` (see stack_instances) --
    same "sample once, reuse for loss + viz" purpose as sample_ce_points."""
    with torch.no_grad():
        return sample_uncertain_points(pred, calculate_uncertainty, num_points,
                                        oversample_ratio, importance_sample_ratio)


def loss_mask_at(pred: torch.Tensor, true: torch.Tensor, coords: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """loss_mask's actual BCE+Dice computation, given already-sampled `coords` (see
    sample_mask_points) and `pred`/`true` (see stack_instances). Returned separately (rather than
    pre-summed) so each can be logged and inspected on its own. Every instance gets the same
    number of points regardless of how many instances/objects are in its own sample or in the
    batch, so both reductions are already a plain per-object mean, not skewed toward samples with
    more objects."""
    with torch.no_grad():
        labels = point_sample(true, coords, align_corners=False).squeeze(1)
    logits = point_sample(pred, coords, align_corners=False).squeeze(1)
    return sigmoid_ce_loss(logits, labels), dice_loss(logits, labels)


def loss_mask(outputs: torch.Tensor, target_classes: list[torch.Tensor],
              instance_masks: list[torch.Tensor], num_points: int = NUM_POINTS,
              oversample_ratio: int = OVERSAMPLE_RATIO,
              importance_sample_ratio: float = IMPORTANCE_SAMPLE_RATIO) -> tuple[torch.Tensor, torch.Tensor]:
    pred, true, _ = stack_instances(outputs, target_classes, instance_masks)
    if pred is None:
        zero = outputs.sum() * 0.0
        return zero, zero
    coords = sample_mask_points(pred, num_points, oversample_ratio, importance_sample_ratio)
    return loss_mask_at(pred, true, coords)
