"""LDMSegVibrationModel: a vibration encoder trained to land in the frozen LDMSeg mask
autoencoder's latent space, so its pretrained decoder can turn a vibration reading directly into
a segmentation mask. See the plan (okay-with-that-in-glowing-parnas.md) for the full design.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]  # src/ for model.*, repo root for utils.* (used by model.arch)

from torchmetrics import Metric  # noqa: E402
from model.boombox import Encoder, TwoStreamEncoder, Decoder  # noqa: E402
from model.arch import create_metrics, CHEAP_SEG_KEYS  # noqa: E402

from src5.ldmseg_ae import load_frozen_ae  # noqa: E402
from src5.losses import NUM_POINTS, loss_ce_at, loss_mask_at, sample_ce_points, sample_mask_points, stack_instances  # noqa: E402

LATENT_CHANNELS, LATENT_RES = 4, 64


def load_partial(model: nn.Module, ckpt_path: Path) -> None:
    """Loads whichever of ckpt_path's tensors match model's own parameter names+shapes, leaving
    the rest at their current (random) init -- warm-starts the encoder (and, since
    coordconv=False, most of the latent head) from a plain BoomboxModel checkpoint like
    pp-baseline-v2, without needing them to be otherwise identical architectures."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_sd = ckpt["state"]["model"]
    dst_sd = model.state_dict()
    loaded = [k for k, v in src_sd.items() if k in dst_sd and dst_sd[k].shape == v.shape]
    skipped = [k for k in src_sd if k not in loaded]
    dst_sd.update({k: src_sd[k] for k in loaded})
    model.load_state_dict(dst_sd)
    print(f"warm-start from {ckpt_path}: loaded {len(loaded)}/{len(src_sd)} tensors "
          f"(skipped: {skipped})")


def _target_class_map(instance_masks: torch.Tensor, target_classes: torch.Tensor) -> torch.Tensor:
    """(n,H,W) bool masks + (n,) int classes -> dense (H,W) int map, background=0."""
    out = torch.zeros(instance_masks.shape[-2:], dtype=torch.long)
    for mask, cls in zip(instance_masks, target_classes):
        out[mask] = cls
    return out


def load_latents(sample_ids: torch.Tensor, data_dir: Path, device, decode_res: int = 512,
                  cache_name: str = "05_ldmseg_latent.npz") -> dict:
    """Loads each sample's precomputed LDMSeg latent (src5/precompute_latents.py) by sample_id.
    Targets are cached at 512x512 (encode-time resolution, fixed by the frozen AE); if the model
    decodes at a lower `decode_res` (see LDMSegVibrationModel's `decode_interpolate`), downsample
    them once (nearest, to keep mask edges binary) to match before the loss ever sees them.

    `cache_name` picks the file written by precompute_latents.py: the default
    "05_ldmseg_latent.npz" (a distinct id per instance) or, with --same-label,
    "05_ldmseg_latent_same_label.npz" (every instance encoded with the same id -- z_gt only
    carries object presence, not identity).

    Profiling a real training step (torch.profiler) showed cudaStreamSynchronize/
    cudaDeviceSynchronize eating ~91% of CPU time, driven by this function doing up to 2 separate
    .to(device) calls PER SAMPLE (instance_masks, target_classes) -- each a blocking sync point.
    Fixed by concatenating across the whole batch first and transferring once."""
    z_gt, class_map, masks_list, classes_list = [], [], [], []
    for sid in sample_ids.tolist():
        npz = np.load(data_dir / "samples" / f"{sid:06d}" / "image" / cache_name)
        masks, classes = torch.from_numpy(npz["masks_512"]), torch.from_numpy(npz["target_classes"])
        z_gt.append(torch.from_numpy(npz["z_gt"]))
        class_map.append(_target_class_map(masks, classes))
        masks_list.append(masks)
        classes_list.append(classes)

    counts = [m.shape[0] for m in masks_list]
    all_masks = torch.cat(masks_list).to(device, non_blocking=True)
    all_classes = torch.cat(classes_list).to(device, non_blocking=True)
    class_map_t = torch.stack(class_map).to(device, non_blocking=True)
    if decode_res != 512:
        size = (decode_res, decode_res)
        class_map_t = F.interpolate(class_map_t[:, None].float(), size=size, mode="nearest")[:, 0].long()
        all_masks = F.interpolate(all_masks[:, None].float(), size=size, mode="nearest")[:, 0].bool()
    return dict(z_gt=torch.stack(z_gt).to(device, non_blocking=True), target_class_map=class_map_t,
                instance_masks=list(torch.split(all_masks, counts)),
                target_classes=list(torch.split(all_classes, counts)))


def foreground_prob(outputs: torch.Tensor, target_classes: list[torch.Tensor], out_h: int, out_w: int) -> torch.Tensor:
    """(B,128,H,W) decoded logits + each sample's own target_classes -> (B,out_h,out_w) predicted
    foreground probability (summed over that sample's own instance classes), downsampled to the
    project's native grid -- lets us reuse the existing MaskMetric suite unchanged."""
    probs = outputs.softmax(1)
    fg = torch.stack([probs[b, classes].sum(0).clamp(max=1.0) if len(classes) else
                       torch.zeros(outputs.shape[-2:], device=outputs.device)
                       for b, classes in enumerate(target_classes)])
    return F.interpolate(fg[:, None], size=(out_h, out_w), mode="bilinear", align_corners=False)[:, 0]


class LatentMetric(Metric):
    """Mean L2 distance or cosine similarity between z_pred and z_gt, same accumulate-then-divide
    pattern as MaskMetric in src/model/arch.py."""
    def __init__(self, key: str):
        super().__init__()
        assert key in ("l2", "cos")
        self.key = key
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, z_pred: torch.Tensor, z_gt: torch.Tensor) -> None:
        pred, true = z_pred.flatten(1), z_gt.flatten(1)
        value = (pred - true).norm(dim=1) if self.key == "l2" else F.cosine_similarity(pred, true, dim=1)
        self.total = self.total + value.sum().to(self.total)
        self.count = self.count + pred.shape[0]

    def compute(self) -> torch.Tensor:
        return self.total / self.count.clamp(min=1)


class LDMSegVibrationModel(ComposerModel):
    def __init__(self, data_dir, data_info, d_model: int = 1024, encoder: str = "single",
                 fuse: str = "concat", freq_dropout: float = 0.0, laser_dropout: float = 0.0,
                 latent_loss_weight: float = 0.1, ce_loss_weight: float = 1.0,
                 mask_loss_weight: float = 1.0, warmstart_checkpoint: str | None = None,
                 compile: bool = False, decode_interpolate: bool = True, same_label: bool = False):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_info = data_info
        self.latent_loss_weight, self.ce_loss_weight, self.mask_loss_weight = (
            latent_loss_weight, ce_loss_weight, mask_loss_weight)
        # same_label: read the "every object same id" cache written by
        # precompute_latents.py --same-label instead of the default per-instance-id cache --
        # z_gt/target_class_map then only ever encode object presence, never identity.
        self.cache_name = "05_ldmseg_latent_same_label.npz" if same_label else "05_ldmseg_latent.npz"
        # decode_interpolate=False: skip the frozen AE's final bilinear upsample (512x512 ->
        # 256x256, see src5/ldmseg_ae.py's interpolation_factor), a real 4x reduction in the
        # single biggest tensor in the pipeline. See src5/SPEEDUP_LOG.md for the measured effect.
        self.decode_res = 512 if decode_interpolate else 512 // 2  # interpolation_factor=2
        self.decode_interpolate = decode_interpolate

        enc_cls = TwoStreamEncoder if encoder == "two-stream" else Encoder
        enc_kwargs = dict(fuse=fuse) if encoder == "two-stream" else {}
        self.encoder = enc_cls(data_info["n_channels"], d_model, data_info["n_laser_rows"],
                                data_info["n_laser_cols"], freq_dropout, laser_dropout, **enc_kwargs)
        # named `decoder` (not `latent_head`) to match pp-baseline-v2's own attribute name --
        # load_partial matches by exact parameter name string, so this is what makes its
        # project/stages/head[0,2] weights (shape-compatible since coordconv=False) actually load.
        self.decoder = Decoder(d_model, LATENT_RES, LATENT_RES, LATENT_CHANNELS, coordconv=False)
        self.mask_ae = load_frozen_ae()

        if warmstart_checkpoint:
            load_partial(self, Path(warmstart_checkpoint))

        if compile:
            # compile the two trainable modules only -- profiled +16% throughput and -36% peak
            # memory (see src5/SPEEDUP_LOG.md); the frozen mask_ae is untested/unnecessary to
            # compile since it never gets gradient updates.
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

        # separate LatentMetric instances per split -- sharing objects would mix train/eval state
        self.train_metrics = create_metrics(data_info, CHEAP_SEG_KEYS) | {
            "latent_l2": LatentMetric("l2"), "latent_cos": LatentMetric("cos")}
        self.val_metrics = create_metrics(data_info) | {
            "latent_l2": LatentMetric("l2"), "latent_cos": LatentMetric("cos")}

    def _to_conv(self, x: torch.Tensor) -> torch.Tensor:
        """(B,L,P,PS,C) -> (B,C,L,P*PS), same reshape as BoomboxModel."""
        return x.flatten(2, 3).permute(0, 3, 1, 2)

    def forward(self, batch: dict) -> dict:
        emb = self.encoder(self._to_conv(batch["fft"]))
        # Decoder.forward returns (B,H,W,out_c) (channels-last) whenever out_c != 1 -- its RGB-
        # output convention -- so permute back to the (B,C,H,W) the frozen AE's decode() expects.
        z_pred = self.decoder(emb).permute(0, 3, 1, 2).contiguous()
        latents = load_latents(batch["info"]["sample_id"], self.data_dir, z_pred.device, self.decode_res,
                                self.cache_name)
        outputs = self.mask_ae.decode(z_pred, interpolate=self.decode_interpolate)
        out_h, out_w = self.data_info["out_h"], self.data_info["out_w"]
        mask_pred = foreground_prob(outputs, latents["target_classes"], out_h, out_w)
        # PointRend sample locations -- drawn ONCE here (not resampled in loss() or at viz time),
        # so the loss and VisualizePointRend's overlay (src5/viz.py) both see exactly the points
        # the loss was actually computed at. ce_points: (B,NUM_POINTS,2) normalized [0,1]x[0,1],
        # one whole-image set per sample. mask_points/mask_batch_idx: each instance in the batch
        # gets its own NUM_POINTS, stacked, with mask_batch_idx mapping each row back to its
        # sample -- instance count varies per sample, unlike ce_points.
        ce_points = sample_ce_points(outputs)
        pred_i, true_i, mask_batch_idx = stack_instances(outputs, latents["target_classes"], latents["instance_masks"])
        if pred_i is None:
            mask_points = outputs.new_zeros(0, NUM_POINTS, 2)
            mask_batch_idx = torch.zeros(0, dtype=torch.long, device=outputs.device)
        else:
            mask_points = sample_mask_points(pred_i)
        return dict(z_pred=z_pred, outputs=outputs, mask_pred=mask_pred,
                    mask_logits=torch.special.logit(mask_pred.clamp(1e-6, 1 - 1e-6)),
                    ce_points=ce_points, mask_points=mask_points, mask_batch_idx=mask_batch_idx, **latents)

    def loss(self, outputs: dict, batch: dict) -> dict:
        latent = F.mse_loss(outputs["z_pred"], outputs["z_gt"])
        ce = loss_ce_at(outputs["outputs"], outputs["target_class_map"], outputs["ce_points"])
        pred_i, true_i, _ = stack_instances(outputs["outputs"], outputs["target_classes"], outputs["instance_masks"])
        if pred_i is None:
            bce = dice = outputs["outputs"].sum() * 0.0
        else:
            bce, dice = loss_mask_at(pred_i, true_i, outputs["mask_points"])
        # a dict WITHOUT a 'total' key: Composer sums these for backward and logs each key
        # (loss/train/latent, loss/train/ce, loss/train/bce, loss/train/dice) plus their sum
        # (loss/train/total). bce/dice logged separately (rather than pre-summed into one "mask"
        # term) so each is inspectable on its own -- both are already per-object means (loss_mask
        # samples the same number of points per instance regardless of object count), not just a
        # per-batch or per-pixel average.
        return dict(latent=self.latent_loss_weight * latent, ce=self.ce_loss_weight * ce,
                    bce=self.mask_loss_weight * bce, dice=self.mask_loss_weight * dice)

    def get_metrics(self, is_train: bool = False):
        return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch: dict, outputs: dict, metric) -> None:
        if isinstance(metric, LatentMetric):
            metric.update(outputs["z_pred"], outputs["z_gt"])
        else:
            metric.update(outputs["mask_logits"], outputs["mask_pred"], batch["mask_true"],
                           batch["info"]["n_objects"])

    def eval_forward(self, batch: dict, outputs: dict | None = None) -> dict:
        return outputs if outputs is not None else self.forward(batch)
