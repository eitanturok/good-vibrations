"""Logs predicted-vs-true mask images at most once per --viz-interval epochs per split,
accumulating across all of that split's batches (up to MAX_WANDB_IMAGES) instead of just
whatever fit in the first batch. Reuses VisualizeSMask's rendering (src/model/callbacks.py).
"""
import numpy as np
import torch
from PIL import Image

from model.callbacks import MAX_WANDB_IMAGES, VisualizeSMask, _is_due, _to_cpu


class VisualizeSMaskAccum(VisualizeSMask):
    def __init__(self, viz_interval, max_samples=MAX_WANDB_IMAGES, force_save=False):
        super().__init__(viz_interval, force_save)
        self.max_samples = max_samples
        self.done: dict[str, int] = {}   # data_name -> epoch already flushed
        self.buf: dict[str, list] = {}   # data_name -> [(mask_pred, mask_true, info), ...]

    def visualize(self, state, logger, data_name):
        due = _is_due(self.viz_interval, state, self.force_save)
        if not due or self.done.get(data_name) == state.timestamp.epoch.value:
            return
        batches = self.buf.setdefault(data_name, [])
        n = self.max_samples - sum(b[0].shape[0] for b in batches)
        if n <= 0:
            return
        info = {k: v[:n] for k, v in state.batch["info"].items()}
        batches.append((_to_cpu(state.outputs["mask_pred"][:n]), _to_cpu(state.batch["mask_true"][:n]), info))

    def _flush(self, logger, data_name):
        batches = self.buf.pop(data_name, None)
        if not batches:
            return
        mask_pred = torch.cat([b[0] for b in batches])
        mask_true = torch.cat([b[1] for b in batches])
        info = {k: torch.cat([b[2][k] for b in batches]) if torch.is_tensor(batches[0][2][k])
                else sum((list(b[2][k]) for b in batches), []) for k in batches[0][2]}
        logger.log_images(self._render_batch(mask_pred, mask_true, info), name=f"SMask/{data_name}",
                           channels_last=True, use_table=False)

    def eval_end(self, state, logger):
        label = state.dataloader_label or "eval"
        self._flush(logger, label)
        self.done[label] = state.timestamp.epoch.value

    def epoch_end(self, state, logger):
        self._flush(logger, "train")
        # Composer advances state.timestamp to the *next* epoch before firing EPOCH_END
        # (trainer.py: `state.timestamp = state.timestamp.to_next_epoch()` precedes
        # `engine.run_event(Event.EPOCH_END)`), so `.epoch.value` here is already epoch+1 --
        # one past the epoch that was actually just accumulated/flushed in `visualize()`.
        # Stamping `done` with the un-adjusted value pre-marks the *next* due epoch (every
        # viz_interval-th one) as already-visualized before it starts, permanently blocking
        # all train image logging after the very first flush. Subtract 1 to match what
        # visualize() saw during that epoch's batches.
        self.done["train"] = state.timestamp.epoch.value - 1


def _points_density(points: torch.Tensor, size: int) -> np.ndarray:
    """Normalized [0,1]x[0,1] (x,y) PointRend coords -> a (size,size) hit-count histogram.
    Resolution-independent (a fraction across the image means the same thing at any grid size),
    so this works whether `points` came from the 256/512-res decode or a downsampled panel.
    With NUM_POINTS in the thousands over a couple hundred pixels a side, most cells get hit at
    least once (see the model's docstring on the sampled fraction) -- a hit-count histogram (not
    plain dots) is what actually shows PointRend's real signal: WHERE it resamples repeatedly
    (the uncertain/boundary regions it biases toward) versus the flat, once-each random fill."""
    xs = points[:, 0].mul(size).clamp(0, size - 1).long().numpy()
    ys = points[:, 1].mul(size).clamp(0, size - 1).long().numpy()
    hist = np.zeros((size, size), dtype=np.float32)
    np.add.at(hist, (ys, xs), 1.0)
    return hist


class VisualizePointRend(VisualizeSMaskAccum):
    """Same predicted-vs-true mask panels as VisualizeSMaskAccum, but overlaid with a red heatmap
    of the actual PointRend sample locations (src5/losses.py's sample_uncertain_points, exposed
    via LDMSegVibrationModel.forward's ce_points/mask_points/mask_batch_idx) used for that exact
    step's loss -- both the whole-image CE points and every instance's own mask-loss points --
    so it's visible exactly where (and how repeatedly) the loss is computed."""

    PANEL = 256

    def visualize(self, state, logger, data_name):
        due = _is_due(self.viz_interval, state, self.force_save)
        if not due or self.done.get(data_name) == state.timestamp.epoch.value:
            return
        batches = self.buf.setdefault(data_name, [])
        n = self.max_samples - sum(b[0].shape[0] for b in batches)
        if n <= 0:
            return
        ce_points = _to_cpu(state.outputs["ce_points"][:n])
        mask_points, mask_idx = _to_cpu(state.outputs["mask_points"]), _to_cpu(state.outputs["mask_batch_idx"])
        points = [torch.cat([ce_points[b], mask_points[mask_idx == b].reshape(-1, 2)])
                  for b in range(len(ce_points))]
        batches.append((_to_cpu(state.outputs["mask_pred"][:n]), _to_cpu(state.batch["mask_true"][:n]), points))

    def _flush(self, logger, data_name):
        batches = self.buf.pop(data_name, None)
        if not batches:
            return
        mask_pred = torch.cat([b[0] for b in batches])
        mask_true = torch.cat([b[1] for b in batches])
        points = sum((b[2] for b in batches), [])
        logger.log_images(self._render_points(mask_pred, mask_true, points), name=f"SMaskPoints/{data_name}",
                           channels_last=True, use_table=False)

    # a cell resampled this many times or more renders as fully-opaque red; below that, alpha
    # scales linearly -- so a once-hit cell (most of the image, since sampling is with
    # replacement over thousands of points) reads as a light wash, while cells PointRend keeps
    # re-sampling for being uncertain (mask boundaries) stand out as solid.
    HOT = 4

    def _render_points(self, mask_pred: torch.Tensor, mask_true: torch.Tensor,
                        points: list[torch.Tensor]) -> list[np.ndarray]:
        size, sep = self.PANEL, 4
        out = []
        for pred, true, pts in zip(mask_pred.numpy(), mask_true.numpy(), points):
            alpha = np.clip(_points_density(pts, size) / self.HOT, 0, 1)[..., None]  # (size,size,1)
            canvas = Image.new("RGB", (size * 2 + sep, size), (255, 255, 255))
            for i, arr in enumerate((pred, true)):
                panel = np.array(Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
                                  .convert("RGB").resize((size, size), Image.NEAREST)).astype(np.float32)
                red = np.array([255.0, 0.0, 0.0])
                blended = (panel * (1 - alpha) + red * alpha).clip(0, 255).astype(np.uint8)
                canvas.paste(Image.fromarray(blended), (i * (size + sep), 0))
            out.append(np.array(canvas))
        return out
