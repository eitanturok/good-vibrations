import os

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from composer.core import State, Time, TimeUnit
from composer import Callback, Logger
from composer.utils import format_name_with_dist
from composer.loggers import WandBLogger

from model.arch import com_distances, mses


MAX_WANDB_IMAGES = 108  # wandb.Image caps any single log_images call at this many items

def _to_cpu(x: torch.Tensor):
    # upcast reduced-precision autocast outputs (bf16/fp16): numpy has no bf16 dtype
    x = x.detach().to('cpu', copy=True)
    return x.float() if x.dtype in (torch.bfloat16, torch.float16) else x

def _is_due(interval: Time, state: State, force: bool) -> bool:
    """True when the trainer timestamp lands on an interval boundary, or `force` overrides it."""
    return force or state.timestamp.get(interval.unit).value % interval.value == 0


class VisualizeSMask(Callback):
    def __init__(self, viz_interval, force_save: bool = False):
        self.viz_interval = Time.from_input(viz_interval, TimeUnit.EPOCH)
        # when True, bypasses the viz_interval due-check entirely: set this around a manual
        # trainer.eval() call made outside the normal training loop (e.g. epoch 0 before any weight
        # update, or the final epoch after fit() ends) that wouldn't otherwise land on a due epoch
        self.force_save = force_save

    def _render(self, pred_np, true_np, info, mse_vals, com_dists, i, scale=8, text_height=40, sep=4, font=ImageFont.load_default(size=14)):
        h, w = pred_np[i].shape
        ph, pw = h * scale, w * scale  # panel size after upscale
        canvas = Image.new("RGB", (pw * 2 + sep, ph + text_height), (255, 255, 255))
        for j, (arr, label) in enumerate([(pred_np[i], "Predicted"), (true_np[i], "Ground Truth")]):
            panel = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).resize((pw, ph), Image.NEAREST)
            canvas.paste(panel, (j * (pw + sep), text_height))
            ImageDraw.Draw(canvas).text((j * (pw + sep) + pw // 2, text_height - 14), label, fill=(0, 0, 0), font=font, anchor="mt")
        text = (f"id={info['sample_id'][i]}  pos={info['position_id'][i]}  spk={info['speaker'][i]}  objs={info['n_objects'][i]}  "
                f"com=({info['x_com'][i]:.1f},{info['y_com'][i]:.1f})  mse={mse_vals[i]:.4f}  com_dist={com_dists[i]:.4f}")
        ImageDraw.Draw(canvas).text((pw + sep // 2, 2), text, fill=(80, 80, 80), font=font, anchor="mt")
        return np.array(canvas)

    def _render_batch(self, mask_pred: torch.Tensor, mask_true: torch.Tensor, info: dict) -> list[np.ndarray]:
        com_dists = com_distances(mask_pred, mask_true, epsilon=1e-6, normalize=True).numpy()
        mse_vals = mses(mask_pred, mask_true).numpy()
        pred_np, true_np = mask_pred.numpy(), mask_true.numpy()
        return [self._render(pred_np, true_np, info, mse_vals, com_dists, i) for i in range(len(pred_np))]

    def visualize(self, state: State, logger: Logger, data_name: str):
        """Render the batch's first MAX_WANDB_IMAGES samples straight from in-memory state, so viz
        needs no OutputSaver and no .pt on disk."""
        if not _is_due(self.viz_interval, state, self.force_save): return
        mask_pred, mask_true = _to_cpu(state.outputs['mask_pred'][:MAX_WANDB_IMAGES]), _to_cpu(state.batch['mask_true'][:MAX_WANDB_IMAGES])
        info = {k: v[:MAX_WANDB_IMAGES] for k, v in state.batch['info'].items()}
        logger.log_images(self._render_batch(mask_pred, mask_true, info), name=f'SMask/{data_name}', channels_last=True, use_table=False)

    def after_forward(self, state, logger): self.visualize(state, logger, state.dataloader_label)
    def eval_after_forward(self, state, logger): self.visualize(state, logger, state.dataloader_label or "eval")

# ***** OutputSaver *****

OUTPUT_EXTRACTORS = {
    'mask_pred':   lambda state: _to_cpu(state.outputs['mask_pred']),
    'mask_logits': lambda state: _to_cpu(state.outputs['mask_logits']),
    'mask_true':   lambda state: _to_cpu(state.batch['mask_true']),
    'fft':         lambda state: _to_cpu(state.batch['fft']),
    'info':        lambda state: state.batch['info'],
}
DEFAULT_OUTPUT_KEYS = ('mask_pred', 'info')


class OutputSaver(Callback):
    def __init__(self, save_interval, folder, filename='ep{epoch:04d}-ba{batch:06d}.pt', overwrite:bool=False, output_keys=DEFAULT_OUTPUT_KEYS):
        self.save_interval, self.folder, self.filename, self.overwrite = Time.from_input(save_interval, TimeUnit.EPOCH), folder, filename, overwrite
        if unknown := set(output_keys) - set(OUTPUT_EXTRACTORS):
            raise ValueError(f'OutputSaver: unknown output_keys {sorted(unknown)}; valid keys are {sorted(OUTPUT_EXTRACTORS)}')
        self.output_keys = tuple(output_keys)
        # when True, bypasses the save_interval due-check entirely: set this around a manual
        # trainer.eval() call made outside the normal training loop (e.g. epoch 0 before any weight
        # update, or the final epoch after fit() ends) that wouldn't otherwise land on a due epoch
        self.force_save = False

    def init(self, state: State, logger: Logger):
        del logger
        self.folder = format_name_with_dist(self.folder, state.run_name)
        os.makedirs(self.folder, exist_ok=True)

    def save_outputs(self, state: State, logger: Logger, data_name: str, batch: int):
        # due-check and filename epoch both use the *trainer* timestamp, so eval dumps taken
        # during training land on the real epoch (ep0050, ep0100, ...) instead of ep0000
        if _is_due(self.save_interval, state, self.force_save):
            outputs = {k: OUTPUT_EXTRACTORS[k](state) for k in self.output_keys}

            # local disk is the ground truth: write it first and let it raise before any logger is touched,
            # so a logger destination never ends up with data that wasn't also saved locally
            path = os.path.join(self.folder, data_name, self.filename.format(epoch=state.timestamp.epoch.value, batch=batch))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path):
                if not self.overwrite: raise FileExistsError(f'OutputSaver: file already exists: {path}\nSet overwrite=True to overwrite.')
                os.remove(path)
            torch.save(outputs, path)

            # save outputs to all loggers except WandB
            for destination in logger.destinations:
                if isinstance(destination, WandBLogger): continue # wandb JSON serialization fails on raw tensor payloads
                destination.log_metrics({f'{data_name}/{k}': v for k, v in outputs.items()})

    def after_forward(self, state, logger): self.save_outputs(state, logger, state.dataloader_label, state.timestamp.batch.value)
    def eval_after_forward(self, state, logger): self.save_outputs(state, logger, f'{state.dataloader_label or "eval"}', state.eval_timestamp.batch.value)
