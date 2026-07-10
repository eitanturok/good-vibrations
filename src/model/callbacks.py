import os

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from composer.core import State, Time, TimeUnit
from composer import Callback, Logger
from composer.utils import format_name_with_dist
from composer.loggers import WandBLogger

from model.model import com_distances, mses

# ***** VizSegMask *****
# Not a composer Callback: OutputSaver owns the after_forward/eval_after_forward hooks (it's the
# thing that knows local disk is the ground truth) and calls VizSegMask.upload(path, ...) itself,
# once the local save has already succeeded. This makes "save-before-upload" structural rather than
# a fact about callback list order in run.py, which would silently break if the list were reordered.

MAX_WANDB_IMAGES = 108  # wandb.Image caps any single log_images call at this many items

class VizSegMask:
    def _render(self, pred_np, true_np, info, mse_vals, com_dists, i, scale=8, text_height=40, sep=4, font=ImageFont.load_default(size=14)):
        h, w = pred_np[i].shape
        ph, pw = h * scale, w * scale  # panel size after upscale
        canvas = Image.new("RGB", (pw * 2 + sep, ph + text_height), (255, 255, 255))
        for j, (arr, label) in enumerate([(pred_np[i], "Predicted"), (true_np[i], "Ground Truth")]):
            panel = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).resize((pw, ph), Image.NEAREST)
            canvas.paste(panel, (j * (pw + sep), text_height))
            ImageDraw.Draw(canvas).text((j * (pw + sep) + pw // 2, text_height - 14), label, fill=(0, 0, 0), font=font, anchor="mt")
        text = (f"id={info['sample_id'][i]}  spk={info['speaker'][i]}  objs={info['n_objects'][i]}  "
                f"com=({info['x_com'][i]:.1f},{info['y_com'][i]:.1f})  mse={mse_vals[i]:.4f}  com_dist={com_dists[i]:.4f}")
        ImageDraw.Draw(canvas).text((pw + sep // 2, 2), text, fill=(80, 80, 80), font=font, anchor="mt")
        return np.array(canvas)

    def _render_batch(self, mask_pred: torch.Tensor, mask_true: torch.Tensor, info: dict) -> list[np.ndarray]:
        h, w = mask_pred.shape[-2:]
        xs, ys = torch.arange(w).float(), torch.arange(h).float()
        com_dists = com_distances(mask_pred, mask_true, xs, ys, epsilon=1e-6, normalize=True).numpy()
        mse_vals = mses(mask_pred, mask_true).numpy()
        pred_np, true_np = mask_pred.numpy(), mask_true.numpy()
        return [self._render(pred_np, true_np, info, mse_vals, com_dists, i) for i in range(len(pred_np))]

    def upload(self, path: str, data_name: str, logger: Logger):
        """Load a .pt file OutputSaver just wrote and log its first MAX_WANDB_IMAGES samples as images."""
        outputs = torch.load(path, map_location='cpu', weights_only=False)
        mask_pred, mask_true, info = outputs['mask_pred'][:MAX_WANDB_IMAGES], outputs['mask_true'][:MAX_WANDB_IMAGES], outputs['info']
        info = {k: v[:MAX_WANDB_IMAGES] for k, v in info.items()}
        logger.log_images(self._render_batch(mask_pred, mask_true, info), name=f'SMask/{data_name}', channels_last=True, use_table=False)

# ***** OutputSaver *****

def _to_cpu(x:torch.Tensor): return x.detach().to('cpu', copy=True)

class OutputSaver(Callback):
    def __init__(self, save_interval, folder, filename='ep{epoch:04d}-ba{batch:06d}.pt', overwrite:bool=False, visualizer: VizSegMask | None = None):
        self.save_interval, self.folder, self.filename, self.overwrite, self.visualizer = Time.from_input(save_interval, TimeUnit.EPOCH), folder, filename, overwrite, visualizer
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
        epoch = state.timestamp.get(self.save_interval.unit).value
        if self.force_save or epoch % self.save_interval.value == 0:
            outputs = dict(mask_pred=_to_cpu(state.outputs['mask_pred']), mask_logits=_to_cpu(state.outputs['mask_logits']),
                           mask_true=_to_cpu(state.batch['mask_true']), info=state.batch['info'], fft=_to_cpu(state.batch['fft']))

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
                if isinstance(destination, WandBLogger): continue # wandb JSON serialization fails on complex tensor fft
                destination.log_metrics({f'{data_name}/{k}': v for k, v in outputs.items()})

            # upload to wandb viz only after the file is safely on disk, reading back from that same file
            if self.visualizer is not None: self.visualizer.upload(path, data_name, logger)

    def after_forward(self, state, logger): self.save_outputs(state, logger, state.dataloader_label, state.timestamp.batch.value)
    def eval_after_forward(self, state, logger): self.save_outputs(state, logger, f'{state.dataloader_label or "eval"}', state.eval_timestamp.batch.value)
