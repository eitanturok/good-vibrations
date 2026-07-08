import os

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from composer.core import State, Time, TimeUnit
from composer import Callback, Logger
from composer.utils import format_name_with_dist
from composer.loggers import WandBLogger

from model import com_distances, mses

# ***** MaskVizualizer *****

class MaskVisualizer(Callback):
    def __init__(self, log_interval):
        self.log_interval = Time.from_input(log_interval, TimeUnit.EPOCH)

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

    def _render_batch(self, state: State) -> list[np.ndarray]:
        mask_pred, mask_true, info = state.outputs['mask_pred'], state.batch['mask_true'], state.batch['info']
        h, w = mask_pred.shape[-2:]
        xs, ys = torch.arange(w, device=mask_pred.device).float(), torch.arange(h, device=mask_pred.device).float()
        com_dists = com_distances(mask_pred, mask_true, xs, ys, epsilon=1e-6, normalize=True).detach().cpu().numpy()
        mse_vals = mses(mask_pred, mask_true).detach().cpu().numpy()
        pred_np, true_np = mask_pred.detach().cpu().numpy(), mask_true.detach().cpu().numpy()
        return [self._render(pred_np, true_np, info, mse_vals, com_dists, i) for i in range(len(pred_np))]

    def _due(self, state: State) -> bool:
        current_time_value = state.timestamp.get(self.log_interval.unit).value
        return current_time_value % self.log_interval.value == 0

    def after_forward(self, state: State, logger: Logger):
        # NOTE: wandb.Image caps any single log_images call at MAX_ITEMS=108
        # so if BS>108, we will only log the first 108 images of the batch
        if self._due(state):
            logger.log_images(self._render_batch(state), name='SMask/train', channels_last=True, use_table=False)

    def eval_after_forward(self, state: State, logger: Logger):
        if self._due(state):
            logger.log_images(self._render_batch(state), name=f'SMask/{state.dataloader_label}', channels_last=True, use_table=False)

# ***** OutputSaver *****

def _to_cpu(x:torch.Tensor): return x.detach().to('cpu', copy=True)

class OutputSaver(Callback):
    def __init__(self, save_interval, folder, filename='ep{epoch:04d}-ba{batch:06d}.pt', overwrite:bool=False, save_fft:bool=True):
        self.save_interval, self.folder, self.filename, self.overwrite, self.save_fft = Time.from_input(save_interval, TimeUnit.EPOCH), folder, filename, overwrite, save_fft

    def init(self, state: State, logger: Logger):
        del logger
        self.folder = format_name_with_dist(self.folder, state.run_name)
        os.makedirs(self.folder, exist_ok=True)

    def save_outputs(self, state: State, logger: Logger, data_name: str, batch: int):
        # due-check and filename epoch both use the *trainer* timestamp, so eval dumps taken
        # during training land on the real epoch (ep0050, ep0100, ...) instead of ep0000
        epoch = state.timestamp.get(self.save_interval.unit).value
        if epoch % self.save_interval.value == 0:
            outputs = {'mask_pred': _to_cpu(state.outputs['mask_pred']), 'mask_true': _to_cpu(state.batch['mask_true']), 'info': state.batch['info']}
            if self.save_fft: outputs['fft'] = _to_cpu(state.batch['fft'])

            # save to loggers
            for destination in logger.destinations:
                if isinstance(destination, WandBLogger): continue # wandb JSON serialization fails on complex tensor fft
                destination.log_metrics({f'{data_name}/{k}': v for k, v in outputs.items()})

            # save locally
            path = os.path.join(self.folder, data_name, self.filename.format(epoch=state.timestamp.epoch.value, batch=batch))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path):
                if not self.overwrite:
                    raise FileExistsError(f'OutputSaver: file already exists: {path}\nSet overwrite=True to overwrite.')
                os.remove(path)
            torch.save(outputs, path)

    def after_forward(self, state, logger): self.save_outputs(state, logger, state.dataloader_label, state.timestamp.batch.value)
    def eval_after_forward(self, state, logger): self.save_outputs(state, logger, f'{state.dataloader_label or "eval"}', state.eval_timestamp.batch.value)
