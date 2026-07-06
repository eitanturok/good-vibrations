import os

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from composer.core import State, Time, TimeUnit
from composer import Callback, Logger
from composer.utils import format_name_with_dist
from composer.loggers import WandBLogger

from model import com_distances

# ***** MaskVizualizer *****

class MaskVisualizer(Callback):
    def __init__(self, num_images, train_interval):
        self.num_images = num_images
        self.train_interval = Time.from_input(train_interval, TimeUnit.EPOCH)
        self.last_train_time_value_logged = -1
        self.last_eval_step_logged = {}

    def _log_image(self, state: State, logger: Logger, data_name: str, scale: int=8, text_height: int=40, sep: int=4):
        mask_pred, mask_true, info = state.outputs['mask_pred'], state.batch['mask_true'], state.batch['info']
        pred_np, true_np = mask_pred.detach().cpu().numpy(), mask_true.detach().cpu().numpy()
        font = ImageFont.load_default(size=14)
        h, w = pred_np.shape[-2:]
        xs, ys = torch.arange(w, device=mask_pred.device).float(), torch.arange(h, device=mask_pred.device).float()
        com_dists = com_distances(mask_pred, mask_true, xs, ys, epsilon=1e-6, normalize=True)
        def _render(i):
            h, w = pred_np[i].shape
            ph, pw = h * scale, w * scale  # panel size after upscale
            canvas = Image.new("RGB", (pw * 2 + sep, ph + text_height), (255, 255, 255))
            for j, (arr, label) in enumerate([(pred_np[i], "Predicted"), (true_np[i], "Ground Truth")]):
                panel = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).resize((pw, ph), Image.NEAREST)
                canvas.paste(panel, (j * (pw + sep), text_height))
                ImageDraw.Draw(canvas).text((j * (pw + sep) + pw // 2, text_height - 14), label, fill=(0, 0, 0), font=font, anchor="mt")
            mse = float(np.mean((pred_np[i] - true_np[i]) ** 2))
            text = (f"id={info['sample_id'][i]}  spk={info['speaker'][i]}  objs={info['n_objects'][i]}  "
                    f"com=({info['x_com'][i]:.1f},{info['y_com'][i]:.1f})  mse={mse:.4f}  com_dist={com_dists[i]:.4f}")
            ImageDraw.Draw(canvas).text((pw + sep // 2, 2), text, fill=(80, 80, 80), font=font, anchor="mt")
            return np.array(canvas)
        imgs = [_render(i) for i in range(len(pred_np))]
        logger.log_images(imgs, name=data_name, channels_last=True, use_table=False)

    def before_loss(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.train_interval.unit).value
        if current_time_value % self.train_interval.value == 0 and current_time_value != self.last_train_time_value_logged:
            self.last_train_time_value_logged = current_time_value
            self._log_image(state, logger, 'Images/train')

    def eval_after_forward(self, state: State, logger: Logger):
        eval_batch = state.eval_timestamp.get(TimeUnit.BATCH).value
        train_step = state.timestamp.batch.value
        evaluator_label = state.dataloader_label or 'eval'
        if eval_batch == 0 and train_step != self.last_eval_step_logged.get(evaluator_label):
            self.last_eval_step_logged[evaluator_label] = train_step
            self._log_image(state, logger, f'Images/{evaluator_label}')

# ***** OutputSaver *****

def _to_cpu(x:torch.Tensor): return x.detach().to('cpu', copy=True)

class OutputSaver(Callback):
    def __init__(self, save_interval, folder, filename='ep{epoch:04d}-ba{batch:06d}.pt', overwrite:bool=False):
        self.save_interval, self.folder, self.filename, self.overwrite = Time.from_input(save_interval, TimeUnit.EPOCH), folder, filename, overwrite

    def init(self, state: State, logger: Logger):
        del logger
        self.folder = format_name_with_dist(self.folder, state.run_name)
        os.makedirs(self.folder, exist_ok=True)

    def save_outputs(self, state: State, logger: Logger, data_name: str, timestamp):
        current_time_value = timestamp.get(self.save_interval.unit).value
        if current_time_value % self.save_interval.value == 0:
            outputs = {'fft': _to_cpu(state.batch['fft']), 'mask_pred': _to_cpu(state.outputs['mask_pred']), 'mask_true': _to_cpu(state.batch['mask_true']), 'info': state.batch['info']}

            # save to loggers
            for destination in logger.destinations:
                if isinstance(destination, WandBLogger): continue # wandb JSON serialization fails on complex tensor fft
                destination.log_metrics({f'{data_name}/{k}': v for k, v in outputs.items()})

            # save locally
            path = os.path.join(self.folder, data_name, self.filename.format(epoch=timestamp.epoch.value, batch=timestamp.batch.value))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path):
                if not self.overwrite:
                    raise FileExistsError(f'OutputSaver: file already exists: {path}\nSet overwrite=True to overwrite.')
                os.remove(path)
            torch.save(outputs, path)

    def after_forward(self, state, logger): self.save_outputs(state, logger, state.dataloader_label, state.timestamp)
    def eval_after_forward(self, state, logger): self.save_outputs(state, logger, f'{state.dataloader_label or "eval"}', state.eval_timestamp)
