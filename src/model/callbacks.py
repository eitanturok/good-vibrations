import os

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from composer.core import State, Time, TimeUnit
from composer import Callback, Logger
from composer.utils import format_name_with_dist
from composer.loggers import WandBLogger

from model.arch import com_distances, mses
from model.attribution import capture_attention, ablate_lasers, ablate_freq_patches


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

    @staticmethod
    def _fit_font(text: str, max_width: int, max_size: int, min_size: int = 6) -> ImageFont.ImageFont:
        """Largest default font that fits `text` within `max_width` px."""
        fonts = (ImageFont.load_default(size=s) for s in range(max_size, min_size, -1))
        return next((f for f in fonts if f.getlength(text) <= max_width), ImageFont.load_default(size=min_size))

    def _render(self, pred: np.ndarray, true: np.ndarray, caption: str, target_w: int = 640, sep: int = 4) -> np.ndarray:
        """Two mask panels side by side under a caption, sized to ~target_w at any out_h/out_w."""
        h, w = pred.shape
        scale = max(1, (target_w - sep) // (2 * w))
        ph, pw = h * scale, w * scale
        canvas_w = pw * 2 + sep

        caption_font = self._fit_font(caption, canvas_w - 2 * sep, max_size=max(7, canvas_w // 24))
        label_font = self._fit_font("Ground Truth", pw - sep, max_size=max(7, canvas_w // 24))
        pad = max(2, caption_font.size // 3)
        caption_y, label_y = pad, pad + caption_font.size + pad
        top = label_y + label_font.size + pad

        canvas = Image.new("RGB", (canvas_w, top + ph), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((canvas_w // 2, caption_y), caption, fill=(80, 80, 80), font=caption_font, anchor="mt")
        for x, arr, label in [(0, pred, "Predicted"), (pw + sep, true, "Ground Truth")]:
            canvas.paste(Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8)).resize((pw, ph), Image.NEAREST), (x, top))
            draw.text((x + pw // 2, label_y), label, fill=(0, 0, 0), font=label_font, anchor="mt")
        return np.array(canvas)

    def _render_batch(self, mask_pred: torch.Tensor, mask_true: torch.Tensor, info: dict) -> list[np.ndarray]:
        com_dists = com_distances(mask_pred, mask_true, epsilon=1e-6, normalize=True).numpy()
        mse_vals = mses(mask_pred, mask_true).numpy()
        captions = [f"pos {info['position_id'][i]}  spk {info['speaker'][i]} (smp {info['sample_id'][i]})  objs={info['n_objects'][i]}  "
                    f"com=({info['x_com'][i]:.1f},{info['y_com'][i]:.1f})  mse={mse_vals[i]:.4f}  com_dist={com_dists[i]:.4f}"
                    for i in range(len(mask_pred))]
        return [self._render(p, t, c) for p, t, c in zip(mask_pred.numpy(), mask_true.numpy(), captions)]

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

# ***** AttributionSaver *****


class AttributionSaver(Callback):
    """Save which lasers / frequency bands the model relies on, per eval batch.

    Two measures (see model/attribution.py for the full argument):
    - cls-row attention over lasers and over freq patches, free with the forward pass;
    - optionally, delta-MSE from zeroing each laser / freq patch, which costs
      n_lasers + n_patches extra forwards and is the metric to actually trust.

    Eval only, deliberately. At train time the 0.3 structured dropout is active, so
    key_padding_mask is non-None and attention rows over dropped tokens are meaningless.
    """

    def __init__(self, save_interval, folder, filename='ep{epoch:04d}-ba{batch:06d}.pt', overwrite: bool = False, ablate: bool = False, max_ablate_batches: int = 1):
        self.save_interval, self.folder, self.filename, self.overwrite = Time.from_input(save_interval, TimeUnit.EPOCH), folder, filename, overwrite
        self.ablate, self.max_ablate_batches = ablate, max_ablate_batches
        self._attn_cm = self._attn = None
        self._n_ablated = 0
        # see OutputSaver.force_save
        self.force_save = False

    def init(self, state: State, logger: Logger):
        del logger
        self.folder = format_name_with_dist(self.folder, state.run_name)
        os.makedirs(self.folder, exist_ok=True)
        # hooks must be attached here, not in __init__: callbacks are constructed before the
        # Trainer compiles the model, so __init__ would hook a module the compiled wrapper skips.
        # (Forward hooks do survive torch.compile -- dynamo just graph-breaks around them.)
        self._attn_cm = capture_attention(self._module(state))
        self._attn = self._attn_cm.__enter__()

    def close(self, state: State, logger: Logger):
        del state, logger
        if self._attn_cm is not None:
            self._attn_cm.__exit__(None, None, None)
            self._attn_cm = self._attn = None

    @staticmethod
    def _module(state: State):
        """The raw VibrationTransformer, unwrapping Composer's/compile's wrappers."""
        m = state.model
        for attr in ('_orig_mod', 'module'):
            while hasattr(m, attr): m = getattr(m, attr)
        return m

    def save_attribution(self, state: State, logger: Logger, data_name: str, batch: int):
        if not _is_due(self.save_interval, state, self.force_save) or not self._attn: return

        n_lasers = state.batch['fft'].shape[1]
        out = {}
        # reduce on GPU before the copy: the un-reduced freq map is ~2.4M floats per sample.
        # do this *before* any ablation, whose extra forwards overwrite the captured maps.
        for axis, w in self._attn.items():
            w = torch.stack([w[i] for i in sorted(w)], dim=1)  # (B_,n_layers,H,S-1)
            if axis == 'freq': w = w.reshape(-1, n_lasers, *w.shape[1:]).mean(dim=1)  # mean over lasers
            out[f'attn_{axis}'] = _to_cpu(w.mean(dim=0))  # (n_layers,H,S-1), keep layer/head split

        if self.ablate and self._n_ablated < self.max_ablate_batches:
            model, X, y = self._module(state), state.batch['fft'], state.batch['mask_true']
            out['ablate_laser'] = torch.from_numpy(ablate_lasers(model, X, y))
            out['ablate_freq'] = torch.from_numpy(ablate_freq_patches(model, X, y))
            self._n_ablated += 1

        path = os.path.join(self.folder, data_name, self.filename.format(epoch=state.timestamp.epoch.value, batch=batch))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            if not self.overwrite: raise FileExistsError(f'AttributionSaver: file already exists: {path}\nSet overwrite=True to overwrite.')
            os.remove(path)
        torch.save(out, path)

        # scalars are wandb-safe (unlike the raw maps), so log the headline rankings there
        scalars = {}
        for axis in ('laser', 'freq'):
            if (a := out.get(f'attn_{axis}')) is not None:
                for i, v in enumerate(a.mean(dim=(0, 1)).tolist()): scalars[f'{data_name}/attn_{axis}/{i:03d}'] = v
            if (b := out.get(f'ablate_{axis}')) is not None:
                for i, v in enumerate(b.tolist()): scalars[f'{data_name}/ablate_{axis}/{i:03d}'] = v
        if scalars: logger.log_metrics(scalars)

    def eval_start(self, state, logger):
        del state, logger
        self._n_ablated = 0

    def eval_after_forward(self, state, logger):
        self.save_attribution(state, logger, state.dataloader_label or "eval", state.eval_timestamp.batch.value)
