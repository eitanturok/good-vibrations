import os, time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from composer import Callback
from composer.callbacks import CheckpointSaver
from composer.core import Event, State
from composer.loggers import Logger
from huggingface_hub import HfApi

def getenv(key:str, default:Any=0): return type(default)(os.getenv(key, default))


def load_from_hf(run_name: str | None = None, repo_id: str = "eturok-weizmann/good-vibrations", **model_kwargs):
    """Download model from HF and load into SignalTransformer. If run_name is None, uses most recent model.pt."""
    from vibration_transformer import SignalTransformer
    api = HfApi()
    if run_name is None:
        files = [f for f in api.list_repo_tree(repo_id, recursive=True) if f.path.endswith("model.pt")]
        run_name = max(files, key=lambda f: f.last_commit.date).path.rsplit("/", 1)[0]
    path = api.hf_hub_download(repo_id=repo_id, filename=f"{run_name}/model.pt")
    model = SignalTransformer(**model_kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()

def fetch_wandb_history(run_id: str, keys: list[str] | None = None,
                        entity: str = "eturok", project: str = "good-vibrations") -> list[dict]:
    """Return logged history rows for a run, sorted by step.

    Args:
        run_id: W&B run ID (e.g. 's3pqt79j') or run name.
        keys: Which keys to fetch. None fetches all keys (slower).
        entity: W&B entity. Check the WandBLogger call in model.py for the current value.
        project: W&B project. Check the WandBLogger call in model.py for the current value.

    Returns:
        List of dicts sorted by '_step', one per logged step.

    Example:
        rows = fetch_wandb_history('s3pqt79j', keys=['_step', 'loss/train/total', 'metrics/eval/mask/iou'])
        for r in rows:
            print(r['_step'], r.get('loss/train/total'), r.get('metrics/eval/mask/iou'))
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    rows = list(run.scan_history(keys=keys))
    rows.sort(key=lambda r: r.get('_step', 0))
    return rows


def fetch_wandb_images(run_id: str, split: str = "eval", epoch: int | None = None,
                       key: str = "prob", download_dir: str = ".",
                       entity: str = "eturok", project: str = "good-vibrations") -> list[Path]:
    """Download mask_viz images from a W&B run and return their local paths.

    Images are logged under mask_viz/{split}/{key} (e.g. mask_viz/eval/prob).
    Each image is a side-by-side comparison: True Mask (gray) | Pred Mask (hot colormap).

    Args:
        run_id: W&B run ID or name.
        split: 'eval' or 'train'.
        epoch: Which epoch to fetch (1-indexed). None fetches the latest.
        key: Image sub-key — 'prob' for the continuous prediction, or 'thresh{t}' for a binarized version.
        download_dir: Local directory to save images into.
        entity: W&B entity. Check the WandBLogger call in model.py for the current value.
        project: W&B project. Check the WandBLogger call in model.py for the current value.

    Returns:
        List of local Paths to the downloaded PNG files.

    Example:
        paths = fetch_wandb_images('s3pqt79j', split='eval', epoch=10)
        # Read them with PIL or pass to Claude's Read tool to view visually.
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    wandb_key = f"mask_viz/{split}/{key}"
    rows = [r for r in run.scan_history(keys=['_step', wandb_key]) if wandb_key in r]
    rows.sort(key=lambda r: r['_step'])
    if not rows:
        raise ValueError(f"No images found for key '{wandb_key}' in run {run_id}")
    row = rows[(epoch - 1) if epoch is not None else -1]
    img_data = row[wandb_key]
    filenames = img_data['filenames'] if isinstance(img_data, dict) else [img_data['path']]
    paths = []
    for fname in filenames:
        run.file(fname).download(root=download_dir, replace=True)
        paths.append(Path(download_dir) / fname)
    return paths


class BestMetricCheckpointSaver(CheckpointSaver):
    """Saves weights-only checkpoints only when a metric hits a new best.

    Wraps Composer's CheckpointSaver (with weights_only=True) and gates
    each save on whether the target eval metric has improved since the
    last save.

    Args:
        metric_name: Key into state.eval_metrics['eval'], e.g. 'x/MulticlassAccuracy'.
        higher_is_better: True if larger metric values are better (e.g. accuracy).
                          False for losses/errors. Default: False.
        **kwargs: Forwarded to CheckpointSaver (folder, save_interval, etc.).
                  weights_only is always set to True.
    """

    def __init__(self, metric_name: str, higher_is_better: bool = False, **kwargs):
        kwargs['weights_only'] = True
        super().__init__(**kwargs)
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.best: Optional[float] = None

    def _get_metric_value(self, state: State) -> Optional[float]:
        m = state.eval_metrics.get('eval', {}).get(self.metric_name)
        if m is None:
            return None
        return m.compute().item() if hasattr(m, 'compute') else float(m)

    def _is_improved(self, val: float) -> bool:
        if self.best is None or (val != val):  # first eval or NaN best
            return True
        return val > self.best if self.higher_is_better else val < self.best

    def epoch_checkpoint(self, state: State, logger: Logger):
        val = self._get_metric_value(state)
        if val is not None and self._is_improved(val):
            self.best = val
            super().epoch_checkpoint(state, logger)


class MaskVisualizationCallback(Callback):
    def __init__(self, n_samples=4, save_dir="visualizations", train_viz_interval=10, thresholds=[]):
        self.n_samples, self.save_dir, self.train_viz_interval, self.thresholds = n_samples, save_dir, train_viz_interval, list(thresholds)
        self._last_eval_batch = self._last_train_batch = None

    def epoch_start(self, state, logger): self._last_train_batch = None
    def batch_end(self, state, logger):
        if self._last_train_batch is None: self._last_train_batch = (state.batch, state.outputs)
    def epoch_end(self, state, logger):
        if state.timestamp.epoch.value % self.train_viz_interval == 0 and self._last_train_batch is not None:
            self._visualize(*self._last_train_batch, state, "train")

    def eval_batch_end(self, state, logger):
        if self._last_eval_batch is None: self._last_eval_batch = (state.batch, state.outputs)
    def eval_end(self, state, logger):
        if self._last_eval_batch is None: return
        self._visualize(*self._last_eval_batch, state, "eval")
        self._last_eval_batch = None

    def _visualize(self, batch, outputs, state, split):
        _, true_masks, _, _ = batch
        _, _, _, mask_logits = outputs
        n = min(self.n_samples, true_masks.shape[0], mask_logits.shape[0])
        probs = mask_logits[:n].sigmoid().detach().cpu().float().numpy()
        true = true_masks[:n].detach().cpu().float().numpy()
        epoch = state.timestamp.epoch.value
        os.makedirs(self.save_dir, exist_ok=True)
        if not wandb.run: return
        log = {}
        for i in range(n):
            # continuous prob map (no threshold)
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))  
            ax0.imshow(true[i], vmin=0, vmax=1, cmap='gray'); ax0.set_title('True Mask'); ax0.axis('off')
            ax1.imshow(probs[i], vmin=0, vmax=1, cmap='hot'); ax1.set_title('Pred Mask (prob)'); ax1.axis('off')
            fig.suptitle(f'Epoch {epoch}, {split.capitalize()} Sample {i}, Prob'); fig.tight_layout()
            log.setdefault(f'mask_viz/{split}/prob', []).append(wandb.Image(fig, caption=f'sample {i}')); plt.close(fig)
            # binarized at each threshold
            for t in self.thresholds:
                fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
                ax0.imshow(true[i], vmin=0, vmax=1, cmap='gray'); ax0.set_title('true'); ax0.axis('off')
                ax1.imshow((probs[i] > t).astype(float), vmin=0, vmax=1, cmap='gray'); ax1.set_title(f'Pred Mask (threshold {t})'); ax1.axis('off')
                fig.suptitle(f'Epoch {epoch}, {split.capitalize()} Sample {i}, threshold {t}'); fig.tight_layout()
                log.setdefault(f'mask_viz/{split}/thresh{t}', []).append(wandb.Image(fig, caption=f'sample {i}')); plt.close(fig)
        wandb.log(log, step=state.timestamp.batch.value)


