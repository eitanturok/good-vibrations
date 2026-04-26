import os, shlex, subprocess, time, math
from pathlib import Path
from typing import Any, Callable, Optional, Union

import psutil
import torch
import wandb
from composer import Callback
from composer.core import Event, State
from composer.loggers import Logger
from huggingface_hub import HfApi


SAMPLE_FILENAME_WIDTH = 6
CLUSTER_HOST = "ethantu@mcluster11.wisdom.weizmann.ac.il"
CLUSTER_REPO_ROOT = "mark_sheinin_lab/code/eitan/good-vibrations"


def getenv(key: str, default: Any = 0):
    return type(default)(os.getenv(key, default))


def sample_npz_path(sample_idx: int) -> str:
    return f"data/sample_{int(sample_idx):0{SAMPLE_FILENAME_WIDTH}d}.npz"


def hf_uri(repo_id: str, path_in_repo: str = "") -> str:
    repo_id = repo_id.removeprefix("hf://").rstrip("/")
    path_in_repo = path_in_repo.lstrip("/")
    return f"hf://{repo_id}/{path_in_repo}" if path_in_repo else f"hf://{repo_id}"


def run_root_path(run_id: str) -> str:
    return f"runs/{run_id}"


def run_predictions_dir(run_id: str) -> str:
    return f"{run_root_path(run_id)}/predictions"


def run_checkpoints_dir(run_id: str) -> str:
    return f"{run_root_path(run_id)}/checkpoints"


def checkpoint_pattern_path(run_id: str) -> str:
    return f"{run_checkpoints_dir(run_id)}/ep{{epoch:07d}}_ba{{batch:010d}}.pt"


def latest_checkpoint_path(run_id: str, rank: int = 0) -> str:
    return f"{run_checkpoints_dir(run_id)}/latest-rank{rank}.pt"


def best_checkpoint_path(run_id: str, filename: str = "best.pt") -> str:
    return f"{run_root_path(run_id)}/{filename}"


def run_visualizations_dir(run_id: str) -> str:
    return f"{run_root_path(run_id)}/visualizations"


def load_from_hf(
    run_name: str | None = None,
    repo_id: str = "eturok-weizmann/good-vibrations",
    **model_kwargs,
):
    """Download model from HF and load into SignalTransformer. If run_name is None, uses most recent model.pt."""
    from vibration_transformer import SignalTransformer

    api = HfApi()
    if run_name is None:
        files = [
            f
            for f in api.list_repo_tree(repo_id, recursive=True)
            if f.path.endswith("model.pt")
        ]
        run_name = max(files, key=lambda f: f.last_commit.date).path.rsplit("/", 1)[0]
    path = api.hf_hub_download(repo_id=repo_id, filename=f"{run_name}/model.pt")
    model = SignalTransformer(**model_kwargs)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def prediction_npz_path(run_id: str, split: str, epoch: int, batch: int) -> str:
    return (
        f"{run_predictions_dir(run_id)}/"
        f"{split}_ep{int(epoch):07d}_ba{int(batch):010d}.npz"
    )


def visualization_image_path(
    run_id: str,
    split: str,
    kind: str,
    epoch: int,
    batch: int,
    sample_idx: int,
) -> str:
    safe_kind = str(kind).replace(".", "p")
    return (
        f"{run_visualizations_dir(run_id)}/{split}/{safe_kind}/"
        f"ep{int(epoch):07d}_ba{int(batch):010d}_sample{int(sample_idx):06d}.png"
    )


def fetch_wandb_history(
    run_id: str,
    keys: list[str] | None = None,
    entity: str = "eturok",
    project: str = "good-vibrations",
) -> list[dict]:
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
    rows.sort(key=lambda r: r.get("_step", 0))
    return rows


def fetch_wandb_images(
    run_id: str,
    split: str = "eval",
    epoch: int | None = None,
    key: str = "prob",
    download_dir: str = ".",
    entity: str = "eturok",
    project: str = "good-vibrations",
) -> list[Path]:
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
    rows = [r for r in run.scan_history(keys=["_step", wandb_key]) if wandb_key in r]
    rows.sort(key=lambda r: r["_step"])
    if not rows:
        raise ValueError(f"No images found for key '{wandb_key}' in run {run_id}")
    row = rows[(epoch - 1) if epoch is not None else -1]
    img_data = row[wandb_key]
    filenames = (
        img_data["filenames"] if isinstance(img_data, dict) else [img_data["path"]]
    )
    paths = []
    for fname in filenames:
        run.file(fname).download(root=download_dir, replace=True)
        paths.append(Path(download_dir) / fname)
    return paths


class BestMetricCheckpointSaver(Callback):
    """Saves a weights-only checkpoint whenever a metric hits a new best.

    Does NOT inherit from CheckpointSaver — intentionally a plain Callback so
    Composer's Trainer does not mistake it for the primary checkpoint saver and
    override save_folder / save_latest_filename for autoresume.

    Args:
        metric_name: Key into state.eval_metrics['eval'], e.g. 'mse'.
        save_path: Full path where the best checkpoint file is written.
        higher_is_better: True if larger metric values are better. Default: False.
    """

    def __init__(self, metric_name: str, save_path: str, higher_is_better: bool = False):
        self.metric_name = metric_name
        self.save_path = save_path
        self.higher_is_better = higher_is_better
        self.best: Optional[float] = None

    def _get_metric_value(self, state: State) -> Optional[float]:
        m = state.eval_metrics.get("eval", {}).get(self.metric_name)
        if m is None:
            return None
        val = m.compute().item() if hasattr(m, "compute") else float(m)
        return val if math.isfinite(val) else None

    def _is_improved(self, val: float) -> bool:
        if self.best is None or not math.isfinite(self.best):
            return True
        return val > self.best if self.higher_is_better else val < self.best

    def epoch_checkpoint(self, state: State, logger: Logger):
        val = self._get_metric_value(state)
        if val is None or not self._is_improved(val):
            return
        self.best = val
        path = Path(self.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        baseline = torch.cuda.memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
        torch.save(state.model.state_dict(), str(path))
        spike = torch.cuda.max_memory_allocated() / 1e9 - baseline
        print(f"[BestMetricCheckpointSaver] new best {self.metric_name}={val:.4f}, saved to {path} (memory spike: {spike:.2f} GB)")
        if wandb.run:
            wandb.log(
                {"memory/gpu/checkpoint_spike_gb": spike},
                step=state.timestamp.batch.value,
            )


class HFUploaderCallback(Callback):
    def __init__(self, repo_id: str, run_id: str):
        self.repo_id = repo_id.removeprefix("hf://")
        self.run_id = run_id
        self.local_run_dir = Path(run_root_path(run_id))
        self.local_predictions_dir = Path(run_predictions_dir(run_id))
        self.local_visualizations_dir = Path(run_visualizations_dir(run_id))
        self.local_best_path = Path(best_checkpoint_path(run_id))
        self.local_latest_path = Path(latest_checkpoint_path(run_id))
        self._uploaded = False

    def fit_end(self, state, logger):
        del state, logger
        if self._uploaded or not self.local_run_dir.exists():
            return
        api = HfApi()
        remote_root = run_root_path(self.run_id)
        if self.local_predictions_dir.exists():
            print(f"[HFUploaderCallback] uploading predictions for {self.run_id}")
            api.upload_folder(
                folder_path=str(self.local_predictions_dir),
                path_in_repo=run_predictions_dir(self.run_id),
                repo_id=self.repo_id,
                repo_type="dataset",
            )
        if self.local_visualizations_dir.exists():
            print(f"[HFUploaderCallback] uploading visualizations for {self.run_id}")
            api.upload_folder(
                folder_path=str(self.local_visualizations_dir),
                path_in_repo=run_visualizations_dir(self.run_id),
                repo_id=self.repo_id,
                repo_type="dataset",
            )
        if self.local_best_path.exists():
            print(f"[HFUploaderCallback] uploading best checkpoint for {self.run_id}")
            api.upload_file(
                path_or_fileobj=str(self.local_best_path),
                path_in_repo=str(Path(remote_root) / self.local_best_path.name),
                repo_id=self.repo_id,
                repo_type="dataset",
            )
        if self.local_latest_path.exists():
            print(f"[HFUploaderCallback] uploading latest checkpoint for {self.run_id}")
            api.upload_file(
                path_or_fileobj=str(self.local_latest_path),
                path_in_repo=latest_checkpoint_path(self.run_id),
                repo_id=self.repo_id,
                repo_type="dataset",
            )
        self._uploaded = True


# class StepTimeCallback(Callback):
#     """Logs per-batch and cumulative train wall-clock time to the active logger."""

#     def batch_end(self, state: State, logger: Logger):
#         logger.log_metrics(
#             {
#                 "time/train_step_sec": state.timestamp.batch_wct.total_seconds(),
#                 "time/train_total_sec": state.timestamp.total_wct.total_seconds(),
#             }
#         )


# class MemoryCallback(Callback):
#     """Logs GPU/CPU memory every batch and streams a Plotly stacked area chart to wandb."""

#     def __init__(self, dataset_gb: float):
#         self.dataset_gb = dataset_gb
#         self._has_cuda = torch.cuda.is_available()
#         self._peak_forward = 0.0
#         self._peak_backward = 0.0
#         # accumulated history — each append is one batch
#         self._steps = []
#         self._weights = []
#         self._optimizer = []
#         self._gradients = []
#         self._other = []
#         self._peak_fwd = []
#         self._peak_bwd = []
#         self._gpu_total = []
#         self._cpu_other = []
#         self._ram_total = []

#     def before_train_batch(self, state, logger):
#         if self._has_cuda:
#             torch.cuda.reset_peak_memory_stats()

#     def after_forward(self, state, logger):
#         if self._has_cuda:
#             self._peak_forward = torch.cuda.max_memory_allocated() / 1e9

#     def after_backward(self, state, logger):
#         if self._has_cuda:
#             self._peak_backward = torch.cuda.max_memory_allocated() / 1e9

#     def batch_end(self, state, logger):
#         model = state.model
#         optimizer = state.optimizers[0]
#         weights_gb = sum(p.data.nbytes for p in model.parameters()) / 1e9
#         grads_gb = (
#             sum(p.grad.nbytes for p in model.parameters() if p.grad is not None) / 1e9
#         )
#         opt_gb = (
#             sum(
#                 v.nbytes
#                 for s in optimizer.state.values()
#                 for v in s.values()
#                 if isinstance(v, torch.Tensor)
#             )
#             / 1e9
#         )
#         allocated_gb = torch.cuda.memory_allocated() / 1e9 if self._has_cuda else 0.0
#         gpu_total_gb = (
#             torch.cuda.get_device_properties(0).total_memory / 1e9 if self._has_cuda else 0.0
#         )
#         act_gb = max(0.0, self._peak_forward - weights_gb) if self._has_cuda else 0.0
#         other_gpu = (
#             max(0.0, allocated_gb - weights_gb - opt_gb - grads_gb) if self._has_cuda else 0.0
#         )
#         vm = psutil.virtual_memory()
#         other_cpu = max(0.0, vm.used / 1e9 - self.dataset_gb)

#         logger.log_metrics(
#             {
#                 "memory/gpu/weights_gb": weights_gb,
#                 "memory/gpu/gradients_gb": grads_gb,
#                 "memory/gpu/optimizer_gb": opt_gb,
#                 "memory/gpu/activations_gb": act_gb,
#                 "memory/gpu/peak_forward_gb": self._peak_forward,
#                 "memory/gpu/peak_backward_gb": self._peak_backward,
#                 "memory/gpu/allocated_gb": allocated_gb,
#                 "memory/gpu/reserved_gb": torch.cuda.memory_reserved() / 1e9 if self._has_cuda else 0.0,
#                 "memory/gpu/total_gb": gpu_total_gb,
#                 "memory/cpu/dataset_gb": self.dataset_gb,
#                 "memory/cpu/ram_used_gb": vm.used / 1e9,
#                 "memory/cpu/ram_available_gb": vm.available / 1e9,
#                 "memory/cpu/ram_total_gb": vm.total / 1e9,
#             }
#         )

#         # accumulate and redraw chart
#         step = state.timestamp.batch.value
#         self._steps.append(step)
#         self._weights.append(weights_gb)
#         self._optimizer.append(opt_gb)
#         self._gradients.append(grads_gb)
#         self._other.append(other_gpu)
#         self._peak_fwd.append(self._peak_forward)
#         self._peak_bwd.append(self._peak_backward)
#         self._gpu_total.append(gpu_total_gb)
#         self._cpu_other.append(other_cpu)
#         self._ram_total.append(vm.total / 1e9)

#         if wandb.run:
#             wandb.log({"memory/breakdown": wandb.Plotly(self._build_fig())}, step=step)

#     def _build_fig(self):
#         import plotly.graph_objects as go
#         from plotly.subplots import make_subplots

#         fig = make_subplots(
#             rows=2,
#             cols=1,
#             shared_xaxes=True,
#             subplot_titles=("GPU Memory", "CPU (RAM) Memory"),
#             vertical_spacing=0.12,
#         )
#         s = self._steps

#         for name, values, color in [
#             ("Weights", self._weights, "#1565C0"),  # dark blue  – stable foundation
#             (
#                 "Optimizer",
#                 self._optimizer,
#                 "#2E7D32",
#             ),  # dark green – grows then stabilises
#             ("Gradients", self._gradients, "#E65100"),  # dark orange
#             ("Other", self._other, "#9E9E9E"),  # grey – misc CUDA allocs
#         ]:
#             fig.add_trace(
#                 go.Scatter(
#                     x=s,
#                     y=values,
#                     name=name,
#                     legendgroup="gpu",
#                     mode="lines",
#                     stackgroup="gpu",
#                     line=dict(width=0),
#                     fillcolor=color,
#                     hovertemplate=f"<b>{name}</b>: %{{y:.3f}} GB<extra></extra>",
#                 ),
#                 row=1,
#                 col=1,
#             )

#         for name, values, color, dash in [
#             ("Peak fwd (incl. activations)", self._peak_fwd, "#42A5F5", "dot"),
#             ("Peak bwd", self._peak_bwd, "#EF5350", "solid"),
#         ]:
#             fig.add_trace(
#                 go.Scatter(
#                     x=s,
#                     y=values,
#                     name=name,
#                     legendgroup="gpu",
#                     mode="lines",
#                     line=dict(color=color, width=1.5, dash=dash),
#                     hovertemplate=f"<b>{name}</b>: %{{y:.3f}} GB<extra></extra>",
#                 ),
#                 row=1,
#                 col=1,
#             )

#         fig.add_trace(
#             go.Scatter(
#                 x=s,
#                 y=self._gpu_total,
#                 name="GPU Capacity",
#                 legendgroup="gpu",
#                 mode="lines",
#                 line=dict(color="black", width=2, dash="dash"),
#                 hovertemplate="<b>GPU Capacity</b>: %{y:.1f} GB<extra></extra>",
#             ),
#             row=1,
#             col=1,
#         )

#         dataset_vals = [self.dataset_gb] * len(s)
#         for name, values, color in [
#             ("Dataset", dataset_vals, "#1565C0"),
#             ("Other", self._cpu_other, "#9E9E9E"),
#         ]:
#             fig.add_trace(
#                 go.Scatter(
#                     x=s,
#                     y=values,
#                     name=f"CPU: {name}",
#                     legendgroup="cpu",
#                     mode="lines",
#                     stackgroup="cpu",
#                     line=dict(width=0),
#                     fillcolor=color,
#                     hovertemplate=f"<b>{name}</b>: %{{y:.3f}} GB<extra></extra>",
#                 ),
#                 row=2,
#                 col=1,
#             )

#         fig.add_trace(
#             go.Scatter(
#                 x=s,
#                 y=self._ram_total,
#                 name="RAM Capacity",
#                 legendgroup="cpu",
#                 mode="lines",
#                 line=dict(color="black", width=2, dash="dash"),
#                 hovertemplate="<b>RAM Capacity</b>: %{y:.1f} GB<extra></extra>",
#             ),
#             row=2,
#             col=1,
#         )

#         fig.update_layout(
#             title="Memory Breakdown",
#             hovermode="x unified",
#             height=700,
#             legend=dict(tracegroupgap=20),
#         )
#         fig.update_yaxes(title_text="GB", rangemode="tozero", row=1, col=1)
#         fig.update_yaxes(title_text="GB", rangemode="tozero", row=2, col=1)
#         fig.update_xaxes(title_text="Step", row=2, col=1)
#         return fig


class MaskVisualizationCallback(Callback):
    def __init__(
        self,
        n_samples=4,
        interval=10,
        alpha=0.4,
        pred_save_dir=None,
        run_id=None,
    ):
        self.n_samples = n_samples
        self.interval = interval
        self.alpha = alpha
        self.pred_save_dir = pred_save_dir
        self.run_id = run_id
        self._train_preds = None
        self._eval_preds = []
        self._tables = {}

    def epoch_start(self, state, logger):
        del state, logger
        self._train_preds = None

    def batch_end(self, state, logger):
        del logger
        if self._should_collect_train_preds(state) and self._train_preds is None:
            self._train_preds = [self._batch_to_pred(state.batch, state.outputs)]

    def epoch_end(self, state, logger):
        if not self._train_preds:
            return
        self._log_images(self._train_preds, state, logger, "Train")
        self._save_predictions(self._train_preds, state, "train")

    def eval_batch_end(self, state, logger):
        del logger
        self._eval_preds.append(self._batch_to_pred(state.batch, state.outputs))

    def _should_collect_train_preds(self, state):
        current_train_epoch = state.timestamp.epoch.value + 1
        return current_train_epoch % self.interval == 0

    def eval_end(self, state, logger):
        if not self._eval_preds:
            return
        self._log_images(self._eval_preds, state, logger, "Eval")
        self._save_predictions(self._eval_preds, state, "eval")
        self._eval_preds = []

    def _batch_to_pred(self, batch, outputs):
        _, true_masks, _, _, meta = batch
        _, mask_logits = outputs
        return {
            "mask_true": true_masks.detach().cpu().float().numpy(),
            "mask_logits": mask_logits.detach().cpu().float().numpy(),
            "sample_idx": list(meta["sample_idx"]),
            "x_position": list(meta["x_position"]),
            "y_position": list(meta["y_position"]),
            "object": list(meta["object"]),
            "n_objects": list(meta["n_objects"]),
        }

    def _save_predictions(self, preds_list, state, split):
        if not self.pred_save_dir or not self.run_id:
            return
        import numpy as np

        mask_true = np.concatenate([p["mask_true"] for p in preds_list])
        mask_logits = np.concatenate([p["mask_logits"] for p in preds_list])
        mask_pred = 1.0 / (1.0 + np.exp(-mask_logits))
        sample_idx = sum([p["sample_idx"] for p in preds_list], [])
        x_position = sum([p["x_position"] for p in preds_list], [])
        y_position = sum([p["y_position"] for p in preds_list], [])
        object_type = sum([p["object"] for p in preds_list], [])
        n_objects = sum([p["n_objects"] for p in preds_list], [])

        epoch = state.timestamp.epoch.value
        batch = state.timestamp.batch.value
        local_path = Path(self.pred_save_dir) / Path(
            prediction_npz_path(self.run_id, split, epoch, batch)
        ).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            local_path,
            mask_true=mask_true,
            mask_logits=mask_logits,
            mask_pred=mask_pred,
            sample_idx=np.array(sample_idx),
            x_position=np.array(x_position, dtype=float),
            y_position=np.array(y_position, dtype=float),
            object_type=np.array(object_type),
            n_objects=np.array(n_objects, dtype=int),
        )

    def _log_images(self, preds_list, state, logger, split):
        import numpy as np

        true = np.concatenate([p["mask_true"] for p in preds_list])
        logits = np.concatenate([p["mask_logits"] for p in preds_list])
        probs = 1.0 / (1.0 + np.exp(-logits))
        sample_ids = sum([p["sample_idx"] for p in preds_list], [])
        n_total = min(len(sample_ids), true.shape[0], probs.shape[0])
        n = min(self.n_samples, n_total)
        true = true[:n]
        probs = probs[:n]
        sample_ids = sample_ids[:n]
        if not wandb.run:
            return
        table = self._get_table(split)
        epoch = int(state.timestamp.epoch.value)
        for i in range(n):
            table.add_data(
                wandb.Image(self._make_panel(true[i], probs[i])),
                int(sample_ids[i]),
                epoch,
                split,
            )
        wandb.log({f"Images/{split}": table}, step=state.timestamp.batch.value)

    def _get_table(self, split):
        table = self._tables.get(split)
        if table is None:
            table = wandb.Table(
                columns=["image", "sample_idx", "epoch", "split"],
                log_mode="MUTABLE",
            )
            self._tables[split] = table
        return table

    def _make_panel(self, true_mask, pred_mask):
        return self._stack_h(self._mask_to_rgb(true_mask), self._mask_to_rgb(pred_mask))

    def _mask_to_rgb(self, mask):
        import numpy as np

        gray = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=-1)

    def _stack_h(self, left, right, gap=8):
        import numpy as np

        h = max(left.shape[0], right.shape[0])
        left = self._pad_to_height(left, h)
        right = self._pad_to_height(right, h)
        spacer = np.full((h, gap, 3), 255, dtype=np.uint8)
        return np.concatenate([left, spacer, right], axis=1)

    def _pad_to_height(self, image, height):
        import numpy as np

        if image.shape[0] == height:
            return image
        pad = height - image.shape[0]
        top = pad // 2
        bottom = pad - top
        return np.pad(
            image,
            ((top, bottom), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )

def fetch_predictions(
    run_id,
    data_dir="eturok-weizmann/good-vibrations",
    cache_dir="~/.cache/good-vibrations",
    max_epochs=None,
):
    """Load saved prediction npz files for a run from local disk or HF Hub.

    Returns:
        {'train': {epoch: dict}, 'eval': {epoch: dict}}
        Each dict has keys: mask_true (N,H,W), mask_pred (N,H,W),
        sample_idx (N,), x_position (N,), y_position (N,),
        object_type (N,), n_objects (N,).

    Example:
        preds = fetch_predictions('my-run-20250101-120000')
        eval_ep10 = preds['eval'][10]
        print(eval_ep10['mask_pred'].shape)  # (N, 40, 20)
    """
    import numpy as np
    from pathlib import Path
    from huggingface_hub import HfApi, hf_hub_download

    def _local_prediction_dirs(base: Path) -> list[Path]:
        run_str = str(run_id)
        return [
            base / run_predictions_dir(run_str),
            base / run_str / "predictions",
            base / "predictions" if base.name == run_str else None,
        ]

    def _parse_prediction_file(path_like) -> tuple[str, int, int] | None:
        stem = Path(path_like).stem
        split, _, rest = stem.partition("_ep")
        if split not in {"train", "eval"}:
            return None
        epoch_str, _, batch_str = rest.partition("_ba")
        if not epoch_str or not batch_str:
            return None
        return split, int(epoch_str), int(batch_str)

    def _limit_files(paths: list[Path | str]) -> list[Path | str]:
        if max_epochs is None:
            return list(paths)
        keep_by_split: dict[str, set[int]] = {"train": set(), "eval": set()}
        parsed = []
        for path in paths:
            meta = _parse_prediction_file(path)
            if meta is None:
                continue
            parsed.append((path, *meta))
        for split in ("train", "eval"):
            epochs = sorted({epoch for _, s, epoch, _ in parsed if s == split})
            keep_by_split[split] = set(epochs[-int(max_epochs):])
        return [
            path
            for path, split, epoch, _ in parsed
            if epoch in keep_by_split[split]
        ]

    def _local_prediction_files() -> tuple[list[Path], str | None, float]:
        t_local = time.perf_counter()
        seen: set[Path] = set()
        candidates: list[Path] = []

        roots = [Path.cwd()]
        data_path = Path(str(data_dir)).expanduser()
        if data_path.exists():
            roots.append(data_path)

        for root in roots:
            for candidate in _local_prediction_dirs(root):
                if candidate is None:
                    continue
                candidate = candidate.resolve()
                if candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)
                if candidate.is_dir():
                    files = _limit_files(sorted(candidate.glob("*.npz")))
                    if files:
                        elapsed = time.perf_counter() - t_local
                        return files, str(candidate), elapsed

        elapsed = time.perf_counter() - t_local
        return [], None, elapsed

    def _fetch_cluster_prediction_files() -> tuple[list[Path], str | None, float]:
        t_cluster = time.perf_counter()
        local_root = cache_dir / "cluster-cache"
        local_run_root = local_root / run_root_path(str(run_id))
        local_predictions_dir = local_run_root / "predictions"
        local_run_root.mkdir(parents=True, exist_ok=True)

        remote_predictions_dir = f"{CLUSTER_REPO_ROOT}/{run_predictions_dir(str(run_id))}"
        check = subprocess.run(
            [
                "ssh",
                CLUSTER_HOST,
                f'test -d "{remote_predictions_dir}"',
            ],
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            return [], None, time.perf_counter() - t_cluster

        local_root_q = shlex.quote(str(local_root))
        remote_root_q = shlex.quote(CLUSTER_REPO_ROOT)
        remote_rel_q = shlex.quote(run_predictions_dir(str(run_id)))
        command = (
            f'mkdir -p {local_root_q} && '
            f'ssh {shlex.quote(CLUSTER_HOST)} '
            f'"tar -C {remote_root_q} -cf - {remote_rel_q}" '
            f'| tar -xf - -C {local_root_q}'
        )
        copy = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
        )
        if copy.returncode != 0:
            raise RuntimeError(
                f"Failed to copy cluster predictions for {run_id}: {copy.stderr.strip()}"
            )

        files = _limit_files(sorted(local_predictions_dir.glob("*.npz")))
        elapsed = time.perf_counter() - t_cluster
        return files, str(local_predictions_dir), elapsed

    cache_dir = Path(cache_dir).expanduser()
    files, local_dir, t_tree = _local_prediction_files()
    source_desc = None

    if files:
        source_desc = f"local={local_dir}"
        print(
            f"[fetch_predictions] run={run_id} source={source_desc} scan={t_tree:.1f}s files={len(files)}"
        )
    else:
        repo_id = str(data_dir).removeprefix("hf://")
        prefix = f"{run_predictions_dir(run_id)}/"
        t0 = time.perf_counter()
        try:
            api = HfApi()
            repo_files = sorted(
                f.path
                for f in api.list_repo_tree(
                    repo_id,
                    repo_type="dataset",
                    path_in_repo=prefix,
                    recursive=False,
                )
                if getattr(f, "path", "").endswith(".npz")
            )
            repo_files = _limit_files(repo_files)
            files = [
                Path(
                    hf_hub_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        filename=path,
                        cache_dir=str(cache_dir),
                    )
                )
                for path in repo_files
            ]
        except Exception as e:
            files = []
            print(f"[fetch_predictions] run={run_id} repo lookup failed: {e}")
        t_tree = time.perf_counter() - t0
        source_desc = f"repo={repo_id}"
        print(
            f"[fetch_predictions] run={run_id} source={source_desc} scan={t_tree:.1f}s files={len(files)}"
        )
        if not files:
            files, cluster_dir, t_cluster = _fetch_cluster_prediction_files()
            if files:
                t_tree = t_cluster
                source_desc = f"cluster={cluster_dir}"
                print(
                    f"[fetch_predictions] run={run_id} source={source_desc} scan={t_tree:.1f}s files={len(files)}"
                )

    result = {"train": {}, "eval": {}}
    t_np_load = 0.0
    total_npz_bytes = 0
    for file_i, local in enumerate(files, start=1):
        # fname: runs/{run_id}/predictions/eval_ep0000010_ba0000000050.npz
        parsed = _parse_prediction_file(local)
        if parsed is None:
            continue
        split, epoch, batch = parsed
        total_npz_bytes += local.stat().st_size
        t1 = time.perf_counter()
        payload = dict(np.load(local, allow_pickle=True))
        t_np_load += time.perf_counter() - t1
        payload["epoch"] = epoch
        payload["batch"] = batch
        prev = result[split].get(epoch)
        if prev is None or batch >= int(prev.get("batch", -1)):
            result[split][epoch] = payload
        if file_i % 100 == 0 or file_i == len(files):
            print(
                f"[fetch_predictions] progress run={run_id} files={file_i}/{len(files)} "
                f"source={source_desc} scan={t_tree:.1f}s np_load={t_np_load:.1f}s bytes={total_npz_bytes/1e6:.1f}MB"
            )
    print(
        f"[fetch_predictions] summary run={run_id} files={len(files)} kept_epochs="
        f"train:{len(result['train'])} eval:{len(result['eval'])} "
        f"source={source_desc} scan={t_tree:.1f}s np_load={t_np_load:.1f}s total_bytes={total_npz_bytes/1e6:.1f}MB"
    )
    return result


def fetch_overhead_images(sample_idxs, repo_id="eturok-weizmann/vibrations"):
    """Fetch cropped overhead images for given sample indices from the HF dataset.

    Returns:
        dict mapping sample_idx (int) -> PIL.Image

    Example:
        images = fetch_overhead_images([5, 23, 41])
        images[5].show()
    """
    from datasets import load_dataset

    sample_set = set(int(i) for i in sample_idxs)
    ds = load_dataset(
        repo_id,
        columns=["sample_idx", "cropped_image"],
        split="train",
        verification_mode="no_checks",
    )
    return {
        int(row["sample_idx"]): row["cropped_image"]
        for row in ds
        if int(row["sample_idx"]) in sample_set
    }
