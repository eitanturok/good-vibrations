import os, time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import psutil
import torch
import wandb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from composer import Callback
from composer.callbacks import CheckpointSaver
from composer.core import Event, State
from composer.loggers import Logger
from huggingface_hub import HfApi


def getenv(key: str, default: Any = 0):
    return type(default)(os.getenv(key, default))


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
        kwargs["weights_only"] = True
        super().__init__(**kwargs)
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.best: Optional[float] = None

    def _get_metric_value(self, state: State) -> Optional[float]:
        m = state.eval_metrics.get("eval", {}).get(self.metric_name)
        if m is None:
            return None
        return m.compute().item() if hasattr(m, "compute") else float(m)

    def _is_improved(self, val: float) -> bool:
        if self.best is None or (val != val):  # first eval or NaN best
            return True
        return val > self.best if self.higher_is_better else val < self.best

    def epoch_checkpoint(self, state: State, logger: Logger):
        val = self._get_metric_value(state)
        if val is not None and self._is_improved(val):
            self.best = val
            baseline = torch.cuda.memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            super().epoch_checkpoint(state, logger)
            spike = torch.cuda.max_memory_allocated() / 1e9 - baseline
            print(f"checkpoint memory spike: {spike:.2f} GB")
            if wandb.run:
                wandb.log(
                    {"memory/gpu/checkpoint_spike_gb": spike},
                    step=state.timestamp.batch.value,
                )


class MemoryCallback(Callback):
    """Logs GPU/CPU memory every batch and streams a Plotly stacked area chart to wandb."""

    def __init__(self, dataset_gb: float):
        self.dataset_gb = dataset_gb
        self._peak_forward = 0.0
        self._peak_backward = 0.0
        # accumulated history — each append is one batch
        self._steps = []
        self._weights = []
        self._optimizer = []
        self._gradients = []
        self._other = []
        self._peak_fwd = []
        self._peak_bwd = []
        self._gpu_total = []
        self._cpu_other = []
        self._ram_total = []

    def before_train_batch(self, state, logger):
        torch.cuda.reset_peak_memory_stats()

    def after_forward(self, state, logger):
        self._peak_forward = torch.cuda.max_memory_allocated() / 1e9

    def after_backward(self, state, logger):
        self._peak_backward = torch.cuda.max_memory_allocated() / 1e9

    def batch_end(self, state, logger):
        model = state.model
        optimizer = state.optimizers[0]
        weights_gb = sum(p.data.nbytes for p in model.parameters()) / 1e9
        grads_gb = (
            sum(p.grad.nbytes for p in model.parameters() if p.grad is not None) / 1e9
        )
        opt_gb = (
            sum(
                v.nbytes
                for s in optimizer.state.values()
                for v in s.values()
                if isinstance(v, torch.Tensor)
            )
            / 1e9
        )
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        act_gb = max(0.0, self._peak_forward - weights_gb)
        other_gpu = max(0.0, allocated_gb - weights_gb - opt_gb - grads_gb)
        vm = psutil.virtual_memory()
        other_cpu = max(0.0, vm.used / 1e9 - self.dataset_gb)

        logger.log_metrics(
            {
                "memory/gpu/weights_gb": weights_gb,
                "memory/gpu/gradients_gb": grads_gb,
                "memory/gpu/optimizer_gb": opt_gb,
                "memory/gpu/activations_gb": act_gb,
                "memory/gpu/peak_forward_gb": self._peak_forward,
                "memory/gpu/peak_backward_gb": self._peak_backward,
                "memory/gpu/allocated_gb": allocated_gb,
                "memory/gpu/reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "memory/gpu/total_gb": gpu_total_gb,
                "memory/cpu/dataset_gb": self.dataset_gb,
                "memory/cpu/ram_used_gb": vm.used / 1e9,
                "memory/cpu/ram_available_gb": vm.available / 1e9,
                "memory/cpu/ram_total_gb": vm.total / 1e9,
            }
        )

        # accumulate and redraw chart
        step = state.timestamp.batch.value
        self._steps.append(step)
        self._weights.append(weights_gb)
        self._optimizer.append(opt_gb)
        self._gradients.append(grads_gb)
        self._other.append(other_gpu)
        self._peak_fwd.append(self._peak_forward)
        self._peak_bwd.append(self._peak_backward)
        self._gpu_total.append(gpu_total_gb)
        self._cpu_other.append(other_cpu)
        self._ram_total.append(vm.total / 1e9)

        if wandb.run:
            wandb.log({"memory/breakdown": wandb.Plotly(self._build_fig())}, step=step)

    def _build_fig(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("GPU Memory", "CPU (RAM) Memory"),
            vertical_spacing=0.12,
        )
        s = self._steps

        for name, values, color in [
            ("Weights", self._weights, "#1565C0"),  # dark blue  – stable foundation
            (
                "Optimizer",
                self._optimizer,
                "#2E7D32",
            ),  # dark green – grows then stabilises
            ("Gradients", self._gradients, "#E65100"),  # dark orange
            ("Other", self._other, "#9E9E9E"),  # grey – misc CUDA allocs
        ]:
            fig.add_trace(
                go.Scatter(
                    x=s,
                    y=values,
                    name=name,
                    legendgroup="gpu",
                    mode="lines",
                    stackgroup="gpu",
                    line=dict(width=0),
                    fillcolor=color,
                    hovertemplate=f"<b>{name}</b>: %{{y:.3f}} GB<extra></extra>",
                ),
                row=1,
                col=1,
            )

        for name, values, color, dash in [
            ("Peak fwd (incl. activations)", self._peak_fwd, "#42A5F5", "dot"),
            ("Peak bwd", self._peak_bwd, "#EF5350", "solid"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=s,
                    y=values,
                    name=name,
                    legendgroup="gpu",
                    mode="lines",
                    line=dict(color=color, width=1.5, dash=dash),
                    hovertemplate=f"<b>{name}</b>: %{{y:.3f}} GB<extra></extra>",
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=s,
                y=self._gpu_total,
                name="GPU Capacity",
                legendgroup="gpu",
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                hovertemplate="<b>GPU Capacity</b>: %{y:.1f} GB<extra></extra>",
            ),
            row=1,
            col=1,
        )

        dataset_vals = [self.dataset_gb] * len(s)
        for name, values, color in [
            ("Dataset", dataset_vals, "#1565C0"),
            ("Other", self._cpu_other, "#9E9E9E"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=s,
                    y=values,
                    name=f"CPU: {name}",
                    legendgroup="cpu",
                    mode="lines",
                    stackgroup="cpu",
                    line=dict(width=0),
                    fillcolor=color,
                    hovertemplate=f"<b>{name}</b>: %{{y:.3f}} GB<extra></extra>",
                ),
                row=2,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=s,
                y=self._ram_total,
                name="RAM Capacity",
                legendgroup="cpu",
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                hovertemplate="<b>RAM Capacity</b>: %{y:.1f} GB<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title="Memory Breakdown",
            hovermode="x unified",
            height=700,
            legend=dict(tracegroupgap=20),
        )
        fig.update_yaxes(title_text="GB", rangemode="tozero", row=1, col=1)
        fig.update_yaxes(title_text="GB", rangemode="tozero", row=2, col=1)
        fig.update_xaxes(title_text="Step", row=2, col=1)
        return fig


class MaskVisualizationCallback(Callback):
    def __init__(
        self,
        n_samples=4,
        save_dir="visualizations",
        train_viz_interval=10,
        thresholds=[],
        pred_save_path=None,
        run_id=None,
    ):
        self.n_samples = n_samples
        self.save_dir = save_dir
        self.train_viz_interval = train_viz_interval
        self.thresholds = list(thresholds)
        self.pred_save_path = (
            pred_save_path  # HF repo ID, e.g. "eturok-weizmann/good-vibrations"
        )
        self.run_id = run_id
        self._last_train_batch = None
        self._eval_preds = []  # accumulates across all eval batches

    def epoch_start(self, state, logger):
        self._last_train_batch = None

    def batch_end(self, state, logger):
        if self._last_train_batch is None:
            self._last_train_batch = (state.batch, state.outputs)

    def epoch_end(self, state, logger):
        if (
            state.timestamp.epoch.value % self.train_viz_interval == 0
            and self._last_train_batch is not None
        ):
            batch, outputs = self._last_train_batch
            self._visualize(batch, outputs, state, "train")
            self._save_predictions(
                [self._batch_to_pred(batch, outputs)], state, "train"
            )

    def eval_batch_end(self, state, logger):
        self._eval_preds.append(self._batch_to_pred(state.batch, state.outputs))

    def eval_end(self, state, logger):
        if not self._eval_preds:
            return
        # use first accumulated batch for the W&B image visualization
        first_batch, first_outputs = self._eval_preds[0]["_raw"]
        self._visualize(first_batch, first_outputs, state, "eval")
        self._save_predictions(self._eval_preds, state, "eval")
        self._eval_preds = []

    def _batch_to_pred(self, batch, outputs):
        _, true_masks, _, _, meta = batch
        _, _, _, mask_logits = outputs
        return {
            "_raw": (batch, outputs),
            "mask_true": true_masks.detach().cpu().float().numpy(),
            "mask_pred": mask_logits.sigmoid().detach().cpu().float().numpy(),
            "sample_idx": list(meta["sample_idx"]),
            "x_position": list(meta["x_position"]),
            "y_position": list(meta["y_position"]),
            "object": list(meta["object"]),
            "n_objects": list(meta["n_objects"]),
        }

    def _save_predictions(self, preds_list, state, split):
        if not self.pred_save_path or not self.run_id:
            return
        import io, numpy as np

        mask_true = np.concatenate([p["mask_true"] for p in preds_list])
        mask_pred = np.concatenate([p["mask_pred"] for p in preds_list])
        sample_idx = sum([p["sample_idx"] for p in preds_list], [])
        x_position = sum([p["x_position"] for p in preds_list], [])
        y_position = sum([p["y_position"] for p in preds_list], [])
        object_type = sum([p["object"] for p in preds_list], [])
        n_objects = sum([p["n_objects"] for p in preds_list], [])

        buf = io.BytesIO()
        np.savez(
            buf,
            mask_true=mask_true,
            mask_pred=mask_pred,
            sample_idx=np.array(sample_idx),
            x_position=np.array(x_position, dtype=float),
            y_position=np.array(y_position, dtype=float),
            object_type=np.array(object_type),
            n_objects=np.array(n_objects, dtype=int),
        )
        buf.seek(0)

        epoch = state.timestamp.epoch.value
        repo_id = self.pred_save_path.removeprefix("hf://")
        path_in_repo = f"predictions/{self.run_id}/{split}_epoch_{epoch:05d}.npz"
        try:
            HfApi().upload_file(
                path_or_fileobj=buf,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"[MaskVisualizationCallback] Failed to save predictions: {e}")

    def _visualize(self, batch, outputs, state, split):
        _, true_masks, _, _, meta = batch
        _, _, _, mask_logits = outputs
        n = min(self.n_samples, true_masks.shape[0], mask_logits.shape[0])
        probs = mask_logits[:n].sigmoid().detach().cpu().float().numpy()
        true = true_masks[:n].detach().cpu().float().numpy()
        epoch = state.timestamp.epoch.value
        os.makedirs(self.save_dir, exist_ok=True)
        if not wandb.run:
            return
        sample_ids = meta["sample_idx"][:n]
        log = {}
        for i in range(n):
            caption = f"sample {sample_ids[i]}"
            # continuous prob map (no threshold)
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
            ax0.imshow(true[i], vmin=0, vmax=1, cmap="gray")
            ax0.set_title("True Mask")
            ax0.axis("off")
            ax1.imshow(probs[i], vmin=0, vmax=1, cmap="gray")
            ax1.set_title("Pred Mask (prob)")
            ax1.axis("off")
            fig.suptitle(
                f"Epoch {epoch}, {split.capitalize()} Sample {sample_ids[i]}, Prob"
            )
            fig.tight_layout()
            log.setdefault(f"mask_viz/{split}/prob", []).append(
                wandb.Image(fig, caption=caption)
            )
            plt.close(fig)
            # binarized at each threshold
            for t in self.thresholds:
                fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 4))
                ax0.imshow(true[i], vmin=0, vmax=1, cmap="gray")
                ax0.set_title("true")
                ax0.axis("off")
                ax1.imshow((probs[i] > t).astype(float), vmin=0, vmax=1, cmap="gray")
                ax1.set_title(f"Pred Mask (threshold {t})")
                ax1.axis("off")
                fig.suptitle(
                    f"Epoch {epoch}, {split.capitalize()} Sample {sample_ids[i]}, threshold {t}"
                )
                fig.tight_layout()
                log.setdefault(f"mask_viz/{split}/thresh{t}", []).append(
                    wandb.Image(fig, caption=caption)
                )
                plt.close(fig)
        wandb.log(log, step=state.timestamp.batch.value)


def fetch_predictions(
    run_id,
    data_dir="eturok-weizmann/good-vibrations",
    cache_dir="~/.cache/good-vibrations",
):
    """Download saved prediction npz files for a run from HF Hub.

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
    from huggingface_hub import hf_hub_download, list_repo_tree

    cache_dir = Path(cache_dir).expanduser()
    repo_id = data_dir.removeprefix("hf://")
    prefix = f"predictions/{run_id}/"

    files = [
        f.rfilename
        for f in list_repo_tree(repo_id, repo_type="dataset", recursive=True)
        if f.rfilename.startswith(prefix) and f.rfilename.endswith(".npz")
    ]

    result = {"train": {}, "eval": {}}
    for fname in sorted(files):
        # fname: predictions/{run_id}/eval_epoch_00150.npz
        stem = Path(fname).stem  # "eval_epoch_00150"
        split, _, epoch_str = stem.partition("_epoch_")
        if split not in result:
            continue
        epoch = int(epoch_str)
        local = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        result[split][epoch] = dict(np.load(local, allow_pickle=True))
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
