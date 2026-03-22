import tempfile, os, shutil
from typing import Any

import torch
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from composer import Callback
from composer.core import Time
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

class HFChkptUploader(Callback):
  def __init__(self, repo: str, interval: str = "1ep", monitor: str | None = None, path_in_repo: str | None = None, save_local: bool = False):
    self.repo, self.interval, self.monitor, self.api = repo, Time.from_timestring(interval), monitor, HfApi()
    self.last_save, self.best, self.higher_is_better = None, None, None
    self._path_in_repo = path_in_repo  # None = auto-derive from wandb.run.name
    self.save_local = save_local
    self.api.create_repo(repo, exist_ok=True)  # create once at init

  def _get_path_in_repo(self) -> str | None:
    if self._path_in_repo is not None: return self._path_in_repo
    return wandb.run.name if wandb.run else None

  def _is_better(self, v) -> bool:
    is_nan = self.best != self.best
    if self.best is None or is_nan: return True
    return v > self.best if self.higher_is_better else v < self.best

  def _push(self, model, msg):
    path_in_repo = self._get_path_in_repo()
    folder = f"checkpoints/{path_in_repo}" if self.save_local else tempfile.mkdtemp()
    os.makedirs(folder, exist_ok=True)
    try:
      torch.save(model.state_dict(), f"{folder}/model.pt")
      self.api.upload_folder(folder_path=folder, repo_id=self.repo, path_in_repo=path_in_repo, commit_message=msg)
      print(f"HFChkptUploader: uploaded to {self.repo}/{path_in_repo}")
    except Exception as e:
      print(f"HFChkptUploader ERROR: {e}")
    finally:
      if not self.save_local: shutil.rmtree(folder, ignore_errors=True)

  def _time_for(self, ts) -> int:
    return ts.get(self.interval.unit).value

  def epoch_end(self, state, logger):
    ts = state.timestamp
    if self.last_save is not None and self._time_for(ts) - self.last_save < self.interval.value: return
    if not self.monitor: return self._save(state, ts, None)
    metric = state.eval_metrics.get('eval', {}).get(self.monitor)
    if metric is None: return
    if self.higher_is_better is None: self.higher_is_better = getattr(metric, 'higher_is_better', False)
    v = metric.compute().item() if hasattr(metric, 'compute') else metric
    if not self._is_better(v): return
    self.best = v
    self._save(state, ts, v)

  def _save(self, state, ts, v):
    self.last_save = self._time_for(ts)
    msg = f"epoch {ts.epoch.value}" + (f" | {self.monitor}={v:.4f}" if v else "")
    print(f"HFChkptUploader: saving checkpoint at {msg}")
    self.api.run_as_future(self._push, state.model, msg)


class MaskVisualizationCallback(Callback):
    """After each eval, renders true vs predicted mask side-by-side, saves locally, and logs to WandB."""

    def __init__(self, n_samples: int = 4, save_dir: str = "visualizations"):
        self.n_samples = n_samples
        self.save_dir = save_dir
        self._last_batch = None

    def eval_batch_end(self, state, logger):
        # Grab only the first eval batch for visualization
        if self._last_batch is None:
            self._last_batch = (state.batch, state.outputs)

    def eval_end(self, state, logger):
        if self._last_batch is None:
            return

        batch, outputs = self._last_batch
        self._last_batch = None  # reset for next eval

        _, true_masks, _, _ = batch          # true_masks: (B, H, W) bool/int
        _, _, _, mask_logits = outputs       # mask_logits: (B, H, W)

        n = min(self.n_samples, true_masks.shape[0])
        pred_probs = mask_logits[:n].sigmoid().detach().cpu().float()
        true_masks = true_masks[:n].detach().cpu().float()

        os.makedirs(self.save_dir, exist_ok=True)
        epoch = state.timestamp.epoch.value
        wandb_images = []

        for i in range(n):
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(true_masks[i].numpy(), vmin=0, vmax=1, cmap='gray')
            axes[0].set_title('True mask')
            axes[0].axis('off')
            axes[1].imshow(pred_probs[i].numpy(), vmin=0, vmax=1, cmap='hot')
            axes[1].set_title('Predicted (prob)')
            axes[1].axis('off')
            fig.suptitle(f'Epoch {epoch} — sample {i}')
            fig.tight_layout()

            path = os.path.join(self.save_dir, f'epoch_{epoch:04d}_sample_{i}.png')
            fig.savefig(path, dpi=100)
            wandb_images.append(wandb.Image(fig, caption=f'sample {i}'))
            plt.close(fig)

        if wandb.run:
            wandb.log({'mask_viz': wandb_images}, step=state.timestamp.batch.value)


# from composer.callbacks import CheckpointSaver

# class BestCheckpointSaver(CheckpointSaver):
#     """Saves checkpoints only when a specified metric improves. Optionally uploads to HuggingFace."""

#     def __init__(self, metric_name:str, hf_repo:str|None=None, **kwargs):
#         super().__init__(overwrite=True, **kwargs)
#         self.metric_name = metric_name
#         self.best = None
#         self.hf_repo = hf_repo
#         if hf_repo:
#             self.hf_api = HfApi()
#             self.hf_api.create_repo(hf_repo, exist_ok=True)

#     def _save_checkpoint(self, state, logger):
#         metric = state.eval_metrics.get('eval', {}).get(self.metric_name)
#         if metric is None: return
#         higher_is_better = getattr(metric, 'higher_is_better', False)
#         val = metric.compute().item() if hasattr(metric, 'compute') else float(metric)
#         is_nan = self.best != self.best
#         ic(val, self.best, is_nan)
#         if self.best is None or is_nan or (val > self.best if higher_is_better else val < self.best):
#             self.best = val
#             super()._save_checkpoint(state, logger)
#             if self.hf_repo:
#                 self._upload_to_hf(state, val)

#     def _upload_to_hf(self, state, val):
#         folder = os.path.join(self.folder, state.run_name) if '{run_name}' in self.folder else self.folder
#         msg = f"epoch {state.timestamp.epoch.value} | {self.metric_name}={val:.4f}"
#         path_in_repo = wandb.run.name if wandb.run else None
#         print(f"BestCheckpointSaver: uploading to {self.hf_repo}/{path_in_repo}")
#         self.hf_api.run_as_future(self.hf_api.upload_folder, folder_path=folder, repo_id=self.hf_repo, path_in_repo=path_in_repo, commit_message=msg)

#     best_ckpt_saver = BestCheckpointSaver(metric_name="x/rMSE", hf_repo="eturok-weizmann/good-vibrations", folder="checkpoints", num_checkpoints_to_keep=1, save_interval=args.eval_interval)
