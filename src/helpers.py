import os, time
from typing import Any

import torch
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from composer import Callback
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

class HFSyncCallback(Callback):
  """Blocking upload to HuggingFace after each local checkpoint save.
  Control frequency via Trainer(save_interval='Nep', ...)."""
  def __init__(self, local_folder: str, repo: str, dir_in_repo: str = "checkpoints"):
    self.local_folder, self.repo, self.dir_in_repo = local_folder, repo, dir_in_repo
    self.api = HfApi()
    self.api.create_repo(repo, exist_ok=True)
    self._t0 = None

  def epoch_end(self, state, logger): self._t0 = time.time()

  def epoch_checkpoint(self, state, logger):
    if self._t0: print(f"HFSyncCallback: local save took {time.time()-self._t0:.1f}s")
    msg = f"epoch {state.timestamp.epoch.value}"
    print(f"HFSyncCallback: uploading to HF at {msg}")
    t0 = time.time()
    try:
      commit = self.api.upload_folder(folder_path=self.local_folder, repo_id=self.repo, path_in_repo=self.dir_in_repo, commit_message=msg)
      print(f"HFSyncCallback: upload done in {time.time()-t0:.1f}s → {commit.commit_url}")
    except Exception as e:
      print(f"HFSyncCallback ERROR: {e}")


class MaskVisualizationCallback(Callback):
    def __init__(self, n_samples=4, save_dir="visualizations", train_viz_interval=10, thresholds=(0.3, 0.5, 0.7, 0.9)):
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
