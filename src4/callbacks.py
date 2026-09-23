"""Composer callback that periodically logs a 4-panel wandb image (spectrogram |
starting image | predicted image | ground truth) for a handful of fixed samples --
the SonicDiffusion analogue of src3/train.py's visualize(), wrapped as a Callback
instead of a manual step loop so it plugs into composer.Trainer directly.

Runs its own multi-step generate() sampling (see composer_model.py:generate), which
is NOT part of the normal training forward pass (that only predicts one noise
residual per step) -- so this is gated on --eval-interval batches, not every batch.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from composer import Callback, Logger
from composer.core import State
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path: sys.path.insert(0, str(REPO / "src"))  # src/data/audio.py imports as `from data.audio import ...`

from data.audio import get_spectrogram  # noqa: E402
from src4.dataset import SonicDiffusionDataset  # noqa: E402
from src4.model.clap_audio_encoder import FMAX as CLAP_FMAX  # noqa: E402
from src4.model.clap_audio_encoder import SAMPLE_RATE  # noqa: E402

MAX_WANDB_IMAGES = 108  # wandb.Image caps any single log_images call at this many items
DB_TOP_DB = 60.0


def _spectrogram_panel(waveform: torch.Tensor, resolution: int) -> Image.Image:
    """Raw waveform -> dB-scale STFT spectrogram, percentile-stretched to a viewable
    grayscale panel (freq low->high bottom->top, matching src3/beats_input.py's
    display convention). Cropped to 0-CLAP_FMAX (14000Hz), not the display
    waveform's own Nyquist (22050Hz at CLAP's 44.1kHz) -- CLAP's mel filterbank
    never sees anything above CLAP_FMAX regardless of stretch mode, so showing that
    dead range just reads as a black bar with no signal (see this conversation)."""
    freqs, _, Sxx = get_spectrogram(waveform.numpy(), SAMPLE_RATE, nperseg=512)
    Sxx = Sxx[freqs <= CLAP_FMAX]
    db = 10 * np.log10(Sxx + 1e-10)
    db = np.maximum(db, db.max() - DB_TOP_DB)
    lo, hi = np.percentile(db, [1, 99.5])
    norm = np.clip((db - lo) / max(hi - lo, 1e-8), 0, 1)
    img = (np.flipud(norm) * 255).astype(np.uint8)
    return Image.fromarray(img).convert("RGB").resize((resolution, resolution))


def fixed_indices(n_total: int, n: int, seed: int) -> np.ndarray:
    """The same n indices every call -- logged panels track the same samples across
    eval intervals instead of a fresh random draw each time."""
    return np.random.RandomState(seed).choice(n_total, min(n, n_total), replace=False)


class SonicDiffusionVisualizer(Callback):
    def __init__(self, train_dataset: SonicDiffusionDataset, eval_dataset: SonicDiffusionDataset,
                 eval_interval_batches: int, n_samples: int = 5, start_strength: float = 0.75,
                 num_inference_steps: int = 30, resolution: int = 512, seed: int = 42):
        self.datasets = {"train_viz": train_dataset, "eval_viz": eval_dataset}
        self.indices = {name: fixed_indices(len(ds), n_samples, seed) for name, ds in self.datasets.items()}
        self.eval_interval_batches = eval_interval_batches
        self.start_strength, self.num_inference_steps = start_strength, num_inference_steps
        self.resolution = resolution
        self.generator_seed = seed

    def _is_due(self, state) -> bool:
        n = state.timestamp.batch.value
        return n > 0 and n % self.eval_interval_batches == 0

    def _visualize(self, state, logger, key_prefix: str):
        dataset = self.datasets[key_prefix]
        model = state.model
        device = next(model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(self.generator_seed)
        start_image = dataset.start_image().resize((self.resolution, self.resolution))

        t_gen = time.perf_counter()
        panels = []
        for i in self.indices[key_prefix]:
            item = dataset[i]
            gt = Image.open(dataset.samples[i][0] / dataset.target_name).convert("RGB") \
                .resize((self.resolution, self.resolution))
            pred = model.generate(item["waveform"], item["prompt"], start_image=start_image,
                                   start_strength=self.start_strength,
                                   num_inference_steps=self.num_inference_steps, generator=generator)
            spec = _spectrogram_panel(item["waveform"], self.resolution)

            panel = Image.new("RGB", (self.resolution * 4, self.resolution))
            for x, im in enumerate([spec, start_image, pred, gt]):
                panel.paste(im, (self.resolution * x, 0))
            panels.append(panel)
        t_gen = time.perf_counter() - t_gen

        t_log = time.perf_counter()
        logger.log_images([np.array(p) for p in panels[:MAX_WANDB_IMAGES]],
                           name=f"{key_prefix}/pred_vs_gt", channels_last=True, use_table=False)
        t_log = time.perf_counter() - t_log
        print(f"[timing] {key_prefix}: generate+compose {len(panels)} panels: {t_gen:.2f}s, "
              f"log_images: {t_log:.2f}s")

    def _run_visualization(self, state, logger) -> None:
        was_training = state.model.training
        state.model.eval()
        with torch.no_grad():
            self._visualize(state, logger, "train_viz")
            self._visualize(state, logger, "eval_viz")
        state.model.train(was_training)

    def fit_start(self, state, logger) -> None:
        """Logs one pre-training baseline panel (step 0, before any weight update) --
        so wandb has a reference point the training-progress panels can be compared
        against, not just the panels starting from wherever step eval_interval lands."""
        self._run_visualization(state, logger)

    def batch_end(self, state, logger) -> None:
        if not self._is_due(state):
            return
        self._run_visualization(state, logger)


class TrainableCheckpointSaver(Callback):
    """Saves only SonicDiffusionModel.trainable_state_dict() (~34.6M params, a few
    hundred MB) at the same cadence as SonicDiffusionVisualizer, instead of
    Composer's default full-model checkpoint (5GB/save here, since it includes the
    frozen VAE/text_encoder/CLAP/UNet backbone -- see composer_model.py's docstring
    on trainable_state_dict). --checkpoint-dir/step_<n>.pt and .../final.pt."""

    def __init__(self, checkpoint_dir: Path, interval_batches: int):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval_batches = interval_batches

    def _save(self, state: State, tag: str) -> None:
        torch.save(state.model.trainable_state_dict(), self.checkpoint_dir / f"{tag}.pt")

    def batch_end(self, state: State, logger: Logger) -> None:
        n = state.timestamp.batch.value
        if n > 0 and n % self.interval_batches == 0:
            self._save(state, f"step_{n}")

    def fit_end(self, state: State, logger: Logger) -> None:
        self._save(state, "final")


class StepTimer(Callback):
    """Logs literal per-step wall-clock time (`time/step_seconds`) to wandb --
    complements composer.callbacks.SpeedMonitor's throughput/samples_per_sec (its
    reciprocal, batched over a rolling window) with a direct per-batch number, and
    composer.callbacks.OptimizerMonitor's l2_norm/grad/global with... nothing, that
    one already logs exactly what's needed on its own. See train.py."""

    def __init__(self):
        self._t0 = None

    def batch_start(self, state: State, logger: Logger) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # otherwise this only measures kernel-launch time, not compute
        self._t0 = time.perf_counter()

    def batch_end(self, state: State, logger: Logger) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.log_metrics({"time/step_seconds": time.perf_counter() - self._t0})
