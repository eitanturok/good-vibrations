"""Fine-tunes SonicDiffusion (composer_model.py) on our vibration data with
composer.Trainer -- same Composer-based training pattern as src/run.py, applied to
the diffusion adapter instead of BoomboxModel.

Trainable parameters: the audio Adapter + the UNet's gated cross-attention adapter
layers only (~34.6M params, continued from the released 'landscape' checkpoint) --
everything else (VAE, CLIP text encoder, CLAP audio encoder, the rest of the UNet)
stays frozen. See composer_model.py's docstring.

Usage:
    python src4/train.py --box gastronorm --target-n-objects 1 --epochs 100
"""
import argparse
import sys
from pathlib import Path

import torch
from composer import Trainer
from composer.algorithms import GradientClipping
from composer.callbacks import OptimizerMonitor, SpeedMonitor
from composer.loggers import WandBLogger

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from src4.callbacks import SonicDiffusionVisualizer, StepTimer, TrainableCheckpointSaver  # noqa: E402
from src4.composer_model import SonicDiffusionModel  # noqa: E402
from src4.dataset import DEFAULT_PROMPT, SonicDiffusionDataset  # noqa: E402


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--box", default="gastronorm", choices=["gastronorm", "plastic", "wood", "cardboard", "shoebox"])
    p.add_argument("--target-n-objects", type=int, nargs="+", default=[1],
                    help="train on positions with exactly these many objects (plus every empty-box sample), "
                         "split by position not by sample -- e.g. `--target-n-objects 1 2` for both 1-cube and "
                         "2-cube positions. See src4/dataset.py:multi_object_split / src3/dataset.py:n_object_split.")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="cap sample count, for a quick smoke test")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="text prompt paired with every sample -- defaults to empty, matching "
                         "SonicDiffusion's own null-text training regime (audio is the real signal).")
    p.add_argument("--target", default="rgb", choices=["rgb", "s_mask"],
                    help="rgb (default): predict the natural overhead photo, SDEdit-sampled starting from "
                         "the box's real empty-box photo. s_mask: predict the segmentation mask instead, "
                         "SDEdit-sampled starting from a synthetic all-black square (there's no meaningful "
                         "'empty-box mask' photo to start from). See src4/dataset.py:SonicDiffusionDataset.")
    p.add_argument("--spectrogram-stretch", default="none", choices=["none", "linear", "log"],
                    help="see src4/spectrogram_stretch.py: stretch our narrowband 50-1000Hz vibration "
                         "audio to use more of CLAP's internal 50-14000Hz mel range before it ever "
                         "reaches CLAP. none (default): no stretch. linear: uniform resample. log: "
                         "STFT + log-frequency warp + Griffin-Lim resynthesis.")
    # sampling / eval viz
    p.add_argument("--start-strength", type=float, default=0.75,
                    help="SDEdit strength for the empty-box start image (1.0=pure noise, 0.0=the start image itself).")
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--eval-interval", type=int, default=200, help="in batches")
    p.add_argument("--eval-samples", type=int, default=5)
    # train
    p.add_argument("--batch-size", type=int, default=10,
                    help="EFFECTIVE batch size (one optimizer step per this many samples). Larger than "
                         "--microbatch-size accumulates gradients over multiple microbatches per step "
                         "(Composer's device_train_microbatch_size) -- averaging the per-step noise-"
                         "prediction loss over more (sample, timestep) pairs, which is what actually "
                         "reduces its step-to-step variance (a lower LR does not, since it doesn't change "
                         "what's being measured, just how big the resulting update is).")
    p.add_argument("--microbatch-size", type=int, default=10,
                    help="max samples actually forwarded/backwarded together on the GPU at once. Re-verified "
                         "empirically on this 16GB GPU at bf16: 10 fits (13282MiB), 11 fits but tight "
                         "(14313MiB, little headroom for eval-viz generate() calls), 12+ hits a hard CUDA OOM "
                         "-- see src4/find_max_batch_size.py. Don't raise this past 10 without re-checking.")
    p.add_argument("--lr", type=float, default=1e-4,
                    help="SonicDiffusion's paper (arXiv 2405.00878 sec 3.6) trains the same components "
                         "(cross-attention adapter + audio projector) at 1e-4; we continue-fine-tune "
                         "from their checkpoint rather than training from scratch, but default to the "
                         "same value rather than src3/AudioToken's unrelated 1e-5 convention.")
    p.add_argument("--grad-clip-norm", type=float, default=0.1,
                    help="global L2 gradient-norm clipping threshold (composer.algorithms.GradientClipping) "
                         "-- a safety net against the rising grad-norm trend observed during training, sized "
                         "well above the typically-observed range (mean 0.008, max 0.03 so far) so it only "
                         "kicks in on real outliers rather than constraining normal updates.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    # run
    p.add_argument("--checkpoint-dir", type=Path, default=REPO / "src4/runs/default")
    p.add_argument("--wandb-project", default="good-vibrations-sonicdiffusion")
    p.add_argument("--wandb-tags", type=str, nargs="+", default=[],
                    help="wandb tags for this run (for filtering/grouping runs in the wandb UI, "
                         "separate from --run-name).")
    p.add_argument("--run-name", default=None)
    return p


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    train_dataset = SonicDiffusionDataset(args.box, "train", args.resolution, limit=args.limit,
                                           prompt=args.prompt, target_n_objects=args.target_n_objects,
                                           spectrogram_stretch=args.spectrogram_stretch, target=args.target)
    eval_dataset = SonicDiffusionDataset(args.box, "eval", args.resolution, limit=args.limit,
                                          prompt=args.prompt, target_n_objects=args.target_n_objects,
                                          spectrogram_stretch=args.spectrogram_stretch, target=args.target)
    print(f"{len(train_dataset)} train samples, {len(eval_dataset)} eval samples")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=2, drop_last=True, persistent_workers=True)

    model = SonicDiffusionModel()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)

    visualizer = SonicDiffusionVisualizer(train_dataset, eval_dataset, eval_interval_batches=args.eval_interval,
                                           n_samples=args.eval_samples, start_strength=args.start_strength,
                                           num_inference_steps=args.num_inference_steps, resolution=args.resolution,
                                           seed=args.seed)
    checkpoint_saver = TrainableCheckpointSaver(args.checkpoint_dir, interval_batches=args.eval_interval)
    # step_seconds (StepTimer) + throughput/samples_per_sec (SpeedMonitor) + l2_norm/grad/global
    # (OptimizerMonitor, logged every 10 batches by default) -- all to wandb via WandBLogger below.
    speed_monitor = SpeedMonitor()
    optimizer_monitor = OptimizerMonitor()
    step_timer = StepTimer()
    grad_clipping = GradientClipping(clipping_type="norm", clipping_threshold=args.grad_clip_norm)

    trainer = Trainer(
        run_name=args.run_name,
        model=model,
        optimizers=optimizer,
        train_dataloader=train_loader,
        device_train_microbatch_size=args.microbatch_size,
        max_duration=f"{args.epochs}ep",
        seed=args.seed,
        device="gpu" if torch.cuda.is_available() else "cpu",
        precision="amp_bf16" if torch.cuda.is_available() else "fp32",
        loggers=[WandBLogger(project=args.wandb_project, name=args.run_name,
                              init_kwargs={"config": vars(args), "tags": args.wandb_tags})],
        algorithms=[grad_clipping],
        callbacks=[visualizer, checkpoint_saver, speed_monitor, optimizer_monitor, step_timer],
        progress_bar=False,
        log_to_console=True,
    )
    trainer.fit()


if __name__ == "__main__":
    main(get_parser().parse_args())
