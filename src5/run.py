"""Composer training entrypoint for LDMSegVibrationModel. Mirrors src/run.py's conventions
(argparse, runs/{run_name}/... layout, AdamW+scheduler, WandBLogger/FileLogger) but trimmed to
just what this one model needs -- no Modal wrapping, no --model/--decoder dispatch (there's only
one model here).
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

import torch  # noqa: E402
from composer import Trainer  # noqa: E402
from composer.utils.reproducibility import seed_all  # noqa: E402
from composer.core import Evaluator  # noqa: E402
from composer.loggers import WandBLogger, FileLogger  # noqa: E402
from composer.algorithms import GradientClipping  # noqa: E402
from composer.callbacks import LRMonitor, SpeedMonitor, NaNMonitor, OptimizerMonitor  # noqa: E402
from composer.optim import CosineAnnealingWithWarmupScheduler  # noqa: E402

from model.dataset import build_dataset  # noqa: E402
from src5.latent_model import LDMSegVibrationModel  # noqa: E402
from src5.viz import VisualizePointRend, VisualizeSMaskAccum  # noqa: E402

DATA_DIR = REPO / "experiments" / "31_07_2026_gastronorm_exp1"
WARMSTART_CHECKPOINT = REPO / "runs" / "pp-baseline-v2" / "checkpoints" / "latest-rank0.pt"


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    # data (matches pp-baseline-v2's own config, since we warm-start from it)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--split", default="gastronorm")
    p.add_argument("--speakers", default=None,
                    help="comma-separated speaker ids to restrict train+eval to, e.g. '1,3,5,7' "
                         "(default: all speakers). Forwarded to the split fn (model.dataset._matches).")
    p.add_argument("--speaker-sample-per-position", type=int, default=None,
                    help="instead of a fixed --speakers set, draw this many speakers at random "
                         "PER POSITION (independently, seeded by --seed) -- see "
                         "model.dataset._sample_speakers_per_position. Mutually exclusive with --speakers.")
    p.add_argument("--out-h", type=int, default=32)
    p.add_argument("--out-w", type=int, default=32)
    p.add_argument("--signal-mode", default="magnitude")
    p.add_argument("--normalize-mode", default="std")
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--augment-fft", type=float, default=0.5,
                    help="probability of applying random_frequency_gain to a train sample's fft "
                         "(build_dataset's own default); 0 disables")
    p.add_argument("--augment-mask", type=float, default=0.5,
                    help="probability of applying noisy_blur (blur+noise) to a train sample's "
                         "mask_true (build_dataset's own default); 0 disables -- eval/train-baseline "
                         "loaders are always unaugmented regardless of this flag")
    # model
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--encoder", choices=("single", "two-stream"), default="single")
    p.add_argument("--warmstart-checkpoint", default=str(WARMSTART_CHECKPOINT))
    p.add_argument("--latent-loss-weight", type=float, default=0.1)
    p.add_argument("--ce-loss-weight", type=float, default=1.0)
    p.add_argument("--mask-loss-weight", type=float, default=1.0)
    p.add_argument("--compile", action="store_true",
                    help="torch.compile the encoder+decoder (not the frozen AE) -- see src5/SPEEDUP_LOG.md")
    p.add_argument("--full-res-decode", action="store_true",
                    help="decode at 512x512 (paper's native res) instead of the default 256x256 -- "
                         "+66%% samples/sec, -49%% memory at 256; see src5/SPEEDUP_LOG.md")
    # train
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--t-warmup", default="100ba")
    p.add_argument("--max-duration", default="500ep")
    p.add_argument("--eval-interval", default="10ep")
    p.add_argument("--checkpoint-interval", default="50ep")
    p.add_argument("--viz-interval", default="50ep",
                    help="log predicted-vs-ground-truth mask images (train + each eval split) every N epochs")
    p.add_argument("--log-points", action="store_true",
                    help="also log SMaskPoints/{split}: the same panels, overlaid with the actual "
                         "PointRend sample locations used for that step's ce/bce/dice loss")
    p.add_argument("--grad-clip", type=float, default=3.0,
                    help="global L2 grad-norm clip threshold, matching the paper's own AE training "
                         "(tools/configs/base/base.yaml's clip_grad: 3.0) -- 0 disables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    return p


def eval_boundary(trainer: Trainer, boundary_loaders: list[Evaluator]) -> None:
    """One-off eval pass outside the normal --eval-interval schedule (e.g. step 0, before any
    weight update) -- force VisualizeSMaskAccum to log images for it regardless of --viz-interval,
    same pattern as src/run.py's own eval_boundary."""
    viz_cbs = [cb for cb in trainer.state.callbacks if isinstance(cb, VisualizeSMaskAccum)]
    for cb in viz_cbs:
        cb.force_save = True
    try:
        trainer.eval(boundary_loaders)
    finally:
        for cb in viz_cbs:
            cb.force_save = False


def main(args: argparse.Namespace) -> None:
    device = "gpu" if torch.cuda.is_available() else "cpu"

    if args.speakers and args.speaker_sample_per_position:
        raise ValueError("--speakers and --speaker-sample-per-position are mutually exclusive")
    # only forwarded when set: gastronorm_one_cube/_two_cube etc. don't all accept
    # speaker_sample_per_position, so passing it unconditionally (even as None) would break them
    split_kwargs = {}
    if args.speakers: split_kwargs["speakers"] = [int(s) for s in args.speakers.split(",")]
    if args.speaker_sample_per_position: split_kwargs["speaker_sample_per_position"] = args.speaker_sample_per_position

    train_loader, eval_loaders, train_eval_loader = build_dataset(
        args.data_dir, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, split=args.split, out_h=args.out_h, out_w=args.out_w,
        signal_mode=args.signal_mode, normalize_mode=args.normalize_mode, patch_size=args.patch_size,
        seed=args.seed, augment_fft=args.augment_fft, augment_mask=args.augment_mask, **split_kwargs)

    # read n_channels/laser grid off the dataset, same as src/run.py
    _, _, _, n_channels = train_loader.dataloader.dataset[0]["fft"].shape
    base = train_loader.dataloader.dataset.dataset
    base = getattr(base, "dataset", base)
    n_laser_rows, n_laser_cols = base.grid_shape
    data_info = dict(out_h=args.out_h, out_w=args.out_w, n_channels=n_channels,
                      n_laser_rows=n_laser_rows, n_laser_cols=n_laser_cols)

    # Trainer(seed=...) only takes effect once the Trainer itself is constructed, which is AFTER
    # the model -- so decoder.head.4 (the one layer that can't warm-start, randomly initialized)
    # would otherwise draw from whatever the ambient RNG state happens to be, unrelated to
    # --seed and different on every process launch. Seed here so step-0 predictions (and hence
    # comparisons across run variants) are actually reproducible.
    seed_all(args.seed)
    model = LDMSegVibrationModel(
        data_dir=args.data_dir, data_info=data_info, d_model=args.d_model, encoder=args.encoder,
        latent_loss_weight=args.latent_loss_weight, ce_loss_weight=args.ce_loss_weight,
        mask_loss_weight=args.mask_loss_weight, warmstart_checkpoint=args.warmstart_checkpoint,
        compile=args.compile, decode_interpolate=args.full_res_decode)

    loggers = [FileLogger(f"runs/{{run_name}}/logs-rank{{rank}}.txt")]
    if args.run_name:
        loggers.append(WandBLogger("mask-autoencoder", name=args.run_name,
                                    init_kwargs={"config": data_info | vars(args), "id": args.run_name, "resume": "allow"}))

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay, fused=(device == "gpu"))
    scheduler = CosineAnnealingWithWarmupScheduler(t_warmup=args.t_warmup)
    algorithms = [GradientClipping(clipping_type="norm", clipping_threshold=args.grad_clip)] if args.grad_clip > 0 else []

    trainer = Trainer(
        run_name=args.run_name, model=model, optimizers=optimizer, schedulers=scheduler, algorithms=algorithms,
        train_dataloader=train_loader, eval_dataloader=eval_loaders, max_duration=args.max_duration,
        eval_interval=args.eval_interval, seed=args.seed, device=device,
        precision="amp_bf16" if device == "gpu" else "fp32", save_metrics=True, log_to_console=True,
        progress_bar=False, autoresume=bool(args.run_name),
        save_folder=f"runs/{{run_name}}/checkpoints" if args.run_name else None,
        save_interval=args.checkpoint_interval, loggers=loggers,
        # OptimizerMonitor logs l2_norm/grad/global (mean-over-weights L2 grad norm) each step.
        # VisualizeSMaskAccum (src5/viz.py) logs predicted-vs-true mask images once per
        # --viz-interval epochs per split, accumulated across all of that split's batches
        # (up to max_samples=108) -- not just the first batch. VisualizePointRend (--log-points)
        # logs the same panels overlaid with the actual PointRend sample locations used that step.
        callbacks=[NaNMonitor(), LRMonitor(), SpeedMonitor(1),
                   OptimizerMonitor(log_optimizer_metrics=True),
                   VisualizeSMaskAccum(args.viz_interval, max_samples=108),
                   *([VisualizePointRend(args.viz_interval, max_samples=108)] if args.log_points else [])])

    # step-0 baseline: eval (train split too, via train_eval_loader -- unaugmented/unshuffled)
    # and log its predicted-vs-true mask images, before any weight update happens. Label is
    # "train-baseline", NOT "train": Composer conflates an Evaluator's metrics tracker with the
    # real training dataloader's own when they share a label, silently swapping train_metrics
    # (CHEAP_SEG_KEYS) for val_metrics (full SEG_KEYS, slower) for the rest of the run.
    boundary_loaders = eval_loaders + [Evaluator(label="train-baseline", dataloader=train_eval_loader,
                                                  device_eval_microbatch_size=args.eval_batch_size)]
    eval_boundary(trainer, boundary_loaders)

    trainer.fit()


if __name__ == "__main__":
    main(get_parser().parse_args())
