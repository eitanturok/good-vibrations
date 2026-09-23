"""Post-hoc OutputSaver dump for an LDMSegVibrationModel checkpoint that was trained without
--output-keys wiring (src5/run.py has no such flag), so viz has nothing to read from its
runs/<run_name>/ dir. Loads a checkpoint eval-only (no training) and replays one eval pass with
OutputSaver attached, producing the same runs/<run_name>/outputs_history/{train,eval/<split>}/
ep####-ba######.pt files src/run.py's --output-keys path would have written live.

    python src5/post_hoc_outputs.py --run-name ldmseg-baseline-v11 \
        --checkpoint-path runs/ldmseg-baseline-v11/checkpoints/ep2000-ba684000-rank0.pt --compile
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

import torch  # noqa: E402
from composer import Trainer  # noqa: E402
from composer.core import Evaluator  # noqa: E402

from model.dataset import build_dataset  # noqa: E402
from model.callbacks import OutputSaver  # noqa: E402
from src5.latent_model import LDMSegVibrationModel  # noqa: E402
from src5.run import get_parser as get_train_parser  # noqa: E402


def get_parser():
    p = get_train_parser()
    p.add_argument("--checkpoint-path", required=True)
    return p


def main(args):
    device = "gpu" if torch.cuda.is_available() else "cpu"

    train_loader, eval_loaders, train_eval_loader = build_dataset(
        args.data_dir, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, split=args.split, out_h=args.out_h, out_w=args.out_w,
        signal_mode=args.signal_mode, normalize_mode=args.normalize_mode, patch_size=args.patch_size,
        seed=args.seed)
    boundary_loaders = eval_loaders + [Evaluator(label="train", dataloader=train_eval_loader,
                                                  device_eval_microbatch_size=args.eval_batch_size)]

    _, _, _, n_channels = train_loader.dataloader.dataset[0]["fft"].shape
    base = train_loader.dataloader.dataset.dataset
    base = getattr(base, "dataset", base)
    n_laser_rows, n_laser_cols = base.grid_shape
    data_info = dict(out_h=args.out_h, out_w=args.out_w, n_channels=n_channels,
                      n_laser_rows=n_laser_rows, n_laser_cols=n_laser_cols)

    model = LDMSegVibrationModel(
        data_dir=args.data_dir, data_info=data_info, d_model=args.d_model, encoder=args.encoder,
        latent_loss_weight=args.latent_loss_weight, ce_loss_weight=args.ce_loss_weight,
        mask_loss_weight=args.mask_loss_weight, warmstart_checkpoint=None,
        compile=False, decode_interpolate=args.full_res_decode)

    output_saver = OutputSaver("1ep", f"runs/{{run_name}}/outputs_history", overwrite=True,
                                output_keys=("mask_pred", "info"))

    trainer = Trainer(
        run_name=args.run_name, model=model, eval_dataloader=eval_loaders, max_duration=None,
        seed=args.seed, device=device, precision="amp_bf16" if device == "gpu" else "fp32",
        log_to_console=True, progress_bar=False, load_path=args.checkpoint_path,
        callbacks=[output_saver])

    output_saver.force_save = True
    trainer.eval(boundary_loaders)


if __name__ == "__main__":
    main(get_parser().parse_args())
