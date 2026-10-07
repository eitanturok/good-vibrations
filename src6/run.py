# patch windows to not log emojis
import os, subprocess, sys
if os.environ.get("PYTHONUTF8") != "1":
    env = dict(os.environ, PYTHONUTF8="1")
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)

# supress warnings
import warnings, logging
warnings.filterwarnings("ignore", message=r"The pynvml package is deprecated.*", category=FutureWarning)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

# add current directory to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import argparse, shutil

import torch
from composer.utils.reproducibility import seed_all
from composer.optim import ConstantScheduler, CosineAnnealingScheduler, CosineAnnealingWithWarmupScheduler, LinearWithWarmupScheduler

BASE_DATA_DIR = Path("/home/ethantu/workspace/good-vibrations/experiments")


SCHEDULERS = {
    "constant": lambda t_warmup: ConstantScheduler(),
    "cosine": lambda t_warmup: CosineAnnealingScheduler(),
    "cosine-warmup": lambda t_warmup: CosineAnnealingWithWarmupScheduler(t_warmup=t_warmup),
    "linear-warmup": lambda t_warmup: LinearWithWarmupScheduler(t_warmup=t_warmup),
    }
def build_scheduler(scheduler: str, t_warmup: str): return SCHEDULERS[scheduler](t_warmup)


def get_parser():
    parser = argparse.ArgumentParser()
    # system
    parser.add_argument("--seed",                       type=int,   default=42)
    parser.add_argument("--debug",                      type=int,   default=0)
    parser.add_argument("--verbose",                    type=int,   default=2, help="If >=2, show torch.compile (TorchDynamo) logs.")

    # build data
    parser.add_argument("--data-dir",                   type=str,  default=BASE_DATA_DIR / "31_07_2026_gastronorm_exp1")
    parser.add_argument("--split",                      type=str,   default="gastronorm", help="How to split the dataset into train/val/test.")
    parser.add_argument("--num-workers",                type=int,   default=4)
    parser.add_argument("--test-size",                  type=float, default=0.2)
    parser.add_argument("--laser-cols",                 type=str,   default=None, help="Comma-separated laser column ids to train on, e.g. '0,1,2,3,4'. Whole columns across every row, so the kept lasers stay a rectangle. Default None = all. The selection is applied where the fft is read off disk, so every normalization and reference statistic is computed over exactly these lasers -- which means each selection gets its own MDS build.")
    parser.add_argument("--laser-rows",                 type=str,   default=None, help="Comma-separated laser row ids to train on, e.g. '0,1,2,3,4,5,6,7'. Whole rows across every kept column, so the kept lasers stay a rectangle. Default None = all. Same single-point selection as --laser-cols (composes with it), each combination gets its own MDS build.")

    # train
    parser.add_argument("--batch-size",                 type=int,   default=128)
    parser.add_argument("--lr",                         type=float, default=1e-4)
    parser.add_argument("--weight-decay",               type=float, default=1e-2)
    parser.add_argument("--scheduler",                  type=str,   default="cosine-warmup", choices=tuple(SCHEDULERS), help="LR schedule. 'constant' reproduces the old no-scheduler behavior.")
    parser.add_argument("--t-warmup",                   type=str,   default="100ep", help="Warmup length for the *-warmup schedulers; ignored by 'constant'.")
    parser.add_argument("--max-duration",               type=str,   default="2000ep")

    # eval
    parser.add_argument("--eval-only",                  type=int,   default=0, choices=(0, 1), help="Skip training, just eval a loaded checkpoint (requires --checkpoint-path).")
    parser.add_argument("--eval-batch-size",            type=int,   default=108) # wandb caps images logged in a single call to 108, so eval batch size should be <= 108 to log all images
    parser.add_argument("--eval-interval",              type=str,   default="50ep")
    parser.add_argument("--viz-interval",               type=str,   default="50ep", help="How often VisualizeSMask logs predicted-vs-true mask images to wandb.")
    parser.add_argument("--eval-before-train",          type=int,   default=1, choices=(0, 1), help="Run the boundary eval pass before training starts.")
    parser.add_argument("--eval-after-train",           type=int,   default=1, choices=(0, 1), help="Run the boundary eval pass after training ends.")

    # run
    parser.add_argument("--run-name",                   type=str,   default=None)
    parser.add_argument("--wandb-group",                type=str,   default="attn-lr-sweep", help="wandb group, for keeping sweep runs together.")

    # viz
    parser.add_argument("--viz-port",                   type=int,   default=8504, help="Port for the auto-launched viz dashboard.")
    parser.add_argument("--no-viz",                     action="store_true", help="Don't auto-launch the viz dashboard.")

    # checkpointing
    parser.add_argument("--checkpoint-path",            type=str,   default=None, help="Checkpoint to load for eval. If not set, defaults to the run's latest checkpoint.")
    parser.add_argument("--checkpoint-interval",        type=str,   default="500ep")
    parser.add_argument("--remote-checkpoint-folder",   type=str, default="eturok-weizmann/laser-vibrations-checkpoints")

    # torch compile faster
    parser.add_argument("--compile",                    type=int,   default=1, choices=(0, 1), help="torch.compile-ing the model before training/eval.")
    parser.add_argument("--compile-mode",               type=str,   default="default", help="torch.compile mode, e.g. 'default', 'reduce-overhead', 'max-autotune'.")

    # precision
    parser.add_argument("--precision",                  type=str,   default="amp_bf16", choices=["fp32", "amp_fp16", "amp_bf16"], help="bf16 matches fp16's tensor-core throughput on Blackwell but keeps fp32's exponent range, so no loss scaling and no underflow on wide-dynamic-range FFT magnitudes.")

    # loss
    # parser.add_argument("--loss-fn",                    type=str,   default='mse', choices=list(LOSSES))

    return parser

def run(**kwargs):
    # parse args
    args = get_parser().parse_args()  # get defaults
    args.__dict__.update(kwargs)  # apply overrides from cli
    assert not args.eval_only or args.checkpoint_path, "--checkpoint-path is required when using --eval-only"

    # set seeds BEFORE initializing model + dataloader
    seed_all(args.seed)
    print(f"seed={args.seed}")

    # device
    device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'{device=}')

    # torch compile
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if args.compile and args.verbose >= 2: torch._logging.set_logs(dynamo=logging.INFO)

    # build dataset
    log_dir = f"runs/{args.run_name}/positions"
    shutil.rmtree(log_dir, ignore_errors=True); os.makedirs(log_dir)

    train_loader, eval_loaders, train_eval_loader = build_dataset(
        args.data_dir,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, num_workers=args.num_workers,
        split=args.split, test_size=args.test_size,
        speakers=args.speakers, n_objects=args.n_objects, box=args.box, n_samples=args.n_samples,
        out_h=args.out_h, out_w=args.out_w,
        signal_mode=args.signal_mode, normalize_mode=args.normalize_mode, patch_size=args.patch_size, seed=args.seed,
        augment_fft=args.augment_fft, augment_mask=args.augment_mask,
        force_rebuild_data=bool(args.force_rebuild_data), rgb=bool(args.rgb), log_dir=log_dir)

if __name__ == "__main__":
    run()
