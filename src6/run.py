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
from torch.utils.data import DataLoader
from composer import Trainer
from composer.core import Evaluator
from composer.callbacks import LRMonitor, SpeedMonitor, NaNMonitor, RuntimeEstimator, OptimizerMonitor, OOMObserver
from composer.loggers import WandBLogger, FileLogger
from composer.utils.reproducibility import seed_all

from dataset import DATASETS, build_dataset
from arch import Boombox
from callbacks import VisualizeSMask, OutputSaver, PositionLogger
from composer.optim import ConstantScheduler, CosineAnnealingScheduler, CosineAnnealingWithWarmupScheduler, LinearWithWarmupScheduler


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
    parser.add_argument("--dataset",                    type=str,   default="gastronorm_plastic", choices=tuple(DATASETS))
    parser.add_argument("--test-size",                  type=float, default=0.2, help="Fraction of each split's positions held out for eval.")
    parser.add_argument("--speakers",                   type=lambda s: s if s == "all" else int(s), default=[1, 3, 7], nargs="+", help="Speakers to train/eval on, e.g. --speakers 1 3 7; 'all' = every speaker, none held out.")
    parser.add_argument("--signal",                     type=str,   default="magnitude", choices=("magnitude", "log_magnitude"))
    parser.add_argument("--norm",                       type=str,   default="std", choices=("std", "z"))
    parser.add_argument("--patch-size",                 type=lambda s: None if s.lower() == "none" else int(s), default=None, help="Freq bins per token; 'none' = no patching, x is (L,F,C).")
    parser.add_argument("--out-h",                      type=int,   default=64)
    parser.add_argument("--out-w",                      type=int,   default=64)
    parser.add_argument("--rgb",                        type=int,   default=0, choices=(0, 1), help="Predict the overhead photo instead of the mask.")
    parser.add_argument("--num-workers",                type=int,   default=4)

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
    parser.add_argument("--eval-interval",              type=str,   default="50ep", help="How often to eval, log predicted-vs-true images to wandb, and save outputs for viz.")
    parser.add_argument("--eval-before-train",          type=int,   default=1, choices=(0, 1), help="Run the boundary eval pass before training starts.")
    parser.add_argument("--eval-after-train",           type=int,   default=1, choices=(0, 1), help="Run the boundary eval pass after training ends.")

    # run
    parser.add_argument("--run-name",                   type=str,   default=None)
    parser.add_argument("--wandb",                      type=int,   default=1, choices=(0, 1), help="Log to wandb; 0 = file logger only (smoke tests).")
    parser.add_argument("--wandb-group",                type=str,   default="attn-lr-sweep", help="wandb group, for keeping sweep runs together.")

    # checkpointing
    parser.add_argument("--checkpoint-path",            type=str,   default=None, help="Checkpoint to load for eval. If not set, defaults to the run's latest checkpoint.")
    parser.add_argument("--checkpoint-interval",        type=str,   default="500ep")

    # torch compile faster
    parser.add_argument("--compile",                    type=int,   default=1, choices=(0, 1), help="torch.compile-ing the model before training/eval.")
    parser.add_argument("--compile-mode",               type=str,   default="default", help="torch.compile mode, e.g. 'default', 'reduce-overhead', 'max-autotune'.")

    # precision
    parser.add_argument("--precision",                  type=str,   default="amp_bf16", choices=["fp32", "amp_fp16", "amp_bf16"], help="bf16 matches fp16's tensor-core throughput on Blackwell but keeps fp32's exponent range, so no loss scaling and no underflow on wide-dynamic-range FFT magnitudes.")

    # model
    parser.add_argument("--d-model",                    type=int,   default=1024, help="Size of the embedding between encoder and decoder.")

    # loss
    parser.add_argument("--loss-fn",                    type=str,   default="bce", choices=("bce", "mse", "dice", "bce-dice"), help="Mask loss.")
    parser.add_argument("--alpha",                      type=float, default=0.5, help="bce-dice only: alpha * bce + (1 - alpha) * dice.")
    parser.add_argument("--object-weight",              type=float, default=0.1, help="Loss weight of the object-class head; 0 = off.")
    parser.add_argument("--n-objects-weight",           type=float, default=0.1, help="Loss weight of the object-count head; 0 = off.")
    parser.add_argument("--com-weight",                 type=float, default=1.0, help="Loss weight of the per-object com head; 0 = off.")
    parser.add_argument("--area-weight",                type=float, default=0.1, help="Loss weight of the mask-area head; 0 = off.")

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
    train_dl, eval_dl, data_info = build_dataset(args.dataset, test_size=args.test_size, speakers=None if "all" in args.speakers else args.speakers, signal=args.signal, norm=args.norm, patch_size=args.patch_size,
        out_h=args.out_h, out_w=args.out_w, rgb=bool(args.rgb), batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers)
    train_eval = Evaluator(label="train", dataloader=DataLoader(train_dl.dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers))

    # build model
    assert args.patch_size is None, "Boombox convolves the raw spectrum: use --patch-size none"
    model = Boombox(args.d_model, data_info, args.out_h, args.out_w, rgb=bool(args.rgb), loss_fn=args.loss_fn, alpha=args.alpha,
                    loss_weights=dict(object=args.object_weight, n_objects=args.n_objects_weight, com=args.com_weight, area=args.area_weight))
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

    # loggers
    loggers = [FileLogger("runs/{run_name}/logs-rank{rank}.txt")]
    if args.wandb and not args.eval_only:
        config = args.__dict__ | data_info | dict(gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                                                  num_parameters=sum(p.numel() for p in model.parameters()))
        loggers.append(WandBLogger("better-tsa", group=args.wandb_group, name=args.run_name,
                                   init_kwargs={"config": config, "id": args.run_name, "resume": "allow", "save_code": True}))

    # callbacks
    callbacks = [
        LRMonitor(), SpeedMonitor(), RuntimeEstimator(skip_batches=64, time_unit="minutes"), NaNMonitor(),
        OptimizerMonitor(log_optimizer_metrics=True, batch_log_interval=10),               # grad + weight norms
        OOMObserver(folder="runs/{run_name}/torch_traces", remote_file_name=None, overwrite=True),  # memory traces if a run OOMs
        VisualizeSMask(args.eval_interval), OutputSaver(args.eval_interval), PositionLogger(),
    ]

    # optimizer + lr schedule
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args.scheduler, args.t_warmup)

    # trainer
    trainer = Trainer(
        run_name=args.run_name, model=model, optimizers=optimizer, schedulers=scheduler,
        train_dataloader=train_dl, eval_dataloader=eval_dl, eval_interval=args.eval_interval,
        max_duration=None if args.eval_only else args.max_duration, seed=args.seed,
        device=device, precision=args.precision if device == "gpu" else "fp32",
        loggers=loggers, callbacks=callbacks, progress_bar=False, log_to_console=True, save_metrics=True,
        load_path=args.checkpoint_path, autoresume=bool(args.run_name) and not args.eval_only,
        save_folder=None if args.eval_only else "runs/{run_name}/checkpoints", save_interval=args.checkpoint_interval,
        compile_config={"mode": args.compile_mode} if args.compile else None,
    )

    # train the model
    if args.eval_before_train: trainer.eval(eval_dl + [train_eval])
    if not args.eval_only:
        trainer.fit()
        if args.eval_after_train: trainer.eval(eval_dl + [train_eval])

if __name__ == "__main__":
    run()
