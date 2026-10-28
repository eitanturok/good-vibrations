"""train one of the src9 models (arch/) on the src9 split:
  .venv/bin/python src9/run.py --model hybrid --run-name s9-hybrid-500ep
  .venv/bin/python src9/run.py --model boombox --run-name s9-boombox-3k
  .venv/bin/python src9/run.py --model transformer --run-name s9-transformer-500ep
every default is the model's own (DEFAULTS); any can be overridden, e.g. --max-duration 500ep"""
import os, subprocess, sys
if os.environ.get("PYTHONUTF8") != "1": sys.exit(subprocess.run([sys.executable] + sys.argv, env=dict(os.environ, PYTHONUTF8="1")).returncode)

import warnings, logging
warnings.filterwarnings("ignore", message=r"The pynvml package is deprecated.*", category=FutureWarning)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import argparse, json

import torch
from composer import Trainer
from composer.callbacks import LRMonitor, SpeedMonitor, NaNMonitor, RuntimeEstimator, OptimizerMonitor
from composer.loggers import WandBLogger, FileLogger
from composer.optim import CosineAnnealingWithWarmupScheduler
from composer.utils.reproducibility import seed_all

from dataset import CATEGORIES, NOT_OOD, build_dataset
from callbacks import VisualizeSMask, OutputSaver, PositionLogger, EvalSummary
from arch.boombox import Boombox
from arch.hybrid import Hybrid
from arch.transformer import Transformer

# boombox: s6-gastro-plastic-cardboard-3k (hann 04_ffts, 3000 ep, 100 ep warmup). hybrid: s8-mag-nopose-imom-region-500ep (tukey, 500 ep, 50 ep warmup, microbatch 512)
COMMON = dict(seed=42, batch_size=1024, microbatch=0, lr=1e-4, weight_decay=0.01, eval_batch_size=108, eval_interval="50ep", image_interval="100ep",
              num_workers=4, compile=1, wandb=1, wandb_group="src9")
AUGS = dict(speaker_gain=0.35, speaker_gain_hz=20.0, laser_jitter=0.5)  # train-time augs (utils.py): log-gain sd, knot spacing in Hz, max laser shift in cells
DEFAULTS = {"boombox": COMMON | dict(window="hann", t_warmup="100ep", max_duration="3000ep", checkpoint_interval="500ep"),
            "hybrid":  COMMON | dict(window="tukey", t_warmup="50ep", max_duration="500ep", checkpoint_interval="250ep", microbatch=512, **AUGS),
            "transformer": COMMON | dict(window="tukey", t_warmup="50ep", max_duration="500ep", checkpoint_interval="250ep", microbatch=256, **AUGS)}

def run():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=tuple(DEFAULTS), required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--eval-only", type=int, default=0)
    p.add_argument("--checkpoint-path", default=None)
    for k, v in DEFAULTS["hybrid"].items(): p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=None)  # None = the model's default
    args = p.parse_args()
    for k, v in DEFAULTS[args.model].items():
        if getattr(args, k) is None: setattr(args, k, v)
    assert not args.eval_only or args.checkpoint_path, "--eval-only needs --checkpoint-path"

    seed_all(args.seed)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    if device == "gpu": torch.set_float32_matmul_precision("high"); torch.backends.cudnn.benchmark = True

    train_dl, eval_dl, train_eval, info = build_dataset(args.window, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, num_workers=args.num_workers)
    augs = {} if args.model == "boombox" else {k: getattr(args, k) for k in AUGS}
    model = {"boombox": Boombox, "hybrid": Hybrid, "transformer": Transformer}[args.model](info, **augs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params:,} parameters")
    if args.run_name: Path(f"runs/{args.run_name}").mkdir(parents=True, exist_ok=True); Path(f"runs/{args.run_name}/args.json").write_text(json.dumps(vars(args), indent=1))

    loggers = [FileLogger("runs/{run_name}/logs-rank{rank}.txt")]
    if args.wandb and not args.eval_only:
        loggers.append(WandBLogger("better-tsa", group=args.wandb_group, name=args.run_name,
                                   init_kwargs={"config": vars(args) | info | dict(num_parameters=n_params), "id": args.run_name, "resume": "allow"}))
    callbacks = [LRMonitor(), SpeedMonitor(), RuntimeEstimator(skip_batches=64, time_unit="minutes"), NaNMonitor(), OptimizerMonitor(batch_log_interval=10),
                 VisualizeSMask(args.image_interval), OutputSaver(args.eval_interval, info["objects"]), PositionLogger(), EvalSummary(CATEGORIES, NOT_OOD)]

    trainer = Trainer(
        run_name=args.run_name, model=model, optimizers=torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay),
        schedulers=None if args.eval_only else CosineAnnealingWithWarmupScheduler(t_warmup=args.t_warmup),
        device_train_microbatch_size=args.microbatch or None, train_dataloader=train_dl, eval_dataloader=eval_dl, eval_interval=args.eval_interval,
        max_duration=None if args.eval_only else args.max_duration, seed=args.seed, device=device, precision="amp_bf16" if device == "gpu" else "fp32",
        loggers=loggers, callbacks=callbacks, progress_bar=False, log_to_console=True,
        load_path=args.checkpoint_path, load_weights_only=bool(args.eval_only), autoresume=bool(args.run_name) and not args.eval_only,
        save_folder=None if args.eval_only else "runs/{run_name}/checkpoints", save_interval=args.checkpoint_interval,
        compile_config={"mode": "default"} if args.compile else None,
    )
    trainer.eval(eval_dl + [train_eval])
    if not args.eval_only:
        trainer.fit()
        trainer.eval(eval_dl + [train_eval])

if __name__ == "__main__":
    run()
