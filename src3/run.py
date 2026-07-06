# supress warnings
import warnings, logging
warnings.filterwarnings("ignore", message=r"The pynvml package is deprecated.*", category=FutureWarning)
# only log errors from this file in order to suppress the warning "Redirects are currently not supported in Windows or MacOs."
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import sys, gc, argparse
from pathlib import Path

# src3 is a scripts dir (bare imports). On Modal it is mounted at /root/src3 -> put it on the path
# so `from model import ...` resolves both locally (run from src3/) and remotely.
try:
    if Path("/root/src3").exists() and "/root/src3" not in sys.path: sys.path.insert(0, "/root/src3")
except PermissionError:
    pass

import torch
import modal
from composer.utils.reproducibility import seed_all
from composer import Trainer
from composer.core import Evaluator
from composer.profiler import JSONTraceHandler, cyclic_schedule
from composer.profiler.profiler import Profiler
from composer.loggers import WandBLogger, FileLogger
from composer.callbacks import RuntimeEstimator, SpeedMonitor, OOMObserver, NaNMonitor, SystemMetricsMonitor
from icecream import install; install()
import wandb

from callbacks import MaskVisualizer, OutputSaver
from dataset import build_dataset
from model import VibrationTransformer

def get_parser():
    parser = argparse.ArgumentParser()
    # system
    parser.add_argument("--seed",                       type=int,   default=42)
    parser.add_argument("--num-workers",                type=int,   default=4)
    parser.add_argument("--debug",                      type=int,   default=0)
    parser.add_argument("--no-train",                   action="store_true", default=False)
    parser.add_argument("--no-eval",                    action="store_true", default=False)
    # data
    parser.add_argument("--n-samples",                  type=int,   default=None)
    parser.add_argument("--mds-path",                   type=str,   default=r"D:/eturok/experiment-22/data/mds/3dd5526e5199d80d")
    parser.add_argument("--test-size",                  type=float, default=0.2)
    parser.add_argument("--out-h",                      type=int,   default=18)
    parser.add_argument("--out-w",                      type=int,   default=44)
    parser.add_argument("--n-laser-rows",               type=int,   default=10)
    parser.add_argument("--n-laser-cols",               type=int,   default=10)
    parser.add_argument("--patch-size",                 type=int,   default=256)
    parser.add_argument("--n-freqs",                    type=int,   default=3328)
    parser.add_argument("--signal-mode",                type=str,   default='magnitude')
    parser.add_argument("--normalize-mode",              type=str,   default='z-global')
    parser.add_argument("--speakers",                   type=int,   default=None)
    parser.add_argument("--n-objects",                  type=int,   default=None)
    parser.add_argument("--box",                        type=str,   default=None)
    parser.add_argument("--dry-run",                    action="store_true", default=False)
    # model
    parser.add_argument("--d-model",                    type=int,   default=128)
    parser.add_argument("--pnt-num-heads",              type=int,   default=2)
    parser.add_argument("--seq-num-heads",              type=int,   default=2)
    parser.add_argument("--pnt-num-layers",             type=int,   default=2)
    parser.add_argument("--seq-num-layers",             type=int,   default=2)
    parser.add_argument("--freq-dropout",               type=float, default=0.5)
    parser.add_argument("--laser-dropout",              type=float, default=0.5)
    # train
    parser.add_argument("--batch-size",                 type=int,   default=64)
    parser.add_argument("--lr",                         type=float, default=1e-4)
    parser.add_argument("--max-duration",               type=str,   default="2500ep")
    # eval
    parser.add_argument("--eval-batch-size",            type=int,   default=64)
    parser.add_argument("--eval-interval",              type=str,   default="50ep")
    parser.add_argument("--checkpoint-path",            type=str,   default=None, help="Checkpoint to load for eval. If not set, defaults to the run's latest checkpoint.")
    parser.add_argument("--forward-outputs-dir",        type=str,   default=None, help="Where to save eval .pt outputs. If not set, defaults to the run's forward_outputs dir.")
    # run
    parser.add_argument("--run-name",                   type=str,   default=None)
    # logging
    parser.add_argument("--num-masks-logged",           type=str,   default=500)
    parser.add_argument("--mask-logging-interval",      type=str,   default=None, help="Interval to log masks. If not set, defaults to eval_interval.")
    # checkpointing
    parser.add_argument("--checkpoint-interval",        type=str,   default="500ep")
    parser.add_argument("--remote-checkpoint-folder",   type=str, default="eturok-weizmann/laser-vibrations-checkpoints")
    parser.add_argument("--save-output-interval",       type=str,   default="100ep")
    return parser

# **** Modal ****

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"})
    .uv_pip_install([
        'ipykernel', 'pip', 'ipywidgets', # for notebooks
        'datasets', 'Pillow', 'torchcodec', 'torch>2.10', 'scikit-learn', 'icecream', 'wandb', 'modal', 'pynvml',
        'psutil', # for wandb to properly log the systems pannels (gpu utilization, gpu memory, etc.)
        'mosaicml-streaming', # for streaming dataset
        "git+https://github.com/eitanturok/composer.git@992d49db", # latest commit on my `hf-object-store` branch of composer
        ])
    .add_local_dir("src3", remote_path="/root/src3")
)

app = modal.App(
    image=image,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
)

# **** memory helpers ****

def log_mem(tag):
    import os, psutil
    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / 1e9
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / 1e9
        vram_res   = torch.cuda.memory_reserved()  / 1e9
        print(f'[mem] {tag}: RSS={rss_gb:.2f} GB  VRAM alloc={vram_alloc:.2f} GB  reserved={vram_res:.2f} GB')
    else:
        print(f'[mem] {tag}: RSS={rss_gb:.2f} GB')

def cleanup(trainer, *others):
    log_mem('before close')
    trainer.close()
    trainer.state.model.cpu()
    trainer.state.outputs = None
    trainer.state.batch   = None
    del trainer, others
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    log_mem('after close')

# **** train / eval ****

def run_train(args, device, model, train_loader, eval_loader, data_info):
    """Fit the model. Returns the trained model (moved back to cpu)."""
    optimizer = torch.optim.Adam(model.parameters(), args.lr, fused=True)

    config = data_info | args.__dict__ | dict(gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", num_parameters=sum([p_.numel() for p_ in model.parameters()]))
    wandb_logger = WandBLogger("better-tsa", group="metal", name=args.run_name, init_kwargs={"settings": wandb.Settings(x_disable_stats=False), "config": config, "save_code": True, "id": args.run_name, "resume": "allow"})
    file_logger = FileLogger(f"runs/{{run_name}}/logs-rank{{rank}}.txt")
    loggers = [wandb_logger, file_logger]

    profiler = None
    if args.debug > 0:
        profiler = Profiler(
            trace_handlers=[JSONTraceHandler(folder=f"runs/{{run_name}}/composer_profiler", merged_trace_filename=f"runs/{{run_name}}/merged_trace_node{{node_rank}}.json",
                                            overwrite=True)],
            schedule=cyclic_schedule(wait=0, warmup=0, active=1, repeat=1),
            torch_prof_folder=f"runs/{{run_name}}/torch_profiler", torch_prof_remote_file_name=f"runs/{{run_name}}/torch_profiler",
            torch_prof_overwrite=True, torch_prof_memory_filename=None,
            )

    callbacks=[SpeedMonitor(1), OOMObserver(folder=f"runs/{{run_name}}/torch_traces", remote_file_name=None), NaNMonitor(), RuntimeEstimator(time_unit="minutes"), SystemMetricsMonitor(),
               MaskVisualizer(args.num_masks_logged, args.mask_logging_interval or args.eval_interval),
               ]
    trainer = Trainer(run_name=args.run_name, model=model, optimizers=optimizer, train_dataloader=train_loader, auto_log_hparams=False,
                    eval_dataloader=eval_loader, max_duration=args.max_duration, seed=args.seed, eval_interval=args.eval_interval,
                    device=device, save_metrics=True, log_to_console=True, progress_bar=False,
                    autoresume=True if args.run_name else None, save_folder=f"runs/{{run_name}}/checkpoints", save_interval=args.checkpoint_interval,
                    loggers=loggers, callbacks=callbacks, profiler=profiler)

    trainer.fit()
    run_name = trainer.state.run_name
    trainer.close()
    return model, run_name

def run_eval(model, train_loader, eval_loader, forward_outputs_dir, load_path=None):
    """Run a full eval pass and save .pt outputs. Returns forward_outputs_dir."""
    forward_outputs_dir = Path(forward_outputs_dir)
    forward_outputs_dir.mkdir(parents=True, exist_ok=True)

    data_loaders = eval_loader + [Evaluator(label='train', dataloader=train_loader)]
    trainer = Trainer(model=model, load_path=load_path, callbacks=OutputSaver('1ep', str(forward_outputs_dir), overwrite=True), progress_bar=False, log_to_console=True)
    trainer.eval(data_loaders)
    cleanup(trainer, data_loaders, eval_loader, train_loader)
    return forward_outputs_dir

@app.function(
    gpu="A10",
    timeout=86_400,  # maximum timeout is 24 hours or 86_400 seconds; see https://modal.com/docs/guide/timeouts#timeouts
    retries=0,
)
def run(**kwargs):

    # parse args
    args = get_parser().parse_args()  # get defaults
    args.__dict__.update(kwargs)  # apply overrides from cli
    assert not args.no_train or args.checkpoint_path, "--checkpoint-path is required when using --no-train"
    assert not args.no_eval or not args.no_train, "nothing to do: both --no-train and --no-eval were passed"

    # set device
    device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using {device=}')

    # set seeds for reproducibility before initializing model + dataloader
    seed_all(args.seed)
    print(f"Set random seed to {args.seed} for reproducibility")

    # make model
    data_info = dict(out_h=args.out_h, out_w=args.out_w, n_laser_rows=args.n_laser_rows, n_laser_cols=args.n_laser_cols, patch_size=args.patch_size, n_freqs=args.n_freqs)
    model = VibrationTransformer(args.d_model, args.pnt_num_heads, args.pnt_num_layers, args.seq_num_heads, args.seq_num_layers, data_info,
                                  freq_dropout=args.freq_dropout, laser_dropout=args.laser_dropout)
    run_name = args.run_name

    # train
    if not args.no_train:
        train_loader, eval_loader = build_dataset(
            args.mds_path, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
            test_size=args.test_size, seed=args.seed, num_workers=args.num_workers,
            speakers=args.speakers, n_objects=args.n_objects, box=args.box, n_samples=args.n_samples)
        model, run_name = run_train(args, device, model, train_loader, eval_loader, data_info)

    # eval
    if not args.no_eval:
        model.save_logits = True
        load_path = str(args.checkpoint_path) if args.no_train else None
        forward_outputs_dir = args.forward_outputs_dir or f"runs/{run_name}/forward_outputs"
        train_loader, eval_loader = build_dataset(
            args.mds_path, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
            test_size=args.test_size, seed=args.seed, num_workers=args.num_workers,
            speakers=args.speakers, n_objects=args.n_objects, box=args.box, n_samples=args.n_samples)
        run_eval(model, train_loader, eval_loader, forward_outputs_dir, load_path=load_path)

@app.local_entrypoint()
def main(*args):
    run.remote(**vars(get_parser().parse_args(args)))  # runs on Modal GPU

if __name__ == "__main__":
    run.local(**vars(get_parser().parse_args()))
