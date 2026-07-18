# supress warnings
import warnings, logging
warnings.filterwarnings("ignore", message=r"The pynvml package is deprecated.*", category=FutureWarning)
# only log errors from this file in order to suppress the warning "Redirects are currently not supported in Windows or MacOs."
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import sys, gc, argparse
from pathlib import Path

# src is a scripts dir (bare imports). On Modal it is mounted at /root/src -> put it on the path
# so `from model import ...` resolves both locally (run from src/) and remotely.
try:
    if Path("/root/src").exists() and "/root/src" not in sys.path: sys.path.insert(0, "/root/src")
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
from composer.callbacks import RuntimeEstimator, SpeedMonitor, OOMObserver, NaNMonitor, SystemMetricsMonitor, OptimizerMonitor
from composer.callbacks.speed_monitor import GPU_AVAILABLE_FLOPS

# RTX 5080 isn't in composer's GPU_AVAILABLE_FLOPS table yet, so MFU can't be computed. Patch it in here
GPU_AVAILABLE_FLOPS['nvidia geforce rtx 5080'] = {
    'fp32': 56.3e12,
    'tf32': 56.3e12,
    'fp16': 225.1e12,
    'amp_fp16': 225.1e12,
    'bf16': 225.1e12,
    'amp_bf16': 225.1e12,
    'fp8': 450.2e12,
    'amp_fp8': 450.2e12,
    'int8': 900.4e12,
}
from icecream import install; install()
import wandb

from model.callbacks import VizSegMask, OutputSaver
from model.dataset import build_dataset
from model.arch import VibrationTransformer, LOSSES

def get_parser():
    parser = argparse.ArgumentParser()
    # system
    parser.add_argument("--seed",                       type=int,   default=42)
    parser.add_argument("--debug",                      type=int,   default=0)
    parser.add_argument("--verbose",                    type=int,   default=2, help="If >=2, show torch.compile (TorchDynamo) logs.")
    # build data
    parser.add_argument("--mds-dir",                    type=str,   default=r"D:/eturok/datasets/000-cylinder-dataset/mds")
    parser.add_argument("--split",                      type=str,   default="exp23", help="Which split method from SPLIT_METHODS to use (e.g. 'exp22', 'exp23').")
    parser.add_argument("--num-workers",                type=int,   default=4)
    parser.add_argument("--test-size",                  type=float, default=0.2)
    parser.add_argument("--out-h",                      type=int,   default=20)
    parser.add_argument("--out-w",                      type=int,   default=40)
    parser.add_argument("--n-laser-rows",               type=int,   default=10)
    parser.add_argument("--n-laser-cols",               type=int,   default=10)
    parser.add_argument("--patch-size",                 type=int,   default=256)
    parser.add_argument("--n-freqs",                    type=int,   default=3328)
    parser.add_argument("--n-channels",                 type=int,   default=2, help="Last dim of X: 2 for magnitude, 4 for complex/mag_phase signal modes.")
    # filter data
    parser.add_argument("--n-samples",                  type=int,   default=None)
    parser.add_argument("--speakers",                   type=int,   default=None)
    parser.add_argument("--n-objects",                  type=int,   default=None)
    parser.add_argument("--box",                        type=str,   default=None)
    # model
    parser.add_argument("--decoder",                     type=str,   default='mlp')
    parser.add_argument("--decoder-num-heads",          type=int,   default=2)
    parser.add_argument("--decoder-num-layers",         type=int,   default=2)
    parser.add_argument("--loss-fn",                    type=str,   default='mse', choices=list(LOSSES))
    parser.add_argument("--d-model",                    type=int,   default=128)
    parser.add_argument("--pnt-num-heads",              type=int,   default=2)
    parser.add_argument("--seq-num-heads",              type=int,   default=2)
    parser.add_argument("--pnt-num-layers",             type=int,   default=2)
    parser.add_argument("--seq-num-layers",             type=int,   default=2)
    parser.add_argument("--freq-dropout",               type=float, default=0.3)
    parser.add_argument("--laser-dropout",              type=float, default=0.3)
    parser.add_argument("--no-compile",                 action="store_true", default=False, help="Disable torch.compile-ing the model before training/eval.")
    parser.add_argument("--compile-mode",               type=str,   default="default", help="torch.compile mode, e.g. 'default', 'reduce-overhead', 'max-autotune'.")
    # train
    parser.add_argument("--batch-size",                 type=int,   default=128)
    parser.add_argument("--lr",                         type=float, default=1e-4)
    parser.add_argument("--max-duration",               type=str,   default="2500ep")
    # eval
    parser.add_argument("--eval-only",                  action="store_true", default=False, help="Skip training, just eval a loaded checkpoint (requires --checkpoint-path).")
    parser.add_argument("--eval-batch-size",            type=int,   default=108) # wandb caps images logged in a single call to 108, so eval batch size should be <= 108 to log all images
    parser.add_argument("--eval-interval",              type=str,   default="50ep")
    parser.add_argument("--outputs-dir",                type=str,   default=None, help="Where to save eval .pt outputs. If not set, defaults to the run's outputs_history dir (same dir the training-time history callback writes to).")
    # run
    parser.add_argument("--run-name",                   type=str,   default=None)
    parser.add_argument("--dry-run",                    action="store_true", default=False)
    # checkpointing
    parser.add_argument("--checkpoint-path",            type=str,   default=None, help="Checkpoint to load for eval. If not set, defaults to the run's latest checkpoint.")
    parser.add_argument("--checkpoint-interval",        type=str,   default="500ep")
    parser.add_argument("--remote-checkpoint-folder",   type=str, default="eturok-weizmann/laser-vibrations-checkpoints")
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
    .add_local_dir("src", remote_path="/root/src")
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

def eval_boundary(trainer, boundary_loaders):
    saver = next(cb for cb in trainer.state.callbacks if isinstance(cb, OutputSaver))
    saver.force_save = True
    try: trainer.eval(boundary_loaders)
    finally: saver.force_save = False

@app.function(
    gpu="A10",
    timeout=86_400,  # maximum timeout is 24 hours or 86_400 seconds; see https://modal.com/docs/guide/timeouts#timeouts
    retries=0,
)
def run(**kwargs):

    # parse args
    args = get_parser().parse_args()  # get defaults
    args.__dict__.update(kwargs)  # apply overrides from cli
    assert not args.eval_only or args.checkpoint_path, "--checkpoint-path is required when using --eval-only"

    # set seeds for reproducibility BEFORE initializing model + dataloader
    seed_all(args.seed)
    print(f"Set random seed to {args.seed} for reproducibility")

    # device
    device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using {device=}')

    # set torch compile and cudnn benchmark for speed
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    if not args.no_compile and args.verbose >= 2: torch._logging.set_logs(dynamo=logging.INFO)

    # model
    data_info = dict(out_h=args.out_h, out_w=args.out_w, n_laser_rows=args.n_laser_rows, n_laser_cols=args.n_laser_cols, patch_size=args.patch_size, n_freqs=args.n_freqs, n_channels=args.n_channels)
    model = VibrationTransformer(args.d_model, args.pnt_num_heads, args.pnt_num_layers, args.seq_num_heads, args.seq_num_layers, data_info, args.decoder, args.decoder_num_heads, args.decoder_num_layers, freq_dropout=args.freq_dropout, laser_dropout=args.laser_dropout, loss_fn=args.loss_fn)
    load_path = str(args.checkpoint_path) if args.checkpoint_path else None

    # dataset
    train_loader, eval_loader = build_dataset(
        args.mds_dir, split=args.split, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        test_size=args.test_size, seed=args.seed, num_workers=args.num_workers,
        speakers=args.speakers, n_objects=args.n_objects, box=args.box, n_samples=args.n_samples)
    boundary_loaders = eval_loader + [Evaluator(label='train', dataloader=train_loader)]

    # logger
    loggers = []
    if not args.eval_only:
        config = data_info | args.__dict__ | dict(gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", num_parameters=sum([p_.numel() for p_ in model.parameters()]))
        wandb_logger = WandBLogger("better-tsa", group="metal", name=args.run_name, init_kwargs={"settings": wandb.Settings(x_disable_stats=False), "config": config, "save_code": True, "id": args.run_name, "resume": "allow"})
        file_logger = FileLogger(f"runs/{{run_name}}/logs-rank{{rank}}.txt")
        loggers = [wandb_logger, file_logger]

    # profiler
    profiler = None
    if not args.eval_only and args.debug > 0:
        profiler = Profiler(
            trace_handlers=[JSONTraceHandler(folder=f"runs/{{run_name}}/composer_profiler", merged_trace_filename=f"runs/{{run_name}}/merged_trace_node{{node_rank}}.json",
                                            overwrite=True)],
            schedule=cyclic_schedule(wait=0, warmup=0, active=1, repeat=1),
            torch_prof_folder=f"runs/{{run_name}}/torch_profiler", torch_prof_remote_file_name=f"runs/{{run_name}}/torch_profiler",
            torch_prof_overwrite=True, torch_prof_memory_filename=None,
            )

    # callbacks
    callbacks = [OutputSaver(args.eval_interval, f"runs/{{run_name}}/outputs_history", overwrite=True, visualizer=VizSegMask()),
                 SpeedMonitor(1), OOMObserver(folder=f"runs/{{run_name}}/torch_traces", remote_file_name=None), NaNMonitor(),
                RuntimeEstimator(skip_batches=64, time_unit="minutes"), SystemMetricsMonitor(),
                OptimizerMonitor(log_optimizer_metrics=True, batch_log_interval=10)]

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr, fused=True) if not args.eval_only else None

    # trainer
    trainer = Trainer(run_name=args.run_name, model=model, optimizers=optimizer, train_dataloader=train_loader, auto_log_hparams=False,
                    eval_dataloader=eval_loader, max_duration=args.max_duration if not args.eval_only else None, seed=args.seed, eval_interval=args.eval_interval,
                    device=device, save_metrics=True, log_to_console=True, progress_bar=False, load_path=load_path,
                    autoresume=True if not args.eval_only and args.run_name else None, save_folder=f"runs/{{run_name}}/checkpoints" if not args.eval_only else None, save_interval=args.checkpoint_interval,
                    loggers=loggers, callbacks=callbacks, profiler=profiler,
                    compile_config=None if args.no_compile else {"mode": args.compile_mode})

    eval_boundary(trainer, boundary_loaders)  # eval before training starts
    if not args.eval_only:
        trainer.fit()
        eval_boundary(trainer, boundary_loaders)  # eval after training ends
    cleanup(trainer, boundary_loaders, eval_loader, train_loader)

@app.local_entrypoint()
def main(*args):
    run.remote(**vars(get_parser().parse_args(args)))  # runs on Modal GPU

if __name__ == "__main__":
    run.local(**vars(get_parser().parse_args()))
