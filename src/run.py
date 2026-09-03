# patch windows to not log emojis
import os, subprocess, sys
if os.environ.get("PYTHONUTF8") != "1":
    env = dict(os.environ, PYTHONUTF8="1")
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)

# supress warnings
import warnings, logging
warnings.filterwarnings("ignore", message=r"The pynvml package is deprecated.*", category=FutureWarning)
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import sys, argparse
from pathlib import Path

try:
    if Path("/root/src").exists() and "/root/src" not in sys.path: sys.path.insert(0, "/root/src")
except PermissionError:
    pass

import torch
import modal
import wandb
from composer.utils.reproducibility import seed_all
from composer import Trainer
from composer.core import Evaluator
from composer.profiler import JSONTraceHandler, cyclic_schedule
from composer.profiler.profiler import Profiler
from composer.loggers import WandBLogger, FileLogger
from composer.callbacks import RuntimeEstimator, SpeedMonitor, OOMObserver, NaNMonitor, SystemMetricsMonitor, OptimizerMonitor, LRMonitor
from composer.optim import ConstantScheduler, CosineAnnealingScheduler, CosineAnnealingWithWarmupScheduler, LinearWithWarmupScheduler
from composer.callbacks.speed_monitor import GPU_AVAILABLE_FLOPS

from icecream import install; install()

from model.callbacks import VisualizeSMask, OutputSaver, AttributionSaver, OUTPUT_EXTRACTORS, DEFAULT_OUTPUT_KEYS
from model.dataset import build_dataset
from model.arch import VibrationTransformer, LOSSES, N_COUNT_CLASSES
from model.boombox import BoomboxModel
from utils.helpers import cleanup
from viz import ensure_viz

BASE_DATA_DIR = Path("/home/ethantu/workspace/good-vibrations/experiments")

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
    parser.add_argument("--data-dir",                    type=str,  default=BASE_DATA_DIR / "31_07_2026_gastronorm_exp1")
    parser.add_argument("--split",                      type=str,   default="gastronorm", help="Which split method from SPLIT_METHODS to use (e.g. 'exp22', 'exp23').")
    parser.add_argument("--num-workers",                type=int,   default=4)
    parser.add_argument("--test-size",                  type=float, default=0.2)
    parser.add_argument("--laser-cols",                 type=str,   default=None, help="Comma-separated laser column ids to train on, e.g. '0,1,2,3,4'. Whole columns across every row, so the kept lasers stay a rectangle. Default None = all. The selection is applied where the fft is read off disk, so every normalization and reference statistic is computed over exactly these lasers -- which means each selection gets its own MDS build.")
    parser.add_argument("--patch-size",                 type=int,   default=32)

    # data normalization
    parser.add_argument("--encoder",                    type=str,   default="single", choices=["single", "two-stream"], help="boombox only. 'single' fuses magnitude and phase in the first conv's weights (the v2 encoder). 'two-stream' gives each its own full-width frequency stack and fuses after it, at ~+7%% params.")
    parser.add_argument("--fuse",                       type=str,   default="concat", choices=["concat", "gate"], help="--encoder two-stream only. 'concat' stacks the two streams; 'gate' admits phase through a sigmoid gate driven by the magnitude stream, biased shut at init so training starts near the no-phase model.")
    parser.add_argument("--trim-pad",                   action="store_true", help="boombox only. Drop tokenize()'s trailing zero-pad before the convs (45/1280 bins at patch_size 64, 211/3072 at 256). The pad is harmless to the transformer, which sees it at fixed positions, but a conv slides over it and then averages it in.")
    parser.add_argument("--learned-collapse",           action="store_true", help="boombox only. Replace the frequency stack's AdaptiveAvgPool with a learned (1,width) conv, so WHERE in the spectrum a filter fired survives the collapse instead of being averaged out. +328K params.")
    parser.add_argument("--freq-mult",                  type=int,   default=1, help="boombox only. Scale every frequency-stack width (32/64/128/256 -> x mult). NOTE this also scales the grid stack's input, so params grow well beyond the frequency stack itself.")
    parser.add_argument("--freq-depth",                 type=int,   default=1, help="boombox only. Stride-1 (1,3) blocks per stage, adding depth at each frequency scale. Unlike --freq-mult the output width is unchanged, so the grid stack costs nothing extra.")
    parser.add_argument("--resize",                     type=str,   default="conv", choices=["conv", "bilinear"], help="boombox decoder. 'conv' absorbs the 24x32 -> out_h x out_w margin in a valid-mode head conv (no interpolation); 'bilinear' is the original resize, needed to load older checkpoints.")
    parser.add_argument("--signal-mode",                type=str,   default="magnitude", choices=["magnitude", "log_magnitude", "complex", "mag_phase", "mag_trig_phase"])
    parser.add_argument("--normalize-mode",             type=str,   default="std", help="Per-sample: std, z, per_laser_z. Train-split statistics: per_bin_z. Append '+token-mean' for token-level normalization.")
    parser.add_argument("--augment-mask",               type=float, default=0.5, help="Probability a sample gets mask augmentation (blur+noise). 0 disables.")
    parser.add_argument("--rgb",                        type=int,   default=0, choices=(0, 1), nargs="?", const=1, help="Predict the downsampled (out_h,out_w,3) overhead photo instead of the segmentation mask. Bare flag means 1. Forces --loss-fn mse/ce-pixel, disables --augment-mask, and drops the mask-only metrics (soft-iou, com-distance, ...). Not supported by --attribution or the viz/ dashboard.")
    parser.add_argument("--augment-fft",                type=float, default=0, help="Probability a sample gets FFT frequency-gain augmentation. 0 disables.")
    parser.add_argument("--subtract-speaker-mean",      type=int,   default=0, choices=(0, 1), nargs="?", const=1, help="Subtract each speaker's offline mean spectrum (computed over train samples only) after the magnitude and before normalization. Bare flag means 1. Requires --signal-mode magnitude or log_magnitude.")
    parser.add_argument("--mag-recipe",                 type=str,   default=None, help="One of the 14 magnitude arms in normalizations MAG_RECIPES")
    parser.add_argument("--phase-arm",                  type=str,   default=None, help="Concatenate phase channels onto the magnitude block. One of normalizations PHASE_ARMS: rel_laser[_w], rel_laser_med[_w] (median laser reference instead of mean), group_delay[_w], both[_w], raw_phase. Orthogonal to --signal-mode: phase is read off the complex fft directly. Doubles/quadruples n_channels, which the model picks up automatically.")
    parser.add_argument("--phase-weight",               type=float, default=1.0, help="Scale on the phase block relative to the std-normalized magnitude block. Only meaningful with --phase-arm.")
    parser.add_argument("--subtract-empty-box",         type=int,   default=0, choices=(0, 1), nargs="?", const=1, help="Subtract each speaker's mean empty-box spectrum, i.e. divide out the box's own transfer function. Requires --signal-mode log_magnitude, where subtracting a log reference IS dividing by it (safe at anti-resonances, unlike linear division). Combines with --subtract-speaker-mean: the empty-box reference is applied first and the speaker mean is then computed on referenced signal. Bare flag means 1.")
    parser.add_argument("--force-rebuild-data",         type=int,   default=0, choices=(0, 1), nargs="?", const=1, help="Discard the cached MDS and rebuild it, re-running mask downsampling and fft precomputation. Bare flag means 1. Needed when the preprocessing code changes, since the cache key only covers the config, not the code.")
    parser.add_argument("--pair-speakers",              type=int,   default=0, choices=(0, 1), nargs="?", const=1, help="Make one sample out of two speakers' captures of the same scene, stacked on the channel axis. The partner is fixed: 1<->3, 2<->4, 7<->5, 8<->6. Doubles n_channels; lasers and freqs are unchanged.")

    # filter data
    parser.add_argument("--n-samples",                  type=int,   default=None)
    parser.add_argument("--speakers",                   type=int,   default=None)
    parser.add_argument("--n-objects",                  type=int,   default=None)
    parser.add_argument("--box",                        type=str,   default=None)

    # arch
    parser.add_argument("--model",                      type=str,   default="transformer", choices=("transformer", "boombox"), help="transformer = VibrationTransformer; boombox = the conv encoder/decoder from arXiv 2105.08052.")
    parser.add_argument("--d-model",                    type=int,   default=128)
    parser.add_argument("--decoder",                    type=str,   default='mlp', choices=('mlp', 'mlp-mid', 'attn', 'attn-no-rope', 'conv'), help="--model transformer only. 'conv' is boombox's transposed-conv decoder on the transformer's cls token, i.e. the freq+laser transformer encoder with the boombox decoder.")
    parser.add_argument("--decoder-num-heads",          type=int,   default=2)
    parser.add_argument("--decoder-num-layers",         type=int,   default=2)
    parser.add_argument("--ffn-dim",                    type=int,   default=None, help="Width of every transformer FFN (freq encoder, laser encoder, attn decoder). Default None = 4*d_model. Torch's own default is a fixed 2048, so pre-2026-08 runs had a 2048-wide FFN regardless of d_model; pass 2048 to reproduce them. No effect on --model boombox.")
    parser.add_argument("--pnt-num-heads",              type=int,   default=2)
    parser.add_argument("--seq-num-heads",              type=int,   default=2)
    parser.add_argument("--pnt-num-layers",             type=int,   default=2)
    parser.add_argument("--seq-num-layers",             type=int,   default=2)
    parser.add_argument("--freq-dropout",               type=float, default=0.0)
    parser.add_argument("--laser-dropout",              type=float, default=0.0)
    parser.add_argument("--out-h",                      type=int,   default=21)
    parser.add_argument("--out-w",                      type=int,   default=30)

    # train faster
    parser.add_argument("--precision",                  type=str,   default="amp_bf16", choices=["fp32", "amp_fp16", "amp_bf16"], help="bf16 matches fp16's tensor-core throughput on Blackwell but keeps fp32's exponent range, so no loss scaling and no underflow on wide-dynamic-range FFT magnitudes.")
    parser.add_argument("--compile",                    type=int,   default=1, choices=(0, 1), help="torch.compile-ing the model before training/eval.")
    parser.add_argument("--compile-mode",               type=str,   default="default", help="torch.compile mode, e.g. 'default', 'reduce-overhead', 'max-autotune'.")

    # loss
    parser.add_argument("--loss-fn",                    type=str,   default='mse', choices=list(LOSSES))
    parser.add_argument("--loss-alpha",                 type=float, default=0.5, help="for the -asym losses: weight on false negatives; >0.5 paints more, <0.5 holds back")
    parser.add_argument("--count-loss-weight",          type=float, default=0.0, help="weight on the auxiliary n_objects classification loss; 0 disables it (the head still runs, so count-acc stays readable as a free probe).")

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
    parser.add_argument("--output-keys",                type=str,   default=list(DEFAULT_OUTPUT_KEYS), nargs="*", choices=list(OUTPUT_EXTRACTORS), help="Payloads to dump per-batch as .pt in runs/<run>/outputs_history. Pass with no values to skip saving outputs entirely; 'fft' and 'mask_logits' are very large.")
    parser.add_argument("--attribution",                type=int,   default=0, help="Save per-eval-batch laser/freq attribution (cls attention maps) to runs/<run>/attribution.")
    parser.add_argument("--attribution-ablate",         type=int,   default=0, help="Also measure delta-MSE from zeroing each laser and freq patch. Costs n_lasers+n_patches extra forwards on the first eval batch; more trustworthy than attention.")

    # run
    parser.add_argument("--run-name",                   type=str,   default=None)
    parser.add_argument("--wandb-group",                type=str,   default="attn-lr-sweep", help="wandb group, for keeping sweep runs together.")
    parser.add_argument("--viz-port",                   type=int,   default=8504, help="Port for the auto-launched viz dashboard.")
    parser.add_argument("--no-viz",                     action="store_true", help="Don't auto-launch the viz dashboard.")

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

# **** train / eval ****

def eval_boundary(trainer, boundary_loaders):
    # OutputSaver is opt-in, so force-save whichever interval callbacks are actually installed
    cbs = [cb for cb in trainer.state.callbacks if isinstance(cb, (OutputSaver, VisualizeSMask))]
    for cb in cbs: cb.force_save = True
    try: trainer.eval(boundary_loaders)
    finally:
        for cb in cbs: cb.force_save = False

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
        # covers the ops autocast keeps in fp32 (and cudnn), which set_float32_matmul_precision doesn't
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if args.compile and args.verbose >= 2: torch._logging.set_logs(dynamo=logging.INFO)

    # dataset
    laser_cols = [int(c) for c in args.laser_cols.split(",") if c.strip()] if args.laser_cols else None
    train_loader, eval_loaders, train_eval_loader = build_dataset(
        args.data_dir, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, num_workers=args.num_workers,
        split=args.split, test_size=args.test_size, speakers=args.speakers, n_objects=args.n_objects, box=args.box, n_samples=args.n_samples,
        out_h=args.out_h, out_w=args.out_w, rgb=bool(args.rgb), signal_mode=args.signal_mode, normalize_mode=args.normalize_mode, patch_size=args.patch_size, seed=args.seed,
        augment_fft=args.augment_fft, augment_mask=args.augment_mask, subtract_speaker_mean=bool(args.subtract_speaker_mean), subtract_empty_box=bool(args.subtract_empty_box), mag_recipe=args.mag_recipe, phase_arm=args.phase_arm, phase_weight=args.phase_weight,
        force_rebuild_data=bool(args.force_rebuild_data), n_classes=N_COUNT_CLASSES, pair_speakers_mode=bool(args.pair_speakers),
        laser_cols=laser_cols)
    boundary_loaders = eval_loaders + [Evaluator(label='train', dataloader=train_eval_loader)]
    ensure_viz(args.data_dir, port=args.viz_port, enabled=not args.no_viz)

    # read n_freqs, n_channels from the dataset
    n_lasers, n_patches, patch_size, n_channels = train_loader.dataloader.dataset[0]['fft'].shape  # (L,P,PS,C)
    # tokenize() zero-pads the frequency axis, so n_patches * patch_size is the padded length
    base = train_loader.dataloader.dataset.dataset  # Subset -> (PairedSpeakerDataset ->) VibrationDataset
    base = getattr(base, "dataset", base)
    n_freqs_real = len(base.pk["freqs"])
    # the grid the model builds must match the tensor it is fed, so take it from the dataset, which
    # recorded it after the laser selection was applied
    n_laser_rows, n_laser_cols = base.grid_shape
    assert n_laser_rows * n_laser_cols == n_lasers, f"grid {base.grid_shape} != {n_lasers} lasers"
    data_info = dict(out_h=args.out_h, out_w=args.out_w, out_c=3 if args.rgb else 1, n_laser_rows=n_laser_rows, n_laser_cols=n_laser_cols, patch_size=args.patch_size, n_freqs=n_patches * patch_size, n_freqs_real=n_freqs_real, n_channels=n_channels)
    print(f"laser grid: {n_laser_rows} rows x {n_laser_cols} cols = {n_lasers} lasers")
    print(f"{n_freqs_real} freq bins -> {n_patches} patches of {patch_size} = {n_patches * patch_size} ({n_patches * patch_size - n_freqs_real} padded)")
    if args.model == "boombox":
        model = BoomboxModel(args.d_model, data_info, loss_fn=args.loss_fn, loss_alpha=args.loss_alpha, count_loss_weight=args.count_loss_weight, freq_dropout=args.freq_dropout, laser_dropout=args.laser_dropout, encoder=args.encoder, fuse=args.fuse, trim_pad=args.trim_pad, learned_collapse=args.learned_collapse, freq_mult=args.freq_mult, freq_depth=args.freq_depth, resize=args.resize)
    else:
        model = VibrationTransformer(args.d_model, args.pnt_num_heads, args.pnt_num_layers, args.seq_num_heads, args.seq_num_layers, data_info, args.decoder, args.decoder_num_heads, args.decoder_num_layers, freq_dropout=args.freq_dropout, laser_dropout=args.laser_dropout, loss_fn=args.loss_fn, loss_alpha=args.loss_alpha, count_loss_weight=args.count_loss_weight, ffn_dim=args.ffn_dim)
    load_path = str(args.checkpoint_path) if args.checkpoint_path else None

    # logger
    loggers = []
    if not args.eval_only:
        config = data_info | args.__dict__ | dict(gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", num_parameters=sum([p_.numel() for p_ in model.parameters()]))
        wandb_logger = WandBLogger("better-tsa", group=args.wandb_group, name=args.run_name, init_kwargs={"settings": wandb.Settings(x_disable_stats=False), "config": config, "save_code": True, "id": args.run_name, "resume": "allow"})
        file_logger = FileLogger(f"runs/{{run_name}}/logs-rank{{rank}}.txt")
        loggers = [wandb_logger, file_logger]

    # profiler
    profiler = None
    if not args.eval_only and args.debug > 0:
        profiler = Profiler(
            trace_handlers=[JSONTraceHandler(folder=f"runs/{{run_name}}/composer_profiler", merged_trace_filename=f"runs/{{run_name}}/merged_trace_node{{node_rank}}.json", overwrite=True)],
            schedule=cyclic_schedule(wait=1, warmup=1, active=3, repeat=1),
            torch_prof_folder=f"runs/{{run_name}}/torch_profiler", torch_prof_remote_file_name=f"runs/{{run_name}}/torch_profiler",
            torch_prof_overwrite=True, torch_prof_memory_filename=None, torch_prof_with_stack=True,
            )

    # callbacks
    callbacks = [VisualizeSMask(args.viz_interval), NaNMonitor(), LRMonitor(), SystemMetricsMonitor(), SpeedMonitor(1),
                 OOMObserver(folder=f"runs/{{run_name}}/torch_traces", remote_file_name=None), RuntimeEstimator(skip_batches=64, time_unit="minutes"),
                OptimizerMonitor(log_optimizer_metrics=True, batch_log_interval=10)]
    if args.output_keys: callbacks.append(OutputSaver(args.eval_interval, f"runs/{{run_name}}/outputs_history", overwrite=True, output_keys=args.output_keys))
    if args.attribution: callbacks.append(AttributionSaver(args.eval_interval, f"runs/{{run_name}}/attribution", overwrite=True, ablate=bool(args.attribution_ablate)))

    # optimizer + lr schedule
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay, fused=True) if not args.eval_only else None
    schedulers = build_scheduler(args.scheduler, args.t_warmup) if not args.eval_only else None

    # trainer
    trainer = Trainer(run_name=args.run_name, model=model, optimizers=optimizer, train_dataloader=train_loader, auto_log_hparams=False,
                    eval_dataloader=eval_loaders, max_duration=args.max_duration if not args.eval_only else None, seed=args.seed, eval_interval=args.eval_interval,
                    device=device, precision=args.precision if device == 'gpu' else 'fp32', save_metrics=True, log_to_console=True, progress_bar=False, load_path=load_path,
                    autoresume=True if not args.eval_only and args.run_name else None, save_folder=f"runs/{{run_name}}/checkpoints" if not args.eval_only else None, save_interval=args.checkpoint_interval,
                    schedulers=schedulers, loggers=loggers, callbacks=callbacks, profiler=profiler,
                    compile_config={"mode": args.compile_mode} if args.compile else None)

    if args.eval_before_train: eval_boundary(trainer, boundary_loaders)  # eval before training starts
    if not args.eval_only:
        trainer.fit()
        if args.eval_after_train: eval_boundary(trainer, boundary_loaders)  # eval after training ends
    cleanup(trainer, boundary_loaders, eval_loaders, train_loader)

@app.local_entrypoint()
def main(*args):
    run.remote(**vars(get_parser().parse_args(args)))  # runs on Modal GPU

if __name__ == "__main__":
    run.local(**vars(get_parser().parse_args()))
