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
from composer.algorithms import GradientClipping
from composer.callbacks import LRMonitor, SpeedMonitor, NaNMonitor, RuntimeEstimator, OptimizerMonitor, OOMObserver
from composer.loggers import WandBLogger, FileLogger
from composer.utils.reproducibility import seed_all

from dataset import DATASETS, CATEGORIES, DELTAS, NOT_OOD, build_dataset
from arch import Boombox
from callbacks import VisualizeSMask, OutputSaver, PositionLogger, ConfidenceLogger, EvalSummary
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
    parser.add_argument("--speakers",                   type=lambda s: s if s == "all" else int(s), default=[1, 3, 5, 7], nargs="+", help="Speakers to train/eval on, e.g. --speakers 1 3 5 7; 'all' = every speaker, none held out.")
    parser.add_argument("--signal",                     type=str,   default="magnitude", choices=("magnitude", "log_magnitude", "sqrt_magnitude", "power", "log_power"))
    parser.add_argument("--laser-norm",                 type=int,   default=0, choices=(0, 1), help="Scale each laser's |fft| to rms 1 before --signal: removes per-laser gain.")
    parser.add_argument("--norm",                       type=str,   default="std", choices=("std", "z"))
    parser.add_argument("--phase",                      type=str,   default="none", choices=("none", "cossin"), help="cossin = add cos and sin of each channel's phase as input channels (C -> 3C), after --phase-ref.")
    parser.add_argument("--phase-ref",                  type=str,   default="none", choices=("none", "avg", "real"), help="Rotate every laser's phase, per sample/freq/channel, so this reference reads 0: avg = mean over lasers; real = 1/2 arg sum_l x^2 (most energy in the real part).")
    parser.add_argument("--window",                     type=str,   default="hann", choices=("hann", "tukey", "deconv"), help="hann = the stored 04_ffts (hann over the whole capture); tukey = recompute from the raw shifts with 13ms tapers; deconv = tukey / the chirp's spectrum.")
    parser.add_argument("--gate-ms",                    type=float, default=0, help="--window tukey|deconv: > 0 = deconvolve by the chirp and keep this many ms of the impulse response.")
    parser.add_argument("--freq-gain",                  type=int,   default=0, choices=(0, 1), help="FreqEncoder: learned smooth per-frequency input gain exp(mlp(log f)), starts at 1.")
    parser.add_argument("--freq-coord",                 type=int,   default=0, choices=(0, 1), help="FreqEncoder: add log10(f) as an input channel (CoordConv).")
    parser.add_argument("--freq-pool",                  type=str,   default="mean", choices=("mean", "flatten"), help="FreqEncoder: average the last 5 freq positions, or flatten them into a linear layer.")
    parser.add_argument("--patch-size",                 type=lambda s: None if s.lower() == "none" else int(s), default=None, help="Freq bins per token; 'none' = no patching, x is (L,F,C).")
    parser.add_argument("--out-h",                      type=int,   default=64)
    parser.add_argument("--out-w",                      type=int,   default=64)
    parser.add_argument("--rgb",                        type=int,   default=0, choices=(0, 1), help="Predict the overhead photo instead of the mask.")
    parser.add_argument("--num-workers",                type=int,   default=4)

    # train
    parser.add_argument("--batch-size",                 type=int,   default=128)
    parser.add_argument("--lr",                         type=float, default=1e-4)
    parser.add_argument("--grad-clip",                  type=float, default=0.0, help="Clip the global gradient norm to this; 0 = off.")
    parser.add_argument("--weight-decay",               type=float, default=1e-2)
    parser.add_argument("--scheduler",                  type=str,   default="cosine-warmup", choices=tuple(SCHEDULERS), help="LR schedule. 'constant' reproduces the old no-scheduler behavior.")
    parser.add_argument("--t-warmup",                   type=str,   default="100ep", help="Warmup length for the *-warmup schedulers; ignored by 'constant'.")
    parser.add_argument("--max-duration",               type=str,   default="1000ep")

    # eval
    parser.add_argument("--eval-only",                  type=int,   default=0, choices=(0, 1), help="Skip training, just eval a loaded checkpoint (requires --checkpoint-path).")
    parser.add_argument("--eval-batch-size",            type=int,   default=108) # wandb caps images logged in a single call to 108, so eval batch size should be <= 108 to log all images
    parser.add_argument("--eval-interval",              type=str,   default="50ep", help="How often to eval and save outputs for viz.")
    parser.add_argument("--image-interval",             type=str,   default="100ep", help="How often to log predicted-vs-true images to wandb; a multiple of --eval-interval.")
    parser.add_argument("--wandb-images",               type=int,   default=16, help="Images logged per loader each --image-interval (first batch only); 0 = none.")
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
    parser.add_argument("--encoder",                    type=str,   default="conv", choices=("conv", "transformer"), help="conv = boombox's freq convs + laser-grid convs; transformer = src's VibrationTransformer encoder (freq transformer per laser, then a laser transformer), needs --patch-size.")
    parser.add_argument("--decoder",                    type=str,   default="conv", choices=("conv", "mlp"), help="conv = boombox's upsampling decoder; mlp = src's default MLP decoder (d_model -> 256 -> every pixel).")
    parser.add_argument("--heads",                      type=int,   default=2, help="--encoder transformer: attention heads.")
    parser.add_argument("--freq-layers",                type=int,   default=2, help="--encoder transformer: layers of the per-laser freq transformer.")
    parser.add_argument("--laser-layers",               type=int,   default=2, help="--encoder transformer: layers of the laser transformer.")
    parser.add_argument("--ffn-dim",                    type=int,   default=None, help="--encoder transformer: FFN width; default 4*d_model.")

    # loss
    parser.add_argument("--loss-fn",                    type=str,   default="bce", choices=("bce", "mse", "dice", "bce-dice"), help="Mask loss.")
    parser.add_argument("--alpha",                      type=float, default=0.5, help="bce-dice only: alpha * bce + (1 - alpha) * dice.")
    parser.add_argument("--object-weight",              type=float, default=None, help="Loss weight of the object-class head; 0 = off. Default 0.1 (mask head), 0 (polar head).")
    parser.add_argument("--n-objects-weight",           type=float, default=None, help="Loss weight of the object-count head; 0 = off. Default 0.1 (mask head), 0 (polar head).")
    parser.add_argument("--com-weight",                 type=float, default=None, help="Loss weight of the per-object com head; 0 = off. Default 1.0 (mask head), 0 (polar head).")
    parser.add_argument("--area-weight",                type=float, default=None, help="Loss weight of the mask-area head; 0 = off. Default 0.1 (mask head), 0 (polar head).")
    parser.add_argument("--box-weight",                 type=float, default=None, help="Loss weight of the box-type head; 0 = off. Default 0.1 (mask head), 0 (polar head).")
    parser.add_argument("--boundary-weight",            type=float, default=0.0, help="Loss weight of the boundary loss on the mask; 0 = off. 10 ~ bce at 6 px from the edge.")
    parser.add_argument("--edge-width",                 type=int,   default=0, help="0 = the outline is 1 extra channel of the mask head; > 0 = a separate 4-conv outline head this wide.")
    parser.add_argument("--edge-weight",                type=float, default=0.0, help="Loss weight of the outline head (bce + dice on the mask's outline); 0 = off.")


    # polar head: PolarMask (Xie et al. 2020), class-agnostic
    parser.add_argument("--head",                       type=str,   default="mask", choices=("mask", "polar", "slots"), help="mask = the decoder's mask convs + --loss-fn; polar = per pixel center + centerness + rays, masks drawn from the peaks; slots = per object exists + com + one outline from the embedding.")
    parser.add_argument("--n-rays",                     type=int,   default=36, help="Polar: rays per outline, every 360/n degrees.")
    parser.add_argument("--ray-shape",                  type=str,   default="free", choices=("free", "fourier", "superellipse"), help="Polar: outline prior. free = every ray its own; fourier = log r(θ) up to --n-harmonics; superellipse = rotated rounded rectangle (4 params).")
    parser.add_argument("--n-harmonics",                type=int,   default=4, help="Polar, --ray-shape fourier: harmonics of log r(θ).")
    parser.add_argument("--center-radius",              type=float, default=1.5, help="Polar: training points = pixels inside the object within this many px of its mass center.")
    parser.add_argument("--peak-thresh",                type=float, default=0.3, help="Polar: a peak of p(center) * centerness above this is an object.")
    parser.add_argument("--rays-weight",                type=float, default=1.0, help="Polar: weight of the polar iou loss on the rays (the center focal loss has weight 1).")
    parser.add_argument("--centerness-weight",          type=float, default=1.0, help="Polar: weight of the centerness bce.")
    parser.add_argument("--render-weight",              type=float, default=None, help="Polar/slots: weight of dice on the drawn masks vs the true mask; 0 = off. Default 0 (polar), 1 (slots).")
    parser.add_argument("--slots-aux-weight",           type=float, default=0.0, help="Mask head: > 0 adds the slots head as an auxiliary loss at this weight (exists + slot-com-weight * com + rays-weight * rays); the mask head still draws the mask.")
    parser.add_argument("--slot-com-weight",            type=float, default=0.1, help="Slots: weight of the L1 com loss, px of the out grid.")

    return parser

def run(**kwargs):
    # parse args
    args = get_parser().parse_args()  # get defaults
    args.__dict__.update(kwargs)  # apply overrides from cli
    assert not args.eval_only or args.checkpoint_path, "--checkpoint-path is required when using --eval-only"

    # aux head weights left unset: the mask head's usual ones; the polar head trains on its own losses only (its peaks give com + count)
    for k, v in dict(object=0.1, n_objects=0.1, com=1.0, area=0.1, box=0.1).items():
        if getattr(args, f"{k}_weight") is None: setattr(args, f"{k}_weight", v if args.head == "mask" else 0.0)
    if args.render_weight is None: args.render_weight = 1.0 if args.head == "slots" else 0.0

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
    train_dl, eval_dl, data_info = build_dataset(args.dataset, test_size=args.test_size, speakers=None if "all" in args.speakers else args.speakers, signal=args.signal, norm=args.norm, patch_size=args.patch_size, window=args.window, gate_ms=args.gate_ms, laser_norm=bool(args.laser_norm),
        phase=args.phase, phase_ref=args.phase_ref, out_h=args.out_h, out_w=args.out_w, rgb=bool(args.rgb), batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers)
    train_eval = Evaluator(label="train", dataloader=DataLoader(train_dl.dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers))

    # build model
    model = Boombox(args.d_model, data_info, args.out_h, args.out_w, rgb=bool(args.rgb), loss_fn=args.loss_fn, alpha=args.alpha,
                    loss_weights=dict(object=args.object_weight, n_objects=args.n_objects_weight, com=args.com_weight, area=args.area_weight, box=args.box_weight,
                                      boundary=args.boundary_weight, edge=args.edge_weight, rays=args.rays_weight, centerness=args.centerness_weight,
                                      render=args.render_weight, slot_com=args.slot_com_weight, slots=args.slots_aux_weight), edge_width=args.edge_width,
                    freq_gain=bool(args.freq_gain), freq_coord=bool(args.freq_coord), freq_pool=args.freq_pool,
                    head=args.head, n_rays=args.n_rays, ray_shape=args.ray_shape, n_harmonics=args.n_harmonics, center_radius=args.center_radius, peak_thresh=args.peak_thresh,
                    slots_aux=args.slots_aux_weight > 0, encoder=args.encoder, decoder=args.decoder, patch_size=args.patch_size,
                    heads=args.heads, freq_layers=args.freq_layers, laser_layers=args.laser_layers, ffn_dim=args.ffn_dim)
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
        VisualizeSMask(args.image_interval, args.wandb_images), OutputSaver(args.eval_interval, data_info["objects"]), PositionLogger(), ConfidenceLogger(data_info["objects"]),
        EvalSummary(CATEGORIES, DELTAS, NOT_OOD),
    ]

    # optimizer + lr schedule
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args.scheduler, args.t_warmup)

    # trainer
    trainer = Trainer(
        run_name=args.run_name, model=model, optimizers=optimizer, schedulers=None if args.eval_only else scheduler,  # eval-only: no max_duration, which composer needs for a schedule
        algorithms=[GradientClipping("norm", args.grad_clip)] if args.grad_clip else None,
        train_dataloader=train_dl, eval_dataloader=eval_dl, eval_interval=args.eval_interval,
        max_duration=None if args.eval_only else args.max_duration, seed=args.seed,
        device=device, precision=args.precision if device == "gpu" else "fp32",
        loggers=loggers, callbacks=callbacks, progress_bar=False, log_to_console=True, save_metrics=True,
        load_path=args.checkpoint_path, load_weights_only=bool(args.eval_only), autoresume=bool(args.run_name) and not args.eval_only,  # eval: no optimizer state (polar heads at weight 0 never get one)
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
