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

import argparse, json, shutil

import torch
from torch.utils.data import DataLoader
from composer import Trainer
from composer.core import Evaluator
from composer.algorithms import GradientClipping
from composer.callbacks import LRMonitor, SpeedMonitor, NaNMonitor, RuntimeEstimator, OptimizerMonitor, OOMObserver
from composer.loggers import WandBLogger, FileLogger
from composer.utils.reproducibility import seed_all

from dataset import DATASETS, CATEGORIES, DELTAS, NOT_OOD, PAPER_CATEGORIES, build_dataset
from arch import Boombox
from callbacks import VisualizeSMask, OutputSaver, PositionLogger, ConfidenceLogger, EvalSummary, EvalExtras, Interpretability
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
    parser.add_argument("--laser-center",               type=int,   default=0, choices=(0, 1), help="Subtract the mean over lasers per (freq, channel) after --signal: cancels the speaker's S(f) in log signals.")
    parser.add_argument("--laser-norm",                 type=int,   default=0, choices=(0, 1), help="Scale each laser's |fft| to rms 1 before --signal: removes per-laser gain.")
    parser.add_argument("--norm",                       type=str,   default="std", choices=("std", "z"))
    parser.add_argument("--phase",                      type=str,   default="none", choices=("none", "cossin"), help="cossin = add cos and sin of each channel's phase as input channels (C -> 3C), after --phase-ref.")
    parser.add_argument("--phase-ref",                  type=str,   default="none", choices=("none", "avg", "real"), help="Rotate every laser's phase, per sample/freq/channel, so this reference reads 0: avg = mean over lasers; real = 1/2 arg sum_l x^2 (most energy in the real part).")
    parser.add_argument("--window",                     type=str,   default="hann", choices=("hann", "tukey", "deconv", "farina"), help="hann = the stored 04_ffts (hann over the whole capture); tukey = recompute from the raw shifts with 13ms tapers; deconv = tukey / the chirp's spectrum.")
    parser.add_argument("--gate-ms",                    type=float, default=0, help="--window tukey|deconv: > 0 = deconvolve by the chirp and keep this many ms of the impulse response.")
    parser.add_argument("--freq-gain",                  type=int,   default=0, choices=(0, 1), help="FreqEncoder: learned smooth per-frequency input gain exp(mlp(log f)), starts at 1.")
    parser.add_argument("--freq-coord",                 type=int,   default=0, choices=(0, 1), help="FreqEncoder: add log10(f) as an input channel (CoordConv).")
    parser.add_argument("--freq-pool",                  type=str,   default="mean", choices=("mean", "flatten"), help="FreqEncoder: average the last 5 freq positions, or flatten them into a linear layer.")
    parser.add_argument("--patch-size",                 type=lambda s: None if s.lower() == "none" else int(s), default=None, help="Freq bins per token; 'none' = no patching, x is (L,F,C).")
    parser.add_argument("--out-h",                      type=int,   default=64)
    parser.add_argument("--out-w",                      type=int,   default=64)
    parser.add_argument("--gray",                       type=int,   default=0, choices=(0, 1), help="Also draw the overhead photo in grayscale: (1 - mask) learned floor + mask appearance (mask detached).")
    parser.add_argument("--gray-weight",                type=float, default=1.0, help="Weight of each grayscale photo loss: whole-image L1, object L1 (true mask), gradient L1 near the object.")
    parser.add_argument("--rgb",                        type=int,   default=0, choices=(0, 1), help="Predict the overhead photo instead of the mask.")
    parser.add_argument("--num-workers",                type=int,   default=4)

    # train
    parser.add_argument("--batch-size",                 type=int,   default=128)
    parser.add_argument("--microbatch",                 type=int,   default=None, help="Per-step microbatch (gradient accumulation to --batch-size); None = the whole batch.")
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
    parser.add_argument("--encoder",                    type=str,   default="conv", choices=("conv", "transformer", "modal"), help="modal = src8's ModalFormer (axial laser x band transformer + pixel-query decoder, its own decoder). ""conv = boombox's freq convs + laser-grid convs; transformer = src's VibrationTransformer encoder (freq transformer per laser, then a laser transformer), needs --patch-size.")
    parser.add_argument("--decoder",                    type=str,   default=None, choices=("conv", "mlp", "query"), help="conv = boombox's upsampling decoder; mlp = src's default MLP decoder (d_model -> 256 -> every pixel); query = ModalFormer's pixel queries. Default: query for --encoder modal, else conv.")
    parser.add_argument("--heads",                      type=int,   default=2, help="--encoder transformer: attention heads.")
    parser.add_argument("--freq-layers",                type=int,   default=2, help="--encoder transformer: layers of the per-laser freq transformer.")
    parser.add_argument("--laser-layers",               type=int,   default=2, help="--encoder transformer: layers of the laser transformer.")
    # src8 ModalFormer (--encoder modal)
    parser.add_argument("--modal-d",                    type=int,   default=128, help="ModalFormer token width.")
    parser.add_argument("--modal-heads",                type=int,   default=4)
    parser.add_argument("--modal-order",                type=str,   default="LFLF", help="Axial blocks: L = attention over lasers per band, F = over bands per laser; '' = none.")
    parser.add_argument("--modal-laser-layers",         type=int,   default=2, help="Blocks over the pooled laser tokens.")
    parser.add_argument("--modal-dec-layers",           type=int,   default=2, help="Decoder layers (cross-attention to the lasers + self-attention over the queries).")
    parser.add_argument("--modal-grid",                 type=int,   default=16, help="Pixel queries per side; the output is 4x this, resized to out_h x out_w.")
    parser.add_argument("--geo",                        type=int,   default=1, choices=(0, 1), help="Learnable floor<->wall prior in the decoder's cross-attention.")
    parser.add_argument("--box-film",                   type=int,   default=0, choices=(0, 1), help="ModalFormer: per-box FiLM (scale, shift) on the frequency stem's channels: filters matched to each box's linewidths.")
    parser.add_argument("--box-norm",                   type=int,   default=1, choices=(0, 1), help="ModalFormer: per-box running standardization of the input per (laser, bin, axis).")
    parser.add_argument("--lin-skip",                   type=int,   default=0, help="ModalFormer: rank of a linear shortcut from the band-averaged input into the embedding; 0 = off.")
    parser.add_argument("--laser-drop",                 type=float, default=0.0, help="Training: drop up to this fraction of each sample's lasers.")
    parser.add_argument("--band-drop",                  type=float, default=0.0, help="Training: drop up to this fraction of each sample's 19 frequency bands (~49 Hz each); ModalFormer only.")
    parser.add_argument("--color-aug",                  type=float, default=0.0, help="Training: random smooth spectral tilt per sample (an unseen speaker), std in input units.")
    parser.add_argument("--color-ell",                  type=float, default=0.0, help="--color-aug: > 0 = a random log-gain with knots every this many Hz (the measured speaker difference: 8-28 Hz); 0 = the old 4-cosine tilt.")
    parser.add_argument("--laser-jitter-p",             type=float, default=1.0, help="Training: fraction of samples --laser-jitter shifts (the rest keep the exact grid).")
    parser.add_argument("--obj-decoder",                type=str,   default="none", choices=("none", "queries", "centernet"), help="Multi-object decoder on the hybrid's conv trunk (instead of the pose branch): queries = Mask2Former object queries, set-matched to the instances; centernet = centre heatmap peaks, each a pose-branch canonical mask + angle. The mask = their union; the direct head trains as an aux.")
    parser.add_argument("--obj-pose",                   type=int,   default=0, choices=(0, 1), help="--obj-decoder queries: each query also gets a pose head (canonical mask + angle, warped to its mask's centroid, added to its mask; angle supervised per matched object).")
    parser.add_argument("--obj-pix",                    type=int,   default=1, choices=(0, 1), help="--obj-decoder queries: pixel embedding = a conv head on the trunk features + the direct logits (1), or a 1x1 linear readout (0, qpose v1/v2: could not localize).")
    parser.add_argument("--obj-init",                   type=str,   default="learned", choices=("learned", "peaks"), help="--obj-decoder queries: learned queries, or two-stage: each starts at one of the top-K local maxima of the direct head's map (its pixel features + position).")
    parser.add_argument("--obj-shape",                  type=str,   default="dot", choices=("dot", "gauss"), help="queries: each mask = <mlp(q), pixel embedding> (dot), or an ellipse from the query's centroid / angle / std devs + a shape canvas in the object's own frame (gauss; moments supervised at --imoment-weight).")
    parser.add_argument("--obj-heat-sigma",             type=str,   default="px", choices=("px", "obj"), help="centernet: heatmap gaussians of 1.5 px (px) or of each object's own covariance / 4 (obj).")
    parser.add_argument("--obj-count-floor",            type=int,   default=0, help="centernet eval: keep at least the count head's argmax best peaks, besides those above --obj-thresh.")
    parser.add_argument("--obj-jitter",                 type=float, default=0.0, help="centernet training: gaussian jitter (px) of the teacher-forced centres.")
    parser.add_argument("--obj-pose-merge",             type=str,   default="sum", choices=("sum", "only"), help="--obj-pose 1: each query's mask = free-form + warped canonical (sum), or the warped canonical alone (only: rotation through the angle).")
    parser.add_argument("--imoment-weight",             type=float, default=0.0, help="Per-instance moment loss (area, centroid, covariance of the predicted mask in each true object's Mahalanobis cell), every scene; 0 = off.")
    parser.add_argument("--n-queries",                  type=int,   default=5, help="--obj-decoder: object queries / heatmap peaks (the most objects it can draw).")
    parser.add_argument("--obj-layers",                 type=int,   default=3, help="--obj-decoder queries: decoder layers.")
    parser.add_argument("--obj-thresh",                 type=float, default=0.3, help="--obj-decoder centernet: eval peaks above this heatmap score are objects.")
    parser.add_argument("--obj-weight",                 type=float, default=1.0, help="--obj-decoder: weight of the per-object losses (matched masks, exists / heatmap).")
    parser.add_argument("--direct-weight",              type=float, default=1.0, help="--obj-decoder: weight of the direct head's mask loss (aux).")
    parser.add_argument("--multi-oversample",           type=int,   default=1, help="Repeat every multi-object train sample this many times per epoch (1 = off).")
    parser.add_argument("--freq-jitter",                type=float, default=0.0, help="Training: stretch each sample's frequency axis by a factor U(1-s, 1+s) (all lasers alike, linear interpolation); 0 = off.")
    parser.add_argument("--laser-jitter",               type=float, default=0.0, help="Training: resample the laser grid shifted by up to this many cells per axis (bilinear); 0 = off.")
    parser.add_argument("--sdf-weight",                 type=float, default=0.0, help="Loss weight of the signed-distance channel; 0 = off.")
    parser.add_argument("--pose-branch",                type=int,   default=0, choices=(0, 1), help="Pose-factored branch: per object a canonical mask warped by its com + angle (STN), merged with the decoder's.")
    parser.add_argument("--pose-merge",                 type=str,   default="or", choices=("or", "sum", "only", "slots"), help="Pose branch merge: or = logaddexp (can only add pixels); sum = residual logits (can add and erase); only = the warped canonical mask is the mask, the decoder a centre heatmap.")
    parser.add_argument("--pose-center",                type=str,   default="com", choices=("com", "mask"), help="Pose branch centres: com = the com head's slots; mask = one slot at the soft-argmax of the direct decoder's mask.")
    parser.add_argument("--pose-window",                type=float, default=0.35, help="Pose branch: the canonical canvas spans this fraction of the box.")
    parser.add_argument("--orient-weight",              type=float, default=0.0, help="Loss weight of the orientation aux head (ecc * (cos 2θ, sin 2θ) of a single object's mask, from the embedding); 0 = off.")
    parser.add_argument("--moment-terms",               type=str,   default="acs", help="Moment loss terms: a = log area, c = centroid, s = covariance (size + orientation).")
    parser.add_argument("--slot-weight",                type=float, default=1.0, help="--pose-merge slots: weight of the per-object losses (exists bce, matched slot mask vs its instance).")
    parser.add_argument("--heatmap-weight",             type=float, default=1.0, help="--pose-merge only: weight of the centre heatmap loss (softmax over pixels vs the true mask).")
    parser.add_argument("--moment-weight",              type=float, default=0.0, help="Loss weight of the area/centroid/covariance (orientation) loss on single objects; 0 = off.")
    parser.add_argument("--ffn-dim",                    type=int,   default=None, help="--encoder transformer: FFN width; default 4*d_model.")

    # loss
    parser.add_argument("--loss-fn",                    type=str,   default="bce", choices=("bce", "mse", "dice", "bce-dice", "bce-tversky"), help="Mask loss. bce-tversky: alpha * bce + (1 - alpha) * Tversky(FP 0.3, FN 0.7).")
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

def build_model(args, data_info):
    """the Boombox run.py trains, from its args (also probe.py / notebooks: argparse.Namespace(**json of runs/<run>/args.json))"""
    return Boombox(args.d_model, data_info, args.out_h, args.out_w, rgb=bool(args.rgb), loss_fn=args.loss_fn, alpha=args.alpha,
                    loss_weights=dict(object=args.object_weight, n_objects=args.n_objects_weight, com=args.com_weight, area=args.area_weight, box=args.box_weight,
                                      boundary=args.boundary_weight, edge=args.edge_weight, rays=args.rays_weight, centerness=args.centerness_weight,
                                      render=args.render_weight, slot_com=args.slot_com_weight, slots=args.slots_aux_weight,
                                      sdf=args.sdf_weight, moment=args.moment_weight, orient=args.orient_weight, heatmap=getattr(args, "heatmap_weight", 1.0), slot_exists=getattr(args, "slot_weight", 1.0), slot_mask=getattr(args, "slot_weight", 1.0),
                                      obj_mask=getattr(args, "obj_weight", 1.0), obj_exists=getattr(args, "obj_weight", 1.0), obj_heat=getattr(args, "obj_weight", 1.0), direct=getattr(args, "direct_weight", 1.0),
                                      imoment=getattr(args, "imoment_weight", 0.0), obj_orient=args.orient_weight, obj_gauss=getattr(args, "imoment_weight", 0.0), gray=getattr(args, "gray_weight", 1.0), gray_obj=getattr(args, "gray_weight", 1.0), gray_grad=getattr(args, "gray_weight", 1.0)), edge_width=args.edge_width,
                    freq_gain=bool(args.freq_gain), freq_coord=bool(args.freq_coord), freq_pool=args.freq_pool,
                    head=args.head, n_rays=args.n_rays, ray_shape=args.ray_shape, n_harmonics=args.n_harmonics, center_radius=args.center_radius, peak_thresh=args.peak_thresh,
                    slots_aux=args.slots_aux_weight > 0, encoder=args.encoder, decoder=args.decoder, patch_size=args.patch_size,
                    heads=args.heads, freq_layers=args.freq_layers, laser_layers=args.laser_layers, ffn_dim=args.ffn_dim, color_aug=args.color_aug,
                    pose_branch=bool(getattr(args, "pose_branch", 0)), pose_window=getattr(args, "pose_window", 0.35), pose_merge=getattr(args, "pose_merge", "or"), pose_center=getattr(args, "pose_center", "com"), moment_terms=getattr(args, "moment_terms", "acs"),
                    color_ell=getattr(args, "color_ell", 0.0), laser_jitter=getattr(args, "laser_jitter", 0.0), laser_jitter_p=getattr(args, "laser_jitter_p", 1.0), freq_jitter=getattr(args, "freq_jitter", 0.0),
                    obj_decoder=getattr(args, "obj_decoder", "none"), n_queries=getattr(args, "n_queries", 5), obj_layers=getattr(args, "obj_layers", 3), obj_thresh=getattr(args, "obj_thresh", 0.3), obj_pose=bool(getattr(args, "obj_pose", 0)), obj_pix=bool(getattr(args, "obj_pix", 0)), obj_init=getattr(args, "obj_init", "learned"), obj_pose_merge=getattr(args, "obj_pose_merge", "sum"),
                    obj_shape=getattr(args, "obj_shape", "dot"), obj_heat_sigma=getattr(args, "obj_heat_sigma", "px"), obj_count_floor=bool(getattr(args, "obj_count_floor", 0)), obj_jitter=getattr(args, "obj_jitter", 0.0), gray=bool(getattr(args, "gray", 0)),
                    modal=dict(d=args.modal_d, heads=args.modal_heads, order=args.modal_order, laser_layers=args.modal_laser_layers,
                               dec_layers=args.modal_dec_layers, grid=args.modal_grid, geo=bool(args.geo), laser_drop=args.laser_drop, band_drop=getattr(args, "band_drop", 0.0), lin_skip=args.lin_skip,
                               box_norm=bool(getattr(args, "box_norm", 1)), box_film=bool(getattr(args, "box_film", 0))))

def run(**kwargs):
    # parse args
    args = get_parser().parse_args()  # get defaults
    args.__dict__.update(kwargs)  # apply overrides from cli
    assert not args.eval_only or args.checkpoint_path, "--checkpoint-path is required when using --eval-only"

    # aux head weights left unset: the mask head's usual ones; the polar head trains on its own losses only (its peaks give com + count)
    for k, v in dict(object=0.1, n_objects=0.1, com=1.0, area=0.1, box=0.1).items():
        if getattr(args, f"{k}_weight") is None: setattr(args, f"{k}_weight", v if args.head == "mask" else 0.0)
    if args.render_weight is None: args.render_weight = 1.0 if args.head == "slots" else 0.0

    if args.decoder is None: args.decoder = "query" if args.encoder == "modal" else "conv"
    assert args.decoder != "query" or args.encoder == "modal", "--decoder query is ModalFormer's"

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
    train_dl, eval_dl, data_info = build_dataset(args.dataset, test_size=args.test_size, speakers=None if "all" in args.speakers else args.speakers, signal=args.signal, laser_center=bool(args.laser_center), norm=args.norm, patch_size=args.patch_size, window=args.window, gate_ms=args.gate_ms, laser_norm=bool(args.laser_norm),
        phase=args.phase, phase_ref=args.phase_ref, out_h=args.out_h, out_w=args.out_w, rgb=bool(args.rgb), gray=bool(getattr(args, "gray", 0)), batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, multi_oversample=getattr(args, "multi_oversample", 1))
    train_eval = Evaluator(label="train", dataloader=DataLoader(train_dl.unique_dataset, batch_size=args.eval_batch_size, num_workers=0))  # no workers: see build_dataset

    # build model
    model = build_model(args, data_info)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    if args.run_name: Path(f"runs/{args.run_name}").mkdir(parents=True, exist_ok=True); Path(f"runs/{args.run_name}/args.json").write_text(json.dumps(vars(args), indent=1))

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
        EvalSummary(PAPER_CATEGORIES if args.dataset == "paper" else CATEGORIES, DELTAS, NOT_OOD), EvalExtras(), Interpretability(),
    ]

    # optimizer + lr schedule
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(args.scheduler, args.t_warmup)

    # trainer
    trainer = Trainer(
        run_name=args.run_name, model=model, optimizers=optimizer, schedulers=None if args.eval_only else scheduler,  # eval-only: no max_duration, which composer needs for a schedule
        algorithms=[GradientClipping("norm", args.grad_clip)] if args.grad_clip else None,
        device_train_microbatch_size=args.microbatch,  # None = the whole batch
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
