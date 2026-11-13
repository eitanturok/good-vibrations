"""train one of the src9 models (arch/) on the src9 split:
  .venv/bin/python src9/run.py --model hybrid --run-name s9-hybrid-500ep
  .venv/bin/python src9/run.py --model boombox --run-name s9-boombox-3k
  .venv/bin/python src9/run.py --model transformer --run-name s9-transformer-500ep
  .venv/bin/python src9/run.py --model transformer3 --run-name std-samplez-v2
  .venv/bin/python src9/run.py --model transformer4 --run-name t4-v1
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
from callbacks import VisualizeSMask, OutputSaver, PositionLogger, EvalSummary, SetEpoch, NaNDebug
from arch.boombox import Boombox
from arch.hybrid import Hybrid
from arch.transformer import Transformer
from arch.transformer2 import Transformer2
from arch.transformer3 import Transformer3
from arch.transformer4 import Transformer4, DECODERS, LATENT_NORMS, FREQ_ROPES
from arch.transformer_calib import TransformerCalib

# boombox: s6-gastro-plastic-cardboard-3k (hann 04_ffts, 3000 ep, 100 ep warmup). hybrid: s8-mag-nopose-imom-region-500ep (tukey, 500 ep, 50 ep warmup, microbatch 512)
COMMON = dict(seed=42, batch_size=1024, microbatch=0, lr=1e-4, weight_decay=0.01, eval_batch_size=108, eval_interval="50ep", image_interval="100ep",
              num_workers=4, compile=1, wandb=1, wandb_group="src9", debug_nan=0)  # debug_nan 1 (transformer3): callbacks.NaNDebug
AUGS = dict(speaker_gain=0.35, speaker_gain_hz=20.0, laser_jitter=0.5)  # train-time augs (utils.py): log-gain sd, knot spacing in Hz, max laser shift in cells
DEFAULTS = {"boombox": COMMON | dict(window="hann", t_warmup="100ep", max_duration="3000ep", checkpoint_interval="500ep", microbatch=256),
            "hybrid":  COMMON | dict(window="tukey", t_warmup="50ep", max_duration="500ep", checkpoint_interval="250ep", microbatch=512, **AUGS),
            "transformer": COMMON | dict(window="tukey", t_warmup="50ep", max_duration="500ep", checkpoint_interval="250ep", microbatch=256, **AUGS),
            "transformer2": COMMON | dict(window="tukey", t_warmup="50ep", max_duration="200ep", checkpoint_interval="100ep", microbatch=256, **AUGS,
                                          height=64, width=64, freq_cls=0, laser_cls=0, freq_layers=2, band_proj=0, laser_proj=0, band_v=0, laser_v=0, laser_layers=2, rope=0, heads=4, readout="ref", latent_dim=128, d=128, norm="box", box_emb=1, pos="add", laser_drop=0.0, band_drop=0.0, outline=1, aux="all", loss="full", dice_w=0.5, area_w=0.0, tversky=0, tv_alpha=0.3, tv_beta=0.7, ms_dice=0, edge_w=0.0)}  # [cls] token instead of gated bands (freq) / the flatten (laser)
DEFAULTS["transformer3"] = COMMON | dict(window="tukey", t_warmup="50ep", max_duration="300ep", checkpoint_interval="100ep", microbatch=128, **AUGS,  # = big-drop20-v1 at 300 ep
                                        height=64, width=64, bins_per_band=65, d=256, heads=8, freq_layers=4, laser_layers=8, latent_dim=512, laser_drop=0.2, band_drop=0.2, dice_w=0.5, objrecall_w=0.0, boundary_w=0.0,
                                        mask_loss="bce_dice", maskae="", readout="per", heatmap_w=0.0, laser_jitter_p=1.0, speaker_gain_p=1.0, gan_w=0.0, norm="sample", log_mag=0,
                                        abspos=0, n_queries=16, query_layers=1, freeze_readout=0, drop_rescale=0, band_overlap=0, tokenizer="linear", imom_w=0.0, outline_w=0.0, imom_norm="", imom_start=0,
                                        compile=0)  # eager by default: drop30-v1 / base300-v1 and earlier transformer3 runs trained compiled
DEFAULTS["transformer4"] = COMMON | dict(window="tukey", t_warmup="50ep", max_duration="300ep", checkpoint_interval="100ep", microbatch=128, **AUGS, compile=0,
                                        decoder="two", latent_norm="ln", freq_rope="linear", laser_jitter_p=1.0, speaker_gain_p=1.0, laser_drop_p=1.0, band_drop_p=1.0, spk_grl_w=0.0, spk_grl_lambda=1.0, spk_grl_ramp=0, adapter_dim=0, init_from="", freeze_base=0, lasers="")  # spk_grl_*: speaker head behind a gradient reversal layer (0 = off); eager; base300-v1: transformer3's training defaults. decoder: two (Up) | one | res | res-coord | res-xattn-freq | res-xattn-laser; latent_norm: ln | rescale; freq_rope: linear | log (arch/transformer4.py)
DEFAULTS["transformer_calib"] = DEFAULTS["transformer"]  # transformer with empty-box calibration instead of per-box stats, bce + dice

def run():
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--model", choices=tuple(DEFAULTS))
    model_name = pre.parse_known_args()[0].model                                     # only this model's knobs are flags (--help lists them)
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=tuple(DEFAULTS), required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--eval-only", type=int, default=0)
    p.add_argument("--checkpoint-path", default=None)
    for k, v in DEFAULTS.get(model_name, {}).items(): p.add_argument(f"--{k.replace('_', '-')}", type=(lambda s: s if s == "auto" else int(s)) if k == "microbatch" else type(v), default=None,  # None = the model's default; microbatch: int or "auto"
                                                               choices={"decoder": DECODERS, "latent_norm": LATENT_NORMS, "freq_rope": FREQ_ROPES}.get(k))
    args = p.parse_args()
    for k, v in DEFAULTS[args.model].items():
        if getattr(args, k) is None: setattr(args, k, v)
    assert not args.eval_only or args.checkpoint_path, "--eval-only needs --checkpoint-path"
    T2 = ("height", "width", "freq_cls", "laser_cls", "freq_layers", "band_proj", "laser_proj", "band_v", "laser_v", "laser_layers", "rope", "heads", "readout", "latent_dim", "d", "norm", "box_emb", "pos", "laser_drop", "band_drop", "outline", "aux", "loss", "dice_w", "area_w", "tversky", "tv_alpha", "tv_beta", "ms_dice", "edge_w")  # transformer2-only knobs
    T3 = ("height", "width", "bins_per_band", "d", "heads", "freq_layers", "laser_layers", "latent_dim", "laser_drop", "band_drop", "dice_w", "objrecall_w", "boundary_w", "mask_loss", "maskae", "readout", "heatmap_w", "laser_jitter_p", "speaker_gain_p", "gan_w", "norm", "log_mag", "abspos", "n_queries", "query_layers", "freeze_readout", "drop_rescale", "band_overlap", "tokenizer", "imom_w", "outline_w", "imom_norm", "imom_start")  # transformer3 knobs
    own = {"transformer2": T2, "transformer3": T3, "transformer4": ("decoder", "latent_norm", "freq_rope", "laser_jitter_p", "speaker_gain_p", "laser_drop_p", "band_drop_p", "spk_grl_w", "spk_grl_lambda", "spk_grl_ramp", "adapter_dim", "lasers")}.get(args.model, ())

    seed_all(args.seed)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    if device == "gpu": torch.set_float32_matmul_precision("high"); torch.backends.cudnn.benchmark = True

    train_dl, eval_dl, train_eval, info = build_dataset(args.window, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, num_workers=args.num_workers,
                                                        height=getattr(args, "height", None) or 64, width=getattr(args, "width", None) or 64, eval_microbatch="auto" if args.microbatch == "auto" else None)

    augs = {} if args.model == "boombox" else {k: getattr(args, k) for k in AUGS}
    augs |= {k: getattr(args, k) for k in own}

    model = {"boombox": Boombox, "hybrid": Hybrid, "transformer": Transformer, "transformer_calib": TransformerCalib, "transformer2": Transformer2, "transformer3": Transformer3, "transformer4": Transformer4}[args.model](info, **augs)
    if getattr(args, "init_from", ""):                                               # fine-tune: start from a checkpoint's weights; only new modules may be missing
        miss, unexp = model.load_state_dict(torch.load(args.init_from, map_location="cpu", weights_only=False)["state"]["model"], strict=False)
        assert not unexp and all(k.startswith(("adapter.", "speaker_head.")) for k in miss), (miss, unexp); print(f"init from {args.init_from}, new: {sorted({k.split('.')[0] for k in miss})}")
    if getattr(args, "freeze_base", 0):                                              # train only the adapter + speaker head
        for n, q in model.named_parameters(): q.requires_grad = n.startswith(("adapter.", "speaker_head."))
        print(f"trainable: {sum(q.numel() for q in model.parameters() if q.requires_grad):,}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params:,} parameters")
    if args.run_name: Path(f"runs/{args.run_name}").mkdir(parents=True, exist_ok=True); Path(f"runs/{args.run_name}/args.json").write_text(json.dumps(vars(args), indent=1))

    loggers = [FileLogger("runs/{run_name}/logs-rank{rank}.txt")]
    if args.wandb and not args.eval_only:
        loggers.append(WandBLogger("better-tsa", group=args.wandb_group, name=args.run_name,
                                   init_kwargs={"config": vars(args) | {k: v for k, v in info.items() if not isinstance(v, torch.Tensor)} | dict(num_parameters=n_params), "id": args.run_name, "resume": "allow"}))
    if args.compile and args.model == "transformer3" and model.encoder.n_bands * args.d % 64:
        # inductor (torch 2.11) comprehensive padding miscompiles the backward when a laser's (n_bands, d) bf16 block isn't 128-byte aligned (d % 32 != 0, e.g. d 400 / 336,
        # pool-m-v1 / query-m-v1): it pads that residual buffer's per-laser stride 7200 -> 7232, the fused LN-backward kernel writes d(embed output) at the dense stride, and
        # the embed weight / bias grad kernels read it at the padded one, past its end (garbage -> nan / inf grads, illegal memory access). aligned d (256) unaffected
        from torch._inductor import config as inductor_config; inductor_config.comprehensive_padding = False
    opt = torch.optim.AdamW([q for n, q in model.named_parameters() if not n.startswith("disc.") and q.requires_grad], args.lr, weight_decay=args.weight_decay)
    if args.debug_nan: assert args.model == "transformer3", "--debug-nan is for transformer3"; model.debug_nan = 1  # + x_in / m_l / m_b in the outputs
    callbacks = [NaNDebug(model, opt)] * bool(args.debug_nan) + [LRMonitor(), SpeedMonitor(), RuntimeEstimator(skip_batches=64, time_unit="minutes"), NaNMonitor(), OptimizerMonitor(batch_log_interval=10),
                 VisualizeSMask(args.image_interval), OutputSaver(args.eval_interval, info["objects"]), PositionLogger(), EvalSummary(CATEGORIES, NOT_OOD), SetEpoch()]

    trainer = Trainer(
        run_name=args.run_name, model=model, optimizers=opt,
        schedulers=None if args.eval_only else CosineAnnealingWithWarmupScheduler(t_warmup=args.t_warmup),
        device_train_microbatch_size=args.microbatch if args.microbatch == "auto" else int(args.microbatch) or None, train_dataloader=train_dl, eval_dataloader=eval_dl, eval_interval=args.eval_interval,
        max_duration=None if args.eval_only else args.max_duration, seed=args.seed, device=device, precision="amp_bf16" if device == "gpu" else "fp32",
        loggers=loggers, callbacks=callbacks, progress_bar=False, log_to_console=True,
        load_path=args.checkpoint_path, load_weights_only=bool(args.eval_only), autoresume=bool(args.run_name) and not args.eval_only,
        save_folder=None if args.eval_only else "runs/{run_name}/checkpoints", save_interval=args.checkpoint_interval,
        compile_config={"mode": "default"} if args.compile else None,
    )
    # eval runs eager: inductor's no_grad graph (torch 2.11) gives wrong masks that change from run to run once a 2nd batch size is seen (|dp| ~1)
    if args.compile: model.eval_forward = lambda batch, outputs=None: type(model).forward(model, batch)
    trainer.eval(eval_dl + [train_eval])
    if not args.eval_only:
        trainer.fit()                    # fit already evals eval_dl at the end (last eval_interval or FIT_END): only train left
        trainer.eval([train_eval])

if __name__ == "__main__":
    run()
