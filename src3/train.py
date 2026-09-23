"""Train src3's AudioToken-style textual inversion model on vibration data: BEATs
(bypassed frontend, see beats_input.py) + the released embedder, continued
fine-tuned to condition a frozen Stable-Diffusion-v1-4 on our vibration data
instead of real audio.

Usage:
    python src3/train.py --box gastronorm --condition-mode laser-freq --target mask --start-mode noise
    python src3/train.py --box gastronorm --condition-mode spectrogram --target photo --start-mode empty-box
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from src3.dataset import AudioTokenDataset  # noqa: E402
from src3.model import AudioTokenModel  # noqa: E402
from src3.beats_input import map_to_image, to_beats_fbank  # noqa: E402
from src3.metrics import compute_metrics  # noqa: E402

MAX_WANDB_IMAGES = 108  # wandb's own per-call cap on wandb.Image, see src/run.py


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--box", default="gastronorm", choices=["gastronorm", "plastic", "wood", "cardboard", "shoebox"])
    p.add_argument("--condition-mode", default="laser-freq", choices=["laser-freq", "spectrogram"],
                    help="laser-freq: the FFT heatmap (lasers x freq bins). spectrogram: a real STFT "
                         "spectrogram of the audio recovered from the lasers+x/y-averaged FFT.")
    p.add_argument("--target", default="mask", choices=["mask", "photo"],
                    help="mask: predict the segmentation mask (IoU/mass/contour/localization are logged). "
                         "photo: predict the natural overhead photo (those metrics are skipped -- no "
                         "trustworthy way to threshold a generated photo into a comparable mask).")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="cap sample count, for a quick smoke test")
    p.add_argument("--prompt", default=None,
                    help="override the default per-target prompt (src3/dataset.py:PROMPTS). The placeholder "
                         "token (<*>) is prepended automatically if you don't include it -- it's how the "
                         "audio-derived embedding reaches the UNet, so it must be in the prompt somewhere.")
    p.add_argument("--target-n-objects", type=int, default=1,
                    help="train on positions with exactly this many objects (plus every empty-box sample) -- "
                         "see src3/dataset.py:n_object_split. Default 1 (a single object); not every box has "
                         "single-object data (e.g. cardboard/plastic only have 0- and 2-object scenes).")
    p.add_argument("--log-stretch", type=int, default=0, choices=[0, 1],
                    help="0 (default): src3/beats_input.py:to_beats_fbank's usual linear placement into a "
                         "narrow mel-bin sub-range. 1: log-Hz warp the 50-1000Hz band across the full 128 "
                         "mel bins instead -- see notebooks/83_spectrogram_stretch_strategies.ipynb.")
    # sampling / eval
    p.add_argument("--start-mode", default="empty-box", choices=["noise", "black", "empty-box"],
                    help="what the diffusion sampler starts FROM at eval/generation time -- training "
                         "always uses the standard noise-prediction loss on the real target regardless "
                         "of this flag; it only changes how evaluate()'s generated images are sampled. "
                         "Defaults to empty-box (SDEdit-style from the box's empty-box photo) rather than "
                         "pure noise.")
    p.add_argument("--start-strength", type=float, default=0.75,
                    help="SDEdit strength for --start-mode black/empty-box (1.0=pure noise, 0.0=the start image itself).")
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-samples", type=int, default=5,
                    help="a fixed number of samples (same ones every eval interval, picked once via --seed) "
                         "visualized/scored from EACH of the train and eval splits")
    # model / train
    p.add_argument("--finetune-beats", action="store_true", help="also train BEATs, not just the embedder")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=None,
                    help="if set, overrides --max-steps: train for this many passes over the train split "
                         "(max_steps = epochs * len(train_loader), so it accounts for --batch-size/drop_last).")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    # run
    p.add_argument("--checkpoint-dir", type=Path, default=REPO / "src3/runs/default")
    p.add_argument("--wandb-project", default="good-vibrations-audiotoken")
    p.add_argument("--run-name", default=None)
    return p


def make_start_image(mode: str, dataset: AudioTokenDataset, resolution: int) -> Image.Image | None:
    if mode == "noise":
        return None
    if mode == "black":
        return Image.new("RGB", (resolution, resolution), (0, 0, 0))
    return dataset.empty_box_image().resize((resolution, resolution))


def fixed_indices(n_total: int, n: int, seed: int) -> np.ndarray:
    """The same n indices every call (same seed, same n) -- so the logged panels track
    the identical samples across eval intervals instead of a fresh random draw each time."""
    return np.random.RandomState(seed).choice(n_total, min(n, n_total), replace=False)


@torch.no_grad()
def visualize(model: AudioTokenModel, dataset: AudioTokenDataset, indices: np.ndarray, args: argparse.Namespace,
              device: str, key_prefix: str, start_image: Image.Image | None, start_panel_img: Image.Image,
              start_label: str, generator: torch.Generator) -> dict:
    pred_images, gt_images, panels = [], [], []
    for i in indices:
        sample_dir, meta = dataset.samples[i]
        gt = Image.open(sample_dir / dataset.target_name).convert("RGB").resize((args.resolution, args.resolution))
        cond_map = dataset.condition_map(sample_dir, meta)
        fbank = to_beats_fbank(cond_map, args.condition_mode, log_stretch=bool(args.log_stretch)).unsqueeze(0).to(device)
        pred = model.generate(fbank, dataset.prompt, start_image=start_image, start_strength=args.start_strength,
                               num_inference_steps=args.num_inference_steps, generator=generator)

        # flipud: cond_map's row 0 is the LOWEST freq bin, but PIL puts array row 0 at the
        # TOP of the image -- flip so low freq renders at the bottom, matching the usual
        # spectrogram convention (e.g. notebooks/81's matplotlib plots use origin="lower",
        # which does this same flip internally; Image.fromarray doesn't, so it must be done here)
        cond_img = Image.fromarray(np.ascontiguousarray(np.flipud(map_to_image(cond_map)))).convert("RGB").resize((args.resolution, args.resolution))
        pred_images.append(pred); gt_images.append(gt)
        # gt last (rightmost) -- easiest reference point to compare pred against when scanning left to right
        panel = Image.new("RGB", (args.resolution * 4, args.resolution))
        panel.paste(cond_img, (0, 0)); panel.paste(start_panel_img, (args.resolution, 0))
        panel.paste(pred, (args.resolution * 2, 0)); panel.paste(gt, (args.resolution * 3, 0))
        panels.append(wandb.Image(
            panel, caption=f"{meta.get('sample_id', sample_dir.name)} | audio condition ({args.condition_mode}) | "
                            f"{start_label} | pred | gt"))

    log = {f"{key_prefix}/pred_vs_gt": panels[:MAX_WANDB_IMAGES]}
    if args.target == "mask":
        log.update({f"{key_prefix}/{k}": v for k, v in compute_metrics(pred_images, gt_images).items()})
    return log


@torch.no_grad()
def evaluate(model: AudioTokenModel, train_dataset: AudioTokenDataset, eval_dataset: AudioTokenDataset,
             args: argparse.Namespace, device: str, step: int) -> None:
    start_image = make_start_image(args.start_mode, eval_dataset, args.resolution)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # start_image is what the diffusion sampler's latent is initialized from (SDEdit img2img
    # when --start-mode != noise) -- a DIFFERENT conditioning mechanism from cond_map (which
    # becomes the audio-token cross-attention embedding). Both get logged, clearly labeled, so
    # it's visually verifiable which one is which -- easy to conflate otherwise (see the panel
    # from the first gastronorm run: "condition" only showed cond_map, an almost-black
    # spectrogram, with no indication the empty-box start image was in play at all).
    start_label = args.start_mode if start_image is None else f"start ({args.start_mode})"
    start_panel_img = Image.new("RGB", (args.resolution, args.resolution), (0, 0, 0)) if start_image is None else start_image

    train_idx = fixed_indices(len(train_dataset), args.eval_samples, args.seed)
    eval_idx = fixed_indices(len(eval_dataset), args.eval_samples, args.seed)

    log = {}
    log.update(visualize(model, train_dataset, train_idx, args, device, "train_viz",
                          start_image, start_panel_img, start_label, generator))
    log.update(visualize(model, eval_dataset, eval_idx, args, device, "eval_viz",
                          start_image, start_panel_img, start_label, generator))
    wandb.log(log, step=step)
    model.clear_eval_cache()


def save_checkpoint(model: AudioTokenModel, checkpoint_dir: Path, tag: str) -> None:
    torch.save(model.embedder.state_dict(), checkpoint_dir / f"embedder_{tag}.bin")
    if model.finetune_beats:
        torch.save(model.aud_encoder.state_dict(), checkpoint_dir / f"beats_{tag}.bin")


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = AudioTokenDataset(args.box, "train", args.condition_mode, args.target, args.resolution,
                                       limit=args.limit, prompt=args.prompt, target_n_objects=args.target_n_objects,
                                       log_stretch=bool(args.log_stretch))
    eval_dataset = AudioTokenDataset(args.box, "eval", args.condition_mode, args.target, args.resolution,
                                      limit=args.limit, prompt=args.prompt, target_n_objects=args.target_n_objects,
                                      log_stretch=bool(args.log_stretch))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=2, drop_last=True)
    print(f"{len(train_dataset)} train samples, {len(eval_dataset)} eval samples")

    if args.epochs is not None:
        args.max_steps = args.epochs * len(train_loader)
        print(f"--epochs {args.epochs} -> max_steps={args.max_steps} ({len(train_loader)} steps/epoch)")

    model = AudioTokenModel(finetune_beats=args.finetune_beats).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)

    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps: break
            step_start = time.perf_counter()

            loss = model(batch["pixel_values"].to(device), batch["fbank"].to(device), batch["prompt"])

            optimizer.zero_grad()
            loss.backward()
            grad_norms = [p.grad.detach().norm(2).item() for p in model.trainable_parameters() if p.grad is not None]
            optimizer.step()

            if device == "cuda": torch.cuda.synchronize()  # otherwise step_time only measures kernel-launch, not compute
            step_time = time.perf_counter() - step_start

            if step % args.log_interval == 0:
                wandb.log({"train/loss": loss.item(), "train/step_time": step_time,
                           "train/grad_norm_avg": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0}, step=step)
                print(f"step {step}: loss={loss.item():.4f}")

            if step % args.eval_interval == 0 and step > 0:
                evaluate(model, train_dataset, eval_dataset, args, device, step)
                save_checkpoint(model, args.checkpoint_dir, str(step))

            step += 1

    evaluate(model, train_dataset, eval_dataset, args, device, step)
    save_checkpoint(model, args.checkpoint_dir, "final")
    wandb.finish()


if __name__ == "__main__":
    main(get_parser().parse_args())
