"""Standalone sampling from a trained src3 checkpoint (mirrors vendor/inference.py,
but on our vibration data instead of real audio) -- generate images for a box's
eval-split samples without running any training.

Usage:
    python src3/infer.py --box gastronorm --condition-mode laser-freq --target mask \
        --embedder-ckpt src3/runs/default/embedder_final.bin --start-mode empty-box --out-dir src3/infer_out
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from src3.dataset import AudioTokenDataset  # noqa: E402
from src3.model import AudioTokenModel  # noqa: E402
from src3.beats_input import to_beats_fbank  # noqa: E402
from src3.train import make_start_image  # noqa: E402


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--box", default="gastronorm", choices=["gastronorm", "plastic", "wood", "cardboard", "shoebox"])
    p.add_argument("--condition-mode", default="laser-freq", choices=["laser-freq", "spectrogram"])
    p.add_argument("--target", default="mask", choices=["mask", "photo"])
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--target-n-objects", type=int, default=1, help="see src3/dataset.py:n_object_split")
    p.add_argument("--embedder-ckpt", type=Path, required=True, help="from a src3/train.py run's checkpoint-dir")
    p.add_argument("--beats-ckpt", type=Path, default=None, help="only needed if the run used --finetune-beats")
    p.add_argument("--start-mode", default="noise", choices=["noise", "black", "empty-box"])
    p.add_argument("--start-strength", type=float, default=0.75)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, required=True)
    return p


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = AudioTokenDataset(args.box, "eval", args.condition_mode, args.target, args.resolution,
                                 target_n_objects=args.target_n_objects)
    model = AudioTokenModel(finetune_beats=args.beats_ckpt is not None, embedder_ckpt=args.embedder_ckpt).to(device)
    if args.beats_ckpt is not None:
        model.aud_encoder.load_state_dict(torch.load(args.beats_ckpt, map_location=device))
    model.eval()

    start_image = make_start_image(args.start_mode, dataset, args.resolution)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    n = min(args.num_samples, len(dataset))
    for i in range(n):
        sample_dir, meta = dataset.samples[i]
        fbank = to_beats_fbank(dataset.condition_map(sample_dir, meta), args.condition_mode).unsqueeze(0).to(device)
        image = model.generate(fbank, dataset.prompt, start_image=start_image, start_strength=args.start_strength,
                                num_inference_steps=args.num_inference_steps, generator=generator)
        sample_id = meta.get("sample_id", sample_dir.name)
        out_path = args.out_dir / f"{sample_id}.png"
        image.save(out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main(get_parser().parse_args())
