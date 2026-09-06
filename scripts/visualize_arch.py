"""Visualize the Boombox conv encoder and conv decoder (model/boombox.py) with visualtorch.

Renders each as a layered box diagram and saves it to a PNG -- one image for the
frequency+laser-grid conv encoder, one for the transposed-conv decoder. They're
visualized separately (rather than the full BoomboxModel) because visualtorch traces a
plain tensor-in/tensor-out nn.Module, and BoomboxModel.forward takes a batch dict.

Install (already done in .venv-vibrate, not added to pyproject.toml):
    pip install visualtorch

Usage:
    python scripts/visualize_arch.py
    python scripts/visualize_arch.py --d-model 512 --out-h 21 --out-w 30
"""

import argparse
import sys
from pathlib import Path

import torch.nn as nn
import visualtorch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # utils.*
sys.path.insert(0, str(ROOT / "src"))  # model.* (boombox.py does `from model.arch import ...`)

from model.boombox import Decoder, Encoder  # noqa: E402

# declutter: skip activation/norm boxes, keep the layers that actually change tensor shape
TYPE_IGNORE = [nn.ReLU, nn.LeakyReLU, nn.BatchNorm2d]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=1024, help="embedding width between encoder and decoder")
    p.add_argument("--n-channels", type=int, default=2, help="fft channels into the encoder (2 = magnitude only)")
    p.add_argument("--n-laser-rows", type=int, default=10)
    p.add_argument("--n-laser-cols", type=int, default=10)
    p.add_argument("--n-freqs", type=int, default=11 * 256, help="patch_size * n_patches fed to the encoder")
    p.add_argument("--out-h", type=int, default=21, help="decoder output mask height")
    p.add_argument("--out-w", type=int, default=30, help="decoder output mask width")
    p.add_argument("--out-dir", type=Path, default=ROOT / "media/images/arch_viz")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_lasers = args.n_laser_rows * args.n_laser_cols

    encoder = Encoder(args.n_channels, args.d_model, args.n_laser_rows, args.n_laser_cols).eval()
    decoder = Decoder(args.d_model, args.out_h, args.out_w).eval()

    # draw_volume=True: real 3D cubes -- screen width from channel count (scale_z), screen
    # height/depth from the two spatial dims (scale_xy). Only the clamps are tightened from
    # the library defaults (min/max_xy 10/2000, min/max_z 10/400): the encoder's frequency
    # axis alone spans 2816 -> 1 and channels span 2 -> 1024, so the un-clamped defaults let
    # one axis blow out into a flat bar. show_dimension prints the exact shape regardless.
    common = dict(type_ignore=TYPE_IGNORE, legend=True, draw_volume=True, show_dimension=True,
                  min_xy=15, max_xy=150, min_z=15, max_z=150)

    print(f"encoder: (B,{args.n_channels},{n_lasers},{args.n_freqs}) -> (B,{args.d_model})")
    enc_img = visualtorch.render(
        encoder, input_shape=(1, args.n_channels, n_lasers, args.n_freqs), style="flow", **common,
    )
    enc_path = args.out_dir / "boombox_encoder.png"
    enc_img.save(enc_path)
    print(f"  saved -> {enc_path}")

    print(f"decoder: (B,{args.d_model}) -> (B,{args.out_h},{args.out_w})")
    dec_img = visualtorch.render(
        decoder, input_shape=(1, args.d_model), style="flow", **common,
    )
    dec_path = args.out_dir / "boombox_decoder.png"
    dec_img.save(dec_path)
    print(f"  saved -> {dec_path}")


if __name__ == "__main__":
    main()
