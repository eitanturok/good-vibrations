"""One-off: compute the global mean/std that src3/beats_input.py:GLOBAL_STATS hardcodes,
so to_beats_fbank normalizes by fixed dataset-level statistics (mirroring BEATs.preprocess's
own fixed-constant scheme) instead of a per-sample z-score. Samples across all 5 boxes so
the stats aren't tied to one box's particular contents.

Computes on the SAME representation to_beats_fbank normalizes -- band-placed into
[MEL_BAND_LO, MEL_BAND_HI) with the rest padded at each sample's own floor value -- not on
the raw cond_map, so the constants printed here can be pasted directly into GLOBAL_STATS.

Usage:
    python scripts/compute_beats_global_stats.py [--n-per-box 60]
"""
import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

import numpy as np
import torch

from src2.data import BOX_DIRS, collect, find_fft
from src3.beats_input import (BEATS_MEL_BINS, MEL_BAND_HI, MEL_BAND_LO, laser_freq_map, spectrogram_map)


def _band_placed(cond_map: np.ndarray) -> np.ndarray:
    """Mirrors to_beats_fbank's resize+pad steps, pre-normalization."""
    import torch.nn.functional as tf
    band_bins = MEL_BAND_HI - MEL_BAND_LO
    t = torch.from_numpy(cond_map).float()[None, None]
    t = tf.interpolate(t, size=(band_bins, cond_map.shape[1]), mode="bilinear", align_corners=False)
    t = t[0, 0].T
    full = torch.full((t.shape[0], BEATS_MEL_BINS), t.min().item())
    full[:, MEL_BAND_LO:MEL_BAND_HI] = t
    return full.numpy()


def main(args: argparse.Namespace) -> None:
    random.seed(0)
    for mode, fn in [("laser-freq", laser_freq_map), ("spectrogram", None)]:
        vals = []
        for box in BOX_DIRS:
            samples = collect(box, limit=None)
            subset = random.sample(samples, min(args.n_per_box, len(samples)))
            for d, m in subset:
                try:
                    fft_path = find_fft(d)
                    cond_map = fn(fft_path) if mode == "laser-freq" else spectrogram_map(fft_path, fps=float(m["fps"]))
                    vals.append(_band_placed(cond_map).ravel())
                except Exception as e:
                    print(f"  skip {d}: {e}")
        vals = np.concatenate(vals)
        print(f'"{mode}": ({vals.mean():.6f}, {vals.std():.6f}),  # n={len(vals)}')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-box", type=int, default=60)
    main(p.parse_args())
