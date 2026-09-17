"""PyTorch Dataset pairing each sample's vibration data (as a BEATs-ready condition
map, see beats_input.py) with a target image, for src3's AudioToken-style textual
inversion training/eval.

Reuses src2/data.py's sample-collection utilities directly rather than re-deriving
them: box sample dirs (BOX_DIRS), FFT lookup (find_fft), and the empty-box reference
sample (find_empty_box_reference) -- src2 already built and tuned all of this against
the same experiments/<box>/samples/ layout.

Does NOT reuse src2/data.py's one_cube_split: that function filters on a hardcoded
per-box layout-name list (ONE_CUBE_LAYOUTS = ("purple-cube", "red-cube")) that only
matches gastronorm's naming convention -- every other box uses different layout
names for its single-object scenes (wood: "one-cube-grid1"), and some boxes
(cardboard, plastic) have NO single-object scenes at all, just empty-box + 2-object
grids. Silently returns an empty split rather than erroring, which is exactly what
happened running this against cardboard. n_object_split() below uses each sample's
own metadata['n_objects'] instead -- box-agnostic, verified correct (0/1/2 exactly
matches empty/one-object/two-object layouts) across every box in this repo.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from src2.data import BOX_DIRS, MASK_NAME, PHOTO_NAME, collect, find_empty_box_reference, find_fft  # noqa: E402
from src3.beats_input import laser_freq_map, spectrogram_map, to_beats_fbank  # noqa: E402

PLACEHOLDER_TOKEN = "<*>"
PROMPTS = {
    "mask": f"a photo of {PLACEHOLDER_TOKEN}, a small white square on a black background",
    "photo": f"a photo of {PLACEHOLDER_TOKEN}, a metal cube on the floor of a metal box from a bird's eye view",
}


def n_object_split(samples: list[tuple[Path, dict]], target_n_objects: int = 1,
                    eval_frac: float = 0.2, seed: int = 42) -> tuple[list, list]:
    """(train, eval): train = every empty-box sample (n_objects==0) + 80% of the
    positions with exactly target_n_objects objects; eval = the remaining 20% of
    those positions. Everything else (other object counts) is dropped. Split by
    position_id, not by sample, so a held-out position never leaks into train from
    a different speaker -- same principle as src2/data.py:one_cube_split, but keyed
    off metadata['n_objects'] instead of a hardcoded layout-name list (see module
    docstring for why)."""
    empty = [s for s in samples if s[1].get("n_objects") == 0]
    target = [s for s in samples if s[1].get("n_objects") == target_n_objects]
    if not target:
        raise ValueError(f"no samples with n_objects == {target_n_objects} in this box")

    by_position: dict[int, list] = {}
    for s in target:
        by_position.setdefault(s[1]["position_id"], []).append(s)

    rng = np.random.default_rng(seed)
    positions = sorted(by_position)
    n_eval = round(eval_frac * len(positions))
    eval_positions = set(rng.permutation(positions)[:n_eval].tolist())

    train_target = [s for p, group in by_position.items() if p not in eval_positions for s in group]
    eval_target = [s for p, group in by_position.items() if p in eval_positions for s in group]

    print(f"n_object_split(target_n_objects={target_n_objects}): {len(empty)} empty-box + "
          f"{len(train_target)} train ({len(positions) - n_eval} positions), "
          f"{len(eval_target)} eval ({n_eval} positions)")
    return empty + train_target, eval_target


def load_image(path: Path, resolution: int) -> torch.Tensor:
    """PIL -> (3,res,res) float tensor in [-1,1], matching SD's VAE input convention."""
    img = Image.open(path).convert("RGB").resize((resolution, resolution), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class AudioTokenDataset(torch.utils.data.Dataset):
    def __init__(self, box: str, split: str, condition_mode: str, target: str,
                 resolution: int = 512, eval_frac: float = 0.2, seed: int = 42, limit: int | None = None,
                 prompt: str | None = None, target_n_objects: int = 1, log_stretch: bool = False):
        assert box in BOX_DIRS
        assert split in ("train", "eval")
        assert condition_mode in ("laser-freq", "spectrogram")
        assert target in ("mask", "photo")
        self.box, self.condition_mode, self.target, self.resolution = box, condition_mode, target, resolution
        self.log_stretch = log_stretch
        self.target_name = MASK_NAME if target == "mask" else PHOTO_NAME
        if prompt is None:
            self.prompt = PROMPTS[target]
        else:
            # the placeholder token is how the audio-derived embedding actually reaches
            # the UNet's cross-attention -- a prompt without it carries no conditioning
            # at all, so prepend it rather than silently ignoring a missing token.
            self.prompt = prompt if PLACEHOLDER_TOKEN in prompt else f"{PLACEHOLDER_TOKEN} {prompt}"

        samples = collect(box, limit)
        train, eval_ = n_object_split(samples, target_n_objects, eval_frac, seed)
        self.samples = train if split == "train" else eval_
        self.empty_box_dir = find_empty_box_reference(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def empty_box_image(self) -> Image.Image:
        """Fixed reference image for --start-mode empty-box: the box's one empty-box
        sample, in the same domain (mask/photo) as this dataset's target."""
        return Image.open(self.empty_box_dir / self.target_name).convert("RGB")

    def condition_map(self, sample_dir: Path, meta: dict) -> np.ndarray:
        fft_path = find_fft(sample_dir)
        if self.condition_mode == "laser-freq":
            return laser_freq_map(fft_path)
        return spectrogram_map(fft_path, fps=float(meta["fps"]))

    def __getitem__(self, i: int) -> dict:
        sample_dir, meta = self.samples[i]
        pixel_values = load_image(sample_dir / self.target_name, self.resolution)
        fbank = to_beats_fbank(self.condition_map(sample_dir, meta), self.condition_mode, log_stretch=self.log_stretch)
        return {"pixel_values": pixel_values, "fbank": fbank, "prompt": self.prompt,
                "sample_id": meta.get("sample_id", sample_dir.name)}
