"""PyTorch Dataset pairing each sample's recovered vibration audio with its target
image (RGB photo, `target="rgb"`, or segmentation mask, `target="s_mask"`) and (for
SDEdit-style sampling) a start image -- the box's empty-box photo for `rgb`, a
synthetic all-black square for `s_mask` (see `start_image()`) -- for training
SonicDiffusion on our data.

Unlike src3 (AudioToken/BEATs), SonicDiffusion's audio path doesn't take a 2D
map/image at all -- its CLAP encoder consumes a raw waveform and builds its own
internal log-mel spectrogram (see model/clap_audio_encoder.py's docstring). Every
sample dir already has a precomputed `recovered_audio.wav` (22050Hz mono, from
src/data/vibrate.py:get_recovered_audio), so we just load that directly -- no FFT
recomputation needed here.

Reuses src2/data.py's sample-collection utilities (BOX_DIRS, PHOTO_NAME, collect,
find_empty_box_reference) and src3/dataset.py's n_object_split/load_image rather
than re-deriving them -- same train/eval split and image-loading convention as the
AudioToken pipeline, so results are comparable across the two.
"""
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from src2.data import BOX_DIRS, MASK_NAME, PHOTO_NAME, collect, find_empty_box_reference  # noqa: E402
from src3.dataset import load_image, n_object_split  # noqa: E402
from src4.model.clap_audio_encoder import preprocess_waveform  # noqa: E402
from src4.spectrogram_stretch import apply_stretch  # noqa: E402

# SonicDiffusion's paper trains mostly with null/empty text (audio is the real
# conditioning signal; text is optional, added at inference time) -- see its
# discussion in this conversation. Default to that rather than a hand-written caption.
DEFAULT_PROMPT = ""

# apply_stretch is deterministic given (wav, sr, mode) -- "log" mode's Griffin-Lim
# resynthesis costs ~2.5s/sample, and __getitem__ would otherwise redo it from
# scratch every epoch for the same 2383 samples (100 epochs -> ~165 CPU-hours of
# pure repeated waste). Cache the stretched (wav, sr) to disk on first use instead.
CACHE_DIR = REPO / "src4" / "cache" / "stretch"


def _cached_stretch(wav: np.ndarray, sr: int, mode: str, box: str, sample_id: str) -> tuple[np.ndarray, int]:
    if mode == "none":
        return wav, sr  # passthrough is already free, no need to cache it
    cache_path = CACHE_DIR / mode / box / f"{sample_id}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        return data["wav"], int(data["sr"])
    stretched_wav, stretched_sr = apply_stretch(wav, sr, mode)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp_path, wav=stretched_wav, sr=stretched_sr)
    os.replace(tmp_path, cache_path)  # atomic -- safe against concurrent DataLoader workers
    return stretched_wav, stretched_sr


def multi_object_split(samples: list[tuple[Path, dict]], target_n_objects: list[int],
                        eval_frac: float = 0.2, seed: int = 42) -> tuple[list, list]:
    """Same position-based 80/20 split as src3/dataset.py:n_object_split, but for
    SEVERAL n_objects values at once (e.g. both 1-cube and 2-cube positions), not
    just one -- train = every empty-box sample + 80% of each n's positions; eval =
    the remaining 20% of each n's positions. Calls n_object_split once per n (same
    seed -> same per-n position split it would give alone) and merges, stripping
    each call's own `empty` prefix off so it isn't duplicated across n's."""
    empty = [s for s in samples if s[1].get("n_objects") == 0]
    train, eval_ = list(empty), []
    for n in target_n_objects:
        train_n, eval_n = n_object_split(samples, n, eval_frac, seed)
        train += train_n[len(empty):]  # n_object_split's train = empty + train_target
        eval_ += eval_n
    return train, eval_


class SonicDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, box: str, split: str, resolution: int = 512, eval_frac: float = 0.2, seed: int = 42,
                 limit: int | None = None, prompt: str = DEFAULT_PROMPT, target_n_objects: int | list[int] = 1,
                 spectrogram_stretch: str = "none", target: str = "rgb"):
        assert box in BOX_DIRS
        assert split in ("train", "eval")
        assert target in ("rgb", "s_mask")
        self.box, self.resolution, self.prompt, self.spectrogram_stretch = box, resolution, prompt, spectrogram_stretch
        self.target = target
        self.target_name = PHOTO_NAME if target == "rgb" else MASK_NAME

        samples = collect(box, limit)
        if isinstance(target_n_objects, int):
            train, eval_ = n_object_split(samples, target_n_objects, eval_frac, seed)
        else:
            train, eval_ = multi_object_split(samples, target_n_objects, eval_frac, seed)
        self.samples = train if split == "train" else eval_
        self.empty_box_dir = find_empty_box_reference(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def start_image(self) -> Image.Image:
        """Fixed reference image for SDEdit-style sampling -- what the sampler starts
        FROM. target='rgb': the box's one empty-box photo (its real 'nothing here'
        state). target='s_mask': a synthetic all-black square -- an 'empty-box mask'
        isn't a meaningful real photo to load, so the 'nothing' starting point is
        just literal black, matching src3/train.py's --start-mode black."""
        if self.target == "s_mask":
            return Image.new("RGB", (self.resolution, self.resolution), (0, 0, 0))
        return Image.open(self.empty_box_dir / PHOTO_NAME).convert("RGB")

    def __getitem__(self, i: int) -> dict:
        sample_dir, meta = self.samples[i]
        pixel_values = load_image(sample_dir / self.target_name, self.resolution)
        wav, sr = sf.read(sample_dir / "recovered_audio.wav", dtype="float32")
        wav, sr = _cached_stretch(wav, sr, self.spectrogram_stretch, self.box, sample_dir.name)
        waveform = preprocess_waveform(torch.from_numpy(wav), sample_rate=sr)
        return {"pixel_values": pixel_values, "waveform": waveform, "prompt": self.prompt,
                "sample_id": meta.get("sample_id", sample_dir.name)}
