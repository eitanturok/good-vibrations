"""Build a parquet dataset pairing each sample's FFT heatmap (conditioning image)
with a target image, for FLUX.2 klein img2img LoRA training. Target is the
segmentation mask by default (--target mask), or the natural overhead photo
(--target photo, used with --edit-mode: true edit-mode training predicts the
photo, not the mask, since "editing an empty-box photo toward a photo of the
object" is the well-posed version of this task -- editing a blank mask into an
object-domain photo isn't).

Standalone on purpose: does not touch src/model/dataset.py's MDS/StreamingDataset
pipeline. Reads sample dirs directly, the way notebooks/77_box_datasets.ipynb does.

Written as parquet (not a plain metadata.jsonl + image folder) because the training
script loads data via `datasets.load_dataset(args.dataset_name, ...)`, and that
generic loader only auto-decodes an image column named 'file_name' -> 'image'. Our
second image column (the conditioning heatmap) needs to already be typed as
datasets.Image() for `dataset["train"][i][cond_image_column]` to hand the script a
PIL Image rather than a path string -- parquet is the one format the generic loader
reads back with column typing (incl. Image()) intact.

Usage:
    python src2/data.py --box gastronorm --out src2/data/gastronorm
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset, Image as HFImage
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent

BOX_DIRS = {
    "gastronorm": REPO / "experiments/31_07_2026_gastronorm_exp1/samples",
    "plastic":    REPO / "experiments/31_08_2026_green_plastic_two_laser_faces/samples",
    "wood":       REPO / "experiments/2026_09_06_wood_box/samples",
    "cardboard":  REPO / "experiments/2026_09_07_cardboard_box/samples",
    "shoebox":    REPO / "experiments/2026_09_08_shoebox/samples",
}

FFT_NAMES = ("vibration/04_ffts.npz", "vibration/04_fft.npz")
MASK_NAME = "image/03_smask.png"
PHOTO_NAME = "image/02_cropped_overhead.png"

# Fixed prompt for every example -- constant across the dataset (no per-sample
# captions), but no longer content-free: it names what the TARGET actually looks
# like, giving the text branch real semantic grounding rather than being a pure
# no-op the model learns to ignore. Still fixed rather than per-sample, since the
# dataset has no per-object captions to draw from.
#
# Two prompts, one per --target: the mask target isn't a photo of a metal cube --
# it's a small white blob on a black background -- so describing it as a photo
# would be describing the wrong domain entirely. The photo prompt (edit-mode)
# still describes the real scene, since --target photo's ground truth is an
# actual overhead photo.
PROMPTS = {
    "mask": "A small white square on a black background",
    "photo": "A metal cube on the floor of a metal box from a bird's eye view",
}


def find_fft(sample_dir: Path) -> Path | None:
    for name in FFT_NAMES:
        p = sample_dir / name
        if p.exists():
            return p
    return None


# log(1 + mag/K) contrast scale from notebooks/79_log_contrast_sweep.ipynb: swept
# k in log(1+x/k) against RMS contrast + Shannon entropy over all 3007 gastronorm
# samples: best k = 0.01138 (median per-sample best 0.01138, mean 0.01334 +/-
# 0.0049) -- this is a real, measured optimum, not the old fixed-epsilon log(mag +
# 1e-3) + percentile-clip this function used before, which was never tuned for
# contrast at all.
LOG_CONTRAST_K = 0.01138


def fft_to_heatmap(fft_path: Path) -> Image.Image:
    """(1,L,F,C) complex64 -> an (L,F') RGB heatmap image of mean |FFT| magnitude
    over the x/y channels, contrast-stretched via log(1+mag/K) (see LOG_CONTRAST_K).
    L = lasers (rows), F = freq bins (columns) -- matches the "100 laser points by
    ~1300 freqs" heatmap described for this task.

    No PIL/matplotlib resize: L is upsampled by an exact integer repeat
    (np.repeat, F // L times) so the image comes out roughly square while every
    laser's row stays a uniform block of its own true value -- no interpolation
    blending adjacent lasers together, unlike a resize. F' = L * (F // L), so
    F'/F is close to but not exactly 1 (e.g. 1235 lasers-per-freq -> repeat 12x ->
    1200, ~3% short of literally square); "roughly square," not a fixed resolution
    -- this is a raw per-pixel-per-laser encoding, meant to be resized like any
    other image once it enters the training pipeline's own resolution handling.
    """
    with np.load(fft_path) as z:
        fft = z["fft"]
    fft = np.squeeze(fft, axis=0) if fft.ndim == 4 and fft.shape[0] == 1 else fft  # (L,F,C)
    mag = np.abs(fft).mean(axis=-1)  # (L,F)

    log_mag = np.log1p(mag / LOG_CONTRAST_K)
    lo, hi = log_mag.min(), log_mag.max()
    norm = (log_mag - lo) / max(hi - lo, 1e-8)

    repeats = max(1, mag.shape[1] // mag.shape[0])  # F // L, so L*repeats ~= F
    norm = np.repeat(norm, repeats, axis=0)

    u8 = (norm * 255).astype(np.uint8)
    return Image.fromarray(u8).convert("RGB")


def read_metadata(sample_dir: Path) -> dict:
    meta = {}
    for line in (sample_dir / "metadata.jsonl").read_text().splitlines():
        line = line.strip()
        if line:
            meta.update(json.loads(line))
    return meta


def collect(box: str, limit: int | None = None) -> list[tuple[Path, dict]]:
    samples_dir = BOX_DIRS[box]
    assert samples_dir.is_dir(), f"missing samples dir for {box}: {samples_dir}"
    sample_dirs = sorted(p for p in samples_dir.glob("*") if p.is_dir())
    if limit:
        sample_dirs = sample_dirs[:limit]

    out = []
    for sample_dir in sample_dirs:
        if find_fft(sample_dir) is None or not (sample_dir / MASK_NAME).exists():
            continue
        if not (sample_dir / "metadata.jsonl").exists():
            continue
        out.append((sample_dir, read_metadata(sample_dir)))
    return out


# The gastronorm capture's layout names for "one cube present": the main purple-cube
# raster (560 samples) and red-cube (40 samples, same shape/mass -- see
# project_attribution memory: acoustically indistinguishable from purple-cube, only
# colour differs, which the vibration signal can't see anyway). Pooled together for
# the 80/20 position split per the user's request, rather than red-cube being a
# pure held-out OOD set the way the existing src/model/dataset.py gastronorm() does it.
ONE_CUBE_LAYOUTS = ("purple-cube", "red-cube")
EMPTY_BOX_LAYOUT = "empty-box"


def one_cube_split(samples: list[tuple[Path, dict]], eval_frac: float = 0.2, seed: int = 42
                    ) -> tuple[list[tuple[Path, dict]], list[tuple[Path, dict]]]:
    """(train, eval) samples: train = every empty-box sample + 80% of the pooled
    purple-cube/red-cube positions; eval = the remaining 20% of those positions.
    Two-object layouts (purple--green-cube-grid*) and everything else are dropped
    entirely -- this split is "empty box vs exactly one cube," nothing else.

    Split by position_id, not by sample, so a held-out position never leaks into
    train from a different speaker -- same principle as
    src/model/dataset.py:split_by_position, reimplemented here (not imported) since
    that function lives on the MDS/StreamingDataset index-list contract and this
    pipeline works directly off (sample_dir, metadata) pairs instead.
    """
    empty = [s for s in samples if s[1].get("layout") == EMPTY_BOX_LAYOUT]
    one_cube = [s for s in samples if s[1].get("layout") in ONE_CUBE_LAYOUTS]

    by_position: dict[int, list[tuple[Path, dict]]] = {}
    for s in one_cube:
        by_position.setdefault(s[1]["position_id"], []).append(s)

    rng = np.random.default_rng(seed)
    positions = sorted(by_position)
    n_eval = round(eval_frac * len(positions))
    eval_positions = set(rng.permutation(positions)[:n_eval].tolist())

    train_cube = [s for p, group in by_position.items() if p not in eval_positions for s in group]
    eval_cube = [s for p, group in by_position.items() if p in eval_positions for s in group]

    train = empty + train_cube
    print(f"one_cube_split: {len(empty)} empty-box + {len(train_cube)} one-cube train "
          f"({len(positions) - n_eval} positions), {len(eval_cube)} one-cube eval ({n_eval} positions)")
    return train, eval_cube


def find_empty_box_reference(samples: list[tuple[Path, dict]]) -> Path:
    """One fixed empty-box sample dir for a box, used as the constant edit-mode
    source image for every row -- not a per-sample match. First empty-box sample
    found (by sorted sample_dir name), so it's deterministic across runs."""
    for sample_dir, meta in samples:
        if meta.get("layout") == EMPTY_BOX_LAYOUT:
            return sample_dir
    raise ValueError(f"no {EMPTY_BOX_LAYOUT!r} sample found to use as the empty-box reference")


def to_dataset(samples: list[tuple[Path, dict]], desc: str, target: str = "mask",
                empty_box_source: Path | None = None) -> Dataset:
    """target='mask': predict the segmentation mask (03_smask.png), unconditioned
    generation from noise, heatmap as the only conditioning image (the original
    img2img setup). target='photo': predict the natural overhead photo
    (02_cropped_overhead.png); when empty_box_source is given, an extra
    'source_file_name' column carries that one fixed empty-box photo for every
    row, for edit-mode training (start the trajectory from the empty box, edit
    it toward the object photo, informed by the heatmap)."""
    target_name = MASK_NAME if target == "mask" else PHOTO_NAME
    prompt = PROMPTS[target]
    images, cond_images, texts = [], [], []
    source_images = [] if empty_box_source is not None else None
    empty_box_img = Image.open(empty_box_source / PHOTO_NAME).convert("RGB") if empty_box_source is not None else None

    for sample_dir, _ in tqdm(samples, desc=desc):
        images.append(Image.open(sample_dir / target_name).convert("RGB"))
        cond_images.append(fft_to_heatmap(find_fft(sample_dir)))
        texts.append(prompt)
        if source_images is not None:
            source_images.append(empty_box_img)

    data = {"file_name": images, "cond_file_name": cond_images, "text": texts}
    cast_cols = ["file_name", "cond_file_name"]
    if source_images is not None:
        data["source_file_name"] = source_images
        cast_cols.append("source_file_name")

    ds = Dataset.from_dict(data)
    for col in cast_cols:
        ds = ds.cast_column(col, HFImage())
    return ds


def build(box: str, out_dir: Path, limit: int | None = None, target: str = "mask",
          edit_mode: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = collect(box, limit)
    empty_box_source = find_empty_box_reference(samples) if edit_mode else None
    ds = to_dataset(samples, desc=f"building {box}", target=target, empty_box_source=empty_box_source)
    ds.to_parquet(str(out_dir / "data.parquet"))
    print(f"wrote {len(ds)} pairs to {out_dir / 'data.parquet'}")


def build_one_cube_split(box: str, out_dir: Path, eval_frac: float = 0.2, seed: int = 42,
                          limit: int | None = None, target: str = "mask", edit_mode: bool = False) -> None:
    """Writes out_dir/train/data.parquet and out_dir/eval/data.parquet: train is
    every empty-box sample plus 80% of purple-cube+red-cube positions, eval is the
    remaining 20% -- see one_cube_split's docstring."""
    samples = collect(box, limit)
    empty_box_source = find_empty_box_reference(samples) if edit_mode else None
    train, eval_ = one_cube_split(samples, eval_frac, seed)
    for name, split_samples in [("train", train), ("eval", eval_)]:
        split_dir = out_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        ds = to_dataset(split_samples, desc=f"building {box}/{name}", target=target, empty_box_source=empty_box_source)
        ds.to_parquet(str(split_dir / "data.parquet"))
        print(f"wrote {len(ds)} pairs to {split_dir / 'data.parquet'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--box", required=True, choices=sorted(BOX_DIRS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="cap sample count, for a quick smoke test")
    p.add_argument("--split", choices=["one-cube"], default=None,
                    help="one-cube: empty-box + 80%% of purple-cube/red-cube positions in train, "
                         "remaining 20%% in eval; writes <out>/train and <out>/eval instead of <out>/data.parquet")
    p.add_argument("--eval-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target", choices=["mask", "photo"], default="mask",
                    help="mask: predict the segmentation mask (default). photo: predict the natural "
                         "overhead photo -- use with --edit-mode.")
    p.add_argument("--edit-mode", action="store_true",
                    help="add a 'source_file_name' column: one fixed empty-box photo per box, repeated "
                         "for every row, for training the flow-matching trajectory to start from the "
                         "empty box and edit toward the target rather than starting from pure noise.")
    args = p.parse_args()
    if args.split == "one-cube":
        build_one_cube_split(args.box, args.out, args.eval_frac, args.seed, args.limit, args.target, args.edit_mode)
    else:
        build(args.box, args.out, args.limit, args.target, args.edit_mode)
