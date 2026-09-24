"""Offline: for each gastronorm sample with real per-object masks (image/smasks/*.npy), build a
dense instance-id map, encode it through the frozen LDMSeg encoder to get z_gt, and read off each
instance's target_class from the decoder's own output (see src5/ldmseg_ae.py's dominant_class).
Caches everything to image/05_ldmseg_latent.npz so training never touches the frozen encoder or
the mask files again. --same-label caches to image/05_ldmseg_latent_same_label.npz instead: every
instance is bit-encoded with the same id, so z_gt only ever encodes where objects are, never which
object -- see build_id_map.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]  # src/ for model.*, repo root for utils.* (used by model.dataset)

from model.dataset import downsample_mask  # noqa: E402
from src5.ldmseg_ae import load_frozen_ae, encode_bitmap, dominant_class  # noqa: E402

MASK_RES = 512
DATA_DIR = REPO / "experiments" / "31_07_2026_gastronorm_exp1"
CACHE_NAME = "05_ldmseg_latent.npz"
CACHE_NAME_SAME_LABEL = "05_ldmseg_latent_same_label.npz"


def build_id_map(instance_paths: list[Path], same_label: bool = False) -> tuple[np.ndarray, list[np.ndarray]]:
    """(id_map at 512x512, [each instance's own 512x512 boolean mask]). Each instance is
    downsampled+thresholded SEPARATELY -- never box-filter a combined multi-id map, since that
    would blend adjacent instances' ids into meaningless fractional values at their boundary.

    same_label=True: every instance is bit-encoded with id=1 instead of a distinct id per
    instance -- the frozen encoder then only ever sees "object present here" (one shared label),
    never which object, so z_gt (and everything downstream of it) can't carry instance identity.
    masks_512 stays per-instance regardless -- only the id fed to the encoder collapses."""
    id_map = np.zeros((MASK_RES, MASK_RES), dtype=np.int64)
    masks_512 = []
    for i, path in enumerate(sorted(instance_paths), start=1):
        raw = np.load(path)
        img = Image.fromarray((raw * 255).astype(np.uint8))
        mask_512 = downsample_mask(img, MASK_RES, MASK_RES) > 0.5
        id_map[mask_512] = 1 if same_label else i
        masks_512.append(mask_512)
    return id_map, masks_512


def precompute_sample(model, sample_dir: Path, same_label: bool = False) -> dict:
    """Empty-box samples (no image/smasks/ dir -- nothing to segment) still need a cache entry:
    gastronorm_split puts them in the train split unconditionally ("empty-box is always train, no
    eval carve-out"), so load_latents will be asked to load one for them too. Their id_map is all
    background (all zeros) -- z_gt just encodes "nothing here", with 0 instances."""
    smasks_dir = sample_dir / "image" / "smasks"
    instance_paths = sorted(smasks_dir.glob("*.npy")) if smasks_dir.exists() else []

    id_map, masks_512 = build_id_map(instance_paths, same_label=same_label)
    bits = encode_bitmap(torch.from_numpy(id_map))[None]
    with torch.no_grad():
        z_gt = model.encode(bits).latent_dist.mode()

    if masks_512:
        argmax_map = model.decode(z_gt, interpolate=True).argmax(1)[0]
        target_classes = [dominant_class(argmax_map, torch.from_numpy(m)) for m in masks_512]
        masks_out = np.stack(masks_512)
    else:
        target_classes = []
        masks_out = np.zeros((0, MASK_RES, MASK_RES), dtype=bool)
    return dict(z_gt=z_gt[0].numpy(), masks_512=masks_out,
                target_classes=np.array(target_classes, dtype=np.int64))


def main(data_dir: Path, n_samples: int | None, same_label: bool = False) -> None:
    cache_name = CACHE_NAME_SAME_LABEL if same_label else CACHE_NAME
    model = load_frozen_ae()
    sample_dirs = sorted((data_dir / "samples").glob("*"))
    todo = [d for d in sample_dirs if not (d / "image" / cache_name).exists()]
    n_already_cached = len(sample_dirs) - len(todo)
    if n_samples is not None:
        todo = todo[:n_samples]
    print(f"{len(sample_dirs)} total samples, {n_already_cached} already cached, "
          f"{len(todo)} selected to process now (cache file: {cache_name})")

    n_done = n_collisions = 0
    for sample_dir in todo:
        result = precompute_sample(model, sample_dir, same_label=same_label)
        tc = result["target_classes"]
        n_collisions += len(tc) != len(set(tc.tolist()))
        out_path = sample_dir / "image" / cache_name
        tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")  # np.savez appends .npz if missing
        np.savez(tmp_path, **result)
        tmp_path.replace(out_path)
        n_done += 1
    print(f"wrote {n_done} caches, {n_collisions} had same-target_class collisions between "
          f"co-occurring instances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--same-label", action="store_true",
                         help="bit-encode every instance with the same id (1) instead of a "
                              "distinct id per instance, so z_gt only ever encodes object "
                              "presence, not identity -- writes to a separate cache file "
                              f"({CACHE_NAME_SAME_LABEL}) so it never clobbers the default cache.")
    args = parser.parse_args()
    main(args.data_dir, args.n_samples, args.same_label)
