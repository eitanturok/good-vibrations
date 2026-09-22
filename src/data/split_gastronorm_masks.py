"""Derive per-object smasks for gastronorm from the merged mask we already have, instead of
re-running SAM3.

Gastronorm's `image/03_smask.npy` is `np.any([...all object masks...], axis=0)` (see
`resegment_gastronorm.py`'s module docstring for the full backstory). Since no two objects in
one photo touch or overlap, that merge is *lossless*: each object is still its own separate
connected component in the merged mask, so plain connected-component labeling recovers exactly
the same per-object masks that would come out of the original per-prompt SAM3 masks -- no GPU,
no re-segmentation, no Modal cost.

We still need to know WHICH component is which named object (the merged mask has no identity
info). metadata.jsonl already has that: `coms` is the exact per-object centroid computed by
record.ipynb from the ORIGINAL (never-saved) per-object masks, grouped by object name in the
same order as `objects`. So: label the merged mask's connected components, compute each
component's centroid, and do an optimal (Hungarian) assignment between components and named
objects by centroid distance. If the two centroid sets truly come from the same underlying
masks, this assignment is exact (see the sanity check below).

Sanity check (`--verify N`): pick N positions, run the real SAM3 Segmenter (src/data/segment.py)
on them to get actual per-object masks, and compare against what this script derives via IoU --
confirms the connected-component shortcut reproduces real per-object masks before trusting it
for the other ~364 positions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.optimize import linear_sum_assignment

DATA_DIR = Path("experiments/31_07_2026_gastronorm_exp1")
STRUCTURE = np.ones((3, 3), dtype=bool)  # 8-connectivity: a single diagonal-touching pixel still joins


def load_meta(sample_dir: Path) -> dict:
    meta = {}
    for line in (sample_dir / "metadata.jsonl").read_text().splitlines():
        if line:
            meta.update(json.loads(line))
    return meta


def collect_positions(data_dir: Path) -> dict[int, list[tuple[Path, dict]]]:
    positions: dict[int, list[tuple[Path, dict]]] = {}
    for sample_dir in sorted((data_dir / "samples").glob("*")):
        if not sample_dir.is_dir():
            continue
        meta = load_meta(sample_dir)
        positions.setdefault(meta["position_id"], []).append((sample_dir, meta))
    return positions


def named_coms(meta: dict) -> list[tuple[str, int, tuple[float, float]]]:
    """metadata's `coms`/`objects` -> [(obj_name, instance_idx, (row, col)), ...], flattened in
    the same object/instance order record.ipynb saved them in."""
    out = []
    for (obj, _), com_list in zip(meta["objects"].items(), meta["coms"]):
        for i, com in enumerate(com_list):
            out.append((obj, i, tuple(com)))  # com is always a (row, col) pair
    return out


def split_position(rep_dir: Path, meta: dict, min_size: int = 20) -> dict:
    """Label connected components in the merged mask and match them to named objects by
    centroid. Returns a report dict; does NOT write anything (caller does, once satisfied)."""
    merged = np.load(rep_dir / "image/03_smask.npy")
    labeled, n_components = label(merged, structure=STRUCTURE)
    sizes = np.bincount(labeled.ravel())[1:]  # size of components 1..n_components
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_size]  # drop speckle noise

    targets = named_coms(meta)
    n_expected = len(targets)
    ok = len(keep) == n_expected

    result = dict(position_id=meta["position_id"], n_expected=n_expected,
                  n_components=len(keep), n_components_raw=n_components, ok=ok, masks=None)
    if not ok:
        return result

    comp_coms = [center_of_mass(labeled == i) for i in keep]
    target_coms = [t[2] for t in targets]
    # cost[i, j] = distance from component i to target object j -> optimal one-to-one assignment
    cost = np.linalg.norm(
        np.array(comp_coms)[:, None, :] - np.array(target_coms)[None, :, :], axis=2
    )
    comp_idx, target_idx = linear_sum_assignment(cost)
    result["max_match_dist_px"] = float(cost[comp_idx, target_idx].max()) if len(comp_idx) else 0.0

    masks = {}
    for ci, ti in zip(comp_idx, target_idx):
        obj, instance_i, _ = targets[ti]
        masks[f"{obj}{instance_i}"] = (labeled == keep[ci])
    result["masks"] = masks
    return result


def write_masks(rep_dir: Path, sibling_dirs: list[Path], masks: dict[str, np.ndarray]) -> None:
    """Writes independent copies into every sibling dir (not symlinks, not hardlinks) --
    each sample dir ends up with its own real `image/smasks/*.npy` bytes on disk, per instance."""
    smasks_dir = rep_dir / "image/smasks"
    smasks_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        np.save(smasks_dir / f"{name}.npy", mask)
    for sibling_dir in sibling_dirs:
        sibling_smasks = sibling_dir / "image/smasks"
        sibling_smasks.mkdir(parents=True, exist_ok=True)
        for name, mask in masks.items():
            np.save(sibling_smasks / f"{name}.npy", mask)


def run_split(data_dir: Path, n_samples: int | None, execute: bool) -> None:
    ok_jobs, bad_jobs, n_empty, n_already_done = [], [], 0, 0
    for position_id, group in collect_positions(data_dir).items():
        rep_dir, meta = group[0]
        if not meta.get("objects"):
            n_empty += 1
            continue  # empty box, nothing to split
        if (rep_dir / "image/smasks").exists():
            n_already_done += 1
            continue  # already has per-object masks
        r = split_position(rep_dir, meta)
        r["rep_dir"], r["sibling_dirs"] = rep_dir, [d for d, _ in group[1:]]
        (ok_jobs if r["ok"] else bad_jobs).append(r)

    if n_samples is not None:
        ok_jobs = ok_jobs[:n_samples]

    print(f"{n_empty} empty-box positions skipped, {n_already_done} already have image/smasks/")
    print(f"{len(ok_jobs) + len(bad_jobs)} positions still need splitting"
          + (f" ({len(ok_jobs)} selected by --n-samples)" if n_samples is not None else ""))
    print(f"  {len(ok_jobs)} matched cleanly (n_components == n_expected_objects)")
    print(f"  {len(bad_jobs)} MISMATCHED (component count != expected object count -- likely "
          f"touching/overlapping objects or segmentation noise; these need the SAM3 fallback "
          f"in resegment_gastronorm.py instead)")
    if bad_jobs:
        print("  mismatched positions:", [j["position_id"] for j in bad_jobs[:20]],
              "..." if len(bad_jobs) > 20 else "")
    if ok_jobs:
        dists = [j["max_match_dist_px"] for j in ok_jobs]
        print(f"  centroid match distance across clean positions: max={max(dists):.4f}px, "
              f"mean={sum(dists)/len(dists):.4f}px (near 0 confirms components == real objects)")

    if execute:
        for j in ok_jobs:
            write_masks(j["rep_dir"], j["sibling_dirs"], j["masks"])
        n_files = sum(len(j["masks"]) for j in ok_jobs)
        n_dirs = sum(1 + len(j["sibling_dirs"]) for j in ok_jobs)
        print(f"\nwrote {n_files} mask files across {n_dirs} sample dirs "
              f"({len(ok_jobs)} positions)")
    else:
        print("\n(dry run -- nothing written. Pass --execute to write image/smasks/*.npy for "
              "the cleanly-matched positions.)")


def verify(data_dir: Path, n_verify: int) -> None:
    """Real sanity check: run actual SAM3 on n_verify positions and compare IoU against what
    connected-component splitting derives from the merged mask, for the SAME positions."""
    import subprocess
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for utils.io_utils
    import modal
    from utils.io_utils import unpack_masks

    result = subprocess.run(
        [sys.executable, "-m", "modal", "deploy", "src/data/segment.py"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    result.check_returncode()
    segmenter = modal.Cls.from_name("segment", "Segmenter")()

    checked = 0
    for position_id, group in collect_positions(data_dir).items():
        if checked >= n_verify:
            break
        rep_dir, meta = group[0]
        if not meta.get("objects"):
            continue
        derived = split_position(rep_dir, meta)
        if not derived["ok"]:
            print(f"position {position_id}: SKIPPED (connected-component match failed, can't compare)")
            continue

        image_bytes = (rep_dir / "image/02_cropped_overhead.png").read_bytes()
        seg_results = unpack_masks(segmenter.run.remote(image_bytes, meta["prompts"], top_k=list(meta["objects"].values())))
        # Match fresh SAM3 masks to derived masks by nearest centroid, NOT by positional index --
        # a second SAM3 call can return same-name instances in a different order than the
        # original capture-time call did (only matters when some object has >1 instance; every
        # gastronorm object today has exactly 1, so this never actually swaps here, but matching
        # by index instead of geometry would silently mismatch names if that ever changes).
        real_masks_flat = [(f"{obj}{i}", m) for (obj, _), r in zip(meta["objects"].items(), seg_results)
                            for i, m in enumerate(r["masks"])]
        real_coms = [center_of_mass(m) for _, m in real_masks_flat]
        derived_items = list(derived["masks"].items())
        derived_coms = [center_of_mass(m) for _, m in derived_items]

        print(f"position {position_id}:")
        if len(real_coms) != len(derived_coms):
            print(f"  MISMATCH: derived {len(derived_coms)} instances, real SAM3 call returned "
                  f"{len(real_coms)} -- skipping IoU comparison")
            continue
        cost = np.linalg.norm(np.array(derived_coms)[:, None, :] - np.array(real_coms)[None, :, :], axis=2)
        derived_idx, real_idx = linear_sum_assignment(cost)
        for di, ri in zip(derived_idx, real_idx):
            derived_name, derived_mask = derived_items[di]
            real_name, real_mask = real_masks_flat[ri]
            intersection = np.logical_and(derived_mask, real_mask).sum()
            union = np.logical_or(derived_mask, real_mask).sum()
            iou = intersection / union if union else float("nan")
            match_note = "" if derived_name == real_name else f" (matched to real {real_name!r})"
            print(f"  {derived_name}{match_note}: IoU = {iou:.4f} "
                  f"(derived {derived_mask.sum()}px, real {real_mask.sum()}px)")
        checked += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=None,
                         help="Only process the first N cleanly-matched positions. Omit for all.")
    parser.add_argument("--execute", action="store_true",
                         help="Write image/smasks/*.npy for cleanly-matched positions. Default "
                              "is dry-run (report only, nothing written).")
    parser.add_argument("--verify", type=int, default=0, metavar="N",
                         help="Instead of splitting, run real SAM3 on N positions via Modal and "
                              "report IoU against the connected-component-derived masks, as a "
                              "sanity check. Does not write anything.")
    args = parser.parse_args()

    if args.verify:
        verify(args.data_dir, args.verify)
    else:
        run_split(args.data_dir, args.n_samples, args.execute)
