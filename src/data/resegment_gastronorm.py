"""Retroactively create per-object smasks for the gastronorm dataset.

Gastronorm (experiments/31_07_2026_gastronorm_exp1) was captured on 2026-07-31, a month before
commit 5c866f9 ("properly save individual smasks") added per-object mask saving to
src/record.ipynb. So gastronorm only ever got the merged `image/03_smask.png` (a logical OR
across every object's mask, per-instance identity discarded) -- unlike shoebox and every
box captured after that fix, which also has `image/smasks/{object}{i}.npy` per instance.

The raw per-object mask arrays cannot be recovered from metadata.jsonl (its `seg_results.masks`
field is a truncated numpy repr() string, not parseable data). But everything SAM3 needs to
RE-segment is still there: `image/02_cropped_overhead.png` (the exact image that was segmented)
plus `objects`/`prompts` in metadata.jsonl (the exact text prompts and per-prompt instance counts
originally used). So we can re-run the same Segmenter.run() call record.ipynb made at capture
time and save the per-object masks it returns, using the SAME naming convention shoebox uses
(`image/smasks/{obj}{i}.npy`) -- no new capture, no new equipment, just a second segmentation
pass over already-saved images.

Dedup: within one position_id, every speaker's `02_cropped_overhead.png` is the same file
(same inode -- verified below), so segmentation only needs to run ONCE per position; the
resulting masks are then copied to every sample dir sharing that position.
"""
import argparse
import json
from pathlib import Path

import numpy as np

DATA_DIR = Path("experiments/31_07_2026_gastronorm_exp1")


def load_meta(sample_dir: Path) -> dict:
    meta = {}
    for line in (sample_dir / "metadata.jsonl").read_text().splitlines():
        if line:
            meta.update(json.loads(line))
    return meta


def collect_positions(data_dir: Path) -> dict[int, list[tuple[Path, dict]]]:
    """{position_id: [(sample_dir, metadata), ...]}, in sample_dir sort order."""
    positions: dict[int, list[tuple[Path, dict]]] = {}
    for sample_dir in sorted((data_dir / "samples").glob("*")):
        if not sample_dir.is_dir():
            continue
        meta = load_meta(sample_dir)
        positions.setdefault(meta["position_id"], []).append((sample_dir, meta))
    return positions


def plan(data_dir: Path) -> list[dict]:
    """One entry per position that needs segmentation: the representative (first) sample dir
    to read the image/prompts from, the sibling dirs to copy results into, and the exact
    Segmenter.run() args (prompts, top_k) that record.ipynb would have used."""
    jobs = []
    for position_id, group in collect_positions(data_dir).items():
        rep_dir, meta = group[0]
        objects = meta.get("objects", {})
        if not objects:
            continue  # empty-box position: nothing to segment
        already_done = (rep_dir / "image/smasks").exists()
        jobs.append(dict(
            position_id=position_id,
            rep_dir=rep_dir,
            sibling_dirs=[d for d, _ in group[1:]],
            image_path=rep_dir / "image/02_cropped_overhead.png",
            prompts=meta["prompts"],
            objects=objects,
            top_k=list(objects.values()),
            already_done=already_done,
        ))
    return jobs


def report(jobs: list[dict], todo: list[dict]) -> None:
    print(f"{len(jobs)} positions have objects to segment ({len(jobs) - len(todo)} already have "
          f"image/smasks/ or are excluded by --n-samples, {len(todo)} selected to run)")
    n_samples = sum(1 + len(j["sibling_dirs"]) for j in todo)
    print(f"would make {len(todo)} Segmenter.run() calls (one per position), "
          f"writing masks into {n_samples} sample dirs total (incl. dedup copies)")

    from collections import Counter
    combo_counts = Counter(tuple(sorted(j["objects"].items())) for j in todo)
    print("\nobject-combo breakdown (selected jobs only):")
    for combo, n in combo_counts.most_common():
        print(f"  {dict(combo)} -> {n} positions")

    print("\nfirst 3 selected jobs:")
    for j in todo[:3]:
        n_files = sum(j["top_k"])
        print(f"  position {j['position_id']}: image={j['image_path']}, prompts={j['prompts']}, "
              f"top_k={j['top_k']} -> {n_files} mask files -> "
              f"{j['rep_dir'] / 'image/smasks'}/{{obj}}{{i}}.npy, "
              f"copied to {len(j['sibling_dirs'])} sibling dirs")


def run_one(segmenter, job: dict) -> dict:
    """Do the actual Modal call + disk writes for one job. Runs on a worker thread --
    modal's .remote() blocks the calling thread until the result is back, so bounding how
    many threads call it concurrently is exactly the same as bounding how many uploads/inference
    calls are in flight at once (what --concurrency controls)."""
    image_bytes = job["image_path"].read_bytes()
    seg_results = segmenter.run.remote(image_bytes, job["prompts"], top_k=job["top_k"])
    smasks_dir = job["rep_dir"] / "image/smasks"
    smasks_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for (obj, _), result in zip(job["objects"].items(), seg_results):
        for i, mask in enumerate(result["masks"]):
            path = smasks_dir / f"{obj}{i}.npy"
            np.save(path, mask)
            saved.append(path)
    for sibling_dir in job["sibling_dirs"]:
        sibling_smasks = sibling_dir / "image/smasks"
        sibling_smasks.mkdir(parents=True, exist_ok=True)
        for src_path in saved:
            dst_path = sibling_smasks / src_path.name
            if not dst_path.exists():
                dst_path.hardlink_to(src_path)
    return dict(position_id=job["position_id"], n_saved=len(saved), n_siblings=len(job["sibling_dirs"]))


def execute(todo: list[dict], concurrency: int) -> None:
    """Actually call SAM3 via Modal (bounded to `concurrency` concurrent calls -- tune this to
    your upload bandwidth: each call uploads one ~1-2MB image, so `concurrency` in-flight calls
    means roughly that many MB in flight at once) and write image/smasks/{obj}{i}.npy."""
    import subprocess
    import sys
    import modal
    from concurrent.futures import ThreadPoolExecutor, as_completed

    result = subprocess.run(
        [sys.executable, "-m", "modal", "deploy", "src/data/segment.py"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    result.check_returncode()
    segmenter = modal.Cls.from_name("segment", "Segmenter")()

    n_done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_one, segmenter, j): j for j in todo}
        for future in as_completed(futures):
            job = futures[future]
            try:
                r = future.result()
            except Exception as e:
                print(f"position {job['position_id']}: FAILED -- {e!r}")
                continue
            n_done += 1
            print(f"[{n_done}/{len(todo)}] position {r['position_id']}: saved {r['n_saved']} masks, "
                  f"copied to {r['n_siblings']} sibling dirs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--n-samples", type=int, default=None,
                         help="Only process the first N positions still needing segmentation "
                              "(after skipping empty-box and already-done positions). Omit to "
                              "process all of them. Use this to control cost/time on a --execute run.")
    parser.add_argument("--concurrency", type=int, default=8,
                         help="Max concurrent Segmenter.run() calls during --execute (each call "
                              "uploads one image, roughly --concurrency MB in flight at once -- "
                              "tune down for slow upload bandwidth, up for a faster connection).")
    parser.add_argument("--execute", action="store_true",
                         help="Actually call SAM3 via Modal and write files. Default is dry-run "
                              "(plan + report only, no Modal calls, no writes).")
    args = parser.parse_args()

    jobs = plan(args.data_dir)
    todo = [j for j in jobs if not j["already_done"]]
    if args.n_samples is not None:
        todo = todo[:args.n_samples]
    report(jobs, todo)
    if args.execute:
        execute(todo, args.concurrency)
    else:
        print(f"\n(dry run -- no Modal calls made, no files written. Pass --execute to run for "
              f"real (--n-samples to limit, --concurrency={args.concurrency} to control parallelism).)")
