"""Benchmark: uploading the FULL raw_vibrations vs only the ROI crops to Modal.

The capture pipeline currently uploads the entire (frames, H, W) raw video (~3 GB)
to Modal just to run pclk, which only ever touches the ROI crops. This measures how
many bytes and how much wall-clock upload time we'd save by stacking the crops first
and uploading only those.

Usage:
    python scripts/benchmark_upload.py --sample-dir D:/eturok/experiment-20/data/cardboard/samples/000000
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src3"))
from vibrations_pipeline import volume  # the modal "samples" Volume

GiB = 2**30
MB = 2**20


def load_rois(sample_dir: Path):
    meta = {k: v for d in (json.loads(l) for l in (sample_dir / "metadata.jsonl").read_text().splitlines() if l.strip()) for k, v in d.items()}
    return meta["roi"]


def build_roi_array(raw, rois):
    # same crop logic as get_shifts in vibrations_pipeline.py
    return np.stack([raw[:, y:y + h, x:x + w] for x, y, w, h in rois])  # (L, T, h, w)


def time_upload(local_path: Path, remote_path: str) -> float:
    t0 = time.perf_counter()
    with volume.batch_upload(force=True) as batch:
        batch.put_file(local_path, remote_path)
    return time.perf_counter() - t0


def write_novel(path: Path, nbytes: int):
    """Write `nbytes` of random data so Modal's block-level dedup can't short-circuit
    the upload — every real capture is novel data, so this is the faithful case."""
    rng = np.random.default_rng()
    chunk = 64 * MB
    with open(path, "wb") as f:
        remaining = nbytes
        while remaining > 0:
            n = min(chunk, remaining)
            f.write(rng.integers(0, 256, size=n, dtype=np.uint8).tobytes())
            remaining -= n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", required=True, type=Path)
    ap.add_argument("--keep-remote", action="store_true", help="don't delete the benchmark files from the volume afterwards")
    args = ap.parse_args()

    raw_path = args.sample_dir / "inputs/00_raw_vibrations.npy"
    rois = load_rois(args.sample_dir)

    print(f"loading {raw_path} (memmap)...")
    raw = np.load(raw_path, mmap_mode="r")
    print(f"  raw_vibrations: shape={raw.shape} dtype={raw.dtype} -> {raw.nbytes/GiB:.3f} GiB")

    print(f"building ROI crops ({len(rois)} ROIs of {rois[0][2]}x{rois[0][3]})...")
    roi_arr = build_roi_array(raw, rois)
    print(f"  roi_crops:      shape={roi_arr.shape} dtype={roi_arr.dtype} -> {roi_arr.nbytes/GiB:.3f} GiB")

    full_bytes = raw_path.stat().st_size
    roi_bytes = int(roi_arr.nbytes) + 128  # +npy header, matches what np.save would write
    del roi_arr

    # Modal dedups identical blocks, so re-uploading the real files (already in the
    # volume) measures hashing, not transfer. Every real capture is novel, so we
    # upload random data of the SAME byte sizes to get the true uplink throughput.
    tag = int(time.time())
    with tempfile.TemporaryDirectory() as tmp:
        full_local = Path(tmp) / "full.bin"
        roi_local = Path(tmp) / "roi.bin"
        print(f"writing {full_bytes/GiB:.3f} GiB of novel data for FULL ...")
        write_novel(full_local, full_bytes)
        print(f"writing {roi_bytes/GiB:.3f} GiB of novel data for ROI ...")
        write_novel(roi_local, roi_bytes)

        print("\nuploading FULL raw_vibrations (novel bytes) ...")
        full_t = time_upload(full_local, f"benchmark/{tag}_full.bin")
        print(f"  full upload: {full_t:.1f} s  ({full_bytes/MB/full_t:.1f} MB/s)")

        print("uploading ROI crops only (novel bytes) ...")
        roi_t = time_upload(roi_local, f"benchmark/{tag}_roi.bin")
        print(f"  roi  upload: {roi_t:.1f} s  ({roi_bytes/MB/roi_t:.1f} MB/s)")

    saved = full_bytes - roi_bytes
    print("\n========== RESULTS ==========")
    print(f"full bytes:   {full_bytes/GiB:6.3f} GiB")
    print(f"roi  bytes:   {roi_bytes/GiB:6.3f} GiB   ({roi_bytes/full_bytes*100:.1f}% of full)")
    print(f"bytes saved:  {saved/GiB:6.3f} GiB   ({saved/full_bytes*100:.1f}% reduction)")
    print(f"full time:    {full_t:6.1f} s")
    print(f"roi  time:    {roi_t:6.1f} s")
    print(f"speedup:      {full_t/roi_t:6.2f}x faster")

    if not args.keep_remote:
        print("\ncleaning up remote benchmark files ...")
        for f in (f"benchmark/{tag}_full.bin", f"benchmark/{tag}_roi.bin"):
            try:
                volume.remove_file(f)
            except Exception as e:
                print(f"  could not remove {f}: {e}")


if __name__ == "__main__":
    main()
