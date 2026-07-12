#!/usr/bin/env python3
"""Benchmark the vibration-processing pipeline locally (no Modal), one timed step at a time.

Runs the same steps as watch_and_process_2.py's remote job, but on the local GPU:
  load_raw -> pclk -> clean_shifts -> fft -> recover_audio -> save_outputs

Usage: python utils/benchmark_local.py --dir D:\\eturok\\experiment-23 -n 3
"""
import argparse, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))           # utils.*
sys.path.insert(0, str(REPO / "src"))   # data.*

import numpy as np
from utils.io_utils import load, save
from data.vibrate import get_shifts, get_clean_shifts, get_fft_shifts, get_recovered_audio

def clip_rois(rois, height, width):
    """Clip ROIs to the frame. Metadata contains x<0 ROIs (laser grid past the frame
    edge); numpy would silently slice them to an empty crop and crash pclk."""
    clipped = []
    for x, y, w, h in rois:
        x2, y2 = min(x + w, width), min(y + h, height)
        x, y = max(x, 0), max(y, 0)
        if x2 - x > 0 and y2 - y > 0: clipped.append([x, y, x2 - x, y2 - y])
    return clipped

def process_sample(sample_dir: Path, batch_size: int, pclk_mode: str, do_save: bool) -> dict[str, float]:
    times = {}
    def step(name, fn):
        t0 = time.perf_counter()
        out = fn()
        times[name] = time.perf_counter() - t0
        print(f"  {name:<16} {times[name]:8.2f} s")
        return out

    metadata = {k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}
    fps = int(metadata["fps"])
    rois = clip_rois(metadata["roi"], metadata["height"], metadata["width"])
    n_clipped = len(metadata["roi"]) - len(rois)
    if n_clipped: print(f"  (clipped {n_clipped} fully out-of-frame ROIs; {len(rois)} remain)")
    if pclk_mode != "sequential":
        # batched modes np.stack the crops, so every ROI must have the same shape:
        # keep only ROIs that survived clipping at full size
        full = max((r[2], r[3]) for r in rois)
        rois = [r for r in rois if (r[2], r[3]) == full]
        print(f"  (batched mode: kept {len(rois)} full-size {full} ROIs)")

    raw = step("load_raw", lambda: load(sample_dir / "inputs/00_raw_vibrations.npy"))
    raw_shifts = step("pclk", lambda: get_shifts(raw, rois, batch_size, pclk_mode))          # (L, T, 2)
    clean_shifts = step("clean_shifts", lambda: get_clean_shifts(raw_shifts[None], fps))     # (B, L, T, 2)
    fft, freqs, n_samples = step("fft", lambda: get_fft_shifts(clean_shifts, fps))           # (B, L, F, 2)
    audio = step("recover_audio", lambda: get_recovered_audio(fft, n_samples, fps))

    def save_outputs():
        save(raw_shifts, sample_dir / "inputs/01_raw_shifts.npy", do_save)
        save(clean_shifts, sample_dir / "inputs/02_clean_shifts.npy", do_save)
        save({"fft": fft, "freqs": freqs, "n_samples": n_samples}, sample_dir / "inputs/03_fft.npz", do_save)
        save((audio, 22050), sample_dir / "inputs/04_recovered_audio.wav", do_save)
    step("save_outputs", save_outputs)

    times["TOTAL"] = sum(times.values())
    return times

def main():
    p = argparse.ArgumentParser(description="Benchmark local vibration processing per step.")
    p.add_argument("--dir", required=True, help="Experiment directory (same as watch_and_process_2.py).")
    p.add_argument("-n", type=int, default=3, help="Number of samples to benchmark.")
    p.add_argument("--pclk-batch-size", type=int, default=1024)
    p.add_argument("--pclk-mode", default="sequential", choices=["sequential", "batched", "batched_optimized"])
    p.add_argument("--no-save", action="store_true", help="Skip writing pipeline outputs to the sample dirs.")
    args = p.parse_args()

    raw_paths = sorted(Path(args.dir).rglob("**/inputs/00_raw_vibrations.npy"))[: args.n]
    if not raw_paths: sys.exit(f"no raw vibration files found under {args.dir}")

    all_times = []
    for raw_path in raw_paths:
        sample_dir = raw_path.parents[1]
        print(f"\nsample {sample_dir.name} ({raw_path.stat().st_size / 2**30:.2f} GB, mode={args.pclk_mode})")
        all_times.append(process_sample(sample_dir, args.pclk_batch_size, args.pclk_mode, not args.no_save))

    print(f"\n{'=' * 44}\nmean over {len(all_times)} sample(s):")
    for name in all_times[0]:
        vals = [t[name] for t in all_times]
        print(f"  {name:<16} {np.mean(vals):8.2f} s  (min {min(vals):.2f} / max {max(vals):.2f})")

if __name__ == "__main__":
    main()
