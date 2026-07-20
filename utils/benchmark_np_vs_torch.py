#!/usr/bin/env python3
"""Benchmark dataset_np.py (numpy/scipy/PIL, cpu-only) vs dataset_torch.py (torch, cpu+cuda)
for both augment_site="getitem" and "collate", at a fixed batch size.

Usage: python utils/benchmark_np_vs_torch.py --mds-path <path/to/mds> --batch-size 128
"""
import argparse, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from torch.utils.data import DataLoader

from model.dataset_np import VibrationDatasetNp, augmenting_collate as augmenting_collate_np
from model.dataset_torch import VibrationDatasetTorch, augmenting_collate as augmenting_collate_torch


def time_iterations(dl, n_batches: int, warmup: int = 3) -> dict:
    it = iter(dl)
    for _ in range(warmup):
        try: next(it)
        except StopIteration: it = iter(dl); break

    times, n_samples = [], []
    t_start = time.perf_counter()
    for _ in range(n_batches):
        t0 = time.perf_counter()
        try: batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        times.append(time.perf_counter() - t0)
        n_samples.append(batch["mask_true"].shape[0])
    total_time = time.perf_counter() - t_start
    total_samples = sum(n_samples)
    return {"mean_batch_s": float(np.mean(times)), "samples_per_sec": total_samples / total_time}


def bench_np(mds_path: Path, batch_size: int, n_batches: int, num_workers: int, prefetch_factor: int, augment_site: str) -> dict:
    dataset = VibrationDatasetNp(local=str(mds_path), augment_site=augment_site, batch_size=batch_size)
    collate_fn = (lambda batch: augmenting_collate_np(batch, dataset.rng, dataset.signal_mode, dataset.normalize_mode, dataset.patch_size, dataset.out_h, dataset.out_w, dataset.freqs)) \
        if augment_site == "collate" else None
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                     pin_memory=True, persistent_workers=num_workers > 0,
                     prefetch_factor=prefetch_factor if num_workers > 0 else None, collate_fn=collate_fn)
    return time_iterations(dl, n_batches)


def bench_torch(mds_path: Path, batch_size: int, n_batches: int, num_workers: int, prefetch_factor: int, augment_site: str, device: str) -> dict:
    dataset = VibrationDatasetTorch(local=str(mds_path), augment_site=augment_site, batch_size=batch_size, device=device)
    collate_fn = (lambda batch: augmenting_collate_torch(batch, dataset.generator, dataset.signal_mode, dataset.normalize_mode, dataset.patch_size, dataset.out_h, dataset.out_w, dataset.freqs, device)) \
        if augment_site == "collate" else None
    # num_workers>0 + device="cuda": each worker subprocess would need its own cuda context,
    # which torch DataLoader workers don't set up automatically -- restrict cuda runs to
    # num_workers=0 (main-process) here; num_workers>0 is only meaningful for device="cpu".
    workers = 0 if device == "cuda" else num_workers
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                     pin_memory=(device == "cpu"), persistent_workers=workers > 0,
                     prefetch_factor=prefetch_factor if workers > 0 else None, collate_fn=collate_fn)
    return time_iterations(dl, n_batches)


def main():
    p = argparse.ArgumentParser(description="Benchmark dataset_np vs dataset_torch (cpu/cuda), getitem vs collate.")
    p.add_argument("--mds-path", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("-n", "--n-batches", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--csv-out", type=Path, default=None)
    args = p.parse_args()

    import torch
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    rows = []
    print(f"\n{'implementation':<14} {'device':<6} {'augment_site':<10} {'workers':>8} {'samples/s':>12} {'ms/batch':>10}")
    print("-" * 66)

    for augment_site in ["getitem", "collate"]:
        stats = bench_np(args.mds_path, args.batch_size, args.n_batches, args.num_workers, args.prefetch_factor, augment_site)
        rows.append({"impl": "numpy", "device": "cpu", "augment_site": augment_site, "num_workers": args.num_workers, **stats})
        print(f"{'numpy':<14} {'cpu':<6} {augment_site:<10} {args.num_workers:>8} {stats['samples_per_sec']:>12.1f} {stats['mean_batch_s']*1e3:>10.1f}")

    for device in devices:
        for augment_site in ["getitem", "collate"]:
            workers = 0 if device == "cuda" else args.num_workers
            stats = bench_torch(args.mds_path, args.batch_size, args.n_batches, args.num_workers, args.prefetch_factor, augment_site, device)
            rows.append({"impl": "torch", "device": device, "augment_site": augment_site, "num_workers": workers, **stats})
            print(f"{'torch':<14} {device:<6} {augment_site:<10} {workers:>8} {stats['samples_per_sec']:>12.1f} {stats['mean_batch_s']*1e3:>10.1f}")

    if args.csv_out:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.csv_out}")


if __name__ == "__main__":
    main()
