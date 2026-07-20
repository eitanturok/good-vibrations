#!/usr/bin/env python3
"""Benchmark the live-augmentation dataloader path: augment_site (none/getitem/collate) x
device (cpu/gpu) x num_workers x prefetch_factor, plus a plain-file vs. MDS decode comparison.

MDS now stores X as the clean complex fft and y as the raw mask (see src/data/post_process.py);
extract_signal/normalize_fft/tokenize + augmentation run live, either per-sample in
VibrationDataset.__getitem__ ("getitem") or batched in augmenting_collate ("collate"). This
script answers three questions:
  1. Which augment_site is faster, and does either keep up with the "none" (no augmentation,
     matches the old fully-baked-MDS throughput) baseline?
  2. Does moving extract_signal/normalize_fft/tokenize/augment_vibration onto the GPU help,
     given per-batch host->device transfer of the (much larger, complex) clean fft array?
  3. Does MDS decode + live compute still beat reading directly from the raw per-sample .npy
     files on disk (i.e. is MDS's sharded-read win still worth it once compute isn't baked in)?

Usage: python utils/benchmark_dataloader.py --mds-path <path/to/mds> -n 30
"""
import argparse, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.dataset import VibrationDataset, augmenting_collate
from data.post_process import process_vibration, process_vibration_torch, process_image, make_rng


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
    return {"mean_batch_s": float(np.mean(times)), "min_batch_s": float(np.min(times)), "max_batch_s": float(np.max(times)),
            "samples_per_sec": total_samples / total_time, "total_samples": total_samples, "total_time_s": total_time}


def bench_augment_site_matrix(mds_path: Path, n_batches: int, batch_size: int, num_workers_list: list[int],
                               prefetch_list: list[int], augment_sites: list[str], verbose: int) -> list[dict]:
    results = []
    for augment_site in augment_sites:
        for num_workers in num_workers_list:
            for prefetch_factor in (prefetch_list if num_workers > 0 else [None]):
                # fresh StreamingDataset per config: reusing one instance across DataLoaders
                # with different num_workers hangs (its internal worker/shm state from a prior
                # iteration doesn't tear down cleanly before the next DataLoader starts).
                # batch_size must be passed to StreamingDataset too (not just the DataLoader),
                # for deterministic resumption when iterated directly (not through a Subset).
                dataset = VibrationDataset(local=str(mds_path), augment_site=augment_site, batch_size=batch_size)
                collate_fn = (lambda batch, ds=dataset: augmenting_collate(batch, 0, ds.signal_mode, ds.normalize_mode, ds.patch_size, ds.freqs, augment=True)) \
                    if augment_site == "collate" else None
                dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True, persistent_workers=num_workers > 0,
                                 prefetch_factor=prefetch_factor, collate_fn=collate_fn)
                stats = time_iterations(dl, n_batches)
                row = {"augment_site": augment_site, "num_workers": num_workers, "prefetch_factor": prefetch_factor, **stats}
                results.append(row)
                if verbose:
                    print(f"  augment_site={augment_site:8s} num_workers={num_workers:2d} prefetch={str(prefetch_factor):4s} "
                          f"-> {stats['samples_per_sec']:7.1f} samples/s  (batch mean {stats['mean_batch_s']*1e3:6.1f} ms)")
                del dl
    return results


def bench_cpu_vs_gpu(mds_path: Path, n_batches: int, batch_size: int, signal_mode: str, normalize_mode: str,
                      patch_size: int, verbose: int) -> list[dict]:
    """Isolates just the process_vibration compute (extract_signal/normalize/tokenize/augment)
    on cpu (numpy) vs cuda (torch), holding the DataLoader/decode cost out of the timing."""
    # augment_site="collate" dataset returns raw complex fft / raw mask per sample (no
    # collate_fn attached here), so plain default collation just stacks them for us.
    dataset = VibrationDataset(local=str(mds_path), augment_site="collate", batch_size=batch_size)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches, it, target = [], iter(dl), min(n_batches, 20)
    while len(batches) < target:
        try: batches.append(next(it)["fft"].numpy())
        except StopIteration: it = iter(dl)  # wrap around if the dataset has fewer batches than requested
    freqs = dataset.freqs

    results = []
    # CPU (numpy)
    rng = make_rng(0, 0)
    t0 = time.perf_counter()
    for fft in batches:
        process_vibration(fft, freqs, signal_mode, normalize_mode, patch_size, rng=rng)
    cpu_time = time.perf_counter() - t0
    results.append({"device": "cpu", "total_time_s": cpu_time, "batches_per_sec": len(batches) / cpu_time})
    if verbose: print(f"  cpu (numpy):  {len(batches) / cpu_time:6.1f} batches/s  ({cpu_time*1e3/len(batches):.2f} ms/batch)")

    if torch.cuda.is_available():
        freqs_t = torch.from_numpy(freqs).cuda()
        # warmup (cuda context / kernel compilation shouldn't count against the timing)
        fft_t = torch.from_numpy(batches[0]).cuda()
        process_vibration_torch(fft_t, freqs_t, signal_mode, normalize_mode, patch_size, rng=rng)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for fft in batches:
            fft_t = torch.from_numpy(fft).cuda(non_blocking=True)
            process_vibration_torch(fft_t, freqs_t, signal_mode, normalize_mode, patch_size, rng=rng)
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - t0
        results.append({"device": "cuda", "total_time_s": gpu_time, "batches_per_sec": len(batches) / gpu_time})
        if verbose: print(f"  cuda (torch): {len(batches) / gpu_time:6.1f} batches/s  ({gpu_time*1e3/len(batches):.2f} ms/batch, incl. host->device transfer)")
    else:
        if verbose: print("  cuda not available -- skipping GPU comparison")
    return results


def bench_mds_vs_plain_files(mds_path: Path, samples_dir: Path | None, n_samples: int, verbose: int) -> list[dict]:
    """Compares raw per-sample decode cost: MDS StreamingDataset.__getitem__ vs. plain np.load
    of the equivalent X.npy/y.npy files, single-process, no augmentation."""
    results = []
    dataset = VibrationDataset(local=str(mds_path), augment_site="none")
    n = min(n_samples, dataset.num_samples if hasattr(dataset, "num_samples") else len(dataset))

    t0 = time.perf_counter()
    for i in range(n): dataset[i]
    mds_time = time.perf_counter() - t0
    results.append({"source": "mds", "total_time_s": mds_time, "samples_per_sec": n / mds_time})
    if verbose: print(f"  mds decode:         {n / mds_time:7.1f} samples/s")

    if samples_dir is not None and samples_dir.exists():
        sample_dirs = sorted(p for p in samples_dir.iterdir() if p.is_dir())[:n]
        t0 = time.perf_counter()
        for sd in sample_dirs:
            np.load(sd / "X.npy")
            np.load(sd / "y.npy")
        file_time = time.perf_counter() - t0
        results.append({"source": "plain_files", "total_time_s": file_time, "samples_per_sec": len(sample_dirs) / file_time})
        if verbose: print(f"  plain files decode: {len(sample_dirs) / file_time:7.1f} samples/s")
    else:
        if verbose: print("  --samples-dir not given/found -- skipping plain-file comparison")
    return results


def main():
    p = argparse.ArgumentParser(description="Benchmark the live-augmentation dataloader path.")
    p.add_argument("--mds-path", required=True, type=Path)
    p.add_argument("--samples-dir", type=Path, default=None, help="Raw samples/ dir (with per-sample X.npy/y.npy) for the MDS-vs-plain-files comparison.")
    p.add_argument("-n", "--n-batches", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers-list", type=int, nargs="+", default=[0, 2, 4, 8, 16])
    p.add_argument("--prefetch-list", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--augment-sites", nargs="+", default=["none", "getitem", "collate"], choices=["none", "getitem", "collate"])
    p.add_argument("--signal-mode", default="magnitude")
    p.add_argument("--normalize-mode", default="std-sample")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--skip-matrix", action="store_true", help="Skip the augment_site x workers x prefetch matrix.")
    p.add_argument("--skip-cpu-gpu", action="store_true")
    p.add_argument("--skip-mds-vs-files", action="store_true")
    args = p.parse_args()

    all_rows = []

    if not args.skip_matrix:
        print(f"\n{'='*70}\n1. augment_site x num_workers x prefetch_factor matrix\n{'='*70}")
        rows = bench_augment_site_matrix(args.mds_path, args.n_batches, args.batch_size, args.workers_list,
                                          args.prefetch_list, args.augment_sites, verbose=1)
        for r in rows: r["bench"] = "augment_site_matrix"
        all_rows += rows

    if not args.skip_cpu_gpu:
        print(f"\n{'='*70}\n2. CPU (numpy) vs GPU (torch) process_vibration compute\n{'='*70}")
        rows = bench_cpu_vs_gpu(args.mds_path, args.n_batches, args.batch_size, args.signal_mode,
                                 args.normalize_mode, args.patch_size, verbose=1)
        for r in rows: r["bench"] = "cpu_vs_gpu"
        all_rows += rows

    if not args.skip_mds_vs_files:
        print(f"\n{'='*70}\n3. MDS decode vs. plain-file decode (no augmentation)\n{'='*70}")
        rows = bench_mds_vs_plain_files(args.mds_path, args.samples_dir, n_samples=200, verbose=1)
        for r in rows: r["bench"] = "mds_vs_files"
        all_rows += rows

    if args.csv_out:
        import csv
        keys = sorted({k for r in all_rows for k in r})
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} rows to {args.csv_out}")


if __name__ == "__main__":
    main()
