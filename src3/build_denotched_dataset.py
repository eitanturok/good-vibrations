import argparse, hashlib, json, os
from pathlib import Path

import numpy as np

from io_utils import copy, symlink, load, save, append, Timing, print_system_usage
from post_process import extract_signal, normalize_fft, tokenize, convert_to_mds, MDS_COLUMNS
from denotch import interpolate_notch

# files that don't depend on the FFT at all and can be symlinked straight from the original sample
UNCHANGED_TOP_FILES = ["audio.wav", "recovered_audio.wav", "overhead.png", "y.npy"]
UNCHANGED_INPUT_FILES = ["04_recovered_audio.wav"]
UNCHANGED_OUTPUT_FILES = ["00_raw_overhead.png", "01_resized_overhead.png", "02_segment_mask.png", "03_downsampled_segment_mask.png", "05_overhead.png"]
COPIED_FILES = ["metadata.jsonl", "times.jsonl", "outputs/04_com.jsonl"]  # read-then-appended-to elsewhere; must be real copies

FFT_NAME = "inputs/03_fft_shifts.npz"
PROCESSED_FFT_NAME = "inputs/05_processed_fft.npy"


def build_sample(src_sample_dir: Path, dst_sample_dir: Path, signal_mode: str, normalize_mode: str, patch_size: int,
                  denotch_fn, do_save: bool, verbose: int) -> dict | None:
    sample_id = src_sample_dir.name
    src_fft_path = src_sample_dir / FFT_NAME
    if not src_fft_path.exists():
        if verbose: print(f"[sample {sample_id}] skip: missing {FFT_NAME}")
        return None

    for f in UNCHANGED_TOP_FILES:
        src = src_sample_dir / f
        if src.exists(): symlink(src, dst_sample_dir / f, do_save)
    for f in UNCHANGED_INPUT_FILES:
        src = src_sample_dir / "inputs" / f
        if src.exists(): symlink(src, dst_sample_dir / "inputs" / f, do_save)
    for f in UNCHANGED_OUTPUT_FILES:
        src = src_sample_dir / "outputs" / f
        if src.exists(): symlink(src, dst_sample_dir / "outputs" / f, do_save)
    for f in COPIED_FILES:
        src = src_sample_dir / f
        if src.exists(): copy(src, dst_sample_dir / f, do_save)

    # denotch the fft, save a real (not symlinked) copy into the new tree
    fft_npz = load(src_fft_path)
    freqs, fft_raw = fft_npz["freqs"], fft_npz["fft"]
    fft_denotched = denotch_fn(fft_raw, freqs)
    save({"fft": fft_denotched, "freqs": freqs, "n_samples": fft_npz["n_samples"]}, dst_sample_dir / FFT_NAME, do_save)
    append({"fft_denotched": True}, dst_sample_dir / "times.jsonl", do_save)

    # regenerate X (extract signal -> normalize -> tokenize), matching what produced the original X.npy/05_processed_fft.npy
    signaled = extract_signal(fft_denotched, signal_mode).astype(np.float32)
    normalized = normalize_fft(signaled, normalize_mode)
    tokenized = tokenize(normalized, patch_size)
    save(tokenized, dst_sample_dir / PROCESSED_FFT_NAME, do_save)
    symlink(dst_sample_dir / PROCESSED_FFT_NAME, dst_sample_dir / "X.npy", do_save)

    metadata = {k: v for d in load(src_sample_dir / "metadata.jsonl") for k, v in d.items()}
    n_lasers, n_freqs = fft_raw.shape[1], fft_raw.shape[2]
    append(dict(signal_mode=signal_mode, normalize_mode=normalize_mode, patch_size=patch_size, n_lasers=n_lasers, n_freqs=n_freqs), dst_sample_dir / "metadata.jsonl", do_save)
    metadata["n_lasers"], metadata["n_freqs"] = n_lasers, n_freqs
    return metadata


def build_denotched_dataset(src_dir: Path, dst_dir: Path, mds_dir: Path, signal_mode: str, normalize_mode: str,
                             patch_size: int, spike_freqs: list[float], notch_half_width_hz: float,
                             force: bool = False, verbose: int = 1, do_save: bool = True):
    # resolve to absolute paths up front: convert_to_mds chdirs internally, which would otherwise
    # break any relative path (e.g. sample_dir / "X.npy") captured before the chdir happens
    src_dir, dst_dir, mds_dir = Path(src_dir).resolve(), Path(dst_dir).resolve(), Path(mds_dir).resolve()
    assert src_dir != dst_dir, "--dst-dir must differ from --src-dir (never write into the original dataset)"

    sample_dirs = sorted(p for p in src_dir.glob("*") if p.is_dir())
    if verbose: print(f"Found {len(sample_dirs)} sample dirs under {src_dir}")

    def denotch_fn(fft, freqs):
        return interpolate_notch(fft, freqs, spike_freqs=spike_freqs, half_width_hz=notch_half_width_hz)

    rows = {}
    for src_sample_dir in sample_dirs:
        with Timing(f"[sample {src_sample_dir.name}] build denotched copy: ", enabled=verbose >= 2):
            meta = build_sample(src_sample_dir, dst_dir / src_sample_dir.name, signal_mode, normalize_mode, patch_size, denotch_fn, do_save, verbose)
            if meta is not None: rows[src_sample_dir.name] = meta
    if verbose: print(f"Built {len(rows)}/{len(sample_dirs)} samples into {dst_dir}")
    print_system_usage(dst_dir, label="[after build]", verbose=verbose)
    if not rows: raise RuntimeError(f"No complete samples found under {src_dir}")

    # hash-keyed MDS dir, same convention as post_process.convert_to_mds callers
    sample_ids = list(rows.keys())
    key = hashlib.sha1(json.dumps({"sample_ids": sample_ids, "spike_freqs": spike_freqs, "notch_half_width_hz": notch_half_width_hz,
                                    "signal_mode": signal_mode, "normalize_mode": normalize_mode, "patch_size": patch_size,
                                    "columns": MDS_COLUMNS}, sort_keys=True).encode()).hexdigest()[:16]
    mds_path = mds_dir / key
    if mds_path.exists() and force:
        import shutil
        if verbose: print(f"--force: deleting cached MDS at {mds_path} and rebuilding")
        shutil.rmtree(mds_path)
    elif mds_path.exists() and (mds_path / "dataset.jsonl").exists():
        if verbose: print(f"Cache hit: reusing existing MDS at {mds_path}")
        return mds_path / key

    out_h, out_w = rows[sample_ids[0]]["out_h"], rows[sample_ids[0]]["out_w"]
    n_lasers, n_freqs = rows[sample_ids[0]]["n_lasers"], rows[sample_ids[0]]["n_freqs"]
    x_shape = (1, n_lasers, n_freqs // patch_size, patch_size, 2)
    y_shape = (out_h, out_w)
    mds_rows = [(dst_dir / sid, rows[sid]) for sid in sample_ids]
    mds_path = convert_to_mds(mds_path, mds_rows, key, x_shape, y_shape, verbose)
    if verbose: print(f"Denotched MDS written to: {mds_path}\nPass this as --mds-path to run.py to train on it.")
    return mds_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=str, required=True, help="original base_sample_dir containing sample dirs (e.g. data/samples)")
    p.add_argument("--dst-dir", type=str, default=None, help="default: sibling '<name>-denotched' directory")
    p.add_argument("--mds-dir", type=str, default=None, help="default: '<dst-dir>/mds'")
    p.add_argument("--signal-mode", type=str, default="magnitude")
    p.add_argument("--normalize-mode", type=str, default="std-sample")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--spike-freqs", type=float, nargs="+", default=[60.0, 100.0])
    p.add_argument("--notch-half-width-hz", type=float, default=3.0)
    p.add_argument("--force", action="store_true", default=False)
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir) if args.dst_dir else src_dir.parent / f"{src_dir.name}-denotched"
    mds_dir = Path(args.mds_dir) if args.mds_dir else dst_dir / "mds"

    build_denotched_dataset(
        src_dir, dst_dir, mds_dir, args.signal_mode, args.normalize_mode, args.patch_size,
        args.spike_freqs, args.notch_half_width_hz, force=args.force, verbose=args.verbose,
        do_save=not args.dry_run)
