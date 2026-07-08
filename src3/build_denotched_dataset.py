import argparse
from pathlib import Path

from io_utils import copy, symlink, Timing, print_system_usage
from post_process import post_process, IMAGE_FILES, VIBRATION_FILES
from denotch import interpolate_notch

# files that don't depend on the FFT and can be symlinked straight from the original sample
UNCHANGED_IMAGE_FILES = IMAGE_FILES  # image/ tree is untouched by denotching
UNCHANGED_VIBRATION_FILES = ["00_raw_vibrations.npy", "01_raw_shifts.npy", "02_clean_shifts.npy"]  # everything upstream of the fft itself
UNCHANGED_TOP_FILES = ["audio.mp3", "recovered_audio.mp3", "overhead.png"]

# files that get regenerated fresh for the denotched copy (not symlinked, not copied)
REGENERATED_VIBRATION_FILES = ["05_fft_signaled.npy", "06_fft_normalized.npy", "07_fft_tokenized.npy"]

def build_sample(src_sample_dir: Path, dst_sample_dir: Path, do_save: bool, verbose: int):
    sample_id = src_sample_dir.name

    # symlink the parts that don't change at all
    for f in UNCHANGED_IMAGE_FILES:
        src = src_sample_dir / "image" / f
        if src.exists(): symlink(src, dst_sample_dir / "image" / f, do_save)
    for f in UNCHANGED_VIBRATION_FILES:
        src = src_sample_dir / "inputs" / f
        if src.exists(): symlink(src, dst_sample_dir / "inputs" / f, do_save)
    for f in UNCHANGED_TOP_FILES:
        src = src_sample_dir / f
        if src.exists(): symlink(src, dst_sample_dir / f, do_save)
    for f in ["outputs/02_segment_mask.png"]:  # needed by post_process_sample's downsample step
        src = src_sample_dir / f
        if src.exists(): symlink(src, dst_sample_dir / f, do_save)

    # real copies (never symlinks) for anything post_process_sample reads-then-mutates/appends to
    for f in ["metadata.jsonl", "times.jsonl"]:
        src = src_sample_dir / f
        if src.exists(): copy(src, dst_sample_dir / f, do_save)

    # real copy of the raw fft: post_process_sample overwrites inputs/03_fft.npz in place when denotching
    src_fft = src_sample_dir / "inputs/03_fft.npz"
    if not src_fft.exists():
        if verbose: print(f"[sample {sample_id}] skip: missing inputs/03_fft.npz")
        return False
    copy(src_fft, dst_sample_dir / "inputs/03_fft.npz", do_save)
    return True


def build_denotched_dataset(src_dir: Path, dst_dir: Path, mds_dir: Path, is_empty_box: bool, out_h: int, out_w: int,
                             signal_mode: str, normalize_mode: str, patch_size: int, spike_freqs: list[float],
                             notch_half_width_hz: float, force: bool = False, verbose: int = 1, do_save: bool = True):
    src_dir, dst_dir, mds_dir = Path(src_dir), Path(dst_dir), Path(mds_dir)
    assert src_dir.resolve() != dst_dir.resolve(), "--dst-dir must differ from --src-dir (never write into the original dataset)"

    sample_dirs = sorted(p for p in src_dir.glob("sample_*") if p.is_dir())
    if verbose: print(f"Found {len(sample_dirs)} sample dirs under {src_dir}")

    built = 0
    for src_sample_dir in sample_dirs:
        with Timing(f"[sample {src_sample_dir.name}] build denotched copy: ", enabled=verbose >= 2):
            ok = build_sample(src_sample_dir, dst_dir / src_sample_dir.name, do_save, verbose)
            built += int(ok)
    if verbose: print(f"Copied/symlinked {built}/{len(sample_dirs)} samples into {dst_dir}")
    print_system_usage(dst_dir, label="[after copy]", verbose=verbose)

    def denotch_fn(fft, freqs):
        return interpolate_notch(fft, freqs, spike_freqs=spike_freqs, half_width_hz=notch_half_width_hz)

    mds_path = post_process(dst_dir, mds_dir, is_empty_box, out_h, out_w, signal_mode, normalize_mode, patch_size,
                             force=force, verbose=verbose, do_save=do_save, denotch_fn=denotch_fn)
    if verbose: print(f"Denotched MDS written to: {mds_path}\nPass this as --mds-path to run.py to train on it.")
    return mds_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=str, required=True, help="original base_sample_dir containing sample_* dirs")
    p.add_argument("--dst-dir", type=str, default=None, help="default: sibling '<src-dir>-denotched' directory")
    p.add_argument("--mds-dir", type=str, default=None, help="default: '<dst-dir>/mds'")
    p.add_argument("--is-empty-box", action="store_true", default=False)
    p.add_argument("--out-h", type=int, default=18)
    p.add_argument("--out-w", type=int, default=44)
    p.add_argument("--signal-mode", type=str, default="magnitude")
    p.add_argument("--normalize-mode", type=str, default="z-sample")
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
        src_dir, dst_dir, mds_dir, args.is_empty_box, args.out_h, args.out_w, args.signal_mode, args.normalize_mode,
        args.patch_size, args.spike_freqs, args.notch_half_width_hz, force=args.force, verbose=args.verbose,
        do_save=not args.dry_run)
