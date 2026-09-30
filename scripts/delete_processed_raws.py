"""Delete 01_raw_vibrations.npy (~1-3 GB each) from samples whose post-processing is COMPLETE.

A raw is deleted only when its sample has every output of one post-processing scheme (below),
each one non-empty and whole:
- .npy: file size matches its header, and the shifts cover the raw's frames (T or T - 1)
- .npz: its zip index opens and has the expected keys
Anything else is listed and kept.

Dry run by default:
    python scripts/delete_processed_raws.py D:/eturok/31_07_2026_gastronorm_exp1
    python scripts/delete_processed_raws.py D:/eturok/31_07_2026_gastronorm_exp1 --delete
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np

# every file post-processing writes, per scheme -- paths relative to the sample dir, globs for
# the recovery-laser suffix (e.g. _laser55_x)
SCHEMES = {
    "older (e.g. 31_07_2026_gastronorm_exp1)": [
        "vibration/02_raw_shifts.npy", "vibration/02_clean_shifts.npy",
        "vibration/03_fft.npz", "vibration/03_fft_laser*.png",
        "vibration/04_recovered_audio_laser*.wav",
        "vibration/05_spectrogram_laser*.npz", "vibration/05_spectrogram_laser*.png", "vibration/05_spectrogram_laser*.mp4",
        "recovered_audio.wav",
    ],
    "record/post_process.py": [
        "vibration/02_raw_shifts.npy", "vibration/03_clean_shifts.npy",
        "vibration/04_ffts.npz", "vibration/04_ffts_laser*.png",
        "vibration/05_recovered_audio_laser*.wav",
        "vibration/06_spectrogram_laser*.npz", "vibration/06_spectrogram_laser*.png", "vibration/06_spectrogram_laser*.mp4",
        "recovered_audio.wav",
    ],
}
NPZ_KEYS = {"fft": {"fft"}, "ffts": {"fft"}, "spectrogram": {"Sxx"}}  # a key each .npz must hold


def npy_header(path):
    """(shape, complete) without reading the data."""
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, _, dtype = read(f)
        return shape, path.stat().st_size == int(np.prod(shape)) * dtype.itemsize + f.tell()


def check(path, T_raw):
    """None if `path` is whole, else what's wrong with it."""
    if path.stat().st_size == 0: return "empty"
    if path.suffix == ".npy":
        shape, complete = npy_header(path)
        if not complete: return "truncated"
        if len(shape) >= 2 and shape[-1] == 2 and shape[-2] not in (T_raw, T_raw - 1):
            return f"{shape[-2]} frames, raw has {T_raw}"
    if path.suffix == ".npz":
        try:
            with zipfile.ZipFile(path) as z: keys = {n.removesuffix(".npy") for n in z.namelist()}
        except zipfile.BadZipFile: return "not a readable npz"
        stage = path.stem.split("_")[1]  # 03_fft -> fft, 04_ffts -> ffts, 05_spectrogram_laser55_x -> spectrogram
        if missing := NPZ_KEYS.get(stage, set()) - keys: return f"missing keys {sorted(missing)}"
    return None


def why_keep(raw):
    """None if the raw is safe to delete, else the reason to keep it."""
    sample = raw.parent.parent
    (T_raw, *_), raw_complete = npy_header(raw)
    problems = {}
    for scheme, patterns in SCHEMES.items():
        bad = []
        for pattern in patterns:
            matches = sorted(sample.glob(pattern))
            if not matches: bad.append(f"{pattern} missing"); continue
            bad += [f"{p.relative_to(sample)} {err}" for p in matches if (err := check(p, T_raw))]
        if not bad: return None
        problems[scheme] = bad
    return min(problems.values(), key=len)  # the scheme it came closest to


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir", type=Path)
    ap.add_argument("--delete", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()

    for scheme, patterns in SCHEMES.items():
        print(f"required ({scheme}): {', '.join(patterns)}")
    raws = sorted(args.experiment_dir.glob("samples/*/vibration/01_raw_vibrations.npy"))
    freed, kept = 0, []
    for raw in raws:
        if reason := why_keep(raw):
            kept.append((raw.parent.parent.name, reason))
            continue
        freed += raw.stat().st_size
        if args.delete: raw.unlink()
    verb = "deleted" if args.delete else "would delete"
    print(f"{len(raws)} raw files: {verb} {len(raws) - len(kept)} ({freed / 1e9:.0f} GB), keeping {len(kept)}")
    for sid, reason in kept:
        print(f"  keep {sid}: {'; '.join(reason)}")


if __name__ == "__main__":
    main()
