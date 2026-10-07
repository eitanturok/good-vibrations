from pathlib import Path

import numpy as np

FFT_FILES = ["vibration/04_ffts.npz", "vibration/04_fft.npz"]
REQUIRED_FILES = ["image/03_smask.npy", "metadata.jsonl"]


# 1. collect samples

def fft_path(sample_dir: Path) -> Path:
    for name in FFT_FILES: if (sample_dir / name).exists(): return sample_dir / name
    raise FileNotFoundError(f"{sample_dir}: none of {FFT_FILES}")

def load_fft(sample_dir: Path, laser_idx: np.ndarray | None = None) -> np.ndarray:
    X = np.load(fft_path(sample_dir))["fft"]  # (1, L, F, C) complex64
    X = np.squeeze(X, axis=0) if X.ndim == 4 and X.shape[0] == 1 else X
    return X if laser_idx is None else X[laser_idx]

def collect_samples(data_dir: Path, verbose: int = 1) -> list[tuple[Path, dict]]:
    samples, missing_by_file = [], {f: [] for f in [*REQUIRED_FILES, FFT_FILES[0]]}
    for sample_dir in tqdm(sorted(data_dir.glob("*")), desc="collecting samples", disable=not verbose):
        if not sample_dir.is_dir(): continue
        missing = [f for f in REQUIRED_FILES if not (sample_dir / f).exists()]
        if not any((sample_dir / f).exists() for f in FFT_FILES): missing.append(FFT_FILES[0])
        if missing:
            for f in missing: missing_by_file[f].append(sample_dir.name)
            continue
        meta = {k: v for d in (json.loads(line) for line in (sample_dir / "metadata.jsonl").read_text().splitlines() if line) for k, v in d.items()}
        samples.append((sample_dir, meta))

    n_skipped = len({sid for ids in missing_by_file.values() for sid in ids})
    if verbose:
        print(f"Found {len(samples)} complete samples ({n_skipped} skipped)")
        for f, ids in missing_by_file.items(): if ids: print(f"missing {f!r}: {ids}")
    return samples

# 2 convert to mds
def mds_columns(augment_fft: bool) -> dict[str, str]:
    x_dtype = "complex64" if augment_fft else "float32"  # raw fft is complex; precomputed signal is real
    return {"X": f"ndarray:{x_dtype}", "y": "ndarray:float32",
            "sample_id": "int", "position_id": "int",
            "n_objects": "int", "speaker": "int", "box": "str", "is_empty_box": "int", "object": "str",
            "downsampled_com_x": "float64", "downsampled_com_y": "float64"}

def convert_to_mds(mds_dir: Path, samples: list[tuple[Path, dict]], out_h: int, out_w: int, verbose: int = 1,
                augment_fft: bool = True, signal_mode: str = "magnitude", normalize_mode: str = "std", patch_size: int = 64,
                subtract_speaker_mean: bool = False, subtract_empty_box: bool = False, rgb: bool = False,
                phase_arm: str | None = None, phase_weight: float = 1.0,
                laser_idx: np.ndarray | None = None, laser_cols=None, laser_rows=None) -> Path:

def load_X(sample_dir: Path) -> np.ndarray:
    if not augment_fft:
        return np.load(sample_dir / precomputed_fft_name(signal_mode, normalize_mode, patch_size, subtract_speaker_mean, subtract_empty_box, phase_arm, phase_weight, laser_cols, laser_rows))
    X = np.load(fft_path(sample_dir))["fft"]  # (1, L, F, C) complex64
    X = np.squeeze(X, axis=0) if X.ndim == 4 and X.shape[0] == 1 else X
    return X if laser_idx is None else X[laser_idx]

def build_dataset(args):
    return train_loader, eval_loaders, train_eval_loader
