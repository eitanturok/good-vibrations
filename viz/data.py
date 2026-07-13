"""Data layer for the vibrations dashboard.

Scans data/samples for metadata, reduces FFTs on demand, and extracts
per-sample predictions + metrics from runs/<name>/outputs_history/*.pt files.
All conventions follow what is actually on disk (notebook 46's naming),
not the newer constants in src3.
"""
import json
import re
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from utils.metrics import center_of_mass

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
RUNS_DIR = ROOT / "runs"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

FFT_NAME = "inputs/03_fft_shifts.npz"
THUMB_NAME = "outputs/01_resized_overhead.png"

# description -> short layout label (there is no explicit layout field)
LAYOUTS = [
    ("empty", "empty"),
    ("knob is face up", "1-cube-knob-up"),
    ("metal strips is face down", "1-cube-strips-down"),
    ("together, touching", "2-touching"),
    ("apart, not touching", "2-apart"),
]


def _layout(description: str) -> str:
    for needle, label in LAYOUTS:
        if needle in description:
            return label
    return "1-cube"


def _audio_label(meta: dict) -> str:
    # only one clip exists today; keep this a real facet so future clips just show up
    return "chirp_50_1000_3.0sec"


def load_metadata(sample_dir: Path) -> dict:
    meta = {}
    for line in (sample_dir / "metadata.jsonl").read_text().splitlines():
        if line.strip():
            meta.update(json.loads(line))
    return meta


# ***** manifest *****

def build_manifest() -> dict:
    samples, out_h, out_w = [], 18, 44
    for d in sorted(SAMPLES_DIR.iterdir()):
        if not (d / "metadata.jsonl").exists() or not (d / FFT_NAME).exists():
            continue
        m = load_metadata(d)
        out_h, out_w = int(m["out_h"]), int(m["out_w"])
        com = m.get("downsampled_com", [-1.0, -1.0])  # (row, col)
        samples.append({
            "sample_id": int(m["sample_id"]),
            "output_id": m["output_id"],
            "speaker": int(m["speaker"]),
            "box": m["box"],
            "object": m["object"] or "none",
            "layout": _layout(m.get("description", "")),
            "n_objects": int(m["n_objects"]),
            "is_empty_box": bool(m["is_empty_box"]),
            "audio": _audio_label(m),
            "com_row": round(float(com[0]), 3),
            "com_col": round(float(com[1]), 3),
            "sample_dir": str(d),
        })
    facets = {
        k: sorted({s[k] for s in samples}, key=str)
        for k in ["speaker", "box", "object", "layout", "n_objects", "audio"]
    }
    return {
        "out_h": out_h,
        "out_w": out_w,
        "samples": samples,
        "facets": facets,
        "runs": list_runs(),
        "empty_box_groups": empty_box_groups(),
    }


def empty_box_groups() -> list[dict]:
    """The distinct empty-box output_id groups (no object, no meaningful com)."""
    groups: dict[str, list[int]] = {}
    for d in sorted(SAMPLES_DIR.iterdir()):
        if not (d / "metadata.jsonl").exists() or not (d / FFT_NAME).exists():
            continue
        m = load_metadata(d)
        if m.get("is_empty_box"):
            groups.setdefault(m["output_id"], []).append(int(m["sample_id"]))
    return [{"output_id": oid, "sample_ids": sorted(ids)} for oid, ids in sorted(groups.items())]


def list_runs() -> list[str]:
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        out = d / "outputs_history"
        if out.is_dir() and any(out.rglob("*.pt")):
            runs.append((d.stat().st_mtime, d.name))
    runs.sort(key=lambda t: t[0], reverse=True)
    return [name for _, name in runs]


def data_version() -> str:
    """Cheap change stamp: sample count + run names + outputs_history mtimes."""
    parts = [str(sum(1 for _ in SAMPLES_DIR.iterdir()))]
    for name in list_runs():
        out = RUNS_DIR / name / "outputs_history"
        parts.append(f"{name}:{max((p.stat().st_mtime for p in out.rglob('*.pt')), default=0):.0f}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


# ***** fft *****

@lru_cache(maxsize=64)
def _magnitude(sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (|fft| as float32 (100, F, 2), freqs (F,))."""
    z = np.load(SAMPLES_DIR / f"{sample_id:06d}" / FFT_NAME)
    return np.abs(z["fft"][0]).astype(np.float32), z["freqs"]


@lru_cache(maxsize=64)
def _magnitude_normalized(sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample std-normalized |fft|: matches src3/post_process.py normalize_fft('std-sample'),
    computed over the whole (laser, freq, dir) tensor before any laser/dir subsetting, so the
    scale factor doesn't change as the user toggles which lasers/dirs are selected."""
    mag, freqs = _magnitude(sample_id)
    std = max(float(mag.astype(np.float64).std(ddof=1)), 1e-8)
    return (mag / std).astype(np.float32), freqs


def fft_curve(sample_id: int, lasers: list[int] | None, dirs: str, norm: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Mean |fft| over the given lasers (None = all 100) and dirs ('x'|'y'|'xy')."""
    mag, freqs = _magnitude_normalized(sample_id) if norm else _magnitude(sample_id)
    if lasers is not None:
        mag = mag[lasers]
    d = {"x": [0], "y": [1], "xy": [0, 1]}[dirs]
    return mag[:, :, d].mean(axis=(0, 2)), freqs


# ***** runs *****

def _run_key(run: str) -> str:
    files = sorted((RUNS_DIR / run / "outputs_history").rglob("*.pt"))
    sig = "|".join(f"{p.relative_to(RUNS_DIR)}:{p.stat().st_mtime:.0f}:{p.stat().st_size}" for p in files)
    return hashlib.md5(sig.encode()).hexdigest()[:12]


def extract_run(run: str) -> dict:
    """Strip a run's .pt batches to per-sample metrics + masks. Cached as npz."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"run_{run}_{_run_key(run)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}

    import torch  # local import: only needed on cache miss

    recs = {k: [] for k in ["sample_id", "epoch", "split", "mse", "com_dist",
                            "pred_row", "pred_col", "masks"]}
    for p in sorted((RUNS_DIR / run / "outputs_history").rglob("*.pt")):
        rel = p.relative_to(RUNS_DIR / run / "outputs_history")
        split = rel.parts[0] if rel.parts[0] != "eval" else rel.parts[1]
        epoch = int(re.match(r"ep(\d+)-ba(\d+)", p.stem).group(1))
        d = torch.load(p, map_location="cpu", weights_only=False)
        pred = d["mask_pred"].float().numpy()
        true = d["mask_true"].float().numpy()
        h, w = pred.shape[-2:]
        mse = ((pred - true) ** 2).mean(axis=(-2, -1))
        com_p, com_t = center_of_mass(pred), center_of_mass(true)
        norm = np.array([h - 1, w - 1], dtype=np.float64)
        com_dist = np.linalg.norm((com_p - com_t) / norm, axis=-1)
        com_dist[true.sum(axis=(-2, -1)) <= 0] = np.nan  # metric skips empty GT
        n = len(pred)
        recs["sample_id"].append(np.asarray(d["info"]["sample_id"], dtype=np.int64))
        recs["epoch"].append(np.full(n, epoch, dtype=np.int64))
        recs["split"].append(np.full(n, split, dtype="U32"))
        recs["mse"].append(mse)
        recs["com_dist"].append(com_dist)
        recs["pred_row"].append(com_p[:, 0])
        recs["pred_col"].append(com_p[:, 1])
        recs["masks"].append(pred.astype(np.float16))

    out = {k: np.concatenate(v) for k, v in recs.items()}
    for stale in CACHE_DIR.glob(f"run_{run}_*.npz"):
        stale.unlink()
    np.savez_compressed(cache, **out)
    return out


def run_payload(run: str, epoch: int | None = None) -> dict:
    """JSON-ready per-sample metrics + per-split aggregates for one epoch."""
    z = extract_run(run)
    epochs = sorted(int(e) for e in np.unique(z["epoch"]))
    epoch = epoch if epoch in epochs else epochs[-1]
    sel = z["epoch"] == epoch
    samples = {
        int(sid): {
            "split": str(sp),
            "mse": round(float(m), 6),
            "com_dist": None if np.isnan(cd) else round(float(cd), 5),
            "pred_row": round(float(pr), 3),
            "pred_col": round(float(pc), 3),
        }
        for sid, sp, m, cd, pr, pc in zip(
            z["sample_id"][sel], z["split"][sel], z["mse"][sel],
            z["com_dist"][sel], z["pred_row"][sel], z["pred_col"][sel])
    }
    aggregates = {}
    for sp in np.unique(z["split"][sel]):
        m = sel & (z["split"] == sp)
        aggregates[str(sp)] = {
            "n": int(m.sum()),
            "mse": round(float(z["mse"][m].mean()), 6),
            "com_dist": round(float(np.nanmean(z["com_dist"][m])), 5),
        }
    return {"run": run, "epoch": epoch, "epochs": epochs,
            "samples": samples, "aggregates": aggregates}


def run_masks(run: str, sample_ids: list[int], epoch: int | None = None) -> dict:
    z = extract_run(run)
    epochs = sorted(int(e) for e in np.unique(z["epoch"]))
    epoch = epoch if epoch in epochs else epochs[-1]
    sel = z["epoch"] == epoch
    ids, masks = z["sample_id"][sel], z["masks"][sel]
    idx = {int(s): i for i, s in enumerate(ids)}
    return {
        str(sid): np.round(masks[idx[sid]].astype(np.float64), 4).tolist()
        for sid in sample_ids if sid in idx
    }


def gt_mask(sample_id: int) -> list:
    y = np.load(SAMPLES_DIR / f"{sample_id:06d}" / "y.npy")
    return np.round(y.astype(np.float64), 4).tolist()
