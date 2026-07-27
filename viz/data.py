"""Data layer for the vibrations dashboard.

Reads sample metadata + X/y from an MDS dataset (datasets/<name>/mds), reduces
FFTs on demand, and extracts per-sample predictions + metrics from
runs/<name>/outputs_history/*.pt files.

The dataset a run trained on isn't recorded in the run, so it's resolved by
matching the run's sample_ids against each dataset's metadata sidecar.
"""
import json
import os
import re
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from utils.metrics import center_of_mass

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
RUNS_DIR = ROOT / "runs"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Which dataset a run was trained on. Runs don't record this, so it's resolved by
# matching the run's sample_ids against each dataset's metadata sidecar (see dataset_for_run).
DEFAULT_DATASET = "016"

# Raw capture dirs (overhead images + audio), which the MDS datasets don't carry.
RAW_SAMPLE_ROOTS = [ROOT / "experiments" / "experiment-25" / "samples"]

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


def _object_label(meta: dict) -> str:
    """Object name(s) for a sample. Newer datasets store an `objects` dict ({name: count},
    empty for an empty box); older ones stored a single `object` string. Support both."""
    objs = meta.get("objects")
    if isinstance(objs, dict):
        return "+".join(sorted(objs)) if objs else "none"
    return meta.get("object") or "none"


@lru_cache(maxsize=8)
def load_metadata(dataset: str) -> list[dict]:
    """All per-sample metadata rows from a dataset's MDS sidecar, in shard order."""
    path = DATASETS_DIR / dataset / "mds" / "metadata.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ***** manifest *****

def build_manifest(dataset: str = DEFAULT_DATASET) -> dict:
    samples, out_h, out_w = [], 20, 40
    for m in load_metadata(dataset):
        out_h, out_w = int(m["out_h"]), int(m["out_w"])
        com = m.get("downsampled_com", [-1.0, -1.0])  # (row, col)
        samples.append({
            "sample_id": int(m["sample_id"]),
            "output_id": m["output_id"],
            "speaker": int(m["speaker"]),
            "box": m["box"],
            "object": _object_label(m),
            # newer datasets carry an explicit `layout`; older ones only implied it via description
            "layout": m.get("layout") or _layout(m.get("description", "")),
            "n_objects": int(m["n_objects"]),
            "is_empty_box": bool(m["is_empty_box"]),
            "audio": _audio_label(m),
            "com_row": round(float(com[0]), 3),
            "com_col": round(float(com[1]), 3),
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
        "dataset": dataset,
        "datasets": list_datasets(),
        "runs": list_runs(),
        "empty_box_groups": empty_box_groups(dataset),
    }


def list_datasets() -> list[str]:
    return sorted(d.name for d in DATASETS_DIR.iterdir() if (d / "mds" / "metadata.jsonl").exists())


@lru_cache(maxsize=32)
def dataset_for_run(run: str) -> str:
    """Which dataset a run trained on: the smallest one whose sample_ids cover the run's.
    Runs don't record this, so it's inferred; falls back to DEFAULT_DATASET on no match."""
    z = extract_run(run)
    run_ids = {int(s) for s in np.unique(z["sample_id"])}
    matches = []
    for name in list_datasets():
        ids = {int(m["sample_id"]) for m in load_metadata(name)}
        if run_ids <= ids:
            matches.append((len(ids), name))
    return min(matches)[1] if matches else DEFAULT_DATASET


def empty_box_groups(dataset: str = DEFAULT_DATASET) -> list[dict]:
    """The distinct empty-box output_id groups (no object, no meaningful com)."""
    groups: dict[str, list[int]] = {}
    for m in load_metadata(dataset):
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
    """Cheap change stamp: dataset names + run names + outputs_history mtimes."""
    parts = ["|".join(list_datasets())]
    for name in list_runs():
        out = RUNS_DIR / name / "outputs_history"
        parts.append(f"{name}:{max((p.stat().st_mtime for p in out.rglob('*.pt')), default=0):.0f}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


# ***** fft *****

@lru_cache(maxsize=4)
def _mds(dataset: str):
    """Read-only handle on a dataset's MDS shards, plus sample_id -> row index."""
    from streaming.base.local import LocalDataset  # local import: heavy, only needed for fft/gt
    ds = LocalDataset(str(DATASETS_DIR / dataset / "mds"))
    index = {int(m["sample_id"]): i for i, m in enumerate(load_metadata(dataset))}
    return ds, index


@lru_cache(maxsize=8)
def _freqs(dataset: str, n_freqs: int) -> np.ndarray:
    """The frequency axis for a dataset's X. Newer datasets ship a freqs.npy sidecar; older
    ones only record the [min_freq, max_freq] band the fft was cropped to, so rebuild it.
    Tokenized X drops the tail freqs that don't fill a patch, hence n_freqs (not metadata's)."""
    path = DATASETS_DIR / dataset / "mds" / "freqs.npy"
    if path.exists():
        return np.load(path)[:n_freqs]
    m = load_metadata(dataset)[0]
    return np.linspace(float(m["min_freq"]), float(m["max_freq"]), int(m["n_freqs"]))[:n_freqs]


@lru_cache(maxsize=64)
def _magnitude(sample_id: int, dataset: str = DEFAULT_DATASET) -> tuple[np.ndarray, np.ndarray]:
    """Returns (|fft| as float32 (n_lasers, F, 2), freqs (F,)).

    X is stored either as the raw complex fft (n_lasers, F, C) or — when the dataset was built
    with augment_fft=False — as a real, already-magnitude, patch-tokenized signal
    (n_lasers, P, patch_size, C). Flatten the patch axes back to a plain frequency axis."""
    ds, index = _mds(dataset)
    X = ds[index[sample_id]]["X"]
    if X.ndim == 4:  # (L, P, patch_size, C) -> (L, P*patch_size, C)
        X = X.reshape(X.shape[0], -1, X.shape[-1])
    return np.abs(X).astype(np.float32), _freqs(dataset, X.shape[1])


@lru_cache(maxsize=64)
def _magnitude_normalized(sample_id: int, dataset: str = DEFAULT_DATASET) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample std-normalized |fft|: matches src/model/dataset.py normalize_fft('std-sample'),
    computed over the whole (laser, freq, dir) tensor before any laser/dir subsetting, so the
    scale factor doesn't change as the user toggles which lasers/dirs are selected."""
    mag, freqs = _magnitude(sample_id, dataset)
    std = max(float(mag.astype(np.float64).std(ddof=1)), 1e-8)
    return (mag / std).astype(np.float32), freqs


def fft_curve(sample_id: int, lasers: list[int] | None, dirs: str, norm: bool = False,
              dataset: str = DEFAULT_DATASET) -> tuple[np.ndarray, np.ndarray]:
    """Mean |fft| over the given lasers (None = all 100) and dirs ('x'|'y'|'xy')."""
    mag, freqs = _magnitude_normalized(sample_id, dataset) if norm else _magnitude(sample_id, dataset)
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
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except (RuntimeError, EOFError):
            continue  # partial/corrupt dump (e.g. a run still writing) — skip, don't kill the run
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


def raw_sample_dir(sample_id: int, dataset: str = DEFAULT_DATASET) -> Path | None:
    """Local dir holding a sample's raw capture assets (overhead images, audio), or None.
    The MDS dataset carries only X/y, and metadata's own `sample_dir` is the Windows path
    from the recording rig, so these come from the local experiment tree instead.
    $VIZ_RAW_SAMPLES overrides the default for datasets captured under a different experiment."""
    roots = [os.environ["VIZ_RAW_SAMPLES"]] if os.environ.get("VIZ_RAW_SAMPLES") else RAW_SAMPLE_ROOTS
    for root in roots:
        d = Path(root) / f"{sample_id:06d}"
        if d.is_dir():
            return d
    return None


def gt_mask(sample_id: int, dataset: str = DEFAULT_DATASET) -> list:
    ds, index = _mds(dataset)
    y = ds[index[sample_id]]["y"]
    return np.round(y.astype(np.float64), 4).tolist()
