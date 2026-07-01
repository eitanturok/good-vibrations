import os, json, hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from composer.core import Evaluator, DataSpec
from streaming import StreamingDataset, MDSWriter

from io_utils import load

MDS_COLUMNS = {
    "X": "ndarray:float32", "y": "ndarray:float32",
    "sample_id": "int", "n_objects": "int", "speaker": "int",
    "box": "str", "is_empty_box": "int", "object": "str",
    "downsampled_com_x": "float64", "downsampled_com_y": "float64",
}
# columns cheap enough to mirror into index.jsonl for loader-side filtering (everything but X/y)
INDEX_KEYS = [k for k in MDS_COLUMNS if k not in ("X", "y")]


def _derive_data_info(info: dict) -> dict:
    """Add the shape keys the model needs, derived from x_shape=(L, P, PS, C)."""
    L, P, PS, _ = info["x_shape"]
    return info | dict(patch_size=PS, n_freqs=P * PS, n_laser_rows=int(L ** 0.5), n_laser_cols=int(L ** 0.5))


# default used by the model when no dataset is loaded (matches the experiment-22 MDS)
DATA_INFO = _derive_data_info({"out_h": 18, "out_w": 44, "x_shape": [100, 13, 256, 2], "n_samples": 0})


def flatten_metadata(sample_dir: Path) -> dict:
    """Merge every {k: v} line of a sample's metadata.jsonl into one dict."""
    return {k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}


def _hash_key(data_dir: Path, out_h: int, out_w: int, sample_ids: list[int]) -> str:
    payload = json.dumps([str(data_dir), out_h, out_w, sorted(sample_ids)], sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def prep_dataset(data_dir: str | Path, out_h: int, out_w: int, mds_root: str | Path | None = None,
                 exist_ok: bool = True, verbose: int = 1) -> Path:
    """Collect all complete (X.npy, y.npy) samples under ``data_dir`` and write one MDS dataset.

    Returns the path to the MDS directory (contains shards, ``index.jsonl`` and ``data_info.json``).
    On a cache hit (same data_dir/out_h/out_w/samples) the existing dir is returned unchanged.
    """
    data_dir = Path(data_dir)
    samples_dir = data_dir / "samples"
    mds_root = Path(mds_root) if mds_root is not None else data_dir / "mds"

    # collect complete samples (both X and y present) + their flattened metadata
    rows = []
    skipped = 0
    for sample_dir in sorted(samples_dir.glob("*")):
        if not (sample_dir / "X.npy").exists() or not (sample_dir / "y.npy").exists():
            skipped += 1
            continue
        rows.append((sample_dir, flatten_metadata(sample_dir)))
    if not rows:
        raise RuntimeError(f"No complete samples (with both X.npy and y.npy) found under {samples_dir}")
    sample_ids = [int(m["sample_id"]) for _, m in rows]
    if verbose: print(f"Found {len(rows)} complete samples ({skipped} skipped, missing X.npy)")

    # hash-keyed output dir -> instant return on cache hit
    key = _hash_key(data_dir, out_h, out_w, sample_ids)
    mds_path = mds_root / key
    if mds_path.exists() and (mds_path / "metadata.jsonl").exists() and exist_ok:
        if verbose: print(f"Cache hit: reusing existing MDS at {mds_path}\nMDS: {mds_path} ({len(rows)} samples)")
        return mds_path

    # write shards + a lightweight metadata sidecar (index.jsonl) for loader-side filtering.
    # Streaming urlparses `out`, so an absolute Windows path (e.g. "D:/...") is misread as a
    # cloud scheme "d:". Chdir to the parent and pass the relative dir name -> empty url scheme.
    if verbose: print(f"Writing MDS to {mds_path} ...")
    index_rows = []
    mds_root.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(mds_root)
    try:
      with MDSWriter(out=key, columns=MDS_COLUMNS, exist_ok=True) as writer:
        for i, (sample_dir, meta) in enumerate(rows):
            X = np.load(sample_dir / "X.npy").astype(np.float32)   # (1, L, P, PS, 2)
            y = np.load(sample_dir / "y.npy").astype(np.float32)   # (out_h, out_w)
            X = np.squeeze(X, axis=0) if X.ndim == 5 and X.shape[0] == 1 else X
            assert y.shape == (out_h, out_w), f"{sample_dir.name}: y.shape={y.shape} != {(out_h, out_w)}"
            if i == 0: x_shape = X.shape
            assert X.shape == x_shape, f"{sample_dir.name}: X.shape={X.shape} != {x_shape}"

            com = meta.get("downsampled_com", [-1.0, -1.0])
            sample = {
                "X": X, "y": y, "sample_id": int(meta["sample_id"]),
                "n_objects": int(meta.get("n_objects", -1)),
                "speaker": int(meta.get("speaker", -1)),
                "box": str(meta.get("box", "")),
                "is_empty_box": int(bool(meta.get("is_empty_box", False))),
                "object": str(meta.get("object", "")),
                "downsampled_com_x": float(com[0]), "downsampled_com_y": float(com[1]),
            }
            writer.write(sample)
            index_rows.append({k: sample[k] for k in INDEX_KEYS})
            if verbose >= 2 and (i + 1) % 50 == 0: print(f"  wrote {i + 1}/{len(rows)}")
    finally:
        os.chdir(cwd)

    # single sidecar (jsonl): line 0 is dataset-level info, then one line of metadata per sample
    # (no arrays) for loader-side filtering.
    data_info = dict(out_h=out_h, out_w=out_w, x_shape=list(x_shape), n_samples=len(rows))
    lines = [json.dumps(data_info)] + [json.dumps(r) for r in index_rows]
    (mds_path / "metadata.jsonl").write_text("\n".join(lines))
    if verbose: print(f"Wrote {len(rows)} samples. data_info={data_info}\nMDS: {mds_path} ({len(rows)} samples)")
    return mds_path


class VibrationDataset(StreamingDataset):
    """Reads the MDS produced by :func:`prep_dataset`."""

    def __init__(self, local: str | Path, shuffle: bool = False, **kwargs):
        super().__init__(local=str(local), shuffle=shuffle, batch_size=kwargs.pop("batch_size", None), **kwargs)

    def __getitem__(self, idx):
        s = super().__getitem__(idx)
        info = dict(sample_id=s["sample_id"], n_objects=s["n_objects"], speaker=s["speaker"],
                    box=s["box"], is_empty_box=s["is_empty_box"],
                    x_com=s["downsampled_com_x"], y_com=s["downsampled_com_y"])
        # Streaming decodes each sample into a fresh read-only buffer; the dataloader's collate
        # copies it into the batch tensor, so we hand the array over without an extra copy.
        return dict(fft=torch.from_numpy(s["X"]), mask_true=torch.from_numpy(s["y"]), info=info)


def _matches(row: dict, speakers, n_objects, box) -> bool:
    if speakers is not None and row["speaker"] not in (speakers if isinstance(speakers, list) else [speakers]): return False
    if n_objects is not None and row["n_objects"] not in (n_objects if isinstance(n_objects, list) else [n_objects]): return False
    if box is not None and row["box"] not in (box if isinstance(box, list) else [box]): return False
    return True


def load_meta(mds_path: str | Path) -> tuple[dict, list[dict]]:
    """Read the metadata.jsonl sidecar -> (data_info with derived model keys, per-sample index rows)."""
    lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
    data_info, index = json.loads(lines[0]), [json.loads(line) for line in lines[1:] if line]
    return _derive_data_info(data_info), index


def make_subset(mds_path: str | Path, speakers=None, n_objects=None, box=None,
                n_samples: int | None = None, shuffle: bool = False, verbose: int = 1):
    """Filter the canonical MDS at read time (no rewrite) and return a (dataset, indices) pair.

    Filtering reads only the tiny meta.json sidecar, then wraps the StreamingDataset in a Subset.
    """
    _, index = load_meta(mds_path)
    indices = [i for i, row in enumerate(index) if _matches(row, speakers, n_objects, box)]
    if n_samples is not None: indices = indices[:n_samples]
    if verbose: print(f"make_subset: {len(indices)}/{len(index)} samples "
                      f"(speakers={speakers}, n_objects={n_objects}, box={box})")
    dataset = VibrationDataset(local=mds_path, shuffle=shuffle)
    return Subset(dataset, indices), indices


def build_dataset(mds_path: str | Path, batch_size: int = 64, eval_batch_size: int = 64, test_size: float = 0.2,
                  seed: int = 42, num_workers: int = 8, speakers=None, n_objects=None, box=None,
                  n_samples: int | None = None, verbose: int = 1):
    """Return (train_loader, eval_loaders, data_info) using an already-written MDS.

    Row filters (speakers/n_objects/box/n_samples) are applied via the index sidecar, then the
    kept samples are split into train/eval.
    """
    generator = torch.Generator().manual_seed(seed)
    dataset = VibrationDataset(local=mds_path)
    data_info, index = load_meta(mds_path)

    # candidate indices after row filters
    keep = [i for i, row in enumerate(index) if _matches(row, speakers, n_objects, box)]
    if n_samples is not None: keep = keep[:n_samples]
    train_idx, eval_idx = train_test_split(keep, test_size=test_size, random_state=seed, shuffle=True)
    if verbose: print(f"{len(keep)} samples -> {len(train_idx)} train, {len(eval_idx)} eval")

    def num_samples(batch): return batch["mask_true"].shape[0]
    def loader(idxs, bs, shuffle):
        dl = DataLoader(Subset(dataset, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers,
                        generator=generator, pin_memory=True, persistent_workers=num_workers > 0,
                        prefetch_factor=4 if num_workers > 0 else None)
        return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

    train_loader = loader(train_idx, batch_size, shuffle=True)
    eval_loaders = [Evaluator(label="eval/base", dataloader=loader(eval_idx, eval_batch_size, shuffle=False))]
    return train_loader, eval_loaders, data_info


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=r"D:/eturok/experiment-22/data")
    p.add_argument("--out-h", type=int, default=18)
    p.add_argument("--out-w", type=int, default=44)
    p.add_argument("--mds-root", default=None)
    args = p.parse_args()
    path = prep_dataset(args.data_dir, args.out_h, args.out_w, args.mds_root, verbose=2)
    print(f"MDS written to: {path}")
