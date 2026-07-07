import os, json, hashlib, shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from composer.core import Evaluator, DataSpec
from streaming import StreamingDataset, MDSWriter

from io_utils import load

# ***** 1. turn dataset into MDS format (sharded, streaming) *****

# default used by the model when no dataset is loaded (matches the experiment-22 MDS)
DATA_INFO = {"out_h": 18, "out_w": 44, "n_samples": 0,
             "n_laser_rows": 10, "n_laser_cols": 10, "patch_size": 256, "n_freqs": 3328}


# def flatten_metadata(sample_dir: Path) -> dict:
#     """Merge every {k: v} line of a sample's metadata.jsonl into one dict."""
#     return {k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}


# def _load_xy(sample_dir: Path) -> tuple[np.ndarray, np.ndarray]:
#     """Load a sample's X (squeezed to (L, P, PS, C)) and y (out_h, out_w)."""
#     X = np.load(sample_dir / "X.npy").astype(np.float32)   # (1, L, P, PS, C)
#     y = np.load(sample_dir / "y.npy").astype(np.float32)   # (out_h, out_w)
#     X = np.squeeze(X, axis=0) if X.ndim == 5 and X.shape[0] == 1 else X
#     return X, y


# def _hash_key(data_dir: Path, sample_ids: list[int]) -> str:
#     payload = json.dumps([str(data_dir), sorted(sample_ids), MDS_COLUMNS], sort_keys=True)
#     return hashlib.sha1(payload.encode()).hexdigest()[:16]


# def prep_dataset(data_dir: str | Path, mds_root: str | Path | None = None,
#                  exist_ok: bool = True, force: bool = False, verbose: int = 1) -> Path:
#     """Collect all complete (X.npy, y.npy) samples under ``data_dir`` and write one MDS dataset.

#     The array shapes (x_shape, out_h, out_w) are inferred from the data and asserted to be uniform
#     across all samples. Returns the path to the MDS dir (shards + ``dataset.jsonl`` sidecar).
#     On a cache hit (same data_dir + sample set + MDS_COLUMNS schema) the existing dir is returned
#     unchanged, unless ``force`` is set, in which case the cached dir is rebuilt from scratch.
#     """
#     data_dir = Path(data_dir)
#     samples_dir = data_dir / "samples"
#     mds_root = Path(mds_root) if mds_root is not None else data_dir / "mds"

#     # collect complete samples (both X and y present) + their flattened metadata
#     rows = []
#     skipped_ids = []
#     for sample_dir in sorted(samples_dir.glob("*")):
#         if not (sample_dir / "X.npy").exists() or not (sample_dir / "y.npy").exists() or not (sample_dir / 'metadata.jsonl').exists():
#             skipped_ids.append(sample_dir.name)
#             continue
#         rows.append((sample_dir, flatten_metadata(sample_dir)))
#     if not rows:
#         raise RuntimeError(f"No complete samples (with both X.npy and y.npy) found under {samples_dir}")
#     sample_ids = [int(m["sample_id"]) for _, m in rows]
#     if verbose: print(f"Found {len(rows)} complete samples ({len(skipped_ids)} skipped, missing X.npy)\nskipped ids: {skipped_ids}")

#     # hash-keyed output dir (depends on MDS_COLUMNS too, so schema changes bust the cache) -> instant return on cache hit
#     key = _hash_key(data_dir, sample_ids)
#     mds_path = mds_root / key
#     if mds_path.exists() and force:
#         if verbose: print(f"--force: deleting cached MDS at {mds_path} and rebuilding")
#         shutil.rmtree(mds_path)
#     elif mds_path.exists() and (mds_path / "dataset.jsonl").exists() and exist_ok:
#         if verbose: print(f"Cache hit: reusing existing MDS at {mds_path}\nMDS: {mds_path} ({len(rows)} samples)")
#         return mds_path

#     # write shards + a lightweight metadata sidecar (index.jsonl) for loader-side filtering.
#     # Streaming urlparses `out`, so an absolute Windows path (e.g. "D:/...") is misread as a
#     # cloud scheme "d:". Chdir to the parent and pass the relative dir name -> empty url scheme.
#     if verbose: print(f"Writing MDS to {mds_path} ...")

#     # parse the model-shape fields once from the first sample: X is (L, P, PS, C)
#     X0, y0 = _load_xy(rows[0][0])
#     L, P, PS, _ = X0.shape
#     shape_info = dict(n_laser_rows=int(L ** 0.5), n_laser_cols=int(L ** 0.5), patch_size=PS, n_freqs=P * PS)
#     x_shape0, out_hw = X0.shape, y0.shape

#     # write to MSD format
#     index_rows = []
#     mds_root.mkdir(parents=True, exist_ok=True)
#     cwd = os.getcwd()
#     os.chdir(mds_root)
#     try:
#       with MDSWriter(out=key, columns=MDS_COLUMNS, exist_ok=True) as writer:
#         for i, (sample_dir, meta) in enumerate(rows):
#             X, y = _load_xy(sample_dir)
#             assert X.shape == x_shape0, f"{sample_dir.name}: X.shape={X.shape} != {x_shape0}"
#             assert y.shape == out_hw, f"{sample_dir.name}: y.shape={y.shape} != {out_hw}"

#             com = meta.get("downsampled_com", [-1.0, -1.0])
#             sample = {
#                 "X": X, "y": y, "sample_id": int(meta["sample_id"]), "output_id": str(meta.get("output_id", "")),
#                 "n_objects": int(meta.get("n_objects", -1)),
#                 "speaker": int(meta.get("speaker", -1)),
#                 "box": str(meta.get("box", "")),
#                 "is_empty_box": int(bool(meta.get("is_empty_box", False))),
#                 "object": str(meta.get("object", "")),
#                 "downsampled_com_x": float(com[0]), "downsampled_com_y": float(com[1]),
#             }
#             writer.write(sample)
#             index_rows.append(meta)  # full per-sample metadata -> sidecar (used for loader-side filtering)
#             if verbose >= 2 and (i + 1) % 50 == 0: print(f"  wrote {i + 1}/{len(rows)}")
#     finally:
#         os.chdir(cwd)

#     # dataset-level sidecar (named dataset.jsonl to distinguish it from each sample's metadata.jsonl):
#     # line 0 is dataset-level info, then one line of full metadata per sample for loader-side filtering.
#     data_info = dict(out_h=out_hw[0], out_w=out_hw[1], n_samples=len(rows)) | shape_info
#     lines = [json.dumps(data_info)] + [json.dumps(r) for r in index_rows]
#     (mds_path / "dataset.jsonl").write_text("\n".join(lines))
#     if verbose: print(f"Wrote {len(rows)} samples. data_info={data_info}\nMDS: {mds_path} ({len(rows)} samples)")
#     return mds_path

#***** 2 create StreamingDataset (like pytorch Dataset but faster) *****

class VibrationDataset(StreamingDataset):
    def __init__(self, local: str | Path, shuffle: bool = False, **kwargs):
        super().__init__(local=str(local), shuffle=shuffle, batch_size=kwargs.pop("batch_size", None), **kwargs)

    def __getitem__(self, idx):
        s = super().__getitem__(idx)
        X, y = s.pop("X"), s.pop("y")
        info = dict(sample_id=s["sample_id"], output_id=s["output_id"], n_objects=s["n_objects"], speaker=s["speaker"], box=s["box"], is_empty_box=s["is_empty_box"], x_com=s["downsampled_com_x"], y_com=s["downsampled_com_y"])
        return dict(fft=torch.from_numpy(X.copy()), mask_true=torch.from_numpy(y.copy()), info=info)

#***** 3 build train/eval DataLoaders with filtering + splitting *****

def _matches(row: dict, speakers, n_objects, box) -> bool:
    if speakers is not None and row["speaker"] not in (speakers if isinstance(speakers, list) else [speakers]): return False
    if n_objects is not None and row["n_objects"] not in (n_objects if isinstance(n_objects, list) else [n_objects]): return False
    if box is not None and row["box"] not in (box if isinstance(box, list) else [box]): return False
    return True


def build_dataset(mds_path: str | Path, batch_size: int = 64, eval_batch_size: int = 64, test_size: float = 0.15,
                  unseen_pos_speaker_frac: float = 0.06, seed: int = 42, num_workers: int = 8,
                  speakers=None, n_objects=None, box=None, n_samples: int | None = None,
                  verbose: int = 1):
    """Return (train_loader, eval_loaders) using an already-written MDS.

    Row filters (speakers/n_objects/box/n_samples) are applied via the index sidecar. The kept
    samples are then split into train + 3 eval sets, split at the output_id level (an output_id
    is a unique object layout/position; several samples, one per speaker, share the same output_id):

    - eval/unseen_layout: all samples whose n_objects > 1.
    - eval/unseen_pos: every sample whose output_id is in held_out_output_ids (the 15%).
    - eval/unseen_pos_speaker: a `unseen_pos_speaker_frac` slice of the *sample_ids* (not output_ids) belonging to train_output_ids, so it spans many (position, speaker) combinations.
    - train: the remaining sample_ids from train_output_ids.
    """
    generator = torch.Generator().manual_seed(seed)
    dataset = VibrationDataset(local=mds_path)

    # load dataset.jsonl (skip line 0: dataset-level info now comes from args, not the sidecar)
    lines = (Path(mds_path) / "dataset.jsonl").read_text().strip().splitlines()
    index = [json.loads(line) for line in lines[1:] if line]

    # filter the dataset by speakers, n_objects, box, n_samples
    keep = [i for i, row in enumerate(index) if _matches(row, speakers, n_objects, box)]
    if n_samples is not None: keep = keep[:n_samples]

    # 1. carve out unseen_layout: all samples with n_objects > 1
    unseen_layout_idx = [i for i in keep if index[i]["n_objects"] > 1]
    remaining_idx = [i for i in keep if index[i]["n_objects"] <= 1]

    # 2. split the remaining output_ids 85/15 -> train_output_ids, unseen_pos_output_ids
    output_ids = sorted({index[i]["output_id"] for i in remaining_idx}) # sort for deterministic output
    train_output_ids, unseen_pos_output_ids = train_test_split(output_ids, test_size=test_size, random_state=seed, shuffle=True)
    train_output_ids, unseen_pos_output_ids = set(train_output_ids), set(unseen_pos_output_ids)

    unseen_pos_idx = [i for i in remaining_idx if index[i]["output_id"] in unseen_pos_output_ids]
    train_pool_idx = [i for i in remaining_idx if index[i]["output_id"] in train_output_ids]

    # 3. carve unseen_pos_speaker_frac of the sample_ids (not output_ids) out of the train pool,
    # so this eval spans many (position, speaker) combinations rather than a few whole layouts
    train_idx, unseen_pos_speaker_idx = train_test_split(train_pool_idx, test_size=unseen_pos_speaker_frac, random_state=seed, shuffle=True)

    if verbose:
        print(f"{len(index)} total samples, {len(keep)} after filtering -> "
              f"train={len(train_idx)}, eval/unseen_pos_speaker={len(unseen_pos_speaker_idx)}, "
              f"eval/unseen_pos={len(unseen_pos_idx)}, eval/unseen_layout={len(unseen_layout_idx)}")

    def num_samples(batch): return batch["mask_true"].shape[0]
    def loader(idxs, bs, shuffle):
        dl = DataLoader(Subset(dataset, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
        return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

    train_loader = loader(train_idx, batch_size, shuffle=True)
    eval_loaders = [
        Evaluator(label="eval/unseen_pos_speaker", dataloader=loader(unseen_pos_speaker_idx, eval_batch_size, shuffle=False)),
        Evaluator(label="eval/unseen_pos", dataloader=loader(unseen_pos_idx, eval_batch_size, shuffle=False)),
        Evaluator(label="eval/unseen_layout", dataloader=loader(unseen_layout_idx, eval_batch_size, shuffle=False)),
    ]
    return train_loader, eval_loaders


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=r"D:/eturok/experiment-22/data")
    p.add_argument("--mds-root", default=None)
    p.add_argument("--force", action="store_true", default=False, help="delete and rebuild the cached MDS even on a cache hit")
    args = p.parse_args()
    path = prep_dataset(args.data_dir, args.mds_root, force=args.force, verbose=2)
    print(f"MDS written to: {path}")
