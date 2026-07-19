import os, json, hashlib, shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from composer.core import Evaluator, DataSpec
from streaming import StreamingDataset, MDSWriter

from utils.io_utils import load

# ***** 1. turn dataset into MDS format (sharded, streaming) *****
DATA_INFO = {"out_h": 18, "out_w": 44, "n_samples": 0,
<<<<<<< Updated upstream
             "n_laser_rows": 10, "n_laser_cols": 10, "patch_size": 256, "n_freqs": 3328, "n_channels": 2}
=======
             "n_laser_rows": 10, "n_laser_cols": 10, "patch_size": 256, "n_freqs": 3328, "n_coords": 2}
>>>>>>> Stashed changes


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


def exp22_split(mds_path: str | Path, batch_size: int = 64, eval_batch_size: int = 64, test_size: float = 0.15,
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

    # load metadata.jsonl (skip line 0: dataset-level info now comes from args, not the sidecar)
    lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
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
    def loader(idxs, bs, shuffle=False, drop_last=False):
        dl = DataLoader(Subset(dataset, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, drop_last=drop_last)
        return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

    train_loader = loader(train_idx, batch_size, shuffle=True, drop_last=True)
    eval_loaders = [
        Evaluator(label="eval/unseen_pos_speaker", dataloader=loader(unseen_pos_speaker_idx, eval_batch_size)),
        Evaluator(label="eval/unseen_pos", dataloader=loader(unseen_pos_idx, eval_batch_size)),
        Evaluator(label="eval/unseen_layout", dataloader=loader(unseen_layout_idx, eval_batch_size)),
    ]
    return train_loader, eval_loaders

def exp23_split(mds_path: str | Path, batch_size: int = 64, eval_batch_size: int = 64, test_size: float = 0.15,
                  seed: int = 42, num_workers: int = 8,
                  speakers=None, n_objects=None, box=None, n_samples: int | None = None,
                  verbose: int = 1):
    """Return (train_loader, eval_loaders) using an already-written MDS, split by `layout`.

    Row filters (speakers/n_objects/box/n_samples) are applied via the index sidecar. The kept
    samples are then split into train + 6 eval sets:

    - eval/unseen_cylinder_2_cube_1: all samples with layout 'cylinder-2-cube-1'.
    - eval/unseen_cylinder_2_stacked: all samples with layout 'cylinder-1-stacked'.
    - eval/unseen_cylinder_{1,2}: a test_size fraction of that layout's output_ids (whole positions held out; an output_id is a unique object layout/position shared by one sample per speaker).
    - eval/unseen_cylinder_{1,2}_speaker_pos: a test_size/2 fraction of that layout's remaining *sample_ids* (not output_ids), so it spans many (position, speaker) combinations.
    - train: everything left in cylinder-1 and cylinder-2.
    """
    generator = torch.Generator().manual_seed(seed)
    dataset = VibrationDataset(local=mds_path)

    # load metadata.jsonl (no dataset-level header line in this format: every line is a sample)
    lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
    index = [json.loads(line) for line in lines if line]

    # filter the dataset by speakers, n_objects, box, n_samples
    keep = [i for i, row in enumerate(index) if _matches(row, speakers, n_objects, box)]
    if n_samples is not None: keep = keep[:n_samples]

    # 1. whole layouts held out entirely
    train_idx, evals = [], {}
    evals["eval/unseen_cylinder_2_cube_1"] = [i for i in keep if index[i]["layout"] == "cylinder-2-cube-1"]
    evals["eval/unseen_cylinder_2_stacked"] = [i for i in keep if index[i]["layout"] == "cylinder-1-stacked"]

    for layout in ("cylinder-1", "cylinder-2"):
        name = layout.replace("-", "_")
        layout_idx = [i for i in keep if index[i]["layout"] == layout]

        # 2. hold out test_size of this layout's output_ids -> eval/unseen_cylinder_{1,2}
        output_ids = sorted({index[i]["output_id"] for i in layout_idx}) # sort for deterministic output
        _, unseen_output_ids = train_test_split(output_ids, test_size=test_size, random_state=seed, shuffle=True)
        unseen_output_ids = set(unseen_output_ids)
        evals[f"eval/unseen_{name}"] = [i for i in layout_idx if index[i]["output_id"] in unseen_output_ids]

        # 3. carve test_size/2 of the remaining sample_ids (not output_ids) -> eval/unseen_cylinder_{1,2}_speaker_pos,
        # so this eval spans many (position, speaker) combinations rather than a few whole layouts
        pool_idx = [i for i in layout_idx if index[i]["output_id"] not in unseen_output_ids]
        layout_train_idx, evals[f"eval/unseen_{name}_speaker_pos"] = train_test_split(pool_idx, test_size=test_size / 2, random_state=seed, shuffle=True)
        train_idx += layout_train_idx

    if verbose:
        counts = ", ".join(f"{label}={len(idxs)}" for label, idxs in evals.items())
        print(f"{len(index)} total samples, {len(keep)} after filtering -> train={len(train_idx)}, {counts}")

    def num_samples(batch): return batch["mask_true"].shape[0]
    def loader(idxs, bs, shuffle=False, drop_last=False):
        dl = DataLoader(Subset(dataset, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, drop_last=drop_last)
        return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

    train_loader = loader(train_idx, batch_size, shuffle=True, drop_last=True)
    eval_loaders = [Evaluator(label=label, dataloader=loader(idxs, eval_batch_size)) for label, idxs in evals.items()]
    return train_loader, eval_loaders

def exp24_split(mds_path: str | Path, batch_size: int = 64, eval_batch_size: int = 64, test_size: float = 0.15,
                  speaker_size: float = 0.05, seed: int = 42, num_workers: int = 8,
                  speakers=None, n_objects=None, box=None, n_samples: int | None = None,
                  verbose: int = 1):
    """Return (train_loader, eval_loaders) using an already-written MDS, split by `layout`.

    Row filters (speakers/n_objects/box/n_samples) are applied via the index sidecar. Every layout
    puts ~test_size + speaker_size of its samples in eval, selected by sample_id:

    - eval/{bullet,cylinder,cylinder_bullet,stacked}: ~test_size of the layout's samples, held out as
      whole output_ids (an output_id is one object position; its ~8 samples, one per speaker, stay
      together) -> unseen positions.
    - eval/{...}_speaker: ~speaker_size of the layout's samples, one sample from each of that many
      *distinct* remaining output_ids -> unseen (position, speaker) pairs, no output_id repeated.
    - eval/empty_box: the empty box has no object, so its output_ids are the same empty scene
      re-recorded and the position/speaker distinction collapses: one flat sample-level split of
      test_size + speaker_size.
    - train: everything left.
    """
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    dataset = VibrationDataset(local=mds_path)

    # load metadata.jsonl (no dataset-level header line in this format: every line is a sample)
    lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
    index = [json.loads(line) for line in lines if line]

    # filter the dataset by speakers, n_objects, box, n_samples
    keep = [i for i, row in enumerate(index) if _matches(row, speakers, n_objects, box)]
    if n_samples is not None: keep = keep[:n_samples]

    train_idx, evals = [], {}
    for layout in sorted({index[i]["layout"] for i in keep}):
        name = layout.replace("-1", "").replace("-", "_") # bullet-1 -> bullet, cylinder-1-bullet-1 -> cylinder_bullet, empty-box -> empty_box
        # sort by sample_id so the split is keyed to sample_ids, not row order in the sidecar
        layout_idx = sorted((i for i in keep if index[i]["layout"] == layout), key=lambda i: index[i]["sample_id"])

        if layout == "empty-box":
            # every output_id is the same empty scene re-recorded -> one flat sample-level split
            layout_train_idx, evals[f"eval/{name}"] = train_test_split(layout_idx, test_size=test_size + speaker_size, random_state=seed, shuffle=True)
            train_idx += layout_train_idx
            continue

        # 1. hold out ~test_size of the samples as whole output_ids -> eval/{name}
        output_ids = sorted({index[i]["output_id"] for i in layout_idx}) # sort for deterministic output
        _, unseen_output_ids = train_test_split(output_ids, test_size=test_size, random_state=seed, shuffle=True)
        unseen_output_ids = set(unseen_output_ids)
        evals[f"eval/{name}"] = [i for i in layout_idx if index[i]["output_id"] in unseen_output_ids]

        # 2. take ~speaker_size of the samples: one sample from each of that many distinct remaining
        # output_ids -> eval/{name}_speaker spans many positions with no output_id repeated
        pool = {}
        for i in layout_idx:
            if index[i]["output_id"] not in unseen_output_ids: pool.setdefault(index[i]["output_id"], []).append(i)
        n_speaker = round(speaker_size * len(layout_idx))
        speaker_idx = {int(rng.choice(pool[o])) for o in rng.choice(sorted(pool), size=n_speaker, replace=False)}
        evals[f"eval/{name}_speaker"] = sorted(speaker_idx)
        train_idx += [i for idxs in pool.values() for i in idxs if i not in speaker_idx]

    if verbose:
        counts = ", ".join(f"{label}={len(idxs)}" for label, idxs in evals.items())
        print(f"{len(index)} total samples, {len(keep)} after filtering -> train={len(train_idx)}, {counts}")

    def num_samples(batch): return batch["mask_true"].shape[0]
    def loader(idxs, bs, shuffle=False, drop_last=False):
        dl = DataLoader(Subset(dataset, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, drop_last=drop_last)
        return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

    train_loader = loader(train_idx, batch_size, shuffle=True, drop_last=True)
    eval_loaders = [Evaluator(label=label, dataloader=loader(idxs, eval_batch_size)) for label, idxs in evals.items()]
    return train_loader, eval_loaders

SPLIT_METHODS = {
    "exp22": exp22_split,
    "exp23": exp23_split,
    "exp24": exp24_split,
}

def build_dataset(mds_path: str | Path, split: str = "exp22", **kwargs):
    if split not in SPLIT_METHODS: raise ValueError(f"Unknown split {split!r}, expected one of {sorted(SPLIT_METHODS)}")
    return SPLIT_METHODS[split](mds_path, **kwargs)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=r"D:/eturok/experiment-22/data")
    p.add_argument("--mds-root", default=None)
    p.add_argument("--force", action="store_true", default=False, help="delete and rebuild the cached MDS even on a cache hit")
    args = p.parse_args()
    path = prep_dataset(args.data_dir, args.mds_root, force=args.force, verbose=2)
    print(f"MDS written to: {path}")
