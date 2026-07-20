import os, json, hashlib, shutil
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from composer.core import Evaluator, DataSpec
from streaming import StreamingDataset, MDSWriter

from utils.io_utils import load
from data.post_process import process_vibration, process_image, make_rng

# ***** 1. turn dataset into MDS format (sharded, streaming) *****
DATA_INFO = {"out_h": 18, "out_w": 44, "n_samples": 0,
             "n_laser_rows": 10, "n_laser_cols": 10, "patch_size": 256, "n_freqs": 3328, "n_channels": 2}


#***** 2 create StreamingDataset (like pytorch Dataset but faster) *****
# MDS now stores X as the clean complex fft (real/imag stacked, shape (L,F,C,2)) and y as the
# raw downsampled mask -- extract_signal/normalize_fft/tokenize and augmentation happen live,
# here or in augmenting_collate below, so they can differ every epoch (see notebooks/53).

AugmentSite = Literal["none", "getitem", "collate"]

class VibrationDataset(StreamingDataset):
    def __init__(self, local: str | Path, shuffle: bool = False, augment_site: AugmentSite = "none",
                 signal_mode: str = "magnitude", normalize_mode: str = "std-sample", patch_size: int = 256, **kwargs):
        super().__init__(local=str(local), shuffle=shuffle, batch_size=kwargs.pop("batch_size", None), **kwargs)
        self.augment_site = augment_site
        self.signal_mode, self.normalize_mode, self.patch_size = signal_mode, normalize_mode, patch_size
        self.freqs = np.load(Path(local) / "freqs.npy")
        self._epoch = 0

    def set_epoch(self, epoch: int):
        # mirrors DistributedSampler.set_epoch -- call once per epoch from the training loop so
        # augmentation seeds (epoch, sample_id) differ every epoch but stay reproducible.
        self._epoch = epoch

    def __getitem__(self, idx):
        s = super().__getitem__(idx)
        X, y = s.pop("X"), s.pop("y")  # X: (L,F,C,2) real/imag fft, y: (H,W) raw mask
        sample_id = s["sample_id"]
        info = dict(sample_id=sample_id, output_id=s["output_id"], n_objects=s["n_objects"], speaker=s["speaker"], box=s["box"], is_empty_box=s["is_empty_box"], x_com=s["downsampled_com_x"], y_com=s["downsampled_com_y"])

        if self.augment_site == "collate":
            # defer all processing to augmenting_collate; hand back raw complex fft + raw mask
            fft_complex = X[..., 0] + 1j * X[..., 1]
            return dict(fft=fft_complex, mask_true=y, info=info)

        rng = make_rng(self._epoch, sample_id) if self.augment_site == "getitem" else None
        fft_complex = (X[..., 0] + 1j * X[..., 1])[None]  # (1,L,F,C)
        fft_processed = process_vibration(fft_complex, self.freqs, self.signal_mode, self.normalize_mode, self.patch_size, rng=rng)[0]
        mask_processed = process_image(y[None], rng=rng)[0]
        return dict(fft=torch.from_numpy(fft_processed.copy()), mask_true=torch.from_numpy(mask_processed.copy()), info=info)


def augmenting_collate(batch: list[dict], epoch: int, signal_mode: str, normalize_mode: str, patch_size: int, freqs: np.ndarray, augment: bool):
    """Batched counterpart to VibrationDataset(augment_site='getitem'): stacks raw complex fft +
    raw mask across the batch, then runs process_vibration/process_image once on the whole
    batch instead of once per sample. Only meaningful when the dataset was built with
    augment_site='collate' (otherwise __getitem__ already returned processed tensors)."""
    fft = np.stack([b["fft"] for b in batch])          # (B,L,F,C) complex
    mask = np.stack([b["mask_true"] for b in batch])   # (B,H,W)
    infos = [b["info"] for b in batch]
    rng = make_rng(epoch, infos[0]["sample_id"]) if augment else None  # one shared draw per batch
    fft_processed = process_vibration(fft, freqs, signal_mode, normalize_mode, patch_size, rng=rng)
    mask_processed = process_image(mask, rng=rng)
    return dict(fft=torch.from_numpy(fft_processed.copy()), mask_true=torch.from_numpy(mask_processed.copy()), info=infos)

#***** 3 build train/eval DataLoaders with filtering + splitting *****

def _matches(row: dict, speakers, n_objects, box) -> bool:
    if speakers is not None and row["speaker"] not in (speakers if isinstance(speakers, list) else [speakers]): return False
    if n_objects is not None and row["n_objects"] not in (n_objects if isinstance(n_objects, list) else [n_objects]): return False
    if box is not None and row["box"] not in (box if isinstance(box, list) else [box]): return False
    return True


def exp22_split(mds_path: str | Path, batch_size: int = 64, eval_batch_size: int = 64, test_size: float = 0.15,
                  unseen_pos_speaker_frac: float = 0.06, seed: int = 42, num_workers: int = 8, augment_site: str = "none",
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
    dataset = VibrationDataset(local=mds_path, augment_site=augment_site)
    eval_dataset = VibrationDataset(local=mds_path, augment_site="none")  # eval never augments, even when train uses augment_site="getitem"

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
    def loader(idxs, bs, shuffle=False, drop_last=False, augment=False):
        # collate site: __getitem__ always hands back raw data regardless of augment, so both
        # train and eval can share `dataset` -- augmenting_collate's own `augment` flag gates it.
        # getitem site (or "none"): augmentation is baked into __getitem__ itself, so eval must
        # use a separate augment_site="none" instance to guarantee it never augments.
        ds = dataset if (augment or augment_site == "collate") else eval_dataset
        collate_fn = (lambda batch: augmenting_collate(batch, dataset._epoch, dataset.signal_mode, dataset.normalize_mode, dataset.patch_size, dataset.freqs, augment)) if augment_site == "collate" else None
        dl = DataLoader(Subset(ds, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, drop_last=drop_last, collate_fn=collate_fn)
        return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

    train_loader = loader(train_idx, batch_size, shuffle=True, drop_last=True, augment=True)
    eval_loaders = [
        Evaluator(label="eval/unseen_pos_speaker", dataloader=loader(unseen_pos_speaker_idx, eval_batch_size)),
        Evaluator(label="eval/unseen_pos", dataloader=loader(unseen_pos_idx, eval_batch_size)),
        Evaluator(label="eval/unseen_layout", dataloader=loader(unseen_layout_idx, eval_batch_size)),
    ]
    return train_loader, eval_loaders

def exp23_split(mds_path: str | Path, batch_size: int = 64, eval_batch_size: int = 64, test_size: float = 0.15,
                  seed: int = 42, num_workers: int = 8, augment_site: str = "none",
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
    dataset = VibrationDataset(local=mds_path, augment_site=augment_site)
    eval_dataset = VibrationDataset(local=mds_path, augment_site="none")  # eval never augments, even when train uses augment_site="getitem"

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
    def loader(idxs, bs, shuffle=False, drop_last=False, augment=False):
        # collate site: __getitem__ always hands back raw data regardless of augment, so both
        # train and eval can share `dataset` -- augmenting_collate's own `augment` flag gates it.
        # getitem site (or "none"): augmentation is baked into __getitem__ itself, so eval must
        # use a separate augment_site="none" instance to guarantee it never augments.
        ds = dataset if (augment or augment_site == "collate") else eval_dataset
        collate_fn = (lambda batch: augmenting_collate(batch, dataset._epoch, dataset.signal_mode, dataset.normalize_mode, dataset.patch_size, dataset.freqs, augment)) if augment_site == "collate" else None
        dl = DataLoader(Subset(ds, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, drop_last=drop_last, collate_fn=collate_fn)
        return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

    train_loader = loader(train_idx, batch_size, shuffle=True, drop_last=True, augment=True)
    eval_loaders = [Evaluator(label=label, dataloader=loader(idxs, eval_batch_size)) for label, idxs in evals.items()]
    return train_loader, eval_loaders

def exp24_split(mds_path: str | Path, batch_size: int = 64, eval_batch_size: int = 64, test_size: float = 0.15,
                  speaker_size: float = 0.05, seed: int = 42, num_workers: int = 8, augment_site: str = "none",
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
    dataset = VibrationDataset(local=mds_path, augment_site=augment_site)
    eval_dataset = VibrationDataset(local=mds_path, augment_site="none")  # eval never augments, even when train uses augment_site="getitem"

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
    def loader(idxs, bs, shuffle=False, drop_last=False, augment=False):
        # collate site: __getitem__ always hands back raw data regardless of augment, so both
        # train and eval can share `dataset` -- augmenting_collate's own `augment` flag gates it.
        # getitem site (or "none"): augmentation is baked into __getitem__ itself, so eval must
        # use a separate augment_site="none" instance to guarantee it never augments.
        ds = dataset if (augment or augment_site == "collate") else eval_dataset
        collate_fn = (lambda batch: augmenting_collate(batch, dataset._epoch, dataset.signal_mode, dataset.normalize_mode, dataset.patch_size, dataset.freqs, augment)) if augment_site == "collate" else None
        dl = DataLoader(Subset(ds, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, drop_last=drop_last, collate_fn=collate_fn)
        return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

    train_loader = loader(train_idx, batch_size, shuffle=True, drop_last=True, augment=True)
    eval_loaders = [Evaluator(label=label, dataloader=loader(idxs, eval_batch_size)) for label, idxs in evals.items()]
    return train_loader, eval_loaders

# each object-type's samples live in two output_id (== image_id) ranges: a "train" range that is
# always train, and one or more "eval-eligible" ranges we may only draw eval samples from.
EXP25_GROUPS = {
    "purple_cube":         {"train_range": (3, 29),   "eval_ranges": [(30, 59)]},
    "green_cube":          {"train_range": (110, 124), "eval_ranges": [(125, 127)]},
    "purple_green_cubes":  {"train_range": (60, 88),  "eval_ranges": [(89, 109)]},
}
EXP25_ALWAYS_TRAIN_RANGE = (0, 2) # empty-box

def exp25_split(mds_path: str | Path, batch_size: int = 64, eval_batch_size: int = 64, test_size: float = 0.20,
                  unseen_pos_frac: float = 0.15, unseen_pos_speaker_frac: float = 0.05, seed: int = 42, num_workers: int = 8, augment_site: str = "none",
                  speakers=None, n_objects=None, box=None, n_samples: int | None = None,
                  verbose: int = 1):
    """Return (train_loader, eval_loaders) using an already-written MDS, split by object-type.

    Each object-type (purple_cube, green_cube, purple_green_cubes) has an output_id (== image_id)
    range that is always train (its "grid-1" layout, or the train-side layout for the mixed type)
    and one or more output_id ranges we may only draw eval samples from (its "grid-2" layout(s)).
    empty-box is always train, no eval carve-out.

    For each type:
    - combined pool = train-range samples + eval-range samples.
    - target_eval = test_size * len(combined pool).
    - If the eval-range pool is smaller than target_eval, the *whole* eval-range pool becomes eval
      candidates (we never dip into the train range to make up the difference) -> this type ends up
      with < test_size eval, and all of its train-range samples stay in train.
    - Otherwise, target_eval samples (whole output_ids) are drawn from the eval-range pool as eval
      candidates, and the leftover eval-range output_ids fall back into train.
    - The eval candidates are then split unseen_pos_frac / unseen_pos_speaker_frac (of the combined
      pool, not of the eval candidates) into:
        - eval/{type}: whole held-out output_ids -> positions never seen in train, from any speaker.
        - eval/{type}_speaker: individual samples from output_ids that DO appear in train -> this
          exact (position, speaker) pair is new, but the position was seen from other speakers.
      If the eval-range pool was capped smaller than target_eval, these two eval sets are scaled down
      proportionally to fit inside what's available.
    """
    generator = torch.Generator().manual_seed(seed)
    dataset = VibrationDataset(local=mds_path, augment_site=augment_site)
    eval_dataset = VibrationDataset(local=mds_path, augment_site="none")  # eval never augments, even when train uses augment_site="getitem"

    # load metadata.jsonl (no dataset-level header line in this format: every line is a sample)
    lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
    index = [json.loads(line) for line in lines if line]

    # filter the dataset by speakers, n_objects, box, n_samples
    keep = [i for i, row in enumerate(index) if _matches(row, speakers, n_objects, box)]
    if n_samples is not None: keep = keep[:n_samples]

    def in_range(i, lo, hi): return lo <= int(index[i]["output_id"]) <= hi

    train_idx = [i for i in keep if in_range(i, *EXP25_ALWAYS_TRAIN_RANGE)]
    evals = {}
    for name, group in EXP25_GROUPS.items():
        train_range_idx = [i for i in keep if in_range(i, *group["train_range"])]
        eval_pool_idx = [i for i in keep if any(in_range(i, lo, hi) for lo, hi in group["eval_ranges"])]
        combined_n = len(train_range_idx) + len(eval_pool_idx)

        target_eval = test_size * combined_n
        eval_pool_output_ids = sorted({index[i]["output_id"] for i in eval_pool_idx})

        if len(eval_pool_idx) <= target_eval:
            # worst case: the whole eval-range pool becomes eval candidates, nothing more
            eval_candidate_idx = eval_pool_idx
            train_idx += train_range_idx
            scale = len(eval_pool_idx) / target_eval if target_eval > 0 else 0.0
        else:
            n_output_ids = round(target_eval / len(eval_pool_idx) * len(eval_pool_output_ids))
            n_output_ids = min(max(n_output_ids, 0), len(eval_pool_output_ids))
            eval_candidate_output_ids, _ = train_test_split(
                eval_pool_output_ids, train_size=n_output_ids, random_state=seed, shuffle=True)
            eval_candidate_output_ids = set(eval_candidate_output_ids)
            eval_candidate_idx = [i for i in eval_pool_idx if index[i]["output_id"] in eval_candidate_output_ids]
            train_idx += train_range_idx + [i for i in eval_pool_idx if index[i]["output_id"] not in eval_candidate_output_ids]
            scale = 1.0

        # split eval candidates' output_ids: unseen_pos_frac -> whole new positions, rest -> pool for _speaker
        pos_frac = min(unseen_pos_frac * scale, unseen_pos_frac + unseen_pos_speaker_frac)
        eval_candidate_output_ids = sorted({index[i]["output_id"] for i in eval_candidate_idx})
        n_unseen_pos = round(pos_frac / test_size * len(eval_candidate_output_ids)) if test_size > 0 else 0
        n_unseen_pos = min(max(n_unseen_pos, 0), len(eval_candidate_output_ids))
        unseen_pos_output_ids, speaker_pool_output_ids = train_test_split(
            eval_candidate_output_ids, train_size=n_unseen_pos, random_state=seed, shuffle=True) if 0 < n_unseen_pos < len(eval_candidate_output_ids) \
            else (eval_candidate_output_ids, []) if n_unseen_pos == len(eval_candidate_output_ids) \
            else ([], eval_candidate_output_ids)
        unseen_pos_output_ids = set(unseen_pos_output_ids)
        evals[f"eval/{name}"] = [i for i in eval_candidate_idx if index[i]["output_id"] in unseen_pos_output_ids]

        # unseen_pos_speaker_frac of the combined pool, taken as individual samples from the
        # speaker-pool output_ids (these output_ids are NOT held out -> they stay available to train too)
        speaker_pool_idx = [i for i in eval_candidate_idx if index[i]["output_id"] in set(speaker_pool_output_ids)]
        n_speaker = round(unseen_pos_speaker_frac * scale / test_size * combined_n) if test_size > 0 else 0
        n_speaker = min(max(n_speaker, 0), len(speaker_pool_idx))
        if n_speaker < len(speaker_pool_idx):
            remainder_idx, speaker_idx = train_test_split(speaker_pool_idx, test_size=n_speaker, random_state=seed, shuffle=True)
        else:
            remainder_idx, speaker_idx = [], speaker_pool_idx
        evals[f"eval/{name}_speaker"] = speaker_idx
        train_idx += remainder_idx

    if verbose:
        counts = ", ".join(f"{label}={len(idxs)}" for label, idxs in evals.items())
        print(f"{len(index)} total samples, {len(keep)} after filtering -> train={len(train_idx)}, {counts}")

    def num_samples(batch): return batch["mask_true"].shape[0]
    def loader(idxs, bs, shuffle=False, drop_last=False, augment=False):
        # collate site: __getitem__ always hands back raw data regardless of augment, so both
        # train and eval can share `dataset` -- augmenting_collate's own `augment` flag gates it.
        # getitem site (or "none"): augmentation is baked into __getitem__ itself, so eval must
        # use a separate augment_site="none" instance to guarantee it never augments.
        ds = dataset if (augment or augment_site == "collate") else eval_dataset
        collate_fn = (lambda batch: augmenting_collate(batch, dataset._epoch, dataset.signal_mode, dataset.normalize_mode, dataset.patch_size, dataset.freqs, augment)) if augment_site == "collate" else None
        dl = DataLoader(Subset(ds, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, drop_last=drop_last, collate_fn=collate_fn)
        return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

    train_loader = loader(train_idx, batch_size, shuffle=True, drop_last=True, augment=True)
    eval_loaders = [Evaluator(label=label, dataloader=loader(idxs, eval_batch_size)) for label, idxs in evals.items()]
    return train_loader, eval_loaders

SPLIT_METHODS = {
    "exp22": exp22_split,
    "exp23": exp23_split,
    "exp24": exp24_split,
    "exp25": exp25_split,
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
