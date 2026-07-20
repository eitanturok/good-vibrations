import hashlib, json, os, shutil
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from composer.core import Evaluator, DataSpec
from streaming import StreamingDataset, MDSWriter

#***** 0 convert to MDS *****
REQUIRED_FILES = ["image/02_smask.png", "vibration/03_fft.npz", "metadata.jsonl"]

MDS_COLUMNS = {"X": "ndarray:float32", "y": "ndarray:float32",
               "sample_id": "int", "output_id": "str",
               "n_objects": "int", "speaker": "int", "box": "str", "is_empty_box": "int", "object": "str",
               "downsampled_com_x": "float64", "downsampled_com_y": "float64"}

def collect_samples(base_sample_dir: Path, verbose: int = 1) -> list[tuple[Path, dict]]:
    samples, missing_by_file = [], {f: [] for f in REQUIRED_FILES}
    for sample_dir in sorted(base_sample_dir.glob("*")):
        if not sample_dir.is_dir(): continue
        missing = [f for f in REQUIRED_FILES if not (sample_dir / f).exists()]
        if missing:
            for f in missing: missing_by_file[f].append(sample_dir.name)
            continue
        meta = {k: v for d in (json.loads(line) for line in (sample_dir / "metadata.jsonl").read_text().splitlines() if line) for k, v in d.items()}
        samples.append((sample_dir, meta))

    n_skipped = len({sid for ids in missing_by_file.values() for sid in ids})
    if verbose:
        print(f"Found {len(samples)} complete samples ({n_skipped} skipped)")
        for f, ids in missing_by_file.items():
            if ids: print(f"missing {f!r}: {ids}")
    return samples

def hash_samples(samples: list[tuple[Path, dict]]) -> str:
    h = hashlib.sha256()
    for sample_dir, meta in sorted(samples, key=lambda s: s[0].name):
        h.update(sample_dir.name.encode())
        h.update(json.dumps(meta, sort_keys=True).encode())
    return h.hexdigest()

def convert_to_mds(mds_dir: Path, samples: list[tuple[Path, dict]], verbose: int = 1) -> Path:

    from PIL import Image
    first_fft = np.load(samples[0][0] / "vibration/03_fft.npz")["fft"]  # (1, L, F, C) complex
    n_lasers, n_freqs = first_fft.shape[1], first_fft.shape[2]
    first_mask = np.array(Image.open(samples[0][0] / "image/02_smask.png"))
    x_shape, y_shape = (n_lasers, n_freqs, 2, 2), first_mask.shape  # X: (L,F,C,2) real/imag fft; y: raw (H,W) mask

    index_rows = []
    mds_dir.parent.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(mds_dir.parent)

    try:
        with MDSWriter(out=mds_dir.name, columns=MDS_COLUMNS, exist_ok=False) as writer:
            for i, (sample_dir, meta) in enumerate(samples):
                fft_npz = np.load(sample_dir / "vibration/03_fft.npz")
                X = fft_npz["fft"]                                          # (1, L, F, C) complex64
                y = (np.array(Image.open(sample_dir / "image/02_smask.png")).astype(np.float32) / 255.0)  # (H,W) in [0,1]
                X = np.squeeze(X, axis=0) if X.ndim == 4 and X.shape[0] == 1 else X  # -> (L,F,C)
                X = np.stack([X.real, X.imag], axis=-1).astype(np.float32)  # complex -> (L,F,C,2) real/imag
                assert X.shape == x_shape, f"{sample_dir.name}: X.shape={X.shape} != {x_shape}"
                assert y.shape == y_shape, f"{sample_dir.name}: y.shape={y.shape} != {y_shape}"

                com = meta.get("downsampled_com", [-1.0, -1.0])
                sample = {
                    "X": X, "y": y,
                    "sample_id": int(meta["sample_id"]), "output_id": str(meta.get("output_id", "")),
                    "n_objects": int(meta.get("n_objects", -1)),
                    "speaker": int(meta.get("speaker", -1)),
                    "box": str(meta.get("box", "")),
                    "is_empty_box": int(bool(meta.get("is_empty_box", False))),
                    "object": str(meta.get("object", "")),
                    "downsampled_com_x": float(com[0]), "downsampled_com_y": float(com[1]),
                }
                writer.write(sample)
                index_rows.append(meta)
                if verbose >= 2 and (i + 1) % 50 == 0: print(f"  wrote {i + 1}/{len(rows)}")
    finally:
        os.chdir(cwd)

    # freqs is identical across every sample (same fft grid) -- one sidecar, not duplicated per-row
    freqs = np.load(samples[0][0] / "vibration/03_fft.npz")["freqs"]
    np.save(mds_dir / "freqs.npy", freqs)

    # save metadata as a sidecar for loader-side filtering
    lines = "\n".join(json.dumps(r) for r in index_rows)
    (mds_dir / "metadata.jsonl").write_text(lines)
    (mds_dir / "samples_hash.txt").write_text(hash_samples(samples))
    if verbose: print(f"Wrote {len(samples)} samples to {mds_dir=}")
    return mds_dir

#***** 1 process image *****

def _gaussian_kernel1d_radius1(sigma: torch.Tensor) -> torch.Tensor:
    # Matches scipy.ndimage._gaussian_kernel1d(sigma, order=0, radius=1) exactly: truncated
    # Gaussian at x in {-1,0,1}, renormalized to sum to 1.
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=sigma.dtype, device=sigma.device)
    w = torch.exp(-0.5 * (x / sigma) ** 2)
    return w / w.sum()

def gaussian_blur(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    # mask: (B,H,W). Separable 3x3 blur via two 1D convs, replicate-padded (matches scipy's
    # mode='nearest', i.e. edge replication).
    sigma_t = torch.as_tensor(sigma, dtype=mask.dtype, device=mask.device)
    k = _gaussian_kernel1d_radius1(sigma_t)
    x = mask.unsqueeze(1)  # (B,1,H,W)
    x = F.pad(x, (0, 0, 1, 1), mode='replicate')
    x = F.conv2d(x, k.view(1, 1, 3, 1))
    x = F.pad(x, (1, 1, 0, 0), mode='replicate')
    x = F.conv2d(x, k.view(1, 1, 1, 3))
    return x.squeeze(1)

def downsample(mask: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    # Box/area resampling -- area-averages over the full H x W mask, matching PIL's Image.BOX
    x = mask.unsqueeze(1)  # (B,1,H,W)
    out = F.interpolate(x, size=(out_h, out_w), mode='area')
    return out.squeeze(1)

def noisy_blur(mask: torch.Tensor, generator: torch.Generator, sigma: float = 0.8, noise_std: float = 0.05) -> torch.Tensor:
    # Gaussian blur, then add noise back onto originally-nonzero pixels only. mask: (B,H,W).
    blurred = gaussian_blur(mask, sigma)
    noise = torch.normal(0.0, noise_std, size=mask.shape, generator=generator, device="cpu", dtype=torch.float32).to(mask.device, mask.dtype)
    out = blurred.clone()
    nonzero = mask != 0
    out[nonzero] = out[nonzero] + noise[nonzero]
    return out.clamp(0, 1)

def process_image(mask: torch.Tensor, out_h: int, out_w: int, augment: bool = True, generator: torch.Generator | None = None, augment_fn=noisy_blur) -> torch.Tensor:
    mask = downsample(mask, out_h, out_w)
    # apply augmentation after downsampling so augmentations don't get washed out
    if augment: mask = augment_fn(mask, generator)
    return mask

#***** 2 process vibration *****

def extract_signal(x: torch.Tensor, signal_mode: str) -> torch.Tensor:
    if signal_mode == "magnitude": return x.abs()
    if signal_mode == "complex": return torch.cat([x.real, x.imag], dim=-1)
    if signal_mode == "mag_phase": return torch.cat([x.abs(), x.angle()], dim=-1)
    raise ValueError(f"Unknown signal mode: {signal_mode}")

def normalize_fft(x: torch.Tensor, normalize_mode: str) -> torch.Tensor:
    if normalize_mode is None: return x
    x64 = x.double()
    if normalize_mode == 'std':
        std = x64.std(dim=(1, 2, 3), correction=1, keepdim=True).clamp_min(1e-8).float()
        return x / std
    if normalize_mode == 'z':
        mean = x64.mean(dim=(1, 2, 3), keepdim=True).float()
        std = x64.std(dim=(1, 2, 3), correction=1, keepdim=True).clamp_min(1e-8).float()
        return (x - mean) / std
    raise ValueError(f"Unknown normalize mode: {normalize_mode}")

def tokenize(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size <= 0: return x
    B, L, F_, C = x.shape
    P = F_ // patch_size
    return x[:, :, :P * patch_size, :].reshape(B, L, P, patch_size, C)

def _hermit_poly(t: torch.Tensor) -> torch.Tensor:
    tt = t[None, :] ** torch.arange(4, device=t.device, dtype=t.dtype)[:, None]
    A = torch.tensor([[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]], dtype=t.dtype, device=t.device)
    return A @ tt

def random_frequency_gain(freqs: torch.Tensor, generator: torch.Generator, n_control: int = 5, gain_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    # Monotone cubic (Hermite spline) interpolant through n_control random control points
    control_freqs = torch.linspace(freqs.min().item(), freqs.max().item(), n_control, dtype=freqs.dtype, device=freqs.device)
    # generator always lives on cpu (see VibrationDataset.__init__); draw there, move after.
    control_gains = torch.normal(1.0, 1.0, size=(n_control,), generator=generator, device="cpu", dtype=torch.float32).to(freqs.device, freqs.dtype)
    idxs = torch.searchsorted(control_freqs[1:], freqs).clamp(max=n_control - 2)
    m = (control_gains[1:] - control_gains[:-1]) / (control_freqs[1:] - control_freqs[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    dx = control_freqs[idxs + 1] - control_freqs[idxs]
    t = (freqs - control_freqs[idxs]) / dx
    H = _hermit_poly(t)
    gain = H[0] * control_gains[idxs] + H[1] * m[idxs] * dx + H[2] * control_gains[idxs + 1] + H[3] * m[idxs + 1] * dx
    gain = (gain - gain.min()) / (gain.max() - gain.min())
    lo, hi = gain_range
    return gain * (hi - lo) + lo

def augment_vibration(fft: torch.Tensor, freqs: torch.Tensor, generator: torch.Generator, n_control: int = 5, gain_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    gain = random_frequency_gain(freqs, generator, n_control, gain_range)
    return fft * gain[None, None, :, None]

def process_vibration(fft: torch.Tensor, freqs: torch.Tensor, signal_mode: str, normalize_mode: str, patch_size: int, augment: bool = True, generator: torch.Generator | None = None, gain_kwargs: dict | None = None) -> torch.Tensor:
    """fft: (B,L,F,C) complex, on the target device. Returns tokenized (B,L,P,patch_size,C) float32."""
    if augment: fft = augment_vibration(fft, freqs, generator, **(gain_kwargs or {}))
    x = extract_signal(fft, signal_mode).float()
    x = normalize_fft(x, normalize_mode)
    return tokenize(x, patch_size)

#***** 3 process batch *****

def process_batch(batch: dict, device: str, signal_mode: str, normalize_mode: str, patch_size: int, out_h: int, out_w: int, freqs: torch.Tensor, generator: torch.Generator | None = None, augment_fft: bool = True, augment_mask: bool = True):
    fft_processed = process_vibration(batch["fft"].to(device), freqs, signal_mode, normalize_mode, patch_size, augment=augment_fft, generator=generator)
    mask_processed = process_image(batch["mask_true"].to(device), out_h, out_w, augment=augment_mask, generator=generator)
    return dict(fft=fft_processed, mask_true=mask_processed, info=batch["info"])

def collate_and_process(samples: list[dict], process_batch_fn) -> dict:
    # default-collate then augment here in the DataLoader worker process, so it overlaps with the main process
    fft = torch.stack([s["fft"] for s in samples])
    mask_true = torch.stack([s["mask_true"] for s in samples])
    info = {k: [s["info"][k] for s in samples] for k in samples[0]["info"]}
    return process_batch_fn(dict(fft=fft, mask_true=mask_true, info=info))

#***** 4 dataset *****

class VibrationDataset(StreamingDataset):
    def __init__(self, local: str | Path, shuffle: bool = False,
                 out_h: int = 20, out_w: int = 40, signal_mode: str = "magnitude", normalize_mode: str = "std",
                 patch_size: int = 256, seed: int = 42, device: str = "cpu", **kwargs):
        super().__init__(local=str(local), shuffle=shuffle, batch_size=kwargs.pop("batch_size", None), **kwargs)
        self.out_h, self.out_w = out_h, out_w
        self.signal_mode, self.normalize_mode, self.patch_size = signal_mode, normalize_mode, patch_size
        self.device = device
        self.freqs = torch.from_numpy(np.load(Path(local) / "freqs.npy")).to(device)
        self.generator = torch.Generator(device=device if device == "cpu" else "cpu")  # cuda generators can't cross process fork boundaries in workers; draw on cpu, move samples after
        self.generator.manual_seed(seed)

    def __getitem__(self, idx):
        s = super().__getitem__(idx)
        X, y = s.pop("X"), s.pop("y")  # X: (L,F,C,2) real/imag fft, y: (H,W) raw mask
        info = dict(sample_id=s["sample_id"], output_id=s["output_id"], n_objects=s["n_objects"], speaker=s["speaker"], box=s["box"], is_empty_box=s["is_empty_box"], x_com=s["downsampled_com_x"], y_com=s["downsampled_com_y"])
        X_t, y_t = torch.from_numpy(X.copy()), torch.from_numpy(y.copy())
        return dict(fft=X_t[..., 0] + 1j * X_t[..., 1], mask_true=y_t, info=info)

#***** 5 split *****

def _matches(row: dict, speakers, n_objects, box) -> bool:
    if speakers is not None and row["speaker"] not in (speakers if isinstance(speakers, list) else [speakers]): return False
    if n_objects is not None and row["n_objects"] not in (n_objects if isinstance(n_objects, list) else [n_objects]): return False
    if box is not None and row["box"] not in (box if isinstance(box, list) else [box]): return False
    return True

EXP25_GROUPS = {
    "purple_cube":         {"train_range": (3, 29),   "eval_ranges": [(30, 59)]},
    "green_cube":          {"train_range": (110, 124), "eval_ranges": [(125, 127)]},
    "purple_green_cubes":  {"train_range": (60, 88),  "eval_ranges": [(89, 109)]},
}
EXP25_ALWAYS_TRAIN_RANGE = (0, 2)  # empty-box

def exp25_split(mds_path: str | Path, test_size: float = 0.20, unseen_pos_frac: float = 0.15,
                 unseen_pos_speaker_frac: float = 0.05, seed: int = 42,
                 speakers=None, n_objects=None, box=None, n_samples: int | None = None,
                 verbose: int = 1) -> dict[str, list[int]]:
    """Return {"train": [...], "eval/<name>": [...], "eval/<name>_speaker": [...], ...}: sample
    indices (into the mds_path metadata.jsonl / dataset) for each split, by object-type.

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
    lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
    index = [json.loads(line) for line in lines if line]

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

    return {"train": train_idx, **evals}

#***** 5 build dataloaders *****

SPLIT_METHODS = {"exp25": exp25_split}

def num_samples(batch): return batch["mask_true"].shape[0]

def loader(dataset, idxs, bs, num_workers, generator, process_batch_fn, shuffle=False, drop_last=False):
    collate_fn = partial(collate_and_process, process_batch_fn=process_batch_fn)
    dl = DataLoader(Subset(dataset, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, drop_last=drop_last, collate_fn=collate_fn)
    return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

def build_dataset(data_dir: str | Path, split: str = "exp25", batch_size: int = 64, eval_batch_size: int = 64,
                   num_workers: int = 8, out_h: int = 20, out_w: int = 40, signal_mode: str = "magnitude",
                   normalize_mode: str = "std", patch_size: int = 256, seed: int = 42,
                   force_mds: bool = False, augment_fft: bool = True, augment_mask: bool = True,
                   verbose: int = 1, **split_kwargs):

    if split not in SPLIT_METHODS: raise ValueError(f"Unknown split {split!r}, expected one of {sorted(SPLIT_METHODS)}")
    data_dir = Path(data_dir)
    if not data_dir.exists(): raise FileNotFoundError(f"{data_dir=} does not exist")
    mds_dir = data_dir / "mds"
    hash_path = mds_dir / "samples_hash.txt"

    samples = collect_samples(data_dir / "samples", verbose)
    stale = not hash_path.exists() or hash_path.read_text() != hash_samples(samples)
    if force_mds or stale:
        if hash_path.exists():
            if not force_mds: raise ValueError(f"{mds_dir=} already exists and {force_mds=}")
            shutil.rmtree(mds_dir)
            if verbose: print(f"Overwriting {mds_dir=}")
        convert_to_mds(mds_dir, samples, verbose=verbose)
    elif verbose:
        print(f"Reusing existing MDS at {mds_dir}")

    splits = SPLIT_METHODS[split](mds_dir, seed=seed, verbose=verbose, **split_kwargs)

    generator = torch.Generator().manual_seed(seed)
    # device=cpu b/c augmentation runs in DataLoader workers via collate_fn, which are forked subprocesses that can't touch CUDA
    dataset = VibrationDataset(local=mds_dir, out_h=out_h, out_w=out_w, signal_mode=signal_mode, normalize_mode=normalize_mode, patch_size=patch_size, seed=seed, device="cpu")
    process_kwargs = dict(device="cpu", signal_mode=signal_mode, normalize_mode=normalize_mode, patch_size=patch_size, out_h=out_h, out_w=out_w, freqs=dataset.freqs, generator=dataset.generator)
    train_transform = partial(process_batch, **process_kwargs, augment_fft=augment_fft, augment_mask=augment_mask)
    eval_transform = partial(process_batch, **process_kwargs, augment_fft=False, augment_mask=False)

    train_loader = loader(dataset, splits["train"], batch_size, num_workers, generator, train_transform, shuffle=True, drop_last=True)
    train_eval_loader = loader(dataset, splits["train"], eval_batch_size, num_workers, generator, eval_transform)
    eval_loaders = [Evaluator(label=label, dataloader=loader(dataset, idxs, eval_batch_size, num_workers, generator, eval_transform)) for label, idxs in splits.items() if label != "train"]
    return train_loader, eval_loaders, train_eval_loader, dataset
