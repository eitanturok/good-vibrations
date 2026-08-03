import hashlib, json, os, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from composer.core import Evaluator, DataSpec
from streaming import StreamingDataset, MDSWriter
from streaming.base.format.mds.encodings import _encodings

# patch streaming to support complex ndarray
_NDArray = _encodings["ndarray"]
_NDArray._int2value_dtype |= {100: "complex64", 101: "complex128"}
_NDArray._value_dtype2int |= {"complex64": 100, "complex128": 101}


def should_augment(p: float) -> bool:
    # per-sample coin flip, so each sample is independently augmented with probability p
    if p <= 0: return False
    if p >= 1: return True
    return torch.rand((), device="cpu").item() < p

#***** 0 collect samples *****

REQUIRED_FILES = ["image/03_smask.npy", "vibration/04_fft.npz", "metadata.jsonl"]

def precomputed_fft_name(signal_mode: str, normalize_mode: str, patch_size: int, subtract_speaker_mean: bool) -> str:
    # the speaker mean is baked into the precomputed array, so it has to be part of the filename
    suffix = "_spkmean" if subtract_speaker_mean else ""
    return f"vibration/05_precomputed_fft_{signal_mode}_{normalize_mode}_{patch_size}{suffix}.npy"

def mds_columns(augment_fft: bool) -> dict[str, str]:
    x_dtype = "complex64" if augment_fft else "float32"  # raw fft is complex; precomputed signal is real
    return {"X": f"ndarray:{x_dtype}", "y": "ndarray:float32",
            "sample_id": "int", "position_id": "int",
            "n_objects": "int", "speaker": "int", "box": "str", "is_empty_box": "int", "object": "str",
            "downsampled_com_x": "float64", "downsampled_com_y": "float64"}

def collect_samples(base_sample_dir: Path, verbose: int = 1) -> list[tuple[Path, dict]]:
    samples, missing_by_file = [], {f: [] for f in REQUIRED_FILES}
    for sample_dir in tqdm(sorted(base_sample_dir.glob("*")), desc="collecting samples", disable=not verbose):
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

def hash_samples(samples: list[tuple[Path, dict]], out_h: int | None = None, out_w: int | None = None,
                  augment_fft: bool = True, signal_mode: str = "magnitude", normalize_mode: str = "std", patch_size: int = 256,
                  subtract_speaker_mean: bool = False) -> str:
    h = hashlib.sha256()
    if out_h is not None: h.update(f"{out_h}x{out_w}".encode())  # resolution is baked into y, so it must invalidate the cache too
    if not augment_fft: h.update(f"{augment_fft}{signal_mode}{normalize_mode}{patch_size}".encode())  # baked into X when not augmenting, so it must invalidate the cache too
    if subtract_speaker_mean: h.update(b"spkmean")  # hashed in both paths, unlike the above
    for sample_dir, meta in sorted(samples, key=lambda s: s[0].name):
        h.update(sample_dir.name.encode())
        h.update(json.dumps(meta, sort_keys=True).encode())
    return h.hexdigest()

#***** 1 convert to mds *****

def convert_to_mds(mds_dir: Path, samples: list[tuple[Path, dict]], out_h: int, out_w: int, verbose: int = 1,
                    augment_fft: bool = True, signal_mode: str = "magnitude", normalize_mode: str = "std", patch_size: int = 256,
                    subtract_speaker_mean: bool = False) -> Path:

    def load_X(sample_dir: Path) -> np.ndarray:
        if not augment_fft:
            return np.load(sample_dir / precomputed_fft_name(signal_mode, normalize_mode, patch_size, subtract_speaker_mean))
        X = np.load(sample_dir / "vibration/04_fft.npz")["fft"]  # (1, L, F, C) complex64
        return np.squeeze(X, axis=0) if X.ndim == 4 and X.shape[0] == 1 else X

    x_shape, y_shape = load_X(samples[0][0]).shape, (out_h, out_w)  # y: downsampled (out_h,out_w) mask

    samples = [(sample_dir.resolve(), meta) for sample_dir, meta in samples]
    index_rows = []
    mds_dir.parent.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(mds_dir.parent)

    try:
        with MDSWriter(out=mds_dir.name, columns=mds_columns(augment_fft), exist_ok=False, size_limit="200MB") as writer:
            for sample_dir, meta in tqdm(samples, desc="writing MDS", disable=not verbose):
                X = load_X(sample_dir)
                assert X.shape == x_shape, f"{sample_dir.name}: X.shape={X.shape} != {x_shape}"

                y = np.load(sample_dir / f"image/04_downsampled_smask_{out_h}h_{out_w}w.npy")  # precomputed by downsample_samples
                assert y.shape == y_shape, f"{sample_dir.name}: y.shape={y.shape} != {y_shape}"

                com = meta.get("downsampled_com", [-1.0, -1.0])
                sample = {
                    "X": X, "y": y,
                    "sample_id": int(meta.get("sample_id", -1)), "position_id": int(meta.get('position_id', meta.get("output_id", -1))),
                    "n_objects": int(meta.get("n_objects", -1)),
                    "speaker": int(meta.get("speaker", -1)),
                    "box": str(meta.get("box", "")),
                    "is_empty_box": bool(meta.get("is_empty_box", False)),
                    "object": str(meta.get("object", "")),
                    "downsampled_com_x": float(com[0]), "downsampled_com_y": float(com[1]),
                }
                writer.write(sample)
                index_rows.append(meta)
    finally:
        os.chdir(cwd)

    # freqs is identical across every sample (same fft grid) -- one sidecar, not duplicated per-row
    freqs = np.load(samples[0][0] / "vibration/04_fft.npz")["freqs"]
    np.save(mds_dir / "freqs.npy", freqs)

    # save metadata as a sidecar for loader-side filtering
    lines = "\n".join(json.dumps(r) for r in index_rows)
    (mds_dir / "metadata.jsonl").write_text(lines)
    if verbose: print(f"Wrote {len(samples)} samples to {mds_dir=}")
    return mds_dir

#***** 2 downsample image *****

def downsample_mask(mask: Image.Image, out_h: int, out_w: int) -> np.ndarray:
    # BOX resampling area-averages over the full H x W mask
    # Convert to float FIRST so BOX
    # would threshold the average back to binary and throw away partial coverage.
    out = np.array(mask.convert("F").resize((out_w, out_h), resample=Image.BOX), dtype=np.float32)
    return np.clip(out / 255.0, 0.0, 1.0)

def downsample_samples(samples: list[tuple[Path, dict]], out_h: int, out_w: int, verbose: int = 1) -> None:
    """Downsample every sample's full-resolution mask to (out_h, out_w). Only called when
    build_dataset's hashed mds_dir doesn't already exist, so no per-sample cache check here."""
    for sample_dir, _ in tqdm(samples, desc="downsampling masks", disable=not verbose):
        out_path = sample_dir / f"image/04_downsampled_smask_{out_h}h_{out_w}w"
        mask = downsample_mask(Image.open(sample_dir / "image/04_smask.png"), out_h, out_w)
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_path.with_suffix(".png"))
        np.save(out_path.with_suffix(".npy"), mask)

#***** 3 process image *****

def _gaussian_kernel1d_radius1(sigma: torch.Tensor) -> torch.Tensor:
    # Matches scipy.ndimage._gaussian_kernel1d(sigma, order=0, radius=1) exactly: truncated
    # Gaussian at x in {-1,0,1}, renormalized to sum to 1.
    x = torch.tensor([-1.0, 0.0, 1.0], dtype=sigma.dtype, device=sigma.device)
    w = torch.exp(-0.5 * (x / sigma) ** 2)
    return w / w.sum()

def gaussian_blur(mask: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # mask: (B,H,W). Separable 3x3 blur via two 1D convs, replicate-padded (matches scipy's mode='nearest', i.e. edge replication).
    k = _gaussian_kernel1d_radius1(sigma)
    x = mask.unsqueeze(1)  # (B,1,H,W)
    x = F.pad(x, (0, 0, 1, 1), mode='replicate')
    x = F.conv2d(x, k.view(1, 1, 3, 1))
    x = F.pad(x, (1, 1, 0, 0), mode='replicate')
    x = F.conv2d(x, k.view(1, 1, 1, 3))
    return x.squeeze(1)

def _sample_scale(lo: float, hi: float) -> torch.Tensor:
    # drawn on cpu, where the worker's global RNG lives
    return torch.empty(()).uniform_(lo, hi)

def noisy_blur(mask: torch.Tensor, blur_noise: tuple = (0.5, 1.0), object_noise: tuple = (0.05, 0.1), background_noise: tuple = (0.05, 0.1)) -> torch.Tensor:
    nonzero = mask != 0

    # blur the segmentation mask so we don't have such sharp edges
    out = gaussian_blur(mask, _sample_scale(*blur_noise).to(mask.device, mask.dtype))

    # add noise to the object
    noise = torch.normal(0.0, _sample_scale(*object_noise).item(), size=mask.shape, device="cpu", dtype=torch.float32).to(mask.device, mask.dtype)
    out[nonzero] = out[nonzero] + noise[nonzero]

    # add noise to the background
    noise = torch.normal(0.0, _sample_scale(*background_noise).item(), size=mask.shape, device="cpu", dtype=torch.float32).to(mask.device, mask.dtype)
    out[~nonzero] = out[~nonzero] + noise[~nonzero]

    return out.clamp(0, 1)

def process_image(mask: torch.Tensor, out_h: int, out_w: int, augment: float = 0.5) -> torch.Tensor:
    assert mask.shape[-2:] == (out_h, out_w), f"mask {tuple(mask.shape[-2:])} != expected ({out_h},{out_w})"
    # apply augmentation after downsampling so augmentations don't get washed out
    if should_augment(augment): mask = noisy_blur(mask)
    return mask

#***** 4 process vibration *****

def extract_signal(x: torch.Tensor, signal_mode: str) -> torch.Tensor:
    if signal_mode == "magnitude": return x.abs()
    if signal_mode == "complex": return torch.cat([x.real, x.imag], dim=-1)
    if signal_mode == "mag_phase": return torch.cat([x.abs(), x.angle()], dim=-1)
    raise ValueError(f"Unknown signal mode: {signal_mode}")

def subtract_speaker_mean(x: torch.Tensor, speaker_mean: torch.Tensor | None) -> torch.Tensor:
    # after the magnitude (per-sample trigger jitter randomizes phase, so complex means cancel toward
    # zero) and before the std (otherwise the std is inflated by the term we're about to remove)
    if speaker_mean is None: return x
    return x - speaker_mean

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
    """(B,L,F,C) -> (B,L,P,patch_size,C), zero-padding F up to a whole number of patches.

    Padding rather than truncating, because truncating silently dropped the top of the excitation
    band: 211 of 1235 bins (838-1000Hz) on the gastronorm capture at patch_size=256.

    No attention mask is needed for the padding. Rounding P up means the last patch is always
    *partly* real (211/256 here), never pure padding, and FreqEncoder.embed collapses each whole
    patch into one token before attention runs -- so the zeros are absorbed by a Linear that sees
    them at fixed input positions, and attention only ever sees real tokens.
    """
    if patch_size <= 0: return x
    B, L, F_, C = x.shape
    P = (F_ + patch_size - 1) // patch_size
    if pad := P * patch_size - F_: x = F.pad(x, (0, 0, 0, pad))  # (0,0) leaves C, (0,pad) grows F
    return x.reshape(B, L, P, patch_size, C)

def _hermit_poly(t: torch.Tensor) -> torch.Tensor:
    tt = t[None, :] ** torch.arange(4, device=t.device, dtype=t.dtype)[:, None]
    A = torch.tensor([[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]], dtype=t.dtype, device=t.device)
    return A @ tt

def random_frequency_gain(freqs: torch.Tensor, n_control: int = 5, gain_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    # Monotone cubic (Hermite spline) interpolant through n_control random control points
    control_freqs = torch.linspace(freqs.min().item(), freqs.max().item(), n_control, dtype=freqs.dtype, device=freqs.device)
    # draw on cpu (the worker's global RNG lives there), then move.
    control_gains = torch.normal(1.0, 1.0, size=(n_control,), device="cpu", dtype=torch.float32).to(freqs.device, freqs.dtype)
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

def augment_vibration(x: torch.Tensor, freqs: torch.Tensor, n_control: int = 5, gain_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    gain = random_frequency_gain(freqs, n_control, gain_range)
    gain = gain[None, None, :, None]
    return x * gain

def process_vibration(fft: torch.Tensor, freqs: torch.Tensor, signal_mode: str, normalize_mode: str, patch_size: int, augment: float = 0.5, gain_kwargs: dict | None = None, speaker_mean: torch.Tensor | None = None) -> torch.Tensor:
    if should_augment(augment): fft = augment_vibration(fft, freqs, **(gain_kwargs or {}))
    x = extract_signal(fft, signal_mode).float()
    x = subtract_speaker_mean(x, speaker_mean)
    x = normalize_fft(x, normalize_mode)
    return tokenize(x, patch_size)

SPEAKER_MEANS_FILE = "speaker_means.npz"

def compute_speaker_means(samples: list[tuple[Path, dict]], keep_idxs: list[int] | None = None, verbose: int = 1) -> dict[int, torch.Tensor]:
    """{speaker: mean magnitude spectrum (1,L,F,C)}, averaged over that speaker's samples.
    keep_idxs restricts the average to a subset (pass the train split to avoid eval leakage)."""
    if keep_idxs is not None:
        keep = set(keep_idxs)
        samples = [s for i, s in enumerate(samples) if i in keep]

    sums, counts = {}, {}
    for sample_dir, meta in tqdm(samples, desc="computing speaker means", disable=not verbose):
        speaker = int(meta.get("speaker", -1))
        X = np.load(sample_dir / "vibration/03_fft.npz")["fft"]  # (1, L, F, C) complex64
        X = np.squeeze(X, axis=0) if X.ndim == 4 and X.shape[0] == 1 else X
        mag = np.abs(X).astype(np.float64)  # float32 sums drift over many samples
        sums[speaker] = sums.get(speaker, 0) + mag
        counts[speaker] = counts.get(speaker, 0) + 1

    if verbose: print("speaker means: " + ", ".join(f"speaker {s}={counts[s]} samples" for s in sorted(counts)))
    return {s: torch.from_numpy((sums[s] / counts[s]).astype(np.float32)).unsqueeze(0) for s in sums}

def save_speaker_means(means: dict[int, torch.Tensor], path: Path) -> None:
    np.savez(path, **{str(s): m.squeeze(0).numpy() for s, m in means.items()})

def load_speaker_means(path: Path) -> dict[int, torch.Tensor]:
    d = np.load(path)
    return {int(s): torch.from_numpy(d[s]).unsqueeze(0) for s in d.files}

def precompute_vibration_samples(samples: list[tuple[Path, dict]], signal_mode: str, normalize_mode: str, patch_size: int, verbose: int = 1, speaker_means: dict[int, torch.Tensor] | None = None) -> None:
    freqs = torch.from_numpy(np.load(samples[0][0] / "vibration/03_fft.npz")["freqs"])
    for sample_dir, meta in tqdm(samples, desc="precomputing fft", disable=not verbose):
        X = np.load(sample_dir / "vibration/03_fft.npz")["fft"]  # (1, L, F, C) complex64
        X = np.squeeze(X, axis=0) if X.ndim == 4 and X.shape[0] == 1 else X
        X = torch.from_numpy(X).unsqueeze(0)
        speaker_mean = speaker_means[int(meta.get("speaker", -1))] if speaker_means is not None else None
        X = process_vibration(X, freqs, signal_mode, normalize_mode, patch_size, augment=0.0, speaker_mean=speaker_mean).squeeze(0).numpy()
        np.save(sample_dir / precomputed_fft_name(signal_mode, normalize_mode, patch_size, speaker_means is not None), X)

#***** 5 define dataset *****

class VibrationDataset(StreamingDataset):
    def __init__(self, local: str | Path, process_kwargs: dict, shuffle: bool = False, seed: int = 42, **kwargs):
        super().__init__(local=str(local), shuffle=shuffle, batch_size=kwargs.pop("batch_size", None), **kwargs)
        self.pk = dict(process_kwargs, freqs=torch.from_numpy(np.load(Path(local) / "freqs.npy")))
        # only needed on the raw-fft path; the precomputed path already had the mean subtracted offline
        means_path = Path(local) / SPEAKER_MEANS_FILE
        self.speaker_means = load_speaker_means(means_path) if means_path.exists() and not self.pk["mds_precomputed_fft"] else None

    def __getitem__(self, idx):
        s = super().__getitem__(idx)
        X = torch.from_numpy(s.pop("X").copy()).unsqueeze(0).to(self.pk["device"])
        y = torch.from_numpy(s.pop("y").copy()).unsqueeze(0).to(self.pk["device"])
        info = dict(sample_id=s["sample_id"], position_id=s["position_id"], n_objects=s["n_objects"], speaker=s["speaker"], box=s["box"], is_empty_box=s["is_empty_box"], x_com=s["downsampled_com_x"], y_com=s["downsampled_com_y"])
        speaker_mean = self.speaker_means[int(s["speaker"])].to(self.pk["device"]) if self.speaker_means is not None else None
        fft = X if self.pk["mds_precomputed_fft"] else process_vibration(X, self.pk["freqs"], self.pk["signal_mode"], self.pk["normalize_mode"], self.pk["patch_size"], augment=self.pk["augment_fft"], speaker_mean=speaker_mean)
        mask_true = process_image(y, self.pk["out_h"], self.pk["out_w"], augment=self.pk["augment_mask"])
        return dict(fft=fft.squeeze(0), mask_true=mask_true.squeeze(0), info=info)

#***** 6 define split *****

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
                 verbose: int = 1, index: list[dict] | None = None) -> dict[str, list[int]]:
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
    if index is None:
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


def split_by_position(index: list[dict], idxs: list[int], percent: float = 0.2, seed: int = 42) -> tuple[list[int], list[int], list[int]]:
    """Split `idxs` (indices into `index`) three ways by position_id. `percent` of the positions are
    held out entirely; the remaining 1 - percent are seen in train.

    - unseen_speaker: at each seen position, every speaker independently lands here with probability
      1/n_speakers (so ~1 of 8 on average). The position is in train, this speaker at it is not.
    - unseen_position: every sample at a held-out position -- never seen from any speaker.
    - train: the remaining speakers at each seen position.
    """
    rng = np.random.default_rng(seed)

    by_position = {} # position_id -> sample_id
    for i in idxs: by_position.setdefault(index[i]["position_id"], []).append(i)

    held_out = set(rng.permutation(sorted(by_position))[:round(percent * len(by_position))].tolist())

    unseen_speaker, unseen_position, train = [], [], []
    for position_id, group in by_position.items():
        if position_id in held_out:
            unseen_position += group
            continue
        to_eval = rng.random(len(group)) < 1 / len(group)
        if to_eval.all(): to_eval[rng.integers(len(group))] = False  # keep the position seen in train
        for i, is_eval in zip(group, to_eval): (unseen_speaker if is_eval else train).append(i)

    return sorted(unseen_speaker), sorted(unseen_position), sorted(train)


def gastronorm(mds_path, test_size=0.2, seed=42, speakers=None, n_objects=None, box=None, n_samples: int | None = None, verbose: int = 1, index: list[dict] | None = None):

    if index is None:
        lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
        index = [json.loads(line) for line in lines if line]

    keep = [i for i, row in enumerate(index) if _matches(row, speakers, n_objects, box)]
    if n_samples is not None: keep = keep[:n_samples]

    lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
    index = [json.loads(line) for line in lines if line]

    # One cube: 80% train, 20% eval.
    # The eval set is further split into unseen speaker and unseen position.
    one_cube_train, one_cube_eval = train_test_split([i for i, row in enumerate(index) if row['layout'] == 'purple-cube'], test_size=0.2, random_state=seed, shuffle=True)
    one_cube_unseen_speaker, one_cube_unseen_position, one_cube_train_2 = split_by_position(index, one_cube_eval, percent=0.2, seed=seed)
    one_cube_train += one_cube_train_2

    # Two cubes: 80% train, 20% eval.
    two_cubes_unseen_speaker, two_cubes_unseen_position, two_cubes_train = split_by_position(index, [i for i, row in enumerate(index) if row['layout'] == 'purple--green-cube-grid4'], percent=0.2, seed=seed)

    # make train
    splits = {}
    splits['train'] = [i for i, row in enumerate(index) if row['layout'] in ['empty-box', 'purple--green-cube-grid1', 'purple--green-cube-grid2', 'purple--green-cube-grid3', 'x-shift', 'y-shift']]
    splits['train'] += one_cube_train + two_cubes_train

    # eval
    splits |= {'eval/1-cube': one_cube_unseen_position, 'eval/1-cube-speaker': one_cube_unseen_speaker,
               'eval/2-cubes': two_cubes_unseen_position, 'eval/2-cubes-speaker': two_cubes_unseen_speaker}

    # ood eval
    splits['eval/3-cubes'] = [i for i, row in enumerate(index) if row['layout'] == 'purple--green-red-cube']
    splits['eval/red-cube'] = [i for i, row in enumerate(index) if row['layout'] == 'red-cube']

    if verbose:
        for label, idxs in splits.items(): print(f"{label}: {len(idxs)} samples")
    return splits

def gastronorm_one_cube(mds_path, test_size=0.2, seed=42, speakers=None, n_objects=None, box=None, n_samples: int | None = None, verbose: int = 1, index: list[dict] | None = None):

    if index is None:
        lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
        index = [json.loads(line) for line in lines if line]

    keep = [i for i, row in enumerate(index) if _matches(row, speakers, n_objects, box)]
    if n_samples is not None: keep = keep[:n_samples]

    one_cube_train, one_cube_eval = train_test_split([i for i, row in enumerate(index) if row['layout'] in ['purple-cube', 'empty-box']], test_size=0.2, random_state=seed, shuffle=True)
    return {'train': one_cube_train, 'eval/1-cube': one_cube_eval}

#***** 8 build dataloaders *****

SPLIT_METHODS = {"exp25": exp25_split, "gastronorm": gastronorm, "gastronorm_one_cube": gastronorm_one_cube}

def num_samples(batch): return batch["mask_true"].shape[0]

def loader(dataset, idxs, bs, num_workers, generator, shuffle=False, drop_last=False):
    dl = DataLoader(Subset(dataset, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True,
                    persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, drop_last=drop_last)
    return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

def build_dataset(data_dir: str | Path, split: str = "exp25", batch_size: int = 64, eval_batch_size: int = 64,
                   num_workers: int = 8, out_h: int = 20, out_w: int = 40, signal_mode: str = "magnitude",
                   normalize_mode: str = "std", patch_size: int = 256, seed: int = 42,
                   force_rebuild_data: bool = False, augment_fft: float = 0.5, augment_mask: float = 0.5,
                   subtract_speaker_mean: bool = False, verbose: int = 1, **split_kwargs):

    if split not in SPLIT_METHODS: raise ValueError(f"Unknown split {split!r}, expected one of {sorted(SPLIT_METHODS)}")
    if not 0 <= augment_fft <= 1: raise ValueError(f"{augment_fft=} must be a probability in [0, 1]")
    if not 0 <= augment_mask <= 1: raise ValueError(f"{augment_mask=} must be a probability in [0, 1]")
    if subtract_speaker_mean and signal_mode != "magnitude":
        raise ValueError(f"{subtract_speaker_mean=} requires signal_mode='magnitude', got {signal_mode!r}")
    data_dir = Path(data_dir)
    if not data_dir.exists(): raise FileNotFoundError(f"{data_dir=} does not exist")

    # the fft gain augmentation needs the raw complex fft, so any nonzero probability means we store it raw
    raw_fft = augment_fft > 0
    samples = collect_samples(data_dir / "samples", verbose)
    mds_dir = data_dir / "mds" / hash_samples(samples, out_h, out_w, raw_fft, signal_mode, normalize_mode, patch_size, subtract_speaker_mean)[:16]
    done = mds_dir / "metadata.jsonl"  # last file convert_to_mds writes -- its presence means the build completed

    if force_rebuild_data and mds_dir.exists():
        shutil.rmtree(mds_dir)
        if verbose: print(f"Overwriting {mds_dir=}")

    if not done.exists():
        if mds_dir.exists(): shutil.rmtree(mds_dir)  # clear out a partial/crashed build

        speaker_means = None
        if subtract_speaker_mean:
            # train-only, so eval spectra don't leak in.
            speaker_means = compute_speaker_means(samples, keep_idxs=SPLIT_METHODS[split](mds_dir, seed=seed, verbose=0, index=[m for _, m in samples], **split_kwargs)["train"], verbose=verbose)

        # downsample image, precompute fft, and convert to mds
        downsample_samples(samples, out_h, out_w, verbose=verbose)
        if not raw_fft: precompute_vibration_samples(samples, signal_mode, normalize_mode, patch_size, verbose=verbose, speaker_means=speaker_means)
        convert_to_mds(mds_dir, samples, out_h, out_w, verbose=verbose, augment_fft=raw_fft, signal_mode=signal_mode, normalize_mode=normalize_mode, patch_size=patch_size, subtract_speaker_mean=subtract_speaker_mean)
        if speaker_means is not None: save_speaker_means(speaker_means, mds_dir / SPEAKER_MEANS_FILE)
    elif verbose:
        print(f"Reusing existing MDS at {mds_dir}")

    splits = SPLIT_METHODS[split](mds_dir, seed=seed, verbose=verbose, **split_kwargs)

    generator = torch.Generator().manual_seed(seed)
    process_kwargs = dict(device="cpu", signal_mode=signal_mode, normalize_mode=normalize_mode, patch_size=patch_size, out_h=out_h, out_w=out_w, mds_precomputed_fft=not raw_fft)
    train_dataset = VibrationDataset(local=mds_dir, seed=seed, process_kwargs=dict(process_kwargs, augment_fft=augment_fft, augment_mask=augment_mask))
    eval_dataset = VibrationDataset(local=mds_dir, seed=seed, process_kwargs=dict(process_kwargs, augment_fft=0.0, augment_mask=0.0))

    train_loader = loader(train_dataset, splits["train"], batch_size, num_workers, generator, shuffle=True, drop_last=True)
    train_eval_loader = loader(eval_dataset, splits["train"], eval_batch_size, num_workers, generator)
    eval_loaders = [Evaluator(label=label, dataloader=loader(eval_dataset, idxs, eval_batch_size, num_workers, generator)) for label, idxs in splits.items() if label != "train"]

    return train_loader, eval_loaders, train_eval_loader
