import hashlib, json, os, random, shutil
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


# captures disagree on the spelling: 04_ffts.npz is what the pipeline writes now, 04_fft.npz is
# what older captures have. Same contents, so resolve per sample rather than forcing one name.
FFT_FILES = ("vibration/04_ffts.npz", "vibration/04_fft.npz")
REQUIRED_FILES = ["image/03_smask.npy", "metadata.jsonl"]

def fft_path(sample_dir: Path) -> Path:
    for name in FFT_FILES:
        if (sample_dir / name).exists(): return sample_dir / name
    raise FileNotFoundError(f"{sample_dir}: none of {FFT_FILES}")
LID_LAYOUT = 'lid-purple-cube'
EMPTY_BOX_LAYOUT = "empty-box"

def should_augment(p: float) -> bool:
    # per-sample coin flip, so each sample is independently augmented with probability p
    if p <= 0: return False
    if p >= 1: return True
    return torch.rand((), device="cpu").item() < p

#***** 0 collect samples *****

def laser_indices(laser_cols, n_laser_rows: int, n_laser_cols: int) -> np.ndarray | None:
    """Column ids -> flat row-major indices into the laser axis L. None means every laser.

    Selection is always whole columns across every row, so the kept set stays a rectangle
    (n_laser_rows, len(laser_cols)) and the model's grid is still well defined. Validated here,
    once, so a bad id fails before the ~10 min MDS build rather than inside a worker.
    """
    if laser_cols is None: return None
    cols = sorted(laser_cols)
    if len(set(cols)) != len(cols): raise ValueError(f"duplicate laser columns in {laser_cols}")
    if not cols: raise ValueError("laser_cols is empty; pass None for all lasers")
    if bad := [c for c in cols if not 0 <= c < n_laser_cols]:
        raise ValueError(f"laser columns {bad} outside [0, {n_laser_cols}), the recorded grid's width")
    return np.array([r * n_laser_cols + c for r in range(n_laser_rows) for c in cols])

LASER_GRID_FILE = "laser_grid.json"

def infer_laser_grid(samples: list[tuple[Path, dict]], n_lasers: int) -> tuple[int, int]:
    """(rows, cols) of the recorded laser grid, read off the per-laser ROI boxes.

    From `rois` -- one [x,y,w,h] per laser, in the same row-major order as the fft's L axis -- and
    NOT from the metadata's own n_rows/n_cols, which are unreliable: the two-laser-faces capture
    records 10x10 while its rois and its fft both hold 80 lasers in 8 rows of 10.

    Every sample must agree, and rows*cols must equal the real L, so a capture whose geometry
    drifted mid-run fails here rather than reshaping into a plausible-looking wrong grid.
    """
    grids = {}
    for sample_dir, meta in samples:
        rois = meta.get("rois")
        if not rois:
            raise ValueError(f"{sample_dir.name}: no 'rois' in metadata, so the laser grid cannot be "
                             f"inferred")
        rows = len({y for _, y, *_ in rois})
        grids.setdefault((rows, len(rois) // rows), sample_dir.name)
    if len(grids) > 1:
        raise ValueError(f"samples disagree on the laser grid: " +
                         ", ".join(f"{g} (e.g. {name})" for g, name in sorted(grids.items())))
    (rows, cols), _ = grids.popitem()
    if rows * cols != n_lasers:
        raise ValueError(f"rois describe a {rows}x{cols} grid but the fft holds {n_lasers} lasers")
    return rows, cols

def laser_tag(laser_cols) -> str:
    """Filename/hash fragment for a column selection. Sorted, so [2,0,1] and [0,1,2] agree."""
    return "" if laser_cols is None else "_lasers" + "-".join(str(c) for c in sorted(laser_cols))

def load_fft(sample_dir: Path, laser_idx: np.ndarray | None = None) -> np.ndarray:
    """The one door to the raw fft: (L,F,C) complex64, already restricted to the kept lasers.

    Every consumer -- the MDS writer, the offline precompute, and all three reference/statistics
    passes -- comes through here, so the unselected lasers never enter the pipeline at all and
    nothing downstream has to know a selection happened.
    """
    X = np.load(fft_path(sample_dir))["fft"]  # (1, L, F, C) complex64
    X = np.squeeze(X, axis=0) if X.ndim == 4 and X.shape[0] == 1 else X
    return X if laser_idx is None else X[laser_idx]

def precomputed_fft_name(signal_mode: str, normalize_mode: str, patch_size: int, subtract_speaker_mean: bool, subtract_empty_box: bool = False, phase_arm: str | None = None, phase_weight: float = 1.0, laser_cols=None) -> str:
    # the speaker mean and empty-box reference are baked into the precomputed array, so they have to
    # be part of the filename. so does the laser selection: normalize_fft reduces over L, so the
    # same sample precomputed on a subset of lasers is a different array -- this file lives next to
    # the raw sample, outside the hashed mds dir, so nothing else would catch the collision.
    suffix = "_spkmean" if subtract_speaker_mean else ""
    suffix += "_emptybox" if subtract_empty_box else ""
    suffix += f"_{phase_arm}w{phase_weight:g}" if phase_arm else ""
    suffix += laser_tag(laser_cols)
    return f"vibration/05_precomputed_fft_{signal_mode}_{normalize_mode}_{patch_size}{suffix}.npy"

def mds_columns(augment_fft: bool) -> dict[str, str]:
    x_dtype = "complex64" if augment_fft else "float32"  # raw fft is complex; precomputed signal is real
    return {"X": f"ndarray:{x_dtype}", "y": "ndarray:float32",
            "sample_id": "int", "position_id": "int",
            "n_objects": "int", "speaker": "int", "box": "str", "is_empty_box": "int", "object": "str",
            "downsampled_com_x": "float64", "downsampled_com_y": "float64"}

def collect_samples(base_sample_dir: Path, verbose: int = 1) -> list[tuple[Path, dict]]:
    samples, missing_by_file = [], {f: [] for f in [*REQUIRED_FILES, FFT_FILES[0]]}
    for sample_dir in tqdm(sorted(base_sample_dir.glob("*")), desc="collecting samples", disable=not verbose):
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
        for f, ids in missing_by_file.items():
            if ids: print(f"missing {f!r}: {ids}")
    return samples

def hash_samples(samples: list[tuple[Path, dict]], out_h: int | None = None, out_w: int | None = None,
                  augment_fft: bool = True, signal_mode: str = "magnitude", normalize_mode: str = "std", patch_size: int = 64,
                  subtract_speaker_mean: bool = False, subtract_empty_box: bool = False,
                  mag_recipe: str | None = None, rgb: bool = False,
                  phase_arm: str | None = None, phase_weight: float = 1.0, laser_cols=None) -> str:
    h = hashlib.sha256()
    if rgb: h.update(b"rgb")  # different target entirely, so it needs its own cache
    if mag_recipe is not None: h.update(f"recipe{mag_recipe}".encode())  # changes the stored tensor
    if phase_arm is not None: h.update(f"phase{phase_arm}{phase_weight}".encode())  # adds channels to X
    if laser_cols is not None: h.update(laser_tag(laser_cols).encode())  # fewer lasers in X, and every statistic recomputed over them
    if out_h is not None: h.update(f"{out_h}x{out_w}".encode())  # resolution is baked into y, so it must invalidate the cache too
    if not augment_fft: h.update(f"{augment_fft}{signal_mode}{normalize_mode}{patch_size}".encode())  # baked into X when not augmenting, so it must invalidate the cache too
    # the sidecars below are written on both paths and depend on signal_mode, so they're hashed unconditionally
    if subtract_speaker_mean: h.update(f"spkmean{signal_mode}".encode())
    if subtract_empty_box: h.update(f"emptybox{signal_mode}".encode())
    if normalize_mode.split('+')[0] in DATASET_STATS_MODES: h.update(f"stats{normalize_mode}{signal_mode}".encode())
    for sample_dir, meta in sorted(samples, key=lambda s: s[0].name):
        h.update(sample_dir.name.encode())
        h.update(json.dumps(meta, sort_keys=True).encode())
    return h.hexdigest()

#***** 1 convert to mds *****

def convert_to_mds(mds_dir: Path, samples: list[tuple[Path, dict]], out_h: int, out_w: int, verbose: int = 1,
                    augment_fft: bool = True, signal_mode: str = "magnitude", normalize_mode: str = "std", patch_size: int = 64,
                    subtract_speaker_mean: bool = False, subtract_empty_box: bool = False, rgb: bool = False,
                    phase_arm: str | None = None, phase_weight: float = 1.0,
                    laser_idx: np.ndarray | None = None, laser_cols=None) -> Path:

    def load_X(sample_dir: Path) -> np.ndarray:
        # the precomputed array was already written from the selected lasers, so it is sliced
        # once (here or there), never twice
        if not augment_fft:
            return np.load(sample_dir / precomputed_fft_name(signal_mode, normalize_mode, patch_size, subtract_speaker_mean, subtract_empty_box, phase_arm, phase_weight, laser_cols))
        return load_fft(sample_dir, laser_idx)

    x_shape = load_X(samples[0][0]).shape
    y_shape = (out_h, out_w, 3) if rgb else (out_h, out_w)  # y: the downsampled mask, or the downsampled rgb photo

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

                y = np.load(sample_dir / f"image/{target_name(rgb, out_h, out_w)}.npy")  # precomputed by downsample_samples
                assert y.shape == y_shape, f"{sample_dir.name}: y.shape={y.shape} != {y_shape}"

                com = meta.get("downsampled_com", [-1.0, -1.0])
                sample = {
                    "X": X, "y": y,
                    "sample_id": int(meta.get("sample_id", -1)), "position_id": int(meta.get('position_id', meta.get("position_id", -1))),
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
    freqs = np.load(fft_path(samples[0][0]))["freqs"]
    np.save(mds_dir / "freqs.npy", freqs)

    # save metadata as a sidecar for loader-side filtering
    lines = "\n".join(json.dumps(r) for r in index_rows)
    (mds_dir / "metadata.jsonl").write_text(lines)
    if verbose: print(f"Wrote {len(samples)} samples to {mds_dir=}")
    return mds_dir

#***** 2 downsample image *****

# the two prediction targets: the (h,w) segmentation mask, or the (h,w,3) overhead photo
def source_name(rgb: bool) -> str: return "02_cropped_overhead.png" if rgb else "03_smask.png"
def target_name(rgb: bool, out_h: int, out_w: int) -> str: return f"04_downsampled_{'overhead' if rgb else 'smask'}_{out_h}h_{out_w}w"

def downsample_mask(mask: Image.Image, out_h: int, out_w: int, rgb: bool = False) -> np.ndarray:
    # BOX resampling area-averages over the full H x W mask
    # Convert to float FIRST so BOX
    # would threshold the average back to binary and throw away partial coverage.
    out = np.array(mask.convert("RGB" if rgb else "F").resize((out_w, out_h), resample=Image.BOX), dtype=np.float32)
    return np.clip(out / 255.0, 0.0, 1.0)

def upsample_mask(mask: np.ndarray, box_w: int, box_h: int, smooth: bool = False) -> np.ndarray:
    """Inverse of downsample_mask: an (h,w) or (h,w,3) grid in [0,1] -> the box's own
    (box_h, box_w) pixels, in [0,1].

    Anisotropic on purpose. downsample_mask squashes every box into the SAME (out_h,out_w)
    grid, so the two scale factors differ per box; applying them both here is what undoes the
    squash and lands each cell back on the pixels it averaged.

    This is HONEST, not lossless -- BOX destroyed information on the way down, and the block
    boundaries sit at fractional pixels (1337/32 = 41.78), so BOX(upsample_mask(g)) returns g
    only to ~1e-2 at those edges. What NEAREST does guarantee is that it invents nothing: the
    result is piecewise constant with at most h*w distinct values, so it never shows a gradient
    the lasers did not measure. smooth=True (BILINEAR) is cosmetic only. Never BICUBIC/LANCZOS
    on probabilities -- they overshoot [0,1] and ring around blob edges.
    """
    resample = Image.BILINEAR if smooth else Image.NEAREST
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim not in (2, 3): raise ValueError(f"expected (h,w) or (h,w,c), got {arr.shape}")
    # resize in float ('F' mode, one channel at a time) rather than round-tripping through
    # uint8: quantizing to 256 levels here would be a second, avoidable loss on top of BOX.
    planes = [arr] if arr.ndim == 2 else [arr[..., c] for c in range(arr.shape[2])]
    out = [np.array(Image.fromarray(pl, mode="F").resize((box_w, box_h), resample=resample), dtype=np.float32)
           for pl in planes]
    return np.clip(out[0] if arr.ndim == 2 else np.stack(out, axis=-1), 0.0, 1.0)

def downsample_samples(samples: list[tuple[Path, dict]], out_h: int, out_w: int, verbose: int = 1, rgb: bool = False) -> None:
    """Downsample every sample's full-resolution target to (out_h, out_w). Only called when
    build_dataset's hashed mds_dir doesn't already exist, so no per-sample cache check here."""
    for sample_dir, _ in tqdm(samples, desc="downsampling targets", disable=not verbose):
        out_path = sample_dir / f"image/{target_name(rgb, out_h, out_w)}"
        mask = downsample_mask(Image.open(sample_dir / f"image/{source_name(rgb)}"), out_h, out_w, rgb)
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
    assert mask.shape[1:3] == (out_h, out_w), f"mask {tuple(mask.shape[1:3])} != expected ({out_h},{out_w})"
    # apply augmentation after downsampling so augmentations don't get washed out
    if should_augment(augment): mask = noisy_blur(mask)
    return mask

#***** 4 process vibration *****

LOG_EPS = 1e-3  # |fft| bottoms out near 1e-9; this floors the dead bins at -6.9 instead of -20 (notebook 65)

def extract_signal(x: torch.Tensor, signal_mode: str) -> torch.Tensor:
    if signal_mode == "magnitude": return x.abs()
    if signal_mode == "log_magnitude": return torch.log(x.abs() + LOG_EPS)  # before normalize_fft: normalizing first doesn't survive the log
    if signal_mode == "complex": return torch.cat([x.real, x.imag], dim=-1)
    if signal_mode == "mag_phase": return torch.cat([x.abs(), x.angle()], dim=-1)
    if signal_mode == "mag_trig_phase": return torch.cat([x.abs(), torch.sin(x.angle()), torch.cos(x.angle())], dim=-1)
    # if signal_mode == "mag_trig_phase"
    raise ValueError(f"Unknown signal mode: {signal_mode}")

def subtract_speaker_mean(x: torch.Tensor, speaker_mean: torch.Tensor | None) -> torch.Tensor:
    # after the magnitude (per-sample trigger jitter randomizes phase, so complex means cancel toward
    # zero) and before the std (otherwise the std is inflated by the term we're about to remove).
    # under log_magnitude this is a mean of logs, so subtracting it divides out the speaker's gain.
    if speaker_mean is None: return x
    return x - speaker_mean

def subtract_empty_box_ref(x: torch.Tensor, empty_box_ref: torch.Tensor | None) -> torch.Tensor:
    """Divide out the box's own transfer function. In log space, subtracting IS dividing --
    and it stays finite at anti-resonances, where a linear divide would explode."""
    if empty_box_ref is None: return x
    return x - empty_box_ref

DATASET_STATS_MODES = ("per_bin_z",)  # these normalize against train-split stats, not the sample itself

def normalize_fft(x: torch.Tensor, normalize_mode: str, stats: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    if normalize_mode is None: return x
    normalize_mode = normalize_mode.split('+')[0]  # trailing parts are token-level, see normalize_token
    x64 = x.double()
    if normalize_mode == 'std':
        std = x64.std(dim=(1, 2, 3), correction=1, keepdim=True).clamp_min(1e-8).float() # (B, L, F, C)
        return x / std
    if normalize_mode == 'z':
        mean = x64.mean(dim=(1, 2, 3), keepdim=True).float()
        std = x64.std(dim=(1, 2, 3), correction=1, keepdim=True).clamp_min(1e-8).float()
        return (x - mean) / std
    if normalize_mode == 'per_laser_z':
        # each laser standardized on its own, so per-laser sensitivity stops riding along
        mean = x64.mean(dim=(2, 3), keepdim=True).float()  # (B,L,1,1)
        std = x64.std(dim=(2, 3), correction=1, keepdim=True).clamp_min(1e-8).float()
        return (x - mean) / std
    if normalize_mode == 'per_bin_z':
        # whitens every (laser,freq,channel) bin against the train split, removing any fixed spectral shape
        if stats is None: raise ValueError("normalize_mode='per_bin_z' requires dataset stats; pass stats=")
        return (x - stats["mean"].to(x.device)) / stats["std"].to(x.device).clamp_min(1e-8)
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

def normalize_token(x: torch.Tensor, normalize_mode: str) -> torch.Tensor:
    """(B,L,P,PS,C) -> same shape. Token-level normalization, applied after tokenize().

    Both modes equalize loudness across frequency bands; pick by what space the signal is in.
    'token-mean' divides each (sample, patch) by its own mean -- only valid when the signal is
    positive. 'token-sub' subtracts it instead, which is the same operation for logs (where means
    cross zero and dividing would explode). Selected after a '+' in normalize_mode, e.g.
    'std+token-mean' -- normalize_fft handles the part before the '+', this handles the rest.

    The channel is pooled into the denominator on purpose: embed is
    nn.Linear(patch_size*n_channels, d_model) over a flattened patch (arch.py), so x and y live in
    one token and the model never sees them apart. A per-channel scale would be something it
    cannot represent.
    """
    if normalize_mode is None or '+' not in normalize_mode: return x
    mode = normalize_mode.split('+', 1)[1]
    mean = x.double().mean(dim=(1, 3, 4), keepdim=True)  # (B,1,P,1,1)
    if mode == 'token-mean': return x / mean.clamp_min(1e-8).float()
    if mode == 'token-sub': return x - mean.float()
    raise ValueError(f"Unknown token normalize mode: {mode!r} (from {normalize_mode!r})")

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

def process_vibration(fft: torch.Tensor, freqs: torch.Tensor, signal_mode: str, normalize_mode: str, patch_size: int, augment: float = 0.5, gain_kwargs: dict | None = None, speaker_mean: torch.Tensor | None = None, stats: dict[str, torch.Tensor] | None = None, empty_box_ref: torch.Tensor | None = None, mag_recipe: str | None = None, phase_arm: str | None = None, phase_weight: float = 1.0) -> torch.Tensor:
    """fft -> tokens.

    `mag_recipe` selects one of the 11 arms in normalizations.MAG_RECIPES and REPLACES
    the two subtract_* steps. It owns both the domain (|Z| vs log|Z|) and the operation
    (divide vs subtract), so `signal_mode` and both references must be in that same
    domain -- build_dataset derives all three from the recipe name.

    `phase_arm` selects one of normalizations.PHASE_ARMS and CONCATENATES real phase
    channels onto the magnitude block, scaled by `phase_weight`. It is orthogonal to
    `mag_recipe`: stage A picks what the magnitude block is, stage B what rides along.

    Order matters: the reference is removed BEFORE normalize_fft, because dividing by
    the std first would rescale the sample but not the reference, so they would no
    longer be commensurable. And std-normalizing last is what makes every arm
    scale-free, so the ablation compares distribution SHAPE rather than gain.
    """
    if should_augment(augment): fft = augment_vibration(fft, freqs, **(gain_kwargs or {}))
    x = extract_signal(fft, signal_mode).float()
    if mag_recipe is not None:
        from model.normalizations import apply_mag_recipe
        x = apply_mag_recipe(x, mag_recipe, empty_box=empty_box_ref, speaker_mean=speaker_mean)
    else:
        x = subtract_empty_box_ref(x, empty_box_ref)
        x = subtract_speaker_mean(x, speaker_mean)
    x = normalize_fft(x, normalize_mode, stats)
    if phase_arm is not None:
        # AFTER normalize_fft, and computed from `fft` rather than `x`: the phase block is
        # already bounded in [-1,1] by construction, so std-normalizing it would distort the
        # circular geometry that makes cos/sin the right encoding. Normalizing the magnitude
        # first is also what puts the two blocks on a common scale, so `phase_weight` sets a
        # meaningful mix instead of fighting the raw |Z| dynamic range.
        from model.normalizations import apply_phase_arm
        x = torch.cat([x, phase_weight * apply_phase_arm(fft, phase_arm).float()], dim=-1)
    return normalize_token(tokenize(x, patch_size), normalize_mode)

SPEAKER_MEANS_FILE = "speaker_means.npz"
DATASET_STATS_FILE = "dataset_stats.npz"
EMPTY_BOX_REF_FILE = "empty_box_ref.npz"

def load_signal(sample_dir: Path, signal_mode: str, laser_idx: np.ndarray | None = None) -> torch.Tensor:
    """(1,L,F,C) raw fft off disk -> extract_signal, in float64 so long sums don't drift."""
    return extract_signal(torch.from_numpy(load_fft(sample_dir, laser_idx)).unsqueeze(0), signal_mode).double()

def _keep(samples, keep_idxs):
    if keep_idxs is None: return samples
    keep = set(keep_idxs)
    return [s for i, s in enumerate(samples) if i in keep]

def compute_speaker_means(samples: list[tuple[Path, dict]], signal_mode: str = "magnitude", keep_idxs: list[int] | None = None, verbose: int = 1, empty_box_ref: dict[int, torch.Tensor] | None = None, laser_idx: np.ndarray | None = None) -> dict[int, torch.Tensor]:
    """{speaker: mean signal (1,L,F,C)}, averaged over that speaker's samples. The mean is taken on
    whatever extract_signal returns, so under log_magnitude it is a mean of logs

    `empty_box_ref` is DEPRECATED and defaults to None. It used to reference the signal before
    averaging, on the reasoning that both terms otherwise carry the speaker gain and it would be
    removed twice. That reasoning is wrong: subtraction makes the empty box cancel outright,
        (x - EB) - mean(x - EB) = x - mean(x),
    so referencing first silently turned the empty box into a no-op rather than avoiding a double
    subtraction. Compute both references on raw signal and subtract each once.

    keep_idxs restricts the average to a subset (pass the train split to avoid eval leakage)."""
    sums, counts = {}, {}
    for sample_dir, meta in tqdm(_keep(samples, keep_idxs), desc="computing speaker means", disable=not verbose):
        speaker = int(meta.get("speaker", -1))
        x = load_signal(sample_dir, signal_mode, laser_idx)
        if empty_box_ref is not None: x = x - empty_box_ref[speaker].double()
        sums[speaker] = sums.get(speaker, 0) + x
        counts[speaker] = counts.get(speaker, 0) + 1

    if verbose: print("speaker means: " + ", ".join(f"speaker {s}={counts[s]} samples" for s in sorted(counts)))
    return {s: (sums[s] / counts[s]).float() for s in sums}

def save_speaker_means(means: dict[int, torch.Tensor], path: Path) -> None:
    np.savez(path, **{str(s): m.squeeze(0).numpy() for s, m in means.items()})

def load_speaker_means(path: Path) -> dict[int, torch.Tensor]:
    d = np.load(path)
    return {int(s): torch.from_numpy(d[s]).unsqueeze(0) for s in d.files}

def compute_empty_box_ref(samples: list[tuple[Path, dict]], signal_mode: str = "log_magnitude", keep_idxs: list[int] | None = None, verbose: int = 1, laser_idx: np.ndarray | None = None) -> dict[int, torch.Tensor]:
    """{speaker: mean empty-box signal (1,L,F,C)} -- the box's own transfer function, per speaker.

    Y(f) = S(f) . H_box(f) . (object perturbation), so the speaker chain S and the box response
    H_box are both multiplicative nuisances shared by every sample from that speaker; one
    per-speaker reference removes both. Per speaker, not global, because S differs by speaker.
    """
    sums, counts = {}, {}
    for sample_dir, meta in tqdm(_keep(samples, keep_idxs), desc="computing empty-box ref", disable=not verbose):
        if meta.get("layout") != EMPTY_BOX_LAYOUT: continue
        speaker = int(meta.get("speaker", -1))
        sums[speaker] = sums.get(speaker, 0) + load_signal(sample_dir, signal_mode, laser_idx)
        counts[speaker] = counts.get(speaker, 0) + 1

    # fail here rather than KeyError-ing later inside a dataloader worker
    if not sums: raise ValueError(f"no {EMPTY_BOX_LAYOUT!r} samples found; cannot build the empty-box reference")
    if verbose: print("empty-box ref: " + ", ".join(f"speaker {s}={counts[s]} samples" for s in sorted(counts)))
    return {s: (sums[s] / counts[s]).float() for s in sums}

def save_empty_box_ref(ref: dict[int, torch.Tensor], path: Path) -> None:
    np.savez(path, **{str(s): m.squeeze(0).numpy() for s, m in ref.items()})

def load_empty_box_ref(path: Path) -> dict[int, torch.Tensor]:
    d = np.load(path)
    return {int(s): torch.from_numpy(d[s]).unsqueeze(0) for s in d.files}

def compute_dataset_stats(samples: list[tuple[Path, dict]], signal_mode: str = "magnitude", keep_idxs: list[int] | None = None, verbose: int = 1, laser_idx: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    """Per-(laser,freq,channel) mean and std over the train split, for normalize_mode='per_bin_z'."""
    samples = _keep(samples, keep_idxs)
    n, total, total_sq = 0, 0, 0
    for sample_dir, _ in tqdm(samples, desc="computing dataset stats", disable=not verbose):
        x = load_signal(sample_dir, signal_mode, laser_idx)
        total, total_sq, n = total + x, total_sq + x ** 2, n + 1

    mean = total / n
    var = (total_sq / n - mean ** 2).clamp_min(0)  # clamp: catastrophic cancellation can go slightly negative
    if verbose: print(f"dataset stats over {n} samples: mean {mean.mean():.4f}, std {var.sqrt().mean():.4f}")
    return {"mean": mean.float(), "std": var.sqrt().float()}

def save_dataset_stats(stats: dict[str, torch.Tensor], path: Path) -> None:
    np.savez(path, **{k: v.squeeze(0).numpy() for k, v in stats.items()})

def load_dataset_stats(path: Path) -> dict[str, torch.Tensor]:
    d = np.load(path)
    return {k: torch.from_numpy(d[k]).unsqueeze(0) for k in d.files}

def precompute_vibration_samples(samples: list[tuple[Path, dict]], signal_mode: str, normalize_mode: str, patch_size: int, verbose: int = 1, speaker_means: dict[int, torch.Tensor] | None = None, stats: dict[str, torch.Tensor] | None = None, empty_box_ref: dict[int, torch.Tensor] | None = None, mag_recipe: str | None = None, phase_arm: str | None = None, phase_weight: float = 1.0, laser_idx: np.ndarray | None = None, laser_cols=None) -> None:
    freqs = torch.from_numpy(np.load(fft_path(samples[0][0]))["freqs"])
    for sample_dir, meta in tqdm(samples, desc="precomputing fft", disable=not verbose):
        X = torch.from_numpy(load_fft(sample_dir, laser_idx)).unsqueeze(0)
        speaker = int(meta.get("speaker", -1))
        speaker_mean = speaker_means[speaker] if speaker_means is not None else None
        ref = empty_box_ref[speaker] if empty_box_ref is not None else None
        X = process_vibration(X, freqs, signal_mode, normalize_mode, patch_size, augment=0.0, speaker_mean=speaker_mean, stats=stats, empty_box_ref=ref, mag_recipe=mag_recipe, phase_arm=phase_arm, phase_weight=phase_weight).squeeze(0).numpy()
        np.save(sample_dir / precomputed_fft_name(signal_mode, normalize_mode, patch_size, speaker_means is not None, empty_box_ref is not None, phase_arm, phase_weight, laser_cols), X)

#***** 5 define dataset *****

class VibrationDataset(StreamingDataset):
    def __init__(self, local: str | Path, process_kwargs: dict, shuffle: bool = False, seed: int = 42, **kwargs):
        super().__init__(local=str(local), shuffle=shuffle, batch_size=kwargs.pop("batch_size", None), **kwargs)
        self.pk = dict(process_kwargs, freqs=torch.from_numpy(np.load(Path(local) / "freqs.npy")))
        # (rows, cols) of the lasers actually in X -- the model builds its grid from this
        self.grid_shape = tuple(json.loads((Path(local) / LASER_GRID_FILE).read_text()))
        # only needed on the raw-fft path; the precomputed path already applied both offline
        raw = not self.pk["mds_precomputed_fft"]
        means_path, stats_path = Path(local) / SPEAKER_MEANS_FILE, Path(local) / DATASET_STATS_FILE
        ref_path = Path(local) / EMPTY_BOX_REF_FILE
        self.speaker_means = load_speaker_means(means_path) if means_path.exists() and raw else None
        self.stats = load_dataset_stats(stats_path) if stats_path.exists() and raw else None
        self.empty_box_ref = load_empty_box_ref(ref_path) if ref_path.exists() and raw else None

    def __getitem__(self, idx):
        s = super().__getitem__(idx)
        X = torch.from_numpy(s.pop("X").copy()).unsqueeze(0).to(self.pk["device"])
        y = torch.from_numpy(s.pop("y").copy()).unsqueeze(0).to(self.pk["device"])
        n_classes = self.pk["n_classes"]
        assert 0 <= s["n_objects"] < n_classes, f'n_objects={s["n_objects"]} outside the {n_classes} count classes'
        n_objects = torch.tensor(s["n_objects"], dtype=torch.long, device=self.pk["device"])
        info = dict(sample_id=s["sample_id"], position_id=s["position_id"], n_objects=n_objects, speaker=s["speaker"], box=s["box"], is_empty_box=s["is_empty_box"], x_com=s["downsampled_com_x"], y_com=s["downsampled_com_y"])
        speaker_mean = self.speaker_means[int(s["speaker"])].to(self.pk["device"]) if self.speaker_means is not None else None
        ref = self.empty_box_ref[int(s["speaker"])].to(self.pk["device"]) if self.empty_box_ref is not None else None
        fft = X if self.pk["mds_precomputed_fft"] else process_vibration(X, self.pk["freqs"], self.pk["signal_mode"], self.pk["normalize_mode"], self.pk["patch_size"], augment=self.pk["augment_fft"], speaker_mean=speaker_mean, stats=self.stats, empty_box_ref=ref, mag_recipe=self.pk.get("mag_recipe"), phase_arm=self.pk.get("phase_arm"), phase_weight=self.pk.get("phase_weight", 1.0))
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

    Each object-type (purple_cube, green_cube, purple_green_cubes) has an position_id (== image_id)
    range that is always train (its "grid-1" layout, or the train-side layout for the mixed type)
    and one or more position_id ranges we may only draw eval samples from (its "grid-2" layout(s)).
    empty-box is always train, no eval carve-out.

    For each type:
    - combined pool = train-range samples + eval-range samples.
    - target_eval = test_size * len(combined pool).
    - If the eval-range pool is smaller than target_eval, the *whole* eval-range pool becomes eval
      candidates (we never dip into the train range to make up the difference) -> this type ends up
      with < test_size eval, and all of its train-range samples stay in train.
    - Otherwise, target_eval samples (whole position_ids) are drawn from the eval-range pool as eval
      candidates, and the leftover eval-range position_ids fall back into train.
    - The eval candidates are then split unseen_pos_frac / unseen_pos_speaker_frac (of the combined
      pool, not of the eval candidates) into:
        - eval/{type}: whole held-out position_ids -> positions never seen in train, from any speaker.
        - eval/{type}_speaker: individual samples from position_ids that DO appear in train -> this
          exact (position, speaker) pair is new, but the position was seen from other speakers.
      If the eval-range pool was capped smaller than target_eval, these two eval sets are scaled down
      proportionally to fit inside what's available.
    """
    if index is None:
        lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
        index = [json.loads(line) for line in lines if line]

    keep = [i for i, row in enumerate(index) if _matches(row, speakers, n_objects, box)]
    if n_samples is not None: keep = keep[:n_samples]

    def in_range(i, lo, hi): return lo <= int(index[i]["position_id"]) <= hi

    train_idx = [i for i in keep if in_range(i, *EXP25_ALWAYS_TRAIN_RANGE)]
    evals = {}
    for name, group in EXP25_GROUPS.items():
        train_range_idx = [i for i in keep if in_range(i, *group["train_range"])]
        eval_pool_idx = [i for i in keep if any(in_range(i, lo, hi) for lo, hi in group["eval_ranges"])]
        combined_n = len(train_range_idx) + len(eval_pool_idx)

        target_eval = test_size * combined_n
        eval_pool_position_ids = sorted({index[i]["position_id"] for i in eval_pool_idx})

        if len(eval_pool_idx) <= target_eval:
            eval_candidate_idx = eval_pool_idx
            train_idx += train_range_idx
            scale = len(eval_pool_idx) / target_eval if target_eval > 0 else 0.0
        else:
            n_position_ids = round(target_eval / len(eval_pool_idx) * len(eval_pool_position_ids))
            n_position_ids = min(max(n_position_ids, 0), len(eval_pool_position_ids))
            eval_candidate_position_ids, _ = train_test_split(
                eval_pool_position_ids, train_size=n_position_ids, random_state=seed, shuffle=True)
            eval_candidate_position_ids = set(eval_candidate_position_ids)
            eval_candidate_idx = [i for i in eval_pool_idx if index[i]["position_id"] in eval_candidate_position_ids]
            train_idx += train_range_idx + [i for i in eval_pool_idx if index[i]["position_id"] not in eval_candidate_position_ids]
            scale = 1.0

        pos_frac = min(unseen_pos_frac * scale, unseen_pos_frac + unseen_pos_speaker_frac)
        eval_candidate_position_ids = sorted({index[i]["position_id"] for i in eval_candidate_idx})
        n_unseen_pos = round(pos_frac / test_size * len(eval_candidate_position_ids)) if test_size > 0 else 0
        n_unseen_pos = min(max(n_unseen_pos, 0), len(eval_candidate_position_ids))
        unseen_pos_position_ids, speaker_pool_position_ids = train_test_split(
            eval_candidate_position_ids, train_size=n_unseen_pos, random_state=seed, shuffle=True) if 0 < n_unseen_pos < len(eval_candidate_position_ids) \
            else (eval_candidate_position_ids, []) if n_unseen_pos == len(eval_candidate_position_ids) \
            else ([], eval_candidate_position_ids)
        unseen_pos_position_ids = set(unseen_pos_position_ids)
        evals[f"eval/{name}"] = [i for i in eval_candidate_idx if index[i]["position_id"] in unseen_pos_position_ids]

        speaker_pool_idx = [i for i in eval_candidate_idx if index[i]["position_id"] in set(speaker_pool_position_ids)]
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

    # One cube
    one_cube_unseen_speaker, one_cube_unseen_position, one_cube_train = split_by_position(index, [i for i, row in enumerate(index) if row['layout'] == 'purple-cube'], percent=test_size, seed=seed)

    # Two cubes: eval comes from grid4 only, so grid1/2/3 stay wholly in train and every spot in the
    # raster is seen at least three times.
    two_cubes_unseen_speaker, two_cubes_unseen_position, two_cubes_train = split_by_position(index, [i for i, row in enumerate(index) if row['layout'] == 'purple--green-cube-grid4'], percent=test_size, seed=seed)

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

# The object-count experiment excludes red-cube: it is a distinct object, so keeping it would
# confound "how many objects" with "which object".
OBJECT_COUNT_EXCLUDED_LAYOUTS = (LID_LAYOUT, 'red-cube')

# Positions held out for eval, per object count. Every position carries 8 samples, so this is
# 40 eval samples per count.
OBJECT_COUNT_EVAL_POSITIONS = 5

# Train positions per run. The 1-object pool is the binding constraint (70 positions once red-cube
# and the lid layout are dropped, minus the 5 held out for eval), so every run gets this many so
# that no run is simply trained on more data than another.
OBJECT_COUNT_TRAIN_POSITIONS = 65


def _object_count_pools(mds_path, seed, speakers, n_objects, box, n_samples, index):
    """Shared setup for the object-count splits: the per-count position pools, and the eval positions
    held out of each.

    The holdout is drawn per object count from a fixed seed, independent of what any given run trains
    on, so all runs in the experiment evaluate on exactly the same samples. Splitting by position_id
    rather than by sample matters because every position is captured once per speaker -- splitting
    samples would leave the same position in both train and eval.

    The object count comes from each row's `n_objects`, not its layout name -- the two disagree on this
    dataset (x-shift/y-shift are 2-object layouts).
    """
    if index is None:
        lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
        index = [json.loads(line) for line in lines if line]

    keep = [i for i, row in enumerate(index)
            if _matches(row, speakers, n_objects, box) and row['layout'] not in OBJECT_COUNT_EXCLUDED_LAYOUTS]
    if n_samples is not None: keep = keep[:n_samples]

    pools, eval_positions = {}, {}
    for count in (1, 2):
        pool = [i for i in keep if index[i]["n_objects"] == count]
        if not pool: raise ValueError(f"no samples with n_objects={count}")
        position_ids = sorted({index[i]["position_id"] for i in pool})
        if len(position_ids) < OBJECT_COUNT_EVAL_POSITIONS + OBJECT_COUNT_TRAIN_POSITIONS:
            raise ValueError(f"n_objects={count} has only {len(position_ids)} positions, need "
                             f"{OBJECT_COUNT_EVAL_POSITIONS + OBJECT_COUNT_TRAIN_POSITIONS}")
        _, held = train_test_split(position_ids, test_size=OBJECT_COUNT_EVAL_POSITIONS, random_state=seed, shuffle=True)
        pools[count], eval_positions[count] = pool, set(held)

    # empty-box samples go wholly into train in every run: they carry no object to localise, so they
    # can only ever be train signal -- they are what teaches the model the box's own resonances.
    empty_box = [i for i in keep if index[i]["n_objects"] == 0]
    return index, pools, eval_positions, empty_box


def _object_count_split(mds_path, train_positions: dict[int, int], test_size=0.05, seed=42, speakers=None,
                        n_objects=None, box=None, n_samples: int | None = None, verbose: int = 1,
                        index: list[dict] | None = None):
    """One run of the object-count experiment. `train_positions` maps object count -> how many positions
    of that count to train on; counts absent from it are not trained on at all.

    Every run evaluates on the same `eval/1-obj` and `eval/2-obj` sets regardless of what it trained on,
    so the numbers line up across runs. `test_size` is accepted for interface compatibility and unused --
    the eval size is fixed by OBJECT_COUNT_EVAL_POSITIONS so that it cannot drift between runs.
    """
    index, pools, eval_positions, empty_box = _object_count_pools(
        mds_path, seed, speakers, n_objects, box, n_samples, index)

    splits = {"train": list(empty_box)}
    for count, n_positions in train_positions.items():
        # train positions are drawn from the same seeded shuffle for every run, so a run training on
        # fewer positions gets a subset of what a run training on more gets, not a different sample.
        available = sorted({index[i]["position_id"] for i in pools[count]} - eval_positions[count])
        rng = random.Random(seed + count)
        rng.shuffle(available)
        chosen = set(available[:n_positions])
        splits["train"] += [i for i in pools[count] if index[i]["position_id"] in chosen]
    splits["train"] = sorted(splits["train"])

    for count in (1, 2):
        splits[f"eval/{count}-obj"] = sorted(i for i in pools[count] if index[i]["position_id"] in eval_positions[count])

    if verbose:
        for label, idxs in splits.items(): print(f"{label}: {len(idxs)} samples")
    return splits


def gastronorm_train1_eval2(*args, **kwargs):
    """Train on 1-object scenes only."""
    return _object_count_split(*args, train_positions={1: OBJECT_COUNT_TRAIN_POSITIONS}, **kwargs)


def gastronorm_train2_eval1(*args, **kwargs):
    """Train on 2-object scenes only."""
    return _object_count_split(*args, train_positions={2: OBJECT_COUNT_TRAIN_POSITIONS}, **kwargs)


def gastronorm_train12_eval12(*args, **kwargs):
    """Budget-matched control: train on both object counts, same total positions as the single-count
    runs. Isolates how a fixed data budget is best spent -- all on one count, or split across both.
    """
    half = OBJECT_COUNT_TRAIN_POSITIONS // 2
    return _object_count_split(*args, train_positions={1: half, 2: OBJECT_COUNT_TRAIN_POSITIONS - half}, **kwargs)


def gastronorm_train12_eval12_full(*args, **kwargs):
    """Exposure-matched control: train on both object counts at the full per-count budget, so each count
    gets as many positions as it does in its own single-count run. Twice the data of the other three
    runs -- it answers whether adding the other count on top of what you have helps, where the
    budget-matched control answers how to divide a fixed budget.
    """
    return _object_count_split(*args, train_positions={1: OBJECT_COUNT_TRAIN_POSITIONS,
                                                       2: OBJECT_COUNT_TRAIN_POSITIONS}, **kwargs)

def green_plastic(mds_path, test_size=0.2, seed=42, speakers=None, n_objects=None, box=None, n_samples: int | None = None, verbose: int = 1, index: list[dict] | None = None):

    if index is None:
        lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
        index = [json.loads(line) for line in lines if line]

    # filter by the requested speakers, n_objects, and box type
    keep = [i for i, row in enumerate(index) if _matches(row, speakers, n_objects, box)]
    if n_samples is not None: keep = keep[:n_samples]

    # empty box not split: it all goes in train
    empty_box = [i for i, row in enumerate(index) if row['layout'] == 'empty-box']

    # put three entire grids of two-cubes in train and the fourth grid in eval, so every position is "seen" at least three times in train
    two_cubes_train = [i for i, row in enumerate(index) if row['layout'] in ['two-cubes-grid1', 'two-cubes-grid2', 'two-cubes-grid3']]
    two_cubes_eval = [i for i, row in enumerate(index) if row['layout'] == 'two-cubes-grid4']
    # split one cube into train and test -- absent from the two-laser-faces capture, which has no
    # one-cube layout at all, so this stays empty there rather than erroring on an empty split
    one_cube = [i for i, row in enumerate(index) if row['layout'] == 'red-cube']
    one_cube_train, one_cube_eval = train_test_split(one_cube, test_size=test_size, random_state=seed, shuffle=True) if one_cube else ([], [])

    splits = {'train': empty_box + one_cube_train + two_cubes_train, 'eval/1-cube': one_cube_eval, 'eval/2-cubes': two_cubes_eval}
    if verbose:
        for label, idxs in splits.items(): print(f"{label}: {len(idxs)} samples")
    return splits


#***** 8 build dataloaders *****

SPLIT_METHODS = {"exp25": exp25_split, "gastronorm": gastronorm, "gastronorm_one_cube": gastronorm_one_cube,
                 "gastronorm_train1_eval2": gastronorm_train1_eval2, "gastronorm_train2_eval1": gastronorm_train2_eval1,
                 "gastronorm_train12_eval12": gastronorm_train12_eval12,
                 "gastronorm_train12_eval12_full": gastronorm_train12_eval12_full,
                 "green_plastic": green_plastic}

#***** 7 pair two speakers into one sample *****

SPEAKER_PARTNER = {1: 3, 3: 1, 2: 4, 4: 2, 7: 5, 5: 7, 8: 6, 6: 8}

class PairedSpeakerDataset(torch.utils.data.Dataset):
    """One item = two speakers' captures of the same position, stacked on the channel axis:
    two (L,P,PS,C) -> (L,P,PS,2C). The partner is fixed by SPEAKER_PARTNER, and must be in `idxs`,
    so pairs never cross the train/eval boundary.
    """
    def __init__(self, dataset, idxs, index):
        self.dataset = dataset
        partner_of = {(index[j]["position_id"], index[j]["speaker"]): j for j in idxs}
        self.pairs = [(i, partner_of[key]) for i in idxs
                      if (key := (index[i]["position_id"], SPEAKER_PARTNER[index[i]["speaker"]])) in partner_of]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, k):
        i, j = self.pairs[k]
        a, b = self.dataset[i], self.dataset[j]
        info = dict(a["info"], speaker_b=b["info"]["speaker"])
        return dict(fft=torch.cat([a["fft"], b["fft"]], dim=-1), mask_true=a["mask_true"], info=info)

def num_samples(batch): return batch["mask_true"].shape[0]

def loader(dataset, idxs, bs, num_workers, generator, shuffle=False, drop_last=False):
    dl = DataLoader(Subset(dataset, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True,
                    persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, drop_last=drop_last)
    return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

def build_dataset(data_dir: str | Path, split: str = "exp25", batch_size: int = 64, eval_batch_size: int = 64,
                   num_workers: int = 8, out_h: int = 20, out_w: int = 40, signal_mode: str = "magnitude",
                   normalize_mode: str = "std", patch_size: int = 64, seed: int = 42,
                   force_rebuild_data: bool = False, augment_fft: float = 0.5, augment_mask: float = 0.5,
                   subtract_speaker_mean: bool = False, subtract_empty_box: bool = False, n_classes: int = 4, verbose: int = 1,
                   mag_recipe: str | None = None, pair_speakers_mode: bool = False, rgb: bool = False,
                   phase_arm: str | None = None, phase_weight: float = 1.0,
                   laser_cols=None, **split_kwargs):

    # A recipe owns the domain and the operation, so it OVERRIDES signal_mode and both
    # subtract_* flags. Deriving signal_mode here is what guarantees the references are
    # built in the same domain the recipe uses them in -- a linear arm dividing by a log
    # reference is silently meaningless, and nothing downstream would catch it.
    if mag_recipe is not None:
        from model.normalizations import MAG_RECIPES
        if mag_recipe not in MAG_RECIPES:
            raise ValueError(f"unknown {mag_recipe=}; expected one of {sorted(MAG_RECIPES)}")
        signal_mode = "log_magnitude" if mag_recipe.startswith("logmag") else "magnitude"
        subtract_empty_box = mag_recipe.endswith(("_eb", "_both"))
        subtract_speaker_mean = mag_recipe.endswith(("_spk", "_both"))
        if verbose:
            print(f"{mag_recipe=} -> {signal_mode=}, {subtract_empty_box=}, {subtract_speaker_mean=}")

    # Phase is orthogonal to stage A: it reads the complex fft directly, so it imposes no
    # constraint on signal_mode. Validated here rather than at use so a typo fails before the
    # ~10 min MDS build rather than after it.
    if phase_arm is not None:
        from model.normalizations import PHASE_ARMS
        if phase_arm not in PHASE_ARMS:
            raise ValueError(f"unknown {phase_arm=}; expected one of {sorted(PHASE_ARMS)}")

    if split not in SPLIT_METHODS: raise ValueError(f"Unknown split {split!r}, expected one of {sorted(SPLIT_METHODS)}")
    if not 0 <= augment_fft <= 1: raise ValueError(f"{augment_fft=} must be a probability in [0, 1]")
    if not 0 <= augment_mask <= 1: raise ValueError(f"{augment_mask=} must be a probability in [0, 1]")
    if mag_recipe is None and subtract_speaker_mean and signal_mode not in ("magnitude", "log_magnitude"):
        raise ValueError(f"{subtract_speaker_mean=} requires signal_mode='magnitude' or 'log_magnitude', got {signal_mode!r}")
    # log space only: the whole point is that subtracting == dividing there. In linear space this
    # would be a subtraction of magnitudes, which is not the multiplicative correction we want.
    if mag_recipe is None and subtract_empty_box and signal_mode != "log_magnitude":
        raise ValueError(f"{subtract_empty_box=} requires signal_mode='log_magnitude' (subtracting a log reference is dividing by it), got {signal_mode!r}")
    data_dir = Path(data_dir)
    if not data_dir.exists(): raise FileNotFoundError(f"{data_dir=} does not exist")

    # the fft gain augmentation needs the raw complex fft, so any nonzero probability means we store it raw
    raw_fft = augment_fft > 0
    samples = collect_samples(data_dir / "samples", verbose)

    # the laser grid is the data's own geometry, so it is read from the data, never configured
    n_lasers = np.load(fft_path(samples[0][0]))["fft"].shape[-3]
    n_laser_rows, n_laser_cols = infer_laser_grid(samples, n_lasers)
    grid = (n_laser_rows, n_laser_cols if laser_cols is None else len(set(laser_cols)))
    if verbose:
        sel = f"columns {sorted(laser_cols)} of {n_laser_cols}" if laser_cols is not None else "all columns"
        print(f"lasers: {n_laser_rows}x{n_laser_cols} recorded, {sel} -> {grid[0]}x{grid[1]} = {grid[0] * grid[1]} of {n_lasers}")

    # Resolved once and applied at the single point where the fft is read off disk, so every
    # statistic (speaker means, empty-box ref, per-bin stats) and every normalization -- which
    # reduce over L -- see only the kept lasers. Slicing any later would normalize against
    # lasers the model never gets.
    laser_idx = laser_indices(laser_cols, n_laser_rows, n_laser_cols)
    mds_dir = data_dir / "mds" / hash_samples(samples, out_h, out_w, raw_fft, signal_mode, normalize_mode, patch_size, subtract_speaker_mean, subtract_empty_box, mag_recipe, rgb, phase_arm, phase_weight, laser_cols)[:16]
    done = mds_dir / "metadata.jsonl"  # last file convert_to_mds writes -- its presence means the build completed

    if force_rebuild_data and mds_dir.exists():
        shutil.rmtree(mds_dir)
        if verbose: print(f"Overwriting {mds_dir=}")

    if not done.exists():
        if mds_dir.exists(): shutil.rmtree(mds_dir)  # clear out a partial/crashed build

        # train-only, so eval spectra don't leak into either set of statistics
        needs_stats = normalize_mode.split('+')[0] in DATASET_STATS_MODES
        train_idxs = SPLIT_METHODS[split](mds_dir, seed=seed, verbose=0, index=[m for _, m in samples], **split_kwargs)["train"] if subtract_speaker_mean or subtract_empty_box or needs_stats else None

        empty_box_ref = compute_empty_box_ref(samples, signal_mode, keep_idxs=train_idxs, verbose=verbose, laser_idx=laser_idx) if subtract_empty_box else None
        speaker_means = compute_speaker_means(samples, signal_mode, keep_idxs=train_idxs, verbose=verbose, empty_box_ref=None, laser_idx=laser_idx) if subtract_speaker_mean else None
        stats = compute_dataset_stats(samples, signal_mode, keep_idxs=train_idxs, verbose=verbose, laser_idx=laser_idx) if needs_stats else None

        # downsample image, precompute fft, and convert to mds
        downsample_samples(samples, out_h, out_w, verbose=verbose, rgb=rgb)
        if not raw_fft: precompute_vibration_samples(samples, signal_mode, normalize_mode, patch_size, verbose=verbose, speaker_means=speaker_means, stats=stats, empty_box_ref=empty_box_ref, mag_recipe=mag_recipe, phase_arm=phase_arm, phase_weight=phase_weight, laser_idx=laser_idx, laser_cols=laser_cols)
        convert_to_mds(mds_dir, samples, out_h, out_w, verbose=verbose, augment_fft=raw_fft, signal_mode=signal_mode, normalize_mode=normalize_mode, patch_size=patch_size, subtract_speaker_mean=subtract_speaker_mean, subtract_empty_box=subtract_empty_box, rgb=rgb, phase_arm=phase_arm, phase_weight=phase_weight, laser_idx=laser_idx, laser_cols=laser_cols)
        if speaker_means is not None: save_speaker_means(speaker_means, mds_dir / SPEAKER_MEANS_FILE)
        if stats is not None: save_dataset_stats(stats, mds_dir / DATASET_STATS_FILE)
        if empty_box_ref is not None: save_empty_box_ref(empty_box_ref, mds_dir / EMPTY_BOX_REF_FILE)
    elif verbose:
        print(f"Reusing existing MDS at {mds_dir}")
    (mds_dir / LASER_GRID_FILE).write_text(json.dumps(grid))

    splits = SPLIT_METHODS[split](mds_dir, seed=seed, verbose=verbose, **split_kwargs)

    generator = torch.Generator().manual_seed(seed)
    if rgb: augment_mask = 0.0  # blur+noise is a mask augmentation, meaningless on a photo
    process_kwargs = dict(device="cpu", signal_mode=signal_mode, normalize_mode=normalize_mode, patch_size=patch_size, out_h=out_h, out_w=out_w, mds_precomputed_fft=not raw_fft, n_classes=n_classes, mag_recipe=mag_recipe, phase_arm=phase_arm, phase_weight=phase_weight)
    train_dataset = VibrationDataset(local=mds_dir, seed=seed, process_kwargs=dict(process_kwargs, augment_fft=augment_fft, augment_mask=augment_mask))
    eval_dataset = VibrationDataset(local=mds_dir, seed=seed, process_kwargs=dict(process_kwargs, augment_fft=0.0, augment_mask=0.0))

    if pair_speakers_mode:
        index = [json.loads(line) for line in (mds_dir / "metadata.jsonl").read_text().strip().splitlines() if line]
        train_dataset = PairedSpeakerDataset(train_dataset, splits["train"], index)
        eval_datasets = {label: PairedSpeakerDataset(eval_dataset, idxs, index) for label, idxs in splits.items()}
        if verbose:
            for label, idxs in splits.items(): print(f"{label}: {len(idxs)} samples -> {len(eval_datasets[label])} pairs")
        splits = {label: list(range(len(ds))) for label, ds in eval_datasets.items()}
    else:
        eval_datasets = {label: eval_dataset for label in splits}

    train_loader = loader(train_dataset, splits["train"], batch_size, num_workers, generator, shuffle=True, drop_last=True)
    train_eval_loader = loader(eval_datasets["train"], splits["train"], eval_batch_size, num_workers, generator)
    eval_loaders = [Evaluator(label=label, dataloader=loader(eval_datasets[label], idxs, eval_batch_size, num_workers, generator)) for label, idxs in splits.items() if label != "train"]

    return train_loader, eval_loaders, train_eval_loader
