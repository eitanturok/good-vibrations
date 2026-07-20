import argparse, json, hashlib, shutil, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.interpolate import PchipInterpolator
from streaming import MDSWriter

from utils.io_utils import load, save, append, symlink, copy
from utils.helpers import Timing, logger, human_size, dir_size
from utils.metrics import center_of_mass

#***** 1 post-process image (blow up + downsample overhead image) *****

def blow_up_mask(mask: Image.Image, object_scale: float) -> Image.Image:
    # Zooms each connected object about its own center of mass (nearest-neighbor inverse map),
    # so multi-object masks scale independently instead of drifting toward a shared centroid.
    # object_scale is linear (area grows ~object_scale**2); 1 is a no-op.
    if object_scale == 1: return mask
    arr = np.array(mask) > 127
    lbl, n = ndimage.label(arr)
    out = np.zeros_like(arr)
    rr, cc = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    for i in range(1, n + 1):
        obj = lbl == i
        cr, cw = ndimage.center_of_mass(obj)
        sr = np.round((rr - cr) / object_scale + cr).astype(int)
        sc = np.round((cc - cw) / object_scale + cw).astype(int)
        ok = (sr >= 0) & (sr < arr.shape[0]) & (sc >= 0) & (sc < arr.shape[1])
        hit = np.zeros_like(arr)
        hit[ok] = obj[sr[ok], sc[ok]]
        out |= hit
    return Image.fromarray((out * 255).astype(np.uint8))

def downsample(mask: Image.Image, out_h: int, out_w: int) -> Image.Image:
    # BOX resampling area-averages over the full H x W mask (unlike a floor-division block
    # reshape, which silently truncates to block_h*out_h x block_w*out_w and drops the
    # bottom/right edge whenever out_h/out_w don't evenly divide H/W).
    return mask.resize((out_w, out_h), resample=Image.BOX)

#***** 2 post-process fft (extract signal, normalize signal, tokenize) *****

def extract_signal(x: np.ndarray, signal_mode: str) -> np.ndarray:
    # Cast to complex128 before abs/angle: np.abs on complex64 loses precision for large values
    # because sqrt(re²+im²) is computed in float32; PyTorch promotes internally so we match it.
    if signal_mode == "magnitude": return np.abs(x.astype(np.complex128))
    if signal_mode == "complex": return np.concatenate([x.real, x.imag], axis=-1)
    if signal_mode == "mag_phase": return np.concatenate([np.abs(x.astype(np.complex128)), np.angle(x.astype(np.complex128))], axis=-1)
    raise ValueError(f"Unknown signal mode: {signal_mode}")

def normalize_fft(x: np.ndarray, normalize_mode: str, verbose:int=0) -> np.ndarray:
    if normalize_mode is None: return x
    # Compute stats in float64 with ddof=1 to match PyTorch's std behavior
    x64 = x.astype(np.float64)
    if normalize_mode == 'std-sample':
        std = np.maximum(x64.std(axis=(1, 2, 3), ddof=1, keepdims=True), 1e-8).astype(np.float32)
        return x / std
    if normalize_mode == 'z-sample':
        mean = x64.mean(axis=(1, 2, 3), keepdims=True).astype(np.float32)
        std = np.maximum(x64.std(axis=(1, 2, 3), ddof=1, keepdims=True), 1e-8).astype(np.float32)
        return (x - mean) / std
    raise ValueError(f"Unknown normalize mode: {normalize_mode}")

def tokenize(x: np.ndarray, patch_size:int):
    # Note: unfold drops entries that do not fully fit into patch_size
    if patch_size <= 0: return x
    B, L, F, C = x.shape
    P = F // patch_size
    return x[:, :, :P * patch_size, :].reshape(B, L, P, patch_size, C)  # (B,L,P,PS,C)

#***** 2b augmentations (ported from notebooks/53_augmentations.ipynb) *****

def make_rng(epoch:int, sample_id:int) -> np.random.Generator:
    # Deterministic per-(epoch, sample_id) seed: augmentation differs every epoch but is
    # reproducible for a given epoch, independent of worker process / draw order.
    seed = (int(epoch) * 2_147_483_647 + int(sample_id)) & 0xFFFFFFFF
    return np.random.default_rng(seed)

def random_frequency_gain(freqs: np.ndarray, rng: np.random.Generator, n_control:int=5, gain_range:tuple=(0.8, 1.2)) -> np.ndarray:
    # Smooth random multiplicative gain curve: random values at a few control frequencies,
    # connected with a monotonic cubic spline, rescaled into gain_range.
    control_freqs = np.linspace(freqs.min(), freqs.max(), n_control)
    control_gains = rng.normal(loc=1.0, scale=1.0, size=n_control)
    spline = PchipInterpolator(control_freqs, control_gains)
    gain = spline(freqs)
    gain = (gain - gain.min()) / (gain.max() - gain.min())  # -> [0, 1]
    lo, hi = gain_range
    return (gain * (hi - lo) + lo).astype(np.float32)

def augment_vibration(fft: np.ndarray, freqs: np.ndarray, rng: np.random.Generator, n_control:int=5, gain_range:tuple=(0.8, 1.2)) -> np.ndarray:
    # fft: (B,L,F,C) complex. One shared gain curve per call (batch); draw a fresh rng per
    # row upstream (e.g. make_rng(epoch, sample_id)) if per-sample gain curves are wanted.
    gain = random_frequency_gain(freqs, rng, n_control, gain_range)
    return fft * gain[None, None, :, None]

def label_smooth(mask: np.ndarray, lo:float=0.05, hi:float=0.9) -> np.ndarray:
    # Deterministic value remap: [0,1] -> [lo,hi], everywhere. No spatial effect.
    return mask * (hi - lo) + lo

def edge_blur(mask: np.ndarray, sigma:float=0.8) -> np.ndarray:
    # Deterministic spatial smoothing (fixed 3x3 kernel); softens boundary transition band,
    # flat interior/exterior regions unaffected. mask: (B,H,W).
    return gaussian_filter(mask, sigma=sigma, radius=1, mode='nearest', axes=(1, 2))

def random_boundary_jitter(mask: np.ndarray, rng: np.random.Generator, shift_range:float=1.0, sigma_range:tuple=(0.4, 1.2), lo_range:tuple=(0.0, 0.08), hi_range:tuple=(0.85, 1.0)) -> np.ndarray:
    # Stochastic: sub-pixel boundary shift + random blur + random label smoothing, meant to
    # be called fresh per sample per epoch. mask: (B,H,W); one draw per call (shared across B).
    shift = rng.uniform(-shift_range, shift_range, size=2)
    jittered = ndimage.shift(mask, (0, *shift), order=1, mode='nearest')
    sigma = rng.uniform(*sigma_range)
    blurred = gaussian_filter(jittered, sigma=sigma, radius=1, mode='nearest', axes=(1, 2))
    lo, hi = rng.uniform(*lo_range), rng.uniform(*hi_range)
    return np.clip(blurred, 0, 1) * (hi - lo) + lo

def noisy_blur(mask: np.ndarray, rng: np.random.Generator, sigma:float=0.8, noise_std:float=0.05) -> np.ndarray:
    # Gaussian blur, then add noise back onto originally-nonzero pixels only. mask: (B,H,W).
    blurred = gaussian_filter(mask, sigma=sigma, radius=1, mode='nearest', axes=(1, 2))
    noise = rng.normal(0.0, noise_std, size=mask.shape)
    out = blurred.copy()
    out[mask != 0] += noise[mask != 0]
    return np.clip(out, 0, 1)

#***** 2c composed per-sample/per-batch processing (usable standalone or from the live dataloader) *****

def process_vibration(fft: np.ndarray, freqs: np.ndarray, signal_mode:str, normalize_mode:str, patch_size:int, rng: np.random.Generator|None=None, gain_kwargs: dict|None=None) -> np.ndarray:
    """fft: (B,L,F,C) complex. Returns tokenized (B,L,P,patch_size,C) float32.
    rng=None skips frequency augmentation (offline baseline / eval)."""
    if rng is not None:
        fft = augment_vibration(fft, freqs, rng, **(gain_kwargs or {}))
    x = extract_signal(fft, signal_mode).astype(np.float32)  # (B,L,F_,C) -> (B,L,F,C)
    x = normalize_fft(x, normalize_mode)
    return tokenize(x, patch_size)

def process_image(mask: np.ndarray, rng: np.random.Generator|None=None, augment_fn=None) -> np.ndarray:
    """mask: (B,H,W) float32 in [0,1]. Returns (B,H,W) float32.
    rng=None skips mask augmentation (offline baseline / eval)."""
    if rng is not None:
        mask = (augment_fn or noisy_blur)(mask, rng)
    return mask

#***** 2d torch/GPU equivalents of process_vibration (for benchmarking CPU vs GPU) *****
# The fft (B,L,F,C) arrays here are large enough that GPU may win despite transfer overhead;
# the mask (B,H,W) arrays in process_image are tiny, so no GPU variant is provided for those --
# transfer latency alone would dominate any compute saved.

def random_frequency_gain_torch(freqs: "torch.Tensor", rng: np.random.Generator, n_control:int=5, gain_range:tuple=(0.8, 1.2)) -> "torch.Tensor":
    import torch
    control_freqs = torch.linspace(freqs.min().item(), freqs.max().item(), n_control, dtype=freqs.dtype, device=freqs.device)
    control_gains = torch.from_numpy(rng.normal(loc=1.0, scale=1.0, size=n_control)).to(freqs.device, freqs.dtype)
    # notebook 53's hand-rolled Hermite-spline interp (monotone cubic through control points)
    idxs = torch.searchsorted(control_freqs[1:], freqs).clamp(max=n_control - 2)
    m = (control_gains[1:] - control_gains[:-1]) / (control_freqs[1:] - control_freqs[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    dx = control_freqs[idxs + 1] - control_freqs[idxs]
    t = (freqs - control_freqs[idxs]) / dx
    tt = t[None, :] ** torch.arange(4, device=t.device, dtype=t.dtype)[:, None]
    H = torch.tensor([[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]], dtype=t.dtype, device=t.device) @ tt
    gain = H[0] * control_gains[idxs] + H[1] * m[idxs] * dx + H[2] * control_gains[idxs + 1] + H[3] * m[idxs + 1] * dx
    gain = (gain - gain.min()) / (gain.max() - gain.min())
    lo, hi = gain_range
    return gain * (hi - lo) + lo

def process_vibration_torch(fft: "torch.Tensor", freqs: "torch.Tensor", signal_mode:str, normalize_mode:str, patch_size:int, rng: np.random.Generator|None=None, gain_kwargs: dict|None=None) -> "torch.Tensor":
    """Same contract as process_vibration but fft/freqs are torch tensors already on the
    target device (cpu or cuda) -- move them there before calling, once per batch."""
    import torch
    if rng is not None:
        gain = random_frequency_gain_torch(freqs, rng, **(gain_kwargs or {}))
        fft = fft * gain[None, None, :, None]
    if signal_mode == "magnitude": x = fft.abs()
    elif signal_mode == "complex": x = torch.cat([fft.real, fft.imag], dim=-1)
    elif signal_mode == "mag_phase": x = torch.cat([fft.abs(), fft.angle()], dim=-1)
    else: raise ValueError(f"Unknown signal mode: {signal_mode}")
    if normalize_mode is not None:
        x64 = x.double()
        if normalize_mode == 'std-sample':
            std = x64.std(dim=(1, 2, 3), correction=1, keepdim=True).clamp_min(1e-8).float()
            x = x / std
        elif normalize_mode == 'z-sample':
            mean = x64.mean(dim=(1, 2, 3), keepdim=True).float()
            std = x64.std(dim=(1, 2, 3), correction=1, keepdim=True).clamp_min(1e-8).float()
            x = (x - mean) / std
        else: raise ValueError(f"Unknown normalize mode: {normalize_mode}")
    if patch_size > 0:
        B, L, F, C = x.shape
        P = F // patch_size
        x = x[:, :, :P * patch_size, :].reshape(B, L, P, patch_size, C)
    return x

#***** 3 post-process a single sample *****

def post_process_sample(sample_dir:Path, out_h:int, out_w:int, signal_mode:str, normalize_mode:str, patch_size:int, object_scale:float, verbose:int, do_save:bool, denotch_fn=None, empty_mean:np.ndarray|None=None):
    sample_id = sample_dir.name
    is_empty_box = bool({k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}.get("is_empty_box", False))

    # post process overhead image (blow up + downsample segment mask, compute new center of mass)
    with Timing(f"[sample {sample_id}] post-process overhead image: ", enabled=verbose >= 2):
        segment_mask = load(sample_dir / "image/02_smask.png")

        # blow up objects about their own center of mass before downsampling, so the tiny
        # objects in this dataset don't collapse to a handful of low-res pixels
        blownup_segment_mask = blow_up_mask(segment_mask, object_scale)
        save(blownup_segment_mask, sample_dir / "image/05b_blownup_smask.png", do_save)

        # downsample segment mask to out_h x out_w
        downsampled_segment_mask = downsample(blownup_segment_mask, out_h, out_w)
        save(downsampled_segment_mask, sample_dir / "image/06_downsampled_smask.png", do_save)
        save(np.array(downsampled_segment_mask, dtype=np.float32) / 255.0, sample_dir / "image/07_downsampled_smask.npy", do_save)
        append({'downsample': datetime.now(timezone.utc).isoformat()}, sample_dir / 'times.jsonl', do_save)

        # compute new center of mass for the downsampled segment mask
        downsampled_com = (-1, -1) if is_empty_box else center_of_mass(downsampled_segment_mask)
        append([{"downsampled_com": downsampled_com}], sample_dir / "metadata.jsonl", do_save)
        append({'process_overhead/com': datetime.now(timezone.utc).isoformat()}, sample_dir / 'times.jsonl', do_save)
        logger.debug(f"[sample {sample_id}] {downsampled_com=}")

    # post process vibrations: denotch + empty-diff only. extract_signal/normalize_fft/tokenize
    # (and any frequency-domain augmentation) run live, later, via process_vibration -- they need
    # to differ every epoch, so they can't be baked in here.
    with Timing(f"[sample {sample_id}] post-process vibrations: ", enabled=verbose >= 2):
        fft_npz = load(sample_dir / 'vibration/03_fft.npz')
        freqs, fft_clean = fft_npz['freqs'], fft_npz['fft']
        if denotch_fn is not None:
            fft_clean = denotch_fn(fft_clean, freqs)
            save({'fft': fft_clean, 'freqs': freqs, 'n_samples': fft_npz['n_samples']}, sample_dir / 'vibration/03_fft.npz', do_save)
            append({"fft_denotched": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)
        logger.debug(f'[sample {sample_id}] {fft_clean.shape=}=(batch, lasers, freqs, x/y)')

        # subtract this speaker's empty-box mean (empty_diff), still on the clean complex fft
        if empty_mean is not None:
            fft_clean = fft_clean - empty_mean
            append({"fft_diffed": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

        save(fft_clean, sample_dir / 'vibration/06_clean_fft.npy', do_save)
        save(freqs, sample_dir / 'vibration/06_freqs.npy', do_save)
        append({"fft_clean_saved": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

    # symlink X.npy, y.npy for model input, output. X is the clean complex fft (pre signal
    # extraction / normalize / tokenize -- those + augmentation run live in the dataloader).
    symlink(sample_dir / 'vibration/06_clean_fft.npy', sample_dir / 'X.npy', do_save)
    symlink(sample_dir / 'image/07_downsampled_smask.npy', sample_dir / 'y.npy', do_save)

    n_lasers, n_freqs = fft_clean.shape[1], fft_clean.shape[2]
    append(dict(out_h=out_h, out_w=out_w, signal_mode=signal_mode, normalize_mode=normalize_mode, patch_size=patch_size, object_scale=object_scale, n_lasers=n_lasers, n_freqs=n_freqs, empty_diff=empty_mean is not None), sample_dir / "metadata.jsonl", do_save)
    append({"post_process": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)


#***** 4 convert to mds *****

MDS_COLUMNS = {"X": "ndarray:float32", "y": "ndarray:float32",
               "sample_id": "int", "output_id": "str",
               "n_objects": "int", "speaker": "int", "box": "str", "is_empty_box": "int", "object": "str",
               "downsampled_com_x": "float64", "downsampled_com_y": "float64",
}

def convert_to_mds(dataset_dir:Path, rows:list, patch_size:int, out_h:int, out_w:int, verbose:int):
    # Streaming urlparses `out`, so an absolute Windows path (e.g. "D:/...") is misread as a
    # cloud scheme "d:". Chdir to dataset_dir and pass the relative "mds" name -> empty url scheme.
    mds_dir = dataset_dir / "mds"
    if verbose: print(f"Writing MDS to {mds_dir} ...")
    if mds_dir.exists(): shutil.rmtree(mds_dir)

    n_lasers, n_freqs = rows[0][1]["n_lasers"], rows[0][1]["n_freqs"]
    x_shape, y_shape = (n_lasers, n_freqs, 2, 2), (out_h, out_w)  # X: (L,F,C,2) real/imag-stacked complex fft

    index_rows = []
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(dataset_dir)

    # write to MSD format
    try:
      with MDSWriter(out="mds", columns=MDS_COLUMNS, exist_ok=False) as writer:
        for i, (sample_dir, meta) in enumerate(rows):
            X = np.load(sample_dir / "X.npy")                          # (1, L, F, C) complex64
            y = np.load(sample_dir / "y.npy").astype(np.float32)       # (out_h, out_w)
            X = np.squeeze(X, axis=0) if X.ndim == 4 and X.shape[0] == 1 else X  # -> (L,F,C)
            X = np.stack([X.real, X.imag], axis=-1).astype(np.float32)  # complex -> (L,F,C,2) real/imag
            assert X.shape == x_shape, f"{sample_dir.name}: X.shape={X.shape} != {x_shape}"
            assert y.shape == y_shape, f"{sample_dir.name}: y.shape={y.shape} != {y_shape}"

            com = meta.get("downsampled_com", [-1.0, -1.0])
            sample = {
                "X": X, "y": y, "sample_id": int(meta["sample_id"]), "output_id": str(meta.get("output_id", "")),
                "n_objects": int(meta.get("n_objects", -1)),
                "speaker": int(meta.get("speaker", -1)),
                "box": str(meta.get("box", "")),
                "is_empty_box": int(bool(meta.get("is_empty_box", False))),
                "object": str(meta.get("object", "")),
                "downsampled_com_x": float(com[0]), "downsampled_com_y": float(com[1]),
            }
            writer.write(sample)
            index_rows.append(meta)  # full per-sample metadata -> sidecar (used for loader-side filtering)
            if verbose >= 2 and (i + 1) % 50 == 0: print(f"  wrote {i + 1}/{len(rows)}")
    finally:
        os.chdir(cwd)

    # freqs is identical across every sample (same fft grid) -- one sidecar, not duplicated per-row
    shutil.copy(rows[0][0] / "vibration/06_freqs.npy", mds_dir / "freqs.npy")

    # dataset-level sidecar for filtering
    lines = "\n".join([json.dumps(r) for r in index_rows])
    (mds_dir / "metadata.jsonl").write_text(lines)
    if verbose: print(f"Wrote {len(rows)} samples to {mds_dir=}")
    return mds_dir


#***** 5 post-process all samples in a base sample directory *****

def compute_empty_means(samples_dir:Path, rows:dict, denotch_fn, verbose:int) -> dict:
    # Per-speaker mean of the clean complex fft over that speaker's empty-box (n_objects == 0)
    # samples. Running sum (not stack) so memory stays at one fft per speaker. Averaged in
    # complex space (pre extract_signal) since extract_signal/normalize/tokenize now run live.
    sums, counts = {}, {}
    for sample_id, meta in rows.items():
        if int(meta.get("n_objects", -1)) != 0: continue
        speaker = int(meta.get("speaker", -1))
        fft_npz = load(samples_dir / sample_id / 'vibration/03_fft.npz')
        freqs, fft_raw = fft_npz['freqs'], fft_npz['fft']
        if denotch_fn is not None: fft_raw = denotch_fn(fft_raw, freqs)
        sums[speaker] = fft_raw if speaker not in sums else sums[speaker] + fft_raw
        counts[speaker] = counts.get(speaker, 0) + 1
    if verbose: print("empty_diff: empty-box samples per speaker: " + (", ".join(f"speaker {spk}: {n}" for spk, n in sorted(counts.items())) or "none"))
    return {spk: s / counts[spk] for spk, s in sums.items()}

def resolve_dataset_dir(base_dataset_dir:Path, dataset_name:str|None, key:str) -> Path:
    # Datasets are versioned as "NNN" / "NNN-<name>" dirs. If an existing dir's hash.txt
    # already matches this exact param/data combination, reuse it (cache hit -- no new
    # number). Otherwise allocate the next number from count.txt, a persistent counter that
    # only ever increments -- deleting dataset dirs never causes a number to be reused.
    suffix = f"-{dataset_name}" if dataset_name else ""
    existing = sorted(p for p in base_dataset_dir.glob("[0-9][0-9][0-9]*") if p.is_dir()) if base_dataset_dir.exists() else []
    for p in existing:
        if p.name[3:] != suffix: continue
        hash_path = p / "hash.txt"
        if hash_path.exists() and hash_path.read_text().strip() == key: return p
    base_dataset_dir.mkdir(parents=True, exist_ok=True)
    count_path = base_dataset_dir / "count.txt"
    n = int(count_path.read_text().strip()) if count_path.exists() else 0
    count_path.write_text(str(n + 1))
    return base_dataset_dir / f"{n:03d}{suffix}"

def post_process(base_sample_dir:Path, base_dataset_dir:Path, dataset_name:str|None, out_h:int, out_w:int, signal_mode:str, normalize_mode:str, patch_size:int, object_scale:float=1, empty_diff:bool=False, force:bool=False, verbose:int=1, do_save:bool=True, denotch_fn=None):

    # collect complete samples + metadata
    REQUIRED_FILES = ["image/02_smask.png", "vibration/03_fft.npz"]
    rows, missing_by_file = {}, {f: [] for f in REQUIRED_FILES}
    for sample_dir in sorted(base_sample_dir.glob("*")):
        if not sample_dir.is_dir(): continue
        missing = [f for f in REQUIRED_FILES if not (sample_dir / f).exists()]
        if missing:
            for f in missing: missing_by_file[f].append(sample_dir.name)
            continue
        rows[sample_dir.name] = {k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}
    sample_ids = list(rows.keys())
    n_skipped = len({sid for ids in missing_by_file.values() for sid in ids})
    if verbose:
        print(f"Found {len(rows)} complete samples ({n_skipped} skipped)")
        for f, ids in missing_by_file.items():
            if ids: print(f"missing {f!r}: {ids}")

    # hash and check for skip
    with Timing("Hashing: ", enter=f"Hashing metadata of {len(sample_ids)} samples to compute cache key ...", enabled=verbose):
        # empty_diff only enters the key when on, so existing dataset dirs keep their cached keys;
        # object_scale always enters so re-running with a different blow-up factor versions a new dataset
        key = hashlib.sha1(json.dumps(rows | MDS_COLUMNS | ({"empty_diff": True} if empty_diff else {}) | {"object_scale": object_scale}, sort_keys=True, default=str).encode()).hexdigest()[:16]

    dataset_dir = resolve_dataset_dir(base_dataset_dir, dataset_name, key)
    samples_dir, mds_dir, hash_path = dataset_dir / "samples", dataset_dir / "mds", dataset_dir / "hash.txt"
    cached_key = hash_path.read_text().strip() if hash_path.exists() else None
    if not force and cached_key == key and mds_dir.exists() and (mds_dir / "metadata.jsonl").exists():
        if verbose: print(f"Cache hit: reusing existing MDS at {mds_dir}\nMDS: {mds_dir} ({len(rows)} samples)")
        return mds_dir

    # symlink sample_dir to data_dir except we copy metadata.jsonl, times.jsonl
    with Timing("Setting up sample dirs: ", enter=f"Setting up {len(sample_ids)} sample dirs ...", enabled=verbose):
        for sample_id in sample_ids:
            src_dir, dst_dir = base_sample_dir / sample_id, samples_dir / sample_id
            for src_path in src_dir.rglob("*"):
                if src_path.is_dir(): continue
                dst_path = dst_dir / src_path.relative_to(src_dir)
                if src_path.name in ("metadata.jsonl", "times.jsonl"): copy(src_path, dst_path, do_save)
                else: symlink(src_path, dst_path, do_save)

    # compute per-speaker empty-box means for the empty_diff subtraction
    empty_means = {}
    if empty_diff:
        with Timing("Empty-box means: ", enter="Computing per-speaker empty-box fft means ...", enabled=verbose):
            empty_means = compute_empty_means(samples_dir, rows, denotch_fn, verbose)

    # post process each sample
    speakers_without_empties = set()
    with Timing("Post-processing: ", enter=f"Post-processing {len(sample_ids)} samples ...", enabled=verbose):
        for sample_id in sample_ids:
            sample_dir = samples_dir / sample_id
            empty_mean = empty_means.get(int(rows[sample_id].get("speaker", -1))) if empty_diff else None
            if empty_diff and empty_mean is None: speakers_without_empties.add(int(rows[sample_id].get("speaker", -1)))
            post_process_sample(sample_dir, out_h, out_w, signal_mode, normalize_mode, patch_size, object_scale, verbose, do_save, denotch_fn, empty_mean)
            rows[sample_id] = {k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}
    if speakers_without_empties:
        logger.warning(f"empty_diff: no empty-box samples for speakers {sorted(speakers_without_empties)} -- their samples were NOT diffed")

    # convert to MDS
    sample_rows = [(samples_dir / sample_id, meta) for sample_id, meta in rows.items()]
    with Timing("Writing MDS: ", enabled=verbose):
        mds_path = convert_to_mds(dataset_dir, sample_rows, patch_size, out_h, out_w, verbose)
    if do_save: hash_path.write_text(key)
    return mds_path


#***** 6 CLI *****

REMOTE_HOST = "ethantu@132.76.83.154"
REMOTE_BASE_DIR = "/home/ethantu/workspace/good-vibrations/datasets"

def scp_to_remote(local_path:Path, remote_host:str, remote_base_dir:str, verbose:int):
    remote_parent = f"{remote_base_dir}/{local_path.parent.name}"
    remote_path = f"{remote_parent}/{local_path.name}"
    local_hash_path = local_path.parent / "hash.txt"

    # skip re-sending if the remote already has this exact dataset (same hash.txt content)
    remote_hash = subprocess.run(["ssh", remote_host, f"cat {remote_parent}/hash.txt"], capture_output=True, text=True).stdout.strip()
    if remote_hash and remote_hash == local_hash_path.read_text().strip():
        if verbose: print(f"Remote already up to date (hash {remote_hash}) -- skipping scp to {remote_host}:{remote_path}")
        return

    with Timing("scp: ", enter=f"Sending {local_path} to {remote_host}:{remote_path} ...", enabled=verbose):
        # inherit stdout/stderr (not captured) so scp's own progress meter streams live; scp -r
        # can't create a missing remote parent dir, so mkdir -p it over ssh first. hash.txt is
        # sent last so a remote hash.txt only appears once the transfer actually completed.
        subprocess.run(["ssh", remote_host, f"mkdir -p {remote_parent}"], check=True)
        subprocess.run(["scp", "-r", str(local_path), f"{remote_host}:{remote_path}"], check=True)
        subprocess.run(["scp", str(local_hash_path), f"{remote_host}:{remote_parent}/hash.txt"], check=True)

def parse_args():
    p = argparse.ArgumentParser(description="Post-process raw samples into an MDS dataset.")
    p.add_argument("base_sample_dir",       type=Path)
    p.add_argument("base_dataset_dir",      type=Path, nargs="?", default=Path(r"D:\eturok\datasets"), help="Parent dir for versioned dataset dirs.")
    p.add_argument("--dataset-name",        type=str, default=None, help="If not given, the versioned dir is just the number (e.g. '000') with no name suffix.")
    p.add_argument("--out-h",               type=int, default=20)
    p.add_argument("--out-w",               type=int, default=40)
    p.add_argument("--signal-mode",         type=str, default="magnitude", choices=["magnitude", "complex", "mag_phase"])
    p.add_argument("--normalize-mode",      type=str, default="std-sample", choices=["std-sample", "z-sample"])
    p.add_argument("--patch-size",          type=int, default=256)
    p.add_argument("--object-scale",        type=float, default=1, help="Linear factor to blow up each mask object about its own center of mass before downsampling (area grows ~object_scale**2). 1 = no-op.")
    p.add_argument("--empty-diff",          action="store_true", help="Per speaker, average the signaled fft of all empty-box (n_objects=0) samples and subtract that speaker's average from each of its samples before normalization.")
    p.add_argument("--force",               action="store_true")
    p.add_argument("--verbose",             type=int, default=1)
    p.add_argument("--no-save",             dest="do_save", action="store_false")
    p.add_argument("--no-scp",              dest="do_scp", action="store_false", help="Skip scp-ing the mds dir to the remote machine after post-processing.")
    p.add_argument("--remote-host",         type=str, default=REMOTE_HOST)
    p.add_argument("--remote-base-dir",     type=str, default=REMOTE_BASE_DIR)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mds_path = post_process(args.base_sample_dir, args.base_dataset_dir, args.dataset_name, args.out_h, args.out_w, args.signal_mode, args.normalize_mode, args.patch_size, object_scale=args.object_scale, empty_diff=args.empty_diff, force=args.force, verbose=args.verbose, do_save=args.do_save)
    n_samples = len((mds_path / "metadata.jsonl").read_text().strip().splitlines())
    print(f"MDS written to {mds_path} ({n_samples} samples)")

    try:
        if args.do_scp: scp_to_remote(mds_path, args.remote_host, args.remote_base_dir, args.verbose)
    finally:
        samples_dir = mds_path.parent / "samples"
        n_sample_dirs = sum(1 for p in samples_dir.iterdir() if p.is_dir())
        print(f"mds:     {n_samples} samples, {human_size(dir_size(mds_path))}, {mds_path.resolve()}")
        print(f"samples: {n_sample_dirs} samples, {human_size(dir_size(samples_dir))} (no symlinks) / {human_size(dir_size(samples_dir, follow_symlinks=True))} (with symlinks), {samples_dir.resolve()}")


