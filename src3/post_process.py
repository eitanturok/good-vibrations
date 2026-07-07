import json, hashlib, shutil, os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from streaming import MDSWriter

from io_utils import load, save, append, symlink, Timing

#***** 1 post-process image (downsample overhead image) *****

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
        if verbose: print(f'Normalize {normalize_mode}\n{std.shape=}\n{std.squeeze()=}')
        return x / std
    if normalize_mode == 'z-sample':
        mean = x64.mean(axis=(1, 2, 3), keepdims=True).astype(np.float32)
        std = np.maximum(x64.std(axis=(1, 2, 3), ddof=1, keepdims=True), 1e-8).astype(np.float32)
        if verbose: print(f'Normalize {normalize_mode}\n{mean.shape=}\t{std.shape=}\n{mean.squeeze()=}\n{std.squeeze()=}')
        return (x - mean) / std
    raise ValueError(f"Unknown normalize mode: {normalize_mode}")

def tokenize(x: np.ndarray, patch_size:int):
    # Note: unfold drops entries that do not fully fit into patch_size
    if patch_size <= 0: return x
    B, L, F, C = x.shape
    P = F // patch_size
    return x[:, :, :P * patch_size, :].reshape(B, L, P, patch_size, C)  # (B,L,P,PS,C)

#***** 3 post-process a single sample *****

def post_process_sample(sample_dir:Path, is_empty_box:bool, out_h:int, out_w:int, signal_mode:str, normalize_mode:str, patch_size:int, verbose:int, do_save:bool):
    sample_id = sample_dir.name

    # post process overhead image (downsample segment mask, compute new center of mass)
    with Timing(f"[sample {sample_id}] post-process overhead image: ", enabled=verbose >= 2):
        segment_mask = load(sample_dir / "outputs/02_segment_mask.png")

        # downsample segment mask to out_h x out_w
        downsampled_segment_mask = downsample(segment_mask, out_h, out_w)
        save(downsampled_segment_mask, sample_dir / "outputs/06_downsampled_smask.png", do_save)
        save(np.array(downsampled_segment_mask, dtype=np.float32) / 255.0, sample_dir / "outputs/07_downsampled_smask.npy", do_save)
        append({'downsample': datetime.now(timezone.utc).isoformat()}, sample_dir / 'times.jsonl', do_save)

        # compute new center of mass for the downsampled segment mask
        downsampled_com = (-1, -1) if is_empty_box else center_of_mass(downsampled_segment_mask)
        append([{"downsampled_com": downsampled_com}], sample_dir / "metadata.jsonl", do_save)
        append({'process_overhead/com': datetime.now(timezone.utc).isoformat()}, sample_dir / 'times.jsonl', do_save)
        if verbose >= 2: print(f"[sample {sample_id}] {downsampled_com=}")

    # post process vibrations (extract signal, normalize signal, tokenize)
    with Timing(f"[sample {sample_id}] post-process vibrations: ", enabled=verbose >= 2):
        fft_raw = load(sample_dir / 'inputs/03_fft.npz')
        if verbose >= 2: print(f'[sample {sample_id}] {fft_raw.shape=}=(batch, lasers, _freqs, x/y)')

        # extract signal from fft
        fft_signaled = extract_signal(fft_raw, signal_mode).astype(np.float32)  # (B,L,F_,C) -> (B,L,F,C)
        if verbose >= 2: print(f'[sample {sample_id}] {fft_signaled.shape=}=(batch, lasers, _freqs, x/y)')
        save(fft_signaled, sample_dir / 'inputs/05_fft_signaled.npy', do_save)
        append({"fft_signaled": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

        # normalize fft
        fft_normalized = normalize_fft(fft_signaled, normalize_mode, verbose)   # (B,L,F,C) -> (B,L,F,C)
        if verbose >= 2: print(f'[sample {sample_id}] {fft_normalized.shape=}=(batch, lasers, _freqs, x/y)')
        save(fft_normalized, sample_dir / 'inputs/06_fft_normalized.npy', do_save)
        append({"fft_normalized": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

        # tokenize fft
        fft_tokenized = tokenize(fft_normalized, patch_size)                    # (B,L,F,C) -> (B,L,P,PS,C)
        if verbose >= 2: print(f'[sample {sample_id}] {fft_tokenized.shape=}=(batch, lasers, num_patches, patch_size, x/y)')
        save(fft_tokenized, sample_dir / 'inputs/07_fft_tokenized.npy', do_save)
        append({"fft_tokenized": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

    # symlink X.npy, y.npy for model input, output
    symlink(sample_dir / 'outputs/05_processed_fft.npy', sample_dir / 'y.npy', do_save)
    symlink(sample_dir / 'inputs/07_fft_tokenized.npy', sample_dir / 'X.npy', do_save)

    n_lasers, n_freqs = fft_raw.shape[1], fft_raw.shape[2]
    append(dict(out_h=out_h, out_w=out_w, signal_mode=signal_mode, normalize_mode=normalize_mode, patch_size=patch_size, n_lasers=n_lasers, n_freqs=n_freqs), sample_dir / "metadata.jsonl", do_save)
    append({"post_process": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)


#***** 4 convert to mds *****

MDS_COLUMNS = {"X": "ndarray:float32", "y": "ndarray:float32",
               "sample_id": "int", "output_id": "str",
               "n_objects": "int", "speaker": "int", "box": "str", "is_empty_box": "int", "object": "str",
               "downsampled_com_x": "float64", "downsampled_com_y": "float64",
}

def convert_to_mds(mds_path:Path, rows:list, key, x_shape, y_shape, verbose:int):
    # write shards + a lightweight metadata sidecar (index.jsonl) for loader-side filtering.
    # Streaming urlparses `out`, so an absolute Windows path (e.g. "D:/...") is misread as a
    # cloud scheme "d:". Chdir to the parent and pass the relative dir name -> empty url scheme.
    if verbose: print(f"Writing MDS to {mds_path} ...")

    index_rows = []
    mds_path.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(mds_path)

    # write to MSD format
    try:
      with MDSWriter(out=key, columns=MDS_COLUMNS, exist_ok=False) as writer:
        for i, (sample_dir, meta) in enumerate(rows):
            X = np.load(sample_dir / "X.npy").astype(np.float32)   # (1, L, P, PS, C)
            y = np.load(sample_dir / "y.npy").astype(np.float32)   # (out_h, out_w)
            # X = np.squeeze(X, axis=0) if X.ndim == 5 and X.shape[0] == 1 else X # todo: do I need this?
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

    # dataset-level sidecar
    lines = "\n".join([json.dumps(r) for r in index_rows])
    (mds_path / "dataset.jsonl").write_text(lines)
    if verbose: print(f"Wrote {len(rows)} samples to {mds_path=}")
    return mds_path


#***** 5 post-process all samples in a base sample directory *****

IMAGE_FILES = ["00_raw.png", "01_cropped.png", "02_smask.png", "03_smask.npy", "04_overhead_with_smask.png", "05_overhead_with_smask_and_speaker.png", "06_downsampled_smask.png", "07_downsampled_smask.npy"]
VIBRATION_FILES = ["00_raw_vibrations.npy", "01_raw_shifts.npy", "02_clean_shifts.npy", "03_fft.npz", "05_fft_signaled.npy", "06_fft_normalized.npy", "07_fft_tokenized.npy"]
SAMPLE_FILES = ["X.npy", "y.npy", "metadata.jsonl", "times.jsonl", "audio.mp3", "recovered_audio.mp3", "overhead.png"]

def post_process(base_sample_dir:Path, mds_dir:Path, is_empty_box:bool, out_h:int, out_w:int, signal_mode:str, normalize_mode:str, patch_size:int, force:bool=False, verbose:int=1, do_save:bool=True):

    # collect complete samples (both X and y present) + their flattened metadata
    rows, skipped_ids = {}, []
    for sample_dir in sorted(base_sample_dir.glob("*")):
        files = [sample_dir / f].exists() for f in SAMPLE_FILES] + [sample_dir / "image" / f].exists() for f in IMAGE_FILES] + [sample_dir / "vibration" / f].exists() for f in VIBRATION_FILES]
        if not any([sample_dir / f].exists() for f in files]):
            skipped_ids.append(sample_dir.name)
            continue
        rows[sample_dir.name] = {k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}
    sample_ids = list(rows.keys())
    if verbose: print(f"Found {len(rows)} complete samples ({len(skipped_ids)} skipped)\nskipped ids: {skipped_ids}")

    # skip if we already post-processed + converted to MDS
    payload = {load(base_sample_dir / sample_id / "metadata.jsonl") | load(base_sample_dir / sample_id / "times.jsonl") for sample_id in sample_ids}
    key = hashlib.sha1(json.dumps(payload | MDS_COLUMNS).encode()).hexdigest()[:16]
    mds_path = mds_dir / key
    if mds_path.exists() and force:
        if verbose: print(f"--force: deleting cached MDS at {mds_path} and rebuilding")
        shutil.rmtree(mds_path)
    elif mds_path.exists() and (mds_path / "dataset.jsonl").exists():
        if verbose: print(f"Cache hit: reusing existing MDS at {mds_path}\nMDS: {mds_path} ({len(rows)} samples)")
        return mds_path

    # symlink base_sample_dir to MDS directory
    symlink(base_sample_dir, mds_path, do_save)

    # post process each sample
    for sample_dir in sorted(base_sample_dir.glob("sample_*")):
        if not sample_dir.is_dir(): continue
        post_process_sample(sample_dir, is_empty_box, out_h, out_w, signal_mode, normalize_mode, patch_size, verbose, do_save)

    # convert to MDS
    n_lasers, n_freqs = rows[sample_ids[0]]["n_lasers"], rows[sample_ids[0]]["n_freqs"]
    x_shape, y_shape = (1, n_lasers, n_freqs // patch_size, patch_size, 2), (out_h, out_w)
    mds_path = convert_to_mds(mds_path, rows, key, x_shape, y_shape, verbose)
    return mds_path


