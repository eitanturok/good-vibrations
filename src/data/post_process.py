import argparse, json, hashlib, shutil, os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from streaming import MDSWriter

from utils.io_utils import load, save, append, symlink, copy, Timing, logger, human_size, dir_size
from utils.metrics import center_of_mass

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

#***** 3 post-process a single sample *****

def post_process_sample(sample_dir:Path, out_h:int, out_w:int, signal_mode:str, normalize_mode:str, patch_size:int, verbose:int, do_save:bool, denotch_fn=None):
    sample_id = sample_dir.name
    is_empty_box = bool({k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}.get("is_empty_box", False))

    # post process overhead image (downsample segment mask, compute new center of mass)
    with Timing(f"[sample {sample_id}] post-process overhead image: ", enabled=verbose >= 2):
        segment_mask = load(sample_dir / "image/02_smask.png")

        # downsample segment mask to out_h x out_w
        downsampled_segment_mask = downsample(segment_mask, out_h, out_w)
        save(downsampled_segment_mask, sample_dir / "image/06_downsampled_smask.png", do_save)
        save(np.array(downsampled_segment_mask, dtype=np.float32) / 255.0, sample_dir / "image/07_downsampled_smask.npy", do_save)
        append({'downsample': datetime.now(timezone.utc).isoformat()}, sample_dir / 'times.jsonl', do_save)

        # compute new center of mass for the downsampled segment mask
        downsampled_com = (-1, -1) if is_empty_box else center_of_mass(downsampled_segment_mask)
        append([{"downsampled_com": downsampled_com}], sample_dir / "metadata.jsonl", do_save)
        append({'process_overhead/com': datetime.now(timezone.utc).isoformat()}, sample_dir / 'times.jsonl', do_save)
        logger.debug(f"[sample {sample_id}] {downsampled_com=}")

    # post process vibrations (extract signal, normalize signal, tokenize)
    with Timing(f"[sample {sample_id}] post-process vibrations: ", enabled=verbose >= 2):
        fft_npz = load(sample_dir / 'vibration/03_fft.npz')
        freqs, fft_raw = fft_npz['freqs'], fft_npz['fft']
        if denotch_fn is not None:
            fft_raw = denotch_fn(fft_raw, freqs)
            save({'fft': fft_raw, 'freqs': freqs, 'n_samples': fft_npz['n_samples']}, sample_dir / 'vibration/03_fft.npz', do_save)
            append({"fft_denotched": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)
        logger.debug(f'[sample {sample_id}] {fft_raw.shape=}=(batch, lasers, _freqs, x/y)')

        # extract signal from fft
        fft_signaled = extract_signal(fft_raw, signal_mode).astype(np.float32)  # (B,L,F_,C) -> (B,L,F,C)
        logger.debug(f'[sample {sample_id}] {fft_signaled.shape=}=(batch, lasers, _freqs, x/y)')
        save(fft_signaled, sample_dir / 'vibration/06_signaled_fft.npy', do_save)
        append({"fft_signaled": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

        # normalize fft
        fft_normalized = normalize_fft(fft_signaled, normalize_mode, verbose)   # (B,L,F,C) -> (B,L,F,C)
        logger.debug(f'[sample {sample_id}] {fft_normalized.shape=}=(batch, lasers, _freqs, x/y)')
        save(fft_normalized, sample_dir / 'vibration/07_normalized_fft.npy', do_save)
        append({"fft_normalized": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

        # tokenize fft
        fft_tokenized = tokenize(fft_normalized, patch_size)                    # (B,L,F,C) -> (B,L,P,PS,C)
        logger.debug(f'[sample {sample_id}] {fft_tokenized.shape=}=(batch, lasers, num_patches, patch_size, x/y)')
        save(fft_tokenized, sample_dir / 'vibration/08_tokenized_fft.npy', do_save)
        append({"fft_tokenized": datetime.now(timezone.utc).isoformat()}, sample_dir / "times.jsonl", do_save)

    # symlink X.npy, y.npy for model input, output
    symlink(sample_dir / 'vibration/08_tokenized_fft.npy', sample_dir / 'X.npy', do_save)
    symlink(sample_dir / 'image/07_downsampled_smask.npy', sample_dir / 'y.npy', do_save)

    n_lasers, n_freqs = fft_raw.shape[1], fft_raw.shape[2]
    append(dict(out_h=out_h, out_w=out_w, signal_mode=signal_mode, normalize_mode=normalize_mode, patch_size=patch_size, n_lasers=n_lasers, n_freqs=n_freqs), sample_dir / "metadata.jsonl", do_save)
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

    # read n_lasers, n_freqs from the first sample's metadata
    n_lasers, n_freqs = rows[0][1]["n_lasers"], rows[0][1]["n_freqs"]
    x_shape, y_shape = (n_lasers, n_freqs // patch_size, patch_size, 2), (out_h, out_w)

    index_rows = []
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(dataset_dir)

    # write to MSD format
    try:
      with MDSWriter(out="mds", columns=MDS_COLUMNS, exist_ok=False) as writer:
        for i, (sample_dir, meta) in enumerate(rows):
            X = np.load(sample_dir / "X.npy").astype(np.float32)   # (1, L, P, PS, C)
            y = np.load(sample_dir / "y.npy").astype(np.float32)   # (out_h, out_w)
            # model.py's VibrationTransformer.forward expects (B,L,P,C,PS) per-sample but the leading 1 here is just X.npy's on-disk batch dim from post-processing so we remove it
            X = np.squeeze(X, axis=0) if X.ndim == 5 and X.shape[0] == 1 else X
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

    # dataset-level sidecar for filtering
    lines = "\n".join([json.dumps(r) for r in index_rows])
    (mds_dir / "metadata.jsonl").write_text(lines)
    if verbose: print(f"Wrote {len(rows)} samples to {mds_dir=}")
    return mds_dir


#***** 5 post-process all samples in a base sample directory *****

def resolve_dataset_dir(base_dataset_dir:Path, dataset_name:str|None, key:str) -> Path:
    # A dataset name may get reused across runs with different params/data -- version it as
    # "NNN" or "NNN-<name>" under base_dataset_dir so different params never silently
    # collide/overwrite. Reuse an existing numbered dir if its hash.txt already matches this
    # exact param/data combination, else allocate the next increasing number.
    suffix = f"-{dataset_name}" if dataset_name else ""
    existing = sorted(base_dataset_dir.glob(f"[0-9][0-9][0-9]{suffix}")) if base_dataset_dir.exists() else []
    for p in existing:
        hash_path = p / "hash.txt"
        if hash_path.exists() and hash_path.read_text().strip() == key: return p
    next_n = max((int(p.name[:3]) for p in existing), default=-1) + 1
    return base_dataset_dir / f"{next_n:03d}{suffix}"

def post_process(base_sample_dir:Path, base_dataset_dir:Path, dataset_name:str|None, out_h:int, out_w:int, signal_mode:str, normalize_mode:str, patch_size:int, force:bool=False, verbose:int=1, do_save:bool=True, denotch_fn=None):

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
        key = hashlib.sha1(json.dumps(rows | MDS_COLUMNS, sort_keys=True, default=str).encode()).hexdigest()[:16]

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

    # post process each sample
    with Timing("Post-processing: ", enter=f"Post-processing {len(sample_ids)} samples ...", enabled=verbose):
        for sample_id in sample_ids:
            sample_dir = samples_dir / sample_id
            post_process_sample(sample_dir, out_h, out_w, signal_mode, normalize_mode, patch_size, verbose, do_save, denotch_fn)
            rows[sample_id] = {k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}

    # convert to MDS
    sample_rows = [(samples_dir / sample_id, meta) for sample_id, meta in rows.items()]
    with Timing("Writing MDS: ", enabled=verbose):
        mds_path = convert_to_mds(dataset_dir, sample_rows, patch_size, out_h, out_w, verbose)
    if do_save: hash_path.write_text(key)
    return mds_path


#***** 6 CLI *****

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
    p.add_argument("--force",               action="store_true")
    p.add_argument("--verbose",             type=int, default=1)
    p.add_argument("--no-save",             dest="do_save", action="store_false")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mds_path = post_process(args.base_sample_dir, args.base_dataset_dir, args.dataset_name, args.out_h, args.out_w, args.signal_mode, args.normalize_mode, args.patch_size, force=args.force, verbose=args.verbose, do_save=args.do_save)
    n_samples = len((mds_path / "metadata.jsonl").read_text().strip().splitlines())
    print(f"MDS written to {mds_path} ({n_samples} samples)")

    samples_dir = mds_path.parent / "samples"
    n_sample_dirs = sum(1 for p in samples_dir.iterdir() if p.is_dir())
    print(f"mds:     {n_samples} samples, {human_size(dir_size(mds_path))}, {mds_path.resolve()}")
    print(f"samples: {n_sample_dirs} samples, {human_size(dir_size(samples_dir))}, {samples_dir.resolve()}")


