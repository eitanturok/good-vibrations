
import bz2, contextlib, functools, logging, os, sys, threading, time, json, shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import psutil
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Audio
from scipy.io.wavfile import write as wav_write, read as wav_read

#***** file helpers *****

def paths_to_str(x):
    if isinstance(x, os.PathLike): return str(x)
    if isinstance(x, dict): return {paths_to_str(k): paths_to_str(v) for k, v in x.items()}
    if isinstance(x, list): return [paths_to_str(i) for i in x]
    if isinstance(x, tuple): return tuple(paths_to_str(i) for i in x)
    return x

def save(x, path:Path, enabled:bool=True):
    if not enabled: return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.wav':
        if not (isinstance(x, tuple) and len(x) == 2): raise ValueError("save to .wav requires a tuple of (samples, sample_rate)")
        samples, sample_rate = x
        wav_write(path, sample_rate, samples)
    elif path.suffix == '.npz':
        if not isinstance(x, dict): raise ValueError("save to .npz requires a dict of arrays")
        np.savez(path, **x)
    elif isinstance(x, np.ndarray):
        with open(path, 'wb') as f: np.save(f, x)
    elif isinstance(x, Image.Image): x.save(path)
    elif isinstance(x, (dict, list)):
        if isinstance(x, dict): x = [x]
        with open(path, 'w') as f:
            for i in x:
                json.dump(paths_to_str(i), f)
                f.write('\n')
    else: raise ValueError(f"Unsupported data type: {type(x)}")

def load(path:Path|str, keys:list[str]|str|None=None, enabled:bool=True):
    if not enabled: return
    if isinstance(path, str): path = Path(path)
    if path.suffix == '.wav':
        sample_rate, samples = wav_read(path)
        return samples, sample_rate
    elif path.suffix == '.npy':
        with open(path, 'rb') as f: return np.load(f)
    elif path.suffix == '.npz':
        d = np.load(path, allow_pickle=True)
        if keys is None: return dict(d)
        if isinstance(keys, str): return d[keys]
        return tuple(d[k] for k in keys)
    elif path.suffix in ['.jpg', '.jpeg', '.png']: return Image.open(path)
    elif path.suffix == '.jsonl':
        with open(path) as f: return [json.loads(line) for line in f]
    else: raise ValueError(f"Unsupported file type: {path.suffix}")

def append(x, path:Path|str, enabled:bool=True):
    if not enabled: return
    if isinstance(path, str): path = Path(path)
    # create it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.wav':
        raise NotImplementedError("Appending to .wav files is not supported.")
    elif path.suffix == '.npy':
        if path.exists(): x = np.concatenate([np.load(path), x], axis=0)
        np.save(path, x)
    elif path.suffix == '.npz':
        if not isinstance(x, dict): raise ValueError("append to .npz requires a dict of arrays")
        existing = dict(np.load(path, allow_pickle=True)) if path.exists() else {}
        existing.update(x)
        np.savez(path, **existing)
    elif path.suffix in ['.jpg', '.jpeg', '.png']:
        raise NotImplementedError("Appending to images is not supported.")
    elif path.suffix == '.jsonl':
        if isinstance(x, dict): x = [x]
        with open(path, 'a') as f:
            for i in x:
                json.dump(paths_to_str(i), f)
                f.write('\n')
    else: raise ValueError(f"Unsupported file type: {path.suffix}")

def copy(src:Path|str, dst:Path|str, enabled:bool=True):
    if not enabled: return
    if isinstance(src, str): src = Path(src)
    if isinstance(dst, str): dst = Path(dst)
    if not src.exists(): raise FileNotFoundError(f"Source path does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir(): shutil.copytree(src, dst, dirs_exist_ok=True)
    else: shutil.copy2(src, dst)

def symlink(src:Path|str, dst:Path|str, enabled:bool=True):
    if not enabled: return
    if isinstance(src, str): src = Path(src)
    if isinstance(dst, str): dst = Path(dst)
    if not src.exists(): raise FileNotFoundError(f"Source file does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and dst.resolve() == src.resolve(): return
    if dst.exists() or dst.is_symlink(): dst.unlink()
    dst.symlink_to(src.resolve())

#***** compress files ******

class Bz2Compressor:
    """Lossless numpy array compressor using bz2 level 9 -- the pick that sits on
    the time/ratio Pareto frontier for this project's raw vibration arrays
    (~4.2x ratio in under a second per 30 MB, see src/data/compress_pareto.png).
    lzma gets a better ratio (~6.8x) but costs ~30 min per 2.7 GB array, too
    slow for per-sample pipeline use."""

    LEVEL = 9

    def compress(self, arr:np.ndarray) -> bytes:
        arr = np.ascontiguousarray(arr)
        header = json.dumps({"shape": arr.shape, "dtype": str(arr.dtype)}).encode()
        return len(header).to_bytes(4, "big") + header + bz2.compress(arr.tobytes(), self.LEVEL)

    def decompress(self, blob:bytes) -> np.ndarray:
        header_len = int.from_bytes(blob[:4], "big")
        meta = json.loads(blob[4:4 + header_len])
        raw = bz2.decompress(blob[4 + header_len:])
        return np.frombuffer(raw, dtype=meta["dtype"]).reshape(meta["shape"])

#***** copy to sample helpers *****

SHARED_FILES = ["00_raw.png", "01_cropped.png", "02_smask.png", "03_smask.npy", "04_overhead_masked.png", "05_overhead_scored.png"]
SHARED_DIRS = ["smasks"]
COPIED_FILES = ["times.jsonl", "metadata.jsonl"]

def copy_to_sample(sample_dir:Path, output_dir:Path, audio_dir:Path, speaker:int, do_save:bool=True):
    sample_id = sample_dir.name

    # symlink the shared+copy artifacts from output_dir to the current sample_dir
    assert all((output_dir / a).exists() for a in SHARED_FILES+SHARED_DIRS+COPIED_FILES), f"[sample {sample_id}] Missing shared or copied artifact"
    for artifact in SHARED_FILES: symlink(output_dir / artifact, sample_dir / f"image/{artifact}", do_save)
    for d in SHARED_DIRS: symlink(output_dir / d, sample_dir / f"image/{d}", do_save)
    for artifact in COPIED_FILES: copy(output_dir / artifact, sample_dir / artifact, do_save)

    # symlink the audio file from audio_dir to the current sample_dir
    symlink(audio_dir / 'audio.wav', sample_dir / "audio.wav", do_save)
    append([{'audio_dir': sample_dir / "audio.wav"}, {'speaker': speaker}], sample_dir / "metadata.jsonl", do_save)
    append({'sample_id': sample_id, "sample_dir": sample_dir}, audio_dir.parent / "samples.jsonl", do_save)

    # the output_dir record all the samples that have been generated from it
    append({"sample_id": sample_id, "sample_dir": sample_dir, "time": datetime.now(timezone.utc).isoformat()}, output_dir / "samples.jsonl", do_save)

    # this sample should record it's own sample_id, sample_dir
    append([{"sample_id": sample_id}, {"sample_dir": sample_dir}], sample_dir / "metadata.jsonl", do_save)

#***** modal helpers *****

SYMLINKS = [("recovered_audio.wav", "vibration/04_recovered_audio.wav")]

def retry(attempts=4, delay=2.0, backoff=2.0, exceptions=(Exception,)):
    """Retry a function on transient failures (e.g. DNS/connection errors under load)
    with exponential backoff. Re-raises the last exception if all attempts fail."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            wait = delay
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == attempts: raise
                    print(f"[retry] {fn.__name__} failed (attempt {attempt}/{attempts}): {type(e).__name__}: {e}. retrying in {wait:.1f}s")
                    time.sleep(wait)
                    wait *= backoff
        return wrapper
    return decorator

@retry()
def modal_upload(volume, sample_dir, verbose:int=1):
    sample_id = sample_dir.name
    raw_path = sample_dir / 'vibration/00_raw_vibrations.npy'
    nbytes = raw_path.stat().st_size + (sample_dir / 'metadata.jsonl').stat().st_size
    t0 = time.perf_counter()
    with volume.batch_upload(force=True) as batch:
            batch.put_file(raw_path, f"{sample_id}/vibration/00_raw_vibrations.npy")
            batch.put_file(sample_dir / 'metadata.jsonl', f"{sample_id}/metadata.jsonl")
    dt = time.perf_counter() - t0
    if verbose >= 1:
        MB = nbytes / 2**20
        print(f"[sample {sample_id}] upload throughput: {MB:.1f} MB in {dt:.2f}s = {MB/dt:.1f} MB/s ({8*MB/dt:.1f} Mbps)")

def measure_upload_capacity():
    """Actively measure real internet UPLOAD capacity (Mbps) by pushing data to a
    speedtest server. Takes ~15s. Requires `pip install speedtest-cli`."""
    import speedtest
    st = speedtest.Speedtest()
    st.get_best_server()
    return st.upload() / 1e6  # bits/sec -> Mbps

@retry()
def modal_download(volume, remote_path, local_path):
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, 'wb') as f:
        for chunk in volume.read_file(str(remote_path)): f.write(chunk)

def fix_symlinks(sample_dir):
    for dst_rel, src_rel in SYMLINKS:
        dst, src = sample_dir / dst_rel, sample_dir / src_rel
        if dst.exists() or dst.is_symlink(): dst.unlink()
        dst.symlink_to(src.relative_to(dst.parent))
