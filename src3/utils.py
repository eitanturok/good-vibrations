
import contextlib
import time
from pathlib import Path
import json

import numpy as np
from PIL import Image
from IPython.display import Audio
from scipy.io.wavfile import write as wav_write, read as wav_read


class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
    def __enter__(self): self.st = time.perf_counter_ns()
    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms" + (self.on_exit(self.et) if self.on_exit else ""))

def save(x, path:Path):
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
                json.dump(i, f)
                f.write('\n')
    else: raise ValueError(f"Unsupported data type: {type(x)}")

def load(path:Path, keys:list[str]|str|None=None):
    if path.suffix == '.wav':
        sample_rate, samples = wav_read(path)
        return Audio(samples, rate=sample_rate)
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

def append(x, path:Path):
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
                json.dump(i, f)
                f.write('\n')
    else: raise ValueError(f"Unsupported file type: {path.suffix}")

def symlink(src:Path, dst:Path):
    if not src.exists(): raise FileNotFoundError(f"Source file does not exist: {src}")
    if not dst.exists(): dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() and dst.resolve() == src.resolve(): return
    if dst.is_symlink(): dst.unlink()
    # must symlink to absolute path, not relative path
    dst.symlink_to(src.resolve())
