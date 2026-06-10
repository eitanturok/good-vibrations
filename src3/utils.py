
import contextlib
import time
from pathlib import Path
import json

import numpy as np
from PIL import Image


class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
    def __enter__(self): self.st = time.perf_counter_ns()
    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms" + (self.on_exit(self.et) if self.on_exit else ""))

def save(x, path:Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(x, np.ndarray):
        with open(path, 'wb') as f: np.save(f, x)
    elif isinstance(x, Image.Image): x.save(path)
    elif isinstance(x, (dict, list)):
        if isinstance(x, dict): x = [x]
        with open(path, 'w') as f:
            for i in x:
                json.dump(i, f)
                f.write('\n')
    else: raise ValueError(f"Unsupported data type: {type(x)}")

def load(path:Path):
    if path.suffix == '.npy':
        with open(path, 'rb') as f: return np.load(f)
    elif path.suffix in ['.jpg', '.jpeg', '.png']: return Image.open(path)
    elif path.suffix == '.jsonl':
        with open(path) as f: return [json.loads(line) for line in f]
    else: raise ValueError(f"Unsupported file type: {path.suffix}")

def append(x, path:Path):
    if path.suffix == '.npy':
        if path.exists(): x = np.concatenate([np.load(path), x], axis=0)
        np.save(path, x)
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
    dst.symlink_to(src)
