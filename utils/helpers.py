
import bz2, contextlib, functools, logging, os, sys, threading, time, json, shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import psutil
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Audio
from scipy.io.wavfile import write as wav_write, read as wav_read


#***** logging *****

logger = logging.getLogger('good_vibrations')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _console = logging.StreamHandler(sys.stdout)
    _console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_console)

def setup_logger(experiment_dir=None, verbose:int=1, do_save:bool=True):
    # console shows what verbose asks for; logs.md always captures everything down to DEBUG
    logger.setLevel(logging.DEBUG)
    for h in list(logger.handlers): logger.removeHandler(h); h.close()
    console = logging.StreamHandler(sys.stdout)
    # verbose 0/1: no text in the notebook (logs.md still gets everything); verbose>=2: text in the notebook too
    console.setLevel(logging.DEBUG if verbose >= 2 else logging.WARNING)
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)
    if experiment_dir is not None and do_save:
        log_path = Path(experiment_dir) / 'logs.md'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_path.exists(): log_path.write_text('# Experiment logs\n\n', encoding='utf-8')
        file = logging.FileHandler(log_path, encoding='utf-8')
        file.setLevel(logging.DEBUG)
        file.setFormatter(logging.Formatter('- `%(asctime)s` **%(levelname)s** %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file)
    return logger

#***** timing *****

class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", enter="", on_exit=None, enabled=True): self.prefix, self.enter, self.on_exit, self.enabled = prefix, enter, on_exit, enabled
    def _log(self, msg): (logger.info if self.enabled else logger.debug)(msg)
    def __enter__(self):
        self.st = time.perf_counter_ns()
        if self.enter: self._log(self.enter)
    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        self._log(f"{self.prefix}{self.et*1e-6:6.2f} ms" + (self.on_exit(self.et) if self.on_exit else ""))

#***** resource usage *****

def print_system_usage(path, label="", verbose=1):
    total, used, _ = shutil.disk_usage(path)
    ram = psutil.virtual_memory()
    GB = 2**30
    prefix = f"{label} " if label else ""
    logger.debug(f"{prefix}disk: {used/GB:.2f}/{total/GB:.2f} GB used | RAM: {ram.used/GB:.2f}/{ram.total/GB:.2f} GB used | threads: {threading.active_count()}/{psutil.Process().num_threads()} active")


def human_size(n:int) -> str:
    return next((f"{n / d:.2f} {suffix}" for d, suffix in [(1 << 30, "GB"), (1 << 20, "MB"), (1 << 10, "KB")] if n >= d), f"{n} B")

def dir_size(path:Path, follow_symlinks:bool=False) -> int:
    # follow_symlinks=False (default): lstat, so symlinked files count their link size, not the
    # target's -- reports actual disk usage inside `path`, not the size of data it merely points
    # at elsewhere. follow_symlinks=True: stat, so symlinked files count their target's real
    # size -- reports the total logical size of the data `path` gives you access to.
    if follow_symlinks:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return sum(p.lstat().st_size for p in path.rglob("*") if p.is_file() or p.is_symlink())
