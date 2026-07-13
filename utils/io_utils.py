
import contextlib, functools, logging, os, sys, threading, time, json, shutil
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

#***** preview helpers *****

def _draw_image(ax, image):
    ax.imshow(np.array(image))
    ax.axis('off')

def _draw_vibration_image(ax, frame_recording):
    MINMAX = (np.percentile(frame_recording[-100:],10), np.percentile(frame_recording[-100:],90))
    ax.imshow(frame_recording[10],vmin=MINMAX[0],vmax=MINMAX[1])

def preview_image(image:np.ndarray):
    fig, ax = plt.subplots(figsize=(12, 6))
    _draw_image(ax, image)
    plt.show()

def preview_vibration_image(frame_recording):
    fig, ax = plt.subplots(figsize=(12, 6))
    _draw_vibration_image(ax, frame_recording)
    plt.show()


def _stretch_contrast(image,minmax=None):
    if minmax is None:
        min_val = np.min(image)
        max_val = np.max(image)
    else:
        min_val,max_val = minmax
    stretched = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return stretched

def preview_vibration_video(frame_recording):
    import cv2, sys, importlib.util, pathlib
    _repo_root = str(pathlib.Path(__file__).parent.parent)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from utils.opencv_video_utils import videoPlayer

    class videoPlayerv2(videoPlayer):
        def additional_loop_control(self,key):
            if key==42:
                self.data_counter=(self.data_counter+100) % self.N_frames
            if key==47:
                self.data_counter=(self.data_counter-100) % self.N_frames

    def get_frameshow(data_counter):
        frame      = frame_recording[data_counter].copy()
        frame_show = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame_show = _stretch_contrast(frame_show)
        return frame_show

    show_frame_number = 1
    N_frames           = frame_recording.shape[0]
    resize_factor      = 2
    get_frame_func     = get_frameshow
    video_player       = videoPlayerv2(get_frame_func,N_frames,resize_factor=resize_factor)
    video_player.play_video(move_window=0,show_frame_number=show_frame_number)

def get_box_coverage_key(metadata, mask): return (metadata['box'], metadata['object'], metadata['n_objects'], mask.shape)

def _draw_box_coverage(ax, box_coverage, sample_dir):
    metadata = {'sample_id': ''} if sample_dir is None else {k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}
    mask = np.array(load(sample_dir / "image/02_smask.png"))
    key = get_box_coverage_key(metadata, mask)
    box, obj, n_objects, shape = key
    im = ax.imshow(box_coverage[key]['mask'], cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(title=f"Box Coverage\n{n_objects} {obj} in {box} box ({shape[1]}×{shape[0]}, {len(box_coverage[key]['sample_ids'])} samples)", xlabel='x (downsampled pixel space)', ylabel='y (downsampled pixel space)')

def preview_box_coverage(box_coverage, sample_dir):
    fig, ax = plt.subplots()
    _draw_box_coverage(ax, box_coverage, sample_dir)
    plt.show()

def preview_sample_row(sample_overhead, raw_vibrations, box_coverage, sample_dir, header=''):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _draw_image(axes[0], sample_overhead)
    axes[0].set_title('Overhead')
    _draw_vibration_image(axes[1], raw_vibrations)
    axes[1].set_title('Speckles')
    _draw_box_coverage(axes[2], box_coverage, sample_dir)
    if header: fig.suptitle(header, fontsize=14, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout()
    plt.show()

def preview_audio(samples, sample_rate): return Audio(samples, rate=sample_rate)

def plot_position(result: dict, image, objects:list[str]|None=None, header:str='', alpha:float=0.45) -> None:
    """Two-panel figure for one overhead position, styled like preview_sample_row.
    Left: masks + boxes + confidence labels (plot_smask). Right: the same
    masks, same colors, no image/boxes/overlap -- what the masks actually
    look like. header is shown as a bold suptitle (e.g. the output/sample id
    range). Shown for every new position captured during an experiment."""
    from src.data.segment import plot_smask, label_map_image

    objects = objects if objects is not None else result["names"]
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    left = plot_smask(result, image, objects, alpha=alpha, show=False)
    _draw_image(axes[0], left)
    axes[0].set_title("masks + boxes + confidence")

    _draw_image(axes[1], label_map_image(result["masks"]))
    axes[1].set_title("masks only")

    if header: fig.suptitle(header, fontsize=14, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout()
    plt.show()

def preview(obj, mode):
    if mode == 'image': return preview_image(obj)
    if mode == 'vibration_image': return preview_vibration_image(obj)
    if mode == 'vibration_video': return preview_vibration_video(obj)
    if mode == 'box_coverage': return preview_box_coverage(*obj)
    if mode == 'sample_row': return preview_sample_row(*obj)
    if mode == 'smask': return plot_position(*obj)
    if mode == 'audio': return preview_audio(*obj)
    else: raise ValueError(f'{mode=} not recognized')

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
