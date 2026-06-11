
import contextlib
import time
from pathlib import Path
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Audio
from scipy.io.wavfile import write as wav_write, read as wav_read


class Timing(contextlib.ContextDecorator):
    def __init__(self, prefix="", enter="", on_exit=None, enabled=True): self.prefix, self.enter, self.on_exit, self.enabled = prefix, enter, on_exit, enabled
    def __enter__(self):
        self.st = time.perf_counter_ns()
        if self.enabled and self.enter: print(self.enter)
    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms" + (self.on_exit(self.et) if self.on_exit else ""))

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
                json.dump(i, f)
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
                json.dump(i, f)
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

def preview_image(image:np.ndarray):
    plt.figure(figsize=(12, 6))
    plt.imshow(np.array(image))
    plt.axis('off')
    plt.show()

def preview_vibration_image(frame_recording):
    plt.figure(figsize=(12,6))
    MINMAX = (np.percentile(frame_recording[-100:],10), np.percentile(frame_recording[-100:],90))
    plt.imshow(frame_recording[10],vmin=MINMAX[0],vmax=MINMAX[1])
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

def preview_box_coverage(box_coverage, sample_dir):
    metadata = {'sample_id': ''} if sample_dir is None else {k: v for d in load(sample_dir / "metadata.jsonl") for k, v in d.items()}
    mask = np.array(load(sample_dir / "outputs/02_segment_mask.png"))
    key = get_box_coverage_key(metadata, mask)
    box, obj, n_objects, shape = key
    fig, ax = plt.subplots()
    im = ax.imshow(box_coverage[key]['mask'], cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(title=f"Box Coverage\n{n_objects} {obj} in {box} box ({shape[1]}×{shape[0]}, {len(box_coverage[key]['sample_ids'])} samples)", xlabel='x (downsampled pixel space)', ylabel='y (downsampled pixel space)')
    plt.show()

def preview_audio(samples, sample_rate): return Audio(samples, rate=sample_rate)

def preview(obj, mode):
    if mode == 'image': return preview_image(obj)
    if mode == 'vibration_image': return preview_vibration_image(obj)
    if mode == 'vibration_video': return preview_vibration_video(obj)
    if mode == 'box_coverage': return preview_box_coverage(*obj)
    if mode == 'audio': return preview_audio(*obj)
    else: raise ValueError(f'{mode=} not recognized')
