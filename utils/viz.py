
import bz2, contextlib, functools, logging, os, sys, threading, time, json, shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import psutil
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Audio
from scipy.io.wavfile import write as wav_write, read as wav_read

#***** video helpers *****

def plot_fft(freqs:np.ndarray, fft:np.ndarray, out_path:Path, enabled:bool=True):
    """Render a static PNG of FFT magnitude vs. frequency."""
    if not enabled: return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mag = np.abs(fft)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    ax.plot(freqs, mag, color='#3b82f6', linewidth=1)
    ax.set(xlabel='Frequency (Hz)', ylabel='Magnitude', title='FFT Magnitude')
    ax.grid(True, alpha=0.3)
    for spine in ('top', 'right'): ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path

def _draw_spectrogram(ax, freqs:np.ndarray, times:np.ndarray, Sxx_db:np.ndarray):
    """Draw the spectrogram image onto `ax`, titled with the clip duration, and with the
    lowest/highest time values always present as explicit x-axis tick labels."""
    duration = times[-1] - times[0]
    im = ax.imshow(Sxx_db, origin='lower', aspect='auto', extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap='viridis')
    ax.set(xlabel='Time (s)', ylabel='Frequency (Hz)', title=f'Spectrogram ({duration:.2f}s)')
    ax.set_xlim(times[0], times[-1])
    n_inner_ticks = max(0, len(ax.get_xticks()) - 2)
    ax.set_xticks(np.linspace(times[0], times[-1], n_inner_ticks + 2))
    return im

def plot_spectrogram(freqs:np.ndarray, times:np.ndarray, Sxx:np.ndarray, out_path:Path, enabled:bool=True):
    """Render a static PNG of the spectrogram (dB scale), same styling as
    make_spectrogram_video's frames but without the playback line or audio mux."""
    if not enabled: return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    im = _draw_spectrogram(ax, freqs, times, Sxx_db)
    fig.colorbar(im, ax=ax, label='Power (dB)')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path

def make_spectrogram_video(freqs:np.ndarray, times:np.ndarray, Sxx:np.ndarray, audio:np.ndarray, sample_rate:int, out_path:Path, fps:int=20, figsize:tuple[float,float]=(6, 3), dpi:int=80, enabled:bool=True):
    """Render an mp4 of the spectrogram (dB scale) with a vertical line tracking playback
    position, muxed with `audio` as the soundtrack. Requires the `imageio-ffmpeg` package
    (bundles a portable ffmpeg binary, no system install needed). Pixel size is figsize*dpi."""
    if not enabled: return
    import subprocess, tempfile
    import cv2
    import imageio_ffmpeg
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = max(1, int(len(audio) / sample_rate * fps))
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    im = _draw_spectrogram(ax, freqs, times, Sxx_db)
    fig.colorbar(im, ax=ax, label='Power (dB)')
    line = ax.axvline(times[0], color='red', linewidth=2)
    fig.tight_layout()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video_path, audio_path = tmp / 'video.mp4', tmp / 'audio.wav'
        wav_write(audio_path, sample_rate, audio)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        try:
            for i in range(n_frames):
                t = i / fps
                line.set_xdata([t, t])
                fig.canvas.draw()
                frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
            plt.close(fig)

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg_exe, '-y', '-i', str(video_path), '-i', str(audio_path),
                         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(out_path)],
                        check=True, capture_output=True)

    return out_path

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

def get_box_coverage_key(metadata, mask): return (metadata['box'], str(metadata['objects'] if 'objects' in metadata else metadata['object']), metadata['n_objects'], mask.shape)

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
    axes[0].set_title('Overhead Image')
    _draw_vibration_image(axes[1], raw_vibrations)
    axes[1].set_title('Laser Speckles')
    _draw_box_coverage(axes[2], box_coverage, sample_dir)
    if header: fig.suptitle(header, fontsize=20, fontweight='bold')
    fig.tight_layout()
    plt.show()

def preview_audio(samples, sample_rate): return Audio(samples, rate=sample_rate)

def plot_position(result: dict, image, objects:list[str]|None=None, header:str='', alpha:float=0.3) -> None:
    """Two-panel figure for one overhead position, styled like preview_sample_row.
    Left: masks + boxes + confidence labels (plot_smask). Right: the same
    masks, same colors, no image/boxes/overlap -- what the masks actually
    look like. header is shown as a bold suptitle (e.g. the output/sample id
    range). Shown for every new position captured during an experiment."""
    from src.data.segment import plot_smask, label_map_image

    objects = objects if objects is not None else result["names"]

    # size the figure to the image's own aspect ratio so imshow doesn't pad
    # each panel with blank space above/below to keep a square axes box
    img_w, img_h = image.size
    fig_w = 20
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_w / 2 * img_h / img_w))

    left = plot_smask(result, image, objects, alpha=alpha, show=False)
    _draw_image(axes[0], left)
    axes[0].set_title("masks + boxes + confidence")

    _draw_image(axes[1], label_map_image(result["masks"]))
    axes[1].set_title("masks only")

    if header: fig.suptitle(header, fontsize=20, fontweight='bold')
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