
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from scipy.io.wavfile import write as wav_write

from utils.io_utils import load

#***** video helpers *****

def _draw_fft(ax, freqs:np.ndarray, fft:np.ndarray, title:str='FFT Magnitude', max_freq:float|None=None):
    """Draw FFT magnitude vs. frequency onto `ax`. `max_freq` caps the x-axis (e.g. the
    audio's f_end from metadata.jsonl) so the plot isn't dominated by empty spectrum."""
    ax.plot(freqs, np.abs(fft), color='#3b82f6', linewidth=1)
    ax.set(xlabel='Frequency (Hz)', ylabel='Magnitude', title=title)
    if max_freq is not None: ax.set_xlim(0, max_freq)
    ax.grid(True, alpha=0.3)
    for spine in ('top', 'right'): ax.spines[spine].set_visible(False)

def plot_fft(freqs:np.ndarray, fft:np.ndarray, out_path:Path, title:str='FFT Magnitude', max_freq:float|None=None, enabled:bool=True):
    """Render a static PNG of FFT magnitude vs. frequency."""
    if not enabled: return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    _draw_fft(ax, freqs, fft, title=title, max_freq=max_freq)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path

def _draw_spectrogram(ax, freqs:np.ndarray, times:np.ndarray, Sxx:np.ndarray, label:str='',  max_freq:float|None=None):
    """Draw the spectrogram image onto `ax`, with the lowest/highest time values always
    present as explicit x-axis tick labels. `label` is the full title; it may contain a
    '{duration}' placeholder which is filled with the clip duration in seconds."""
    duration = times[-1] - times[0]
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    title = label.format(duration=f'{duration:.2f}') if label else f'Spectrogram ({duration:.2f}s)'
    im = ax.imshow(Sxx_db, origin='lower', aspect='auto', extent=[times[0], times[-1], freqs[0], freqs[-1]], cmap='viridis')
    ax.set(xlabel='Time (s)', ylabel='Frequency (Hz)', title=title)
    ax.set_xlim(times[0], times[-1])
    n_inner_ticks = max(0, len(ax.get_xticks()) - 2)
    ax.set_xticks(np.linspace(times[0], times[-1], n_inner_ticks + 2))
    if max_freq is not None: ax.set_ylim(freqs[0], max_freq)
    return im 


def plot_spectrogram(freqs:np.ndarray, times:np.ndarray, Sxx:np.ndarray, out_path:Path, label:str='', max_freq:float|None=None, enabled:bool=True):
    """Render a static PNG of the spectrogram (dB scale), same styling as
    make_spectrogram_video's frames but without the playback line or audio mux."""
    if not enabled: return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    im = _draw_spectrogram(ax, freqs, times, Sxx, label, max_freq=max_freq)
    fig.colorbar(im, ax=ax, label='Power (dB)')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path

def make_spectrogram_video(freqs:np.ndarray, times:np.ndarray, Sxx:np.ndarray, audio:np.ndarray, sample_rate:int, out_path:Path, fps:int=20, figsize:tuple[float,float]=(6, 3), dpi:int=80, label:str='', max_freq:float|None=None, enabled:bool=True):
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

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    im = _draw_spectrogram(ax, freqs, times, Sxx, label, max_freq)
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
    # estimate contrast percentiles from a strided sample of the last 100 frames — one pass
    # over ~1/16 of the pixels instead of two passes over all of them, visually identical
    vmin, vmax = np.percentile(frame_recording[-100:, ::4, ::4], [10, 90])
    ax.imshow(frame_recording[10], vmin=vmin, vmax=vmax)

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

def preview_audio(samples, sample_rate): return Audio(samples, rate=sample_rate)

def _draw_overhead_full(ax, result:dict, image, objects:list[str]|None=None, alpha:float=0.3):
    """Overhead image with masks + boxes + confidence labels (plot_smask) drawn onto `ax`."""
    from src.data.segment import plot_smask
    objects = objects if objects is not None else result["names"]
    _draw_image(ax, plot_smask(result, image, objects, alpha=alpha, show=False))

def _draw_masks_only(ax, result:dict):
    """Just the segmentation masks (flat colors, no image/boxes/scores) drawn onto `ax`."""
    from src.data.segment import label_map_image
    _draw_image(ax, label_map_image(result["masks"]))

def _draw_coms(ax, result:dict, image=None):
    """Overlay per-object COM markers and the average COM (over objects with score > 0)
    onto whatever is already drawn on `ax`. COMs are computed from result['masks'].
    If `image` (what's drawn on `ax`) is larger than the masks — e.g. draw_speaker's padded
    canvas — the markers are shifted by the (symmetric) padding so they stay aligned."""
    from utils.metrics import center_of_mass
    dx = dy = 0
    if image is not None:
        img_h, img_w = np.array(image).shape[:2]
        mask_h, mask_w = result["masks"].shape[-2:]
        dx, dy = (img_w - mask_w) / 2, (img_h - mask_h) / 2
    coms = [center_of_mass(mask) for mask in result["masks"]]
    for (row, col) in coms:
        if (row, col) != (-1, -1): ax.plot(col + dx, row + dy, marker='o', markersize=6, color='white', markeredgecolor='black')
    real_coms = [com for com, score in zip(coms, result["scores"]) if score > 0 and tuple(com) != (-1, -1)]
    if real_coms:
        avg_row, avg_col = np.mean(real_coms, axis=0)
        ax.plot(avg_col + dx, avg_row + dy, marker='X', markersize=12, color='red', markeredgecolor='white')

def _slice_fft(fft:np.ndarray, xy_idx:int=0) -> np.ndarray:
    """Reduce a (B, L, F, x/y) laser-shift FFT (as returned by process_vibrations with a
    single laser) to the 1-D spectrum for plotting; 1-D input passes through unchanged."""
    return fft[0, 0, :, xy_idx] if fft.ndim == 4 else fft

def plot_live_image(result:dict, image, objects:list[str]|None, audio_samples:np.ndarray, audio_fs:int,
                    freqs:np.ndarray, fft:np.ndarray, spec_freqs:np.ndarray, spec_times:np.ndarray, Sxx:np.ndarray,
                    max_freq:float|None=None, header:str='', alpha:float=0.3) -> None:
    """Live 4-panel row shown once per overhead image during recording:
    overhead (masks+boxes+confidence+COMs), masks only, FFT of the original audio,
    spectrogram of the original audio — followed by a playable Audio widget of it."""
    # overhead panel is double-width, so the row has 5 unit slots like plot_live_sample's
    fig, axes = plt.subplots(1, 4, figsize=(30, 5), dpi=80, gridspec_kw={'width_ratios': [2, 1, 1, 1]})
    _draw_overhead_full(axes[0], result, image, objects, alpha=alpha)
    _draw_coms(axes[0], result)
    axes[0].set_title('Overhead')
    _draw_masks_only(axes[1], result)
    axes[1].set_title('Masks Only')
    _draw_fft(axes[2], freqs, fft, title='Original Audio FFT', max_freq=max_freq)
    im = _draw_spectrogram(axes[3], spec_freqs, spec_times, Sxx, label='Original Audio Spectrogram: {duration}s', max_freq=max_freq)
    fig.colorbar(im, ax=axes[3], label='Power (dB)')
    if header: fig.suptitle(header, fontsize=20, fontweight='bold')
    fig.tight_layout()
    plt.show()
    display(Audio(audio_samples, rate=audio_fs))

def plot_live_sample(sample_overhead, raw_vibrations, box_coverage, sample_dir, result:dict,
                     freqs:np.ndarray, fft:np.ndarray, xy_idx:int, laser_idx:int|None, audio_samples:np.ndarray, audio_fs:int,
                     spec_freqs:np.ndarray, spec_times:np.ndarray, Sxx:np.ndarray, max_freq:float|None=None, header:str='') -> None:
    """Live 5-panel row shown once per speaker/sample during recording:
    overhead (+COMs), laser speckles, box coverage, FFT of the recovered audio,
    spectrogram of the recovered audio — followed by a playable Audio widget of it.
    laser_idx/xy_idx name the laser and x/y channel the audio was recovered from."""
    recovery_label = f"Laser {laser_idx}, {'x' if xy_idx == 0 else 'y'}-axis" if laser_idx is not None else f"{'x' if xy_idx == 0 else 'y'}-axis"
    fig, axes = plt.subplots(1, 5, figsize=(30, 5), dpi=80)
    _draw_image(axes[0], sample_overhead)
    _draw_coms(axes[0], result, image=sample_overhead)
    axes[0].set_title('Overhead')
    _draw_vibration_image(axes[1], raw_vibrations)
    axes[1].set_title('Laser Speckles')
    _draw_box_coverage(axes[2], box_coverage, sample_dir)
    _draw_fft(axes[3], freqs, _slice_fft(fft, xy_idx), title=f'Recovered Audio FFT, {recovery_label}', max_freq=max_freq)
    im = _draw_spectrogram(axes[4], spec_freqs, spec_times, Sxx, label=f'Recovered Audio Spectrogram: {{duration}}s, {recovery_label}', max_freq=max_freq)
    fig.colorbar(im, ax=axes[4], label='Power (dB)')
    if header: fig.suptitle(header, fontsize=20, fontweight='bold')
    fig.tight_layout()
    plt.show()
    display(Audio(audio_samples, rate=audio_fs))

def preview(obj, mode):
    if mode == 'image': return preview_image(obj)
    if mode == 'vibration_image': return preview_vibration_image(obj)
    if mode == 'vibration_video': return preview_vibration_video(obj)
    if mode == 'box_coverage': return preview_box_coverage(*obj)
    if mode == 'plot_live_sample': return plot_live_sample(*obj)
    if mode == 'plot_live_image': return plot_live_image(*obj)
    if mode == 'audio': return preview_audio(*obj)
    else: raise ValueError(f'{mode=} not recognized')