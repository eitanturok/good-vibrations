"""Vibration post-processing, all local: pclk -> clean -> fft -> recover audio -> spectrogram.

post_process(raw_path): the full multi-ROI save, run one sample at a time by a Watcher on the
samples dir. preview_vibrations(...): the same math on one ROI, nothing written, for the live
preview right after capture.
"""
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

from data.pclk import compute_shifts_for_all_rois_batched_optimized
from data.audio import get_spectrogram
from utils.io_utils import save, load_metadata
from utils.viz import plot_fft, plot_spectrogram, make_spectrogram_video
from record2.log import Timing

PCLK_BATCH_SIZE = 256  # ~7x faster than sequential on the RTX 5070 Ti; 1024 spills VRAM
AUDIO_SAMPLE_RATE = 22050


# ---- 1 pclk: speckle video -> per-ROI (x, y) shift per frame ----
def pclk(raw_vibrations, rois, batch_size=PCLK_BATCH_SIZE, use_PC=True, desc=None, progress=True):
    crops = np.stack([raw_vibrations[:, y:y + h, x:x + w] for x, y, w, h in rois])  # (L, T, h, w)
    return compute_shifts_for_all_rois_batched_optimized(crops, batch_size, desc=desc, progress=progress, use_PC=use_PC)  # (L, T, 2)


# The band [min_freq, max_freq] is the chirp's own [f_start, f_end] -- saved in each sample's
# metadata.jsonl, so post-processing always matches the chirp that was actually played.

# ---- 2 clean: band-pass to the chirp's band, then a hann window ----
def clean(shifts, fps, min_freq, max_freq, order=5):
    sos = butter(order, [min_freq, max_freq], fs=fps, btype="band", output="sos")
    return sosfiltfilt(sos, shifts, axis=1) * np.hanning(shifts.shape[1])[None, :, None]  # (L, T, 2)


# ---- 3 fft, keeping only the chirp's band ----
def band_mask(n_samples, fps, min_freq, max_freq):
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fps)
    return (freqs >= min_freq) & (freqs <= max_freq), freqs


def fft(clean_shifts, fps, min_freq, max_freq):
    mask, freqs = band_mask(clean_shifts.shape[1], fps, min_freq, max_freq)
    return np.fft.rfft(clean_shifts, axis=1).astype(np.complex64)[:, mask], freqs[mask]  # (L, F, 2), (F,)


# ---- 4 recover audio: inverse fft of one ROI/axis, resampled to audio rate, int16 ----
def recover_audio(fft_1d, n_samples, fps, min_freq, max_freq):
    mask, _ = band_mask(n_samples, fps, min_freq, max_freq)
    spectrum = np.zeros(len(mask), dtype=np.complex64)
    spectrum[mask] = fft_1d
    audio = resample(np.fft.irfft(spectrum, n=n_samples), int(AUDIO_SAMPLE_RATE * n_samples / fps))
    return (audio / (np.abs(audio).max() + 1e-8) * 32767).astype(np.int16)


def post_process(raw_path):
    """`raw_path` is a sample's vibration/01_raw_vibrations.npy. Reads fps, rois, the chirp's band and
    which laser/axis to listen to (recovery_laser/recovery_axis) from metadata.jsonl, writes the
    vibration/0N_* files, and deletes the raw file only once everything succeeded."""
    raw_path = Path(raw_path)
    sample_dir = raw_path.parent.parent
    with Timing(sample_dir.name, "post process", sample_dir):
        _post_process(raw_path, sample_dir)


def _post_process(raw_path, sample_dir):
    vib, sid = sample_dir / "vibration", sample_dir.name

    metadata = load_metadata(sample_dir / "metadata.jsonl")
    fps, rois = float(metadata["fps"]), metadata["rois"]
    min_freq, max_freq = float(metadata["min_freq"]), float(metadata["max_freq"])
    if rois is None:
        raise ValueError("metadata has rois=None (recorded wide-open, before an ROI grid existed)")
    laser, axis = min(int(metadata["recovery_laser"]), len(rois) - 1), metadata["recovery_axis"]
    channel, suffix = "xy".index(axis), f"_laser{laser}_{axis}"

    shifts = pclk(np.load(raw_path), rois, desc=f"[sample {sid}]")
    save(shifts, vib / "02_raw_shifts.npy")

    clean_shifts = clean(shifts, fps, min_freq, max_freq)
    save(clean_shifts[None], vib / "03_clean_shifts.npy")  # [None]: on-disk files keep a batch dim

    fft_shifts, freqs = fft(clean_shifts, fps, min_freq, max_freq)
    n_samples = shifts.shape[1]
    save({"fft": fft_shifts[None], "freqs": freqs, "n_samples": n_samples}, vib / "04_ffts.npz")
    plot_fft(freqs, fft_shifts[laser, :, channel], vib / f"04_ffts{suffix}.png",
             title=f"Recovered FFT, Laser {laser}, {axis}-axis", max_freq=max_freq)

    audio = recover_audio(fft_shifts[laser, :, channel], n_samples, fps, min_freq, max_freq)
    save((audio, AUDIO_SAMPLE_RATE), vib / f"05_recovered_audio{suffix}.wav")
    save((audio, AUDIO_SAMPLE_RATE), sample_dir / "recovered_audio.wav")  # a real file, never a symlink

    spec_freqs, spec_times, Sxx = get_spectrogram(audio, AUDIO_SAMPLE_RATE)
    save({"freqs": spec_freqs, "times": spec_times, "Sxx": Sxx}, vib / f"06_spectrogram{suffix}.npz")
    label = f"Recovered Spectrogram: {{duration}}s, Laser {laser}, {axis}-axis"
    plot_spectrogram(spec_freqs, spec_times, Sxx, vib / f"06_spectrogram{suffix}.png", label=label, max_freq=max_freq)
    make_spectrogram_video(spec_freqs, spec_times, Sxx, audio, AUDIO_SAMPLE_RATE, vib / f"06_spectrogram{suffix}.mp4",
                           label=label, max_freq=max_freq)

    raw_path.unlink()  # ~1.9 GB each; only after every output above is written


def preview_vibrations(raw_vibrations, roi, fps, laser_idx, channel, min_freq, max_freq, pclk_batch_size=PCLK_BATCH_SIZE, use_PC=True) -> dict:
    """Live preview: the same steps on one ROI and one channel (0 = x, 1 = y), nothing written to disk."""
    shifts = pclk(raw_vibrations, [roi], pclk_batch_size, use_PC=use_PC, progress=False)  # (1, T, 2)
    fft_shifts, freqs = fft(clean(shifts, fps, min_freq, max_freq), fps, min_freq, max_freq)
    n_samples = shifts.shape[1]
    audio = recover_audio(fft_shifts[0, :, channel], n_samples, fps, min_freq, max_freq)
    return {"shifts": shifts, "fft": fft_shifts, "freqs": freqs, "fps": fps, "laser_idx": laser_idx,
            "recovered_audio": audio, "audio_sample_rate": AUDIO_SAMPLE_RATE}


def warmup_pclk():
    """Absorb cupy/cuFFT init (~4s) at startup instead of on the first sample."""
    compute_shifts_for_all_rois_batched_optimized(np.zeros((1, 65, 32, 32), dtype=np.uint8), batch_size=64, progress=False)
