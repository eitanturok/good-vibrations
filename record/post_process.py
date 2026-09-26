"""Vibration post-processing, all local: pclk -> clean -> fft -> recover audio -> spectrogram.

post_process(sample_dir): the full multi-ROI save, run by the @watch background worker one
sample at a time. preview_vibrations(...): the same math on one ROI, nothing written, for the
live preview right after capture.
"""
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample, sosfiltfilt

from data.pclk import compute_shifts_for_all_rois_batched_optimized
from data.audio import get_spectrogram
from utils.io_utils import save, load_metadata
from utils.viz import plot_fft, plot_spectrogram, make_spectrogram_video
from record.utils.watcher import watch

MIN_FREQ, MAX_FREQ = 50, 1000  # Hz -- the chirp's band
AUDIO_SAMPLE_RATE = 22050
RECOVERY_LASER, RECOVERY_XY = 55, 0  # which ROI / axis becomes recovered_audio.wav


# ---- 1 pclk: speckle video -> per-ROI (x, y) shift per frame ----
def pclk(raw_vibrations, rois, batch_size=256, use_PC=True, desc=None, progress=True):
    crops = np.stack([raw_vibrations[:, y:y + h, x:x + w] for x, y, w, h in rois])  # (L, T, h, w)
    return compute_shifts_for_all_rois_batched_optimized(crops, batch_size, desc=desc, progress=progress, use_PC=use_PC)  # (L, T, 2)


# ---- 2 clean: band-pass to the chirp's band, then a hann window ----
def clean(shifts, fps, order=5):
    sos = butter(order, [MIN_FREQ, MAX_FREQ], fs=fps, btype="band", output="sos")
    return sosfiltfilt(sos, shifts, axis=1) * np.hanning(shifts.shape[1])[None, :, None]  # (L, T, 2)


# ---- 3 fft, keeping only the chirp's band ----
def band_mask(n_samples, fps):
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fps)
    return (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ), freqs


def fft(clean_shifts, fps):
    mask, freqs = band_mask(clean_shifts.shape[1], fps)
    return np.fft.rfft(clean_shifts, axis=1).astype(np.complex64)[:, mask], freqs[mask]  # (L, F, 2), (F,)


# ---- 4 recover audio: inverse fft of one ROI/axis, resampled to audio rate, int16 ----
def recover_audio(fft_1d, n_samples, fps):
    mask, _ = band_mask(n_samples, fps)
    spectrum = np.zeros(len(mask), dtype=np.complex64)
    spectrum[mask] = fft_1d
    audio = resample(np.fft.irfft(spectrum, n=n_samples), int(AUDIO_SAMPLE_RATE * n_samples / fps))
    return (audio / (np.abs(audio).max() + 1e-8) * 32767).astype(np.int16)


@watch(pattern="**/vibration/01_raw_vibrations.npy")
def post_process(sample_dir):
    """Accepts the sample dir (direct .submit from save_raw_vibration) or the raw file path
    (what the watcher's periodic scan finds). Reads fps/rois from metadata.jsonl, writes the
    usual vibration/0N_* files, and deletes the raw file only once everything succeeded."""
    sample_dir = Path(sample_dir)
    if sample_dir.is_file():
        sample_dir = sample_dir.parent.parent
    vib, sid = sample_dir / "vibration", sample_dir.name
    raw_path = vib / "01_raw_vibrations.npy"

    metadata = load_metadata(sample_dir / "metadata.jsonl")
    fps, rois = float(metadata["fps"]), metadata["rois"]
    if rois is None:
        raise ValueError("metadata has rois=None (recorded wide-open, before an ROI grid existed)")
    laser, axis = min(RECOVERY_LASER, len(rois) - 1), "xy"[RECOVERY_XY]
    suffix = f"_laser{laser}_{axis}"

    shifts = pclk(np.load(raw_path), rois, desc=f"[sample {sid}]")
    save(shifts, vib / "02_raw_shifts.npy")

    clean_shifts = clean(shifts, fps)
    save(clean_shifts[None], vib / "03_clean_shifts.npy")  # [None]: on-disk files keep a batch dim

    fft_shifts, freqs = fft(clean_shifts, fps)
    n_samples = shifts.shape[1]
    save({"fft": fft_shifts[None], "freqs": freqs, "n_samples": n_samples}, vib / "04_ffts.npz")
    plot_fft(freqs, fft_shifts[laser, :, RECOVERY_XY], vib / f"04_ffts{suffix}.png",
             title=f"Recovered FFT, Laser {laser}, {axis}-axis", max_freq=MAX_FREQ)

    audio = recover_audio(fft_shifts[laser, :, RECOVERY_XY], n_samples, fps)
    save((audio, AUDIO_SAMPLE_RATE), vib / f"05_recovered_audio{suffix}.wav")
    save((audio, AUDIO_SAMPLE_RATE), sample_dir / "recovered_audio.wav")  # a real file, never a symlink

    spec_freqs, spec_times, Sxx = get_spectrogram(audio, AUDIO_SAMPLE_RATE)
    save({"freqs": spec_freqs, "times": spec_times, "Sxx": Sxx}, vib / f"06_spectrogram{suffix}.npz")
    label = f"Recovered Spectrogram: {{duration}}s, Laser {laser}, {axis}-axis"
    plot_spectrogram(spec_freqs, spec_times, Sxx, vib / f"06_spectrogram{suffix}.png", label=label, max_freq=MAX_FREQ)
    make_spectrogram_video(spec_freqs, spec_times, Sxx, audio, AUDIO_SAMPLE_RATE, vib / f"06_spectrogram{suffix}.mp4",
                           label=label, max_freq=MAX_FREQ)

    raw_path.unlink()  # ~1.9 GB each; only after every output above is written


def preview_vibrations(raw_vibrations, roi, fps, laser_idx, pclk_batch_size=256, use_PC=True) -> dict:
    """Live preview: the same steps on one ROI, nothing written to disk."""
    shifts = pclk(raw_vibrations, [roi], pclk_batch_size, use_PC=use_PC, progress=False)  # (1, T, 2)
    fft_shifts, freqs = fft(clean(shifts, fps), fps)
    n_samples = shifts.shape[1]
    audio = recover_audio(fft_shifts[0, :, 0], n_samples, fps)
    spec_freqs, spec_times, Sxx = get_spectrogram(audio, AUDIO_SAMPLE_RATE)
    return {"shifts": shifts, "fft": fft_shifts, "freqs": freqs, "n_samples": n_samples,
            "recovered_audio": audio, "audio_sample_rate": AUDIO_SAMPLE_RATE,
            "spec_freqs": spec_freqs, "spec_times": spec_times, "Sxx": Sxx, "laser_idx": laser_idx}


def warmup_pclk():
    """Absorb cupy/cuFFT init (~4s) at startup instead of on the first sample."""
    compute_shifts_for_all_rois_batched_optimized(np.zeros((1, 65, 32, 32), dtype=np.uint8), batch_size=64, progress=False)
