"""Full multi-ROI vibration post-processing: file-triggered, background, one sample at a
time. All the real math is data.vibrate's existing, proven pipeline; this module is a thin
@watch-decorated wrapper around it, plus a fast single-ROI live-preview path called
directly (not through the watcher) by record.ipynb's run_experiment.
"""

from pathlib import Path

import numpy as np

from data.vibrate import _process_vibrations_local, get_clean_shifts, get_fft_shifts, get_recovered_audio, MIN_FREQ, MAX_FREQ
from data.pclk import compute_shifts_for_all_rois_batched_optimized
from data.audio import get_spectrogram
from record.utils.watcher import watch


@watch(pattern="**/vibration/01_raw_vibrations.npy")
def post_process(sample_dir):
    """File-triggered full multi-ROI save: pclk -> clean -> fft -> recover audio ->
    spectrogram -> write results into the already-saved sample dir -> delete the raw file.
    Delegates entirely to data.vibrate's existing, proven _process_vibrations (do_save=True,
    cleanup_raw_vibrations='delete' -- raw arrays are ~2.7GB each, not worth keeping).

    Reads sample_dir/metadata.jsonl for `fps`/`rois` -- this only works because
    save_raw_vibration (record.ipynb Section 9) writes a minimal metadata.jsonl seed
    (just fps + rois) immediately, before segmentation/save_sample ever runs, so this
    trigger (which fires off the raw vibration file alone) never has to wait on
    segmentation finishing."""
    sample_dir = Path(sample_dir)
    _process_vibrations_local(sample_dir, do_save=True, cleanup_raw_vibrations="delete")


def preview_vibrations(raw_vibrations: np.ndarray, roi: tuple[int, int, int, int], fps: float,
                        laser_idx: int, pclk_batch_size: int = 256, use_PC: bool = True) -> dict:
    """Fast single-ROI live preview: same math as data.vibrate.process_vibrations_2 (pclk on
    just the one ROI crop, then clean/fft/recover/spectrogram), but takes `roi`/`fps`
    directly as arguments instead of reading them from sample_dir/metadata.jsonl -- this
    fires right after capture, from run_experiment, before save_sample necessarily has (or
    even could have -- it's waiting on Modal segmentation) written that file. Never writes
    to disk; `post_process` above is what persists results, for every ROI, in the
    background."""
    x, y, w, h = roi
    crop = np.ascontiguousarray(raw_vibrations[:, y:y + h, x:x + w])[None]  # (1, T, h, w)
    raw_shifts = compute_shifts_for_all_rois_batched_optimized(crop, pclk_batch_size, progress=False, use_PC=use_PC)  # (1, T, 2)
    clean_shifts = get_clean_shifts(raw_shifts[None], fps, MIN_FREQ, MAX_FREQ)  # (1, 1, T, 2)
    fft, freqs, n_samples = get_fft_shifts(clean_shifts, fps, MIN_FREQ, MAX_FREQ)
    audio_sample_rate = 22050
    recovered_audio = get_recovered_audio(fft, n_samples, fps, audio_sample_rate, MIN_FREQ, MAX_FREQ, laser_idx=0, xy_idx=0)
    spec_freqs, spec_times, Sxx = get_spectrogram(recovered_audio, audio_sample_rate)
    return {"shifts": raw_shifts, "fft": fft, "freqs": freqs, "n_samples": n_samples,
            "recovered_audio": recovered_audio, "audio_sample_rate": audio_sample_rate,
            "spec_freqs": spec_freqs, "spec_times": spec_times, "Sxx": Sxx, "laser_idx": laser_idx}
