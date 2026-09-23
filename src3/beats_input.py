"""Feed our vibration data into BEATs (AudioToken's audio encoder) instead of real
audio, by bypassing its raw-waveform -> mel-fbank frontend and handing our own 2D
map directly to its pretrained patch-embedding + transformer (see
vendor/modules/BEATs/BEATs.py: BEATs.preprocess / BEATs.extract_features).

Two ways to build that 2D map from a sample's `vibration/04_fft.npz`
((1,L,F,C) complex64: lasers x freq bins x x/y-channel -- see src/model/dataset.py
and src2/data.py:fft_to_heatmap for the existing conventions this reuses):

  laser_freq_map(): the FFT heatmap itself, mean |FFT| magnitude over x/y,
  log-contrast stretched (src2/data.py's tuned LOG_CONTRAST_K), freq on the row
  axis.

  spectrogram_map(): average the FFT over lasers AND x/y down to one spectrum,
  recover an audio-like waveform via IFFT (generalizes
  src/data/vibrate.py:get_recovered_audio, which does this for one laser/channel
  at a time), then a real STFT spectrogram via src/data/audio.py:get_spectrogram.

Both return (F, L_or_T) float32 arrays with freq on axis 0 (rows/y), matching how a
spectrogram is normally drawn -- purely a display/storage convention. to_beats_fbank()
below is the one place that reshapes into whatever BEATs' patch-embedding conv
actually expects: (T, 128) with freq as the trailing/column axis.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tf
from scipy.signal import resample

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))  # src/data/vibrate.py imports as `from data.audio import ...`

from data.audio import get_spectrogram
from data.vibrate import MIN_FREQ, MAX_FREQ

# log(1+mag/K) contrast scale tuned in notebooks/79_log_contrast_sweep.ipynb -- see
# src2/data.py, which this constant is copied from (kept in sync manually; small
# enough not to warrant a shared module for one float).
LOG_CONTRAST_K = 0.01138

BEATS_MEL_BINS = 128  # fixed by the pretrained patch_embedding conv (num_mel_bins in BEATs.preprocess)
DEFAULT_AUDIO_SAMPLE_RATE = 22050


def _load_fft(fft_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    with np.load(fft_path) as z:
        fft, freqs, n_samples = z["fft"], z["freqs"], int(z["n_samples"])
    fft = np.squeeze(fft, axis=0) if fft.ndim == 4 and fft.shape[0] == 1 else fft  # (L,F,C)
    return fft, freqs, n_samples


def laser_freq_map(fft_path: Path) -> np.ndarray:
    """(F, L) float32 -- mean |FFT| magnitude over x/y, log-contrast stretched, freq on rows."""
    fft, _, _ = _load_fft(fft_path)
    mag = np.abs(fft).mean(axis=-1)  # (L,F)
    log_mag = np.log1p(mag / LOG_CONTRAST_K)
    return log_mag.T.astype(np.float32)  # (F,L)


DB_TOP_DB = 60.0  # dynamic range kept below the max, matches utils/viz.py's plot_spectrogram usage


def spectrogram_map(fft_path: Path, fps: float, audio_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
                     nperseg: int = 512) -> np.ndarray:
    """(F, T) float32 dB-scale STFT spectrogram of the audio recovered from the
    lasers+x/y-averaged FFT (not any single laser/channel). nperseg is much smaller
    than src/data/audio.py:get_spectrogram's own default (4096, tuned for longer
    chirp recordings) -- our recovered clip is only ~3s, and BEATs' patch_embedding
    needs at least `input_patch_size` (16) time frames to run at all.

    dB scale (10*log10), matching utils/viz.py:_draw_spectrogram's own convention --
    NOT the previous log1p(Sxx). Measured in notebooks/81_spectrogram_playground.ipynb
    section 11: the recovered audio's raw STFT magnitude is tiny (normalized to
    max-abs=1 before the STFT), so log1p(Sxx) landed almost entirely within 1e-3 of
    zero (std ~4.7e-5) -- essentially no dynamic range survived, which is why it looked
    like "not a lot of detail" even after percentile-stretching for display. dB scale
    doesn't have that collapse-to-zero problem (std ~9 in the same notebook), so it
    carries real dynamic range both for display and as actual model input."""
    fft, _, n_samples = _load_fft(fft_path)
    avg_spectrum = fft.mean(axis=(0, 2)).astype(np.complex64)  # (F,) -- average over lasers and x/y

    full_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fps)
    mask = (full_freqs >= MIN_FREQ) & (full_freqs <= MAX_FREQ)
    spectrum = np.zeros(len(full_freqs), dtype=np.complex64)
    spectrum[mask] = avg_spectrum
    signal = np.fft.irfft(spectrum, n=n_samples)

    audio = resample(signal, int(audio_sample_rate * len(signal) / fps))
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    _, _, Sxx = get_spectrogram(audio.astype(np.float32), audio_sample_rate, nperseg=nperseg)
    db = 10 * np.log10(Sxx + 1e-10)
    db = np.maximum(db, db.max() - DB_TOP_DB)  # clip the noise floor, same top_db idea as utils/viz.py
    return db.astype(np.float32)  # (F,T)


def map_to_image(cond_map: np.ndarray) -> np.ndarray:
    """(F,L_or_T) float32 -> (F,L_or_T) uint8, for wandb/PNG viewing (not model input --
    to_beats_fbank does its own separate z-score normalization for that).

    Stretches by the 1st/99.5th percentile, not literal min/max: spectrogram_map's
    energy is extremely concentrated (measured on a real sample: ~99% of pixels sit
    within 1% of the max value -- one loud narrowband bin dominates a mostly-quiet
    recovering), so a plain min/max stretch renders it almost pure black. Percentile
    clipping ignores that single extreme outlier and actually shows the structure in
    the other 99% of the map. laser_freq_map (already contrast-stretched via
    LOG_CONTRAST_K) is unaffected in practice -- its distribution isn't this skewed."""
    lo, hi = np.percentile(cond_map, [1, 99.5])
    norm = np.clip((cond_map - lo) / max(hi - lo, 1e-8), 0, 1)
    return (norm * 255).astype(np.uint8)


# Our data (both laser_freq_map and spectrogram_map) is already band-limited to
# MIN_FREQ..MAX_FREQ (50-1000Hz -- the FFT itself is cropped to that band upstream, see
# _load_fft's source .npz). BEATs' real 128-bin mel filterbank spans the full 0-8000Hz
# (16kHz sample rate) range, and its bin centers are NOT linear in Hz (mel scale) --
# computed via torchaudio.compliance.kaldi.get_mel_banks with fbank()'s own defaults
# (25ms window -> 512-sample padded FFT, low_freq=20): our [50,1000]Hz band's bin centers
# land entirely within bins [2,44) of 128 (~33% of the bins). Previously we bilinear-
# stretched our whole freq axis across all 128 bins, smearing our narrow-band signal over
# bins the pretrained conv expects to hold mid/high-frequency content it never actually
# gets from us. Instead: place our data only in bins [MEL_BAND_LO,MEL_BAND_HI), and leave
# the rest at a floor value -- closer to what a real recording with no energy above 1000Hz
# would look like to the pretrained conv.
MEL_BAND_LO, MEL_BAND_HI = 2, 44

# Global normalization stats (see notebooks/81_spectrogram_playground.ipynb section 10),
# computed once via scripts/compute_beats_global_stats.py over ~60 samples/box across all
# 5 boxes (post dB-scale + mel-band placement, pre-normalization) -- NOT per-sample.
# Mirrors BEATs.preprocess's own fixed-constant scheme (fbank_mean=15.41663,
# fbank_std=6.55582) instead of a per-sample z-score, which forces every sample's std to
# 1.0 regardless of content and landed ~3x larger in scale than real BEATs input.
GLOBAL_STATS = {
    "laser-freq": (1.295103, 1.526317),
    "spectrogram": (-81.723145, 3.501121),
}
BEATS_TARGET_STD = 0.32  # measured std of real BEATs fbank input after its own fixed-constant normalize


def _log_freq_resample(band: torch.Tensor, target_h: int) -> torch.Tensor:
    """(F, L) -> (target_h, L): resample rows onto a log-Hz axis, assuming band's rows are
    linearly spaced across MIN_FREQ..MAX_FREQ. Gives low frequencies within our 50-1000Hz
    band proportionally more output rows than a plain linear resize would -- see
    notebooks/83_spectrogram_stretch_strategies.ipynb (strategy 2) for the derivation and a
    visual comparison against linear resize (strategy 1) and tiling (strategy 3)."""
    f_bins = band.shape[0]
    log_f = torch.linspace(np.log(MIN_FREQ), np.log(MAX_FREQ), f_bins)
    target_log = torch.linspace(log_f[0], log_f[-1], target_h)
    idx = torch.searchsorted(log_f, target_log).clamp(1, f_bins - 1)
    lo, hi = idx - 1, idx
    w = ((target_log - log_f[lo]) / (log_f[hi] - log_f[lo]).clamp_min(1e-8))[:, None]
    return band[lo] * (1 - w) + band[hi] * w


def to_beats_fbank(cond_map: np.ndarray, condition_mode: str, target_freq_bins: int = BEATS_MEL_BINS,
                    log_stretch: bool = False) -> torch.Tensor:
    """(F, L_or_T) -> (L_or_T, target_freq_bins) tensor shaped like BEATs' own fbank, freq
    axis resized then transposed so time/laser is the row axis and freq is the column axis
    (BEATs.preprocess's own convention).

    log_stretch=False (default): bilinear-resize into the [MEL_BAND_LO,MEL_BAND_HI) sub-range
    of the pretrained patch-embedding's fixed 128 mel bins (see above), padded with a
    per-sample floor value elsewhere. Normalized by GLOBAL_STATS[condition_mode] (fixed,
    dataset-level -- not per-sample), rescaled to BEATS_TARGET_STD.

    log_stretch=True: log-Hz warp the band (see _log_freq_resample) across the FULL
    target_freq_bins instead of a narrow sub-range -- trades the (already-approximate)
    alignment to BEATs' real mel-bin positions for full use of its frequency resolution.
    This changes the value distribution entirely (no floor padding, full-range warped data),
    and no global stats have been computed for it yet (unlike GLOBAL_STATS above, which
    scripts/compute_beats_global_stats.py measured for the log_stretch=False scheme) --
    normalized per-sample instead."""
    if log_stretch:
        t = torch.from_numpy(cond_map).float()  # (F, L)
        full = _log_freq_resample(t, target_freq_bins).T  # (L, target_freq_bins)
        mean, std = full.mean(), full.std().clamp_min(1e-8)
        return (full - mean) / std * BEATS_TARGET_STD

    band_bins = MEL_BAND_HI - MEL_BAND_LO
    t = torch.from_numpy(cond_map).float()[None, None]  # (1,1,F,L)
    t = tf.interpolate(t, size=(band_bins, cond_map.shape[1]), mode="bilinear", align_corners=False)
    t = t[0, 0].T  # (L,band_bins)

    full = torch.full((t.shape[0], target_freq_bins), t.min().item())
    full[:, MEL_BAND_LO:MEL_BAND_HI] = t

    mean, std = GLOBAL_STATS[condition_mode]
    return (full - mean) / std * BEATS_TARGET_STD


def beats_extract_features(aud_encoder, fbank: torch.Tensor):
    """Mirrors vendor/modules/BEATs/BEATs.py:BEATs.extract_features exactly, minus
    the `fbank = self.preprocess(source, ...)` line at the top -- `fbank` here is
    already what that line would have produced (see to_beats_fbank), batched: (B,T,128).
    Returns the same (x, layers_sum, layers) tuple; callers use `[1]` (layers_sum),
    same as vendor/inference.py and vendor/train.py do."""
    x = fbank.unsqueeze(1)  # (B,1,T,128)
    x = aud_encoder.patch_embedding(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.transpose(1, 2)
    x = aud_encoder.layer_norm(x)
    if aud_encoder.post_extract_proj is not None:
        x = aud_encoder.post_extract_proj(x)
    x = aud_encoder.dropout_input(x)
    return aud_encoder.encoder(x, padding_mask=None)
