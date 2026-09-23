"""Waveform-domain frequency stretching, so our narrowband 50-1000Hz vibration audio
uses more of CLAP's internal 50-14000Hz mel range instead of only the bottom ~7% of
it (CLAP has no injection point for a precomputed spectrogram image -- it only takes
a raw waveform and builds its own log-mel spectrogram internally, see
model/clap_audio_encoder.py's docstring). Applied in dataset.py, BEFORE
clap_audio_encoder.py:preprocess_waveform's resample-to-44.1kHz/10s step.

Three modes, selected via --spectrogram-stretch (train.py):
  none:   passthrough -- real content stays in the bottom ~7% of CLAP's mel range.
  linear: "speeds up" the waveform by a fixed factor so MAX_FREQ maps to ~CLAP's
          fmax -- every frequency shifts by the same multiplicative factor, so this
          is a plain resample (same sample-rate label, shorter array -- reinterpreting
          a time-compressed signal at the ORIGINAL sample rate is exactly what
          multiplies every frequency by that factor), no spectrogram round-trip needed.
  log:    STFT -> log-frequency-warp the magnitude spectrogram to spread the band
          across CLAP's mel range on a log (not linear) scale, matching how natural
          mel-spectrograms allocate resolution -- Griffin-Lim then reconstructs a
          waveform from the warped magnitude (phase is estimated, not preserved,
          since the original phase doesn't apply to a warped frequency axis). Image-
          domain analog: notebooks/83_spectrogram_stretch_strategies.ipynb.

          Unlike `linear`, this needs real Fourier bins up at CLAP_FMAX (14000Hz),
          which a signal natively sampled at 22050Hz cannot represent (Nyquist
          11025Hz < 14000Hz) -- so `log` first resamples up to CLAP's own SAMPLE_RATE
          (44100Hz, Nyquist 22050Hz) before the STFT, and returns that new sample
          rate (every mode returns (waveform, sr) for exactly this reason: `log`'s
          output sample rate differs from its input, `linear`'s deliberately doesn't).
"""
import sys
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import resample

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path: sys.path.insert(0, str(REPO / "src"))

from data.vibrate import MAX_FREQ, MIN_FREQ  # noqa: E402 -- 50, 1000
from src4.model.clap_audio_encoder import FMAX as CLAP_FMAX, FMIN as CLAP_FMIN  # noqa: E402
from src4.model.clap_audio_encoder import SAMPLE_RATE as CLAP_SAMPLE_RATE  # noqa: E402

STRETCH_MODES = ("none", "linear", "log")


def stretch_linear(waveform: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    factor = CLAP_FMAX / MAX_FREQ
    stretched = resample(waveform, max(1, int(len(waveform) / factor))).astype(np.float32)
    return stretched, sr  # same sr label on purpose -- see module docstring


def stretch_log(waveform: np.ndarray, sr: int, n_fft: int = 1024, hop_length: int = 256,
                 griffinlim_iters: int = 16) -> tuple[np.ndarray, int]:
    if sr != CLAP_SAMPLE_RATE:  # need Nyquist >= CLAP_FMAX to place content there at all
        waveform = resample(waveform, int(CLAP_SAMPLE_RATE * len(waveform) / sr)).astype(np.float32)
        sr = CLAP_SAMPLE_RATE

    D = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    src_mask = (freqs >= MIN_FREQ) & (freqs <= MAX_FREQ)
    band, band_freqs = mag[src_mask], freqs[src_mask]  # (f_band, T)

    target_mask = (freqs >= CLAP_FMIN) & (freqs <= CLAP_FMAX)
    target_freqs = freqs[target_mask]

    # Match NORMALIZED position within each axis's own log-Hz span, not literal Hz
    # value -- source spans log(MIN_FREQ)..log(MAX_FREQ), target spans a much WIDER
    # log(CLAP_FMIN)..log(CLAP_FMAX) (that's the whole point of stretching). Matching
    # literal log-Hz value instead (an earlier version of this function did) means
    # ~93% of target bins fall outside the source's range entirely and np.interp
    # flat-extrapolates them to one constant value -- not actually warped, just a
    # broadband plateau Griffin-Lim then reconstructs as noise-like haze.
    src_pos = (np.log(band_freqs) - np.log(MIN_FREQ)) / (np.log(MAX_FREQ) - np.log(MIN_FREQ))
    target_pos = (np.log(target_freqs) - np.log(CLAP_FMIN)) / (np.log(CLAP_FMAX) - np.log(CLAP_FMIN))
    warped = np.stack([np.interp(target_pos, src_pos, band[:, t]) for t in range(band.shape[1])], axis=1)

    new_mag = np.zeros_like(mag)
    new_mag[target_mask] = warped  # actually moved to the wider target bins, not the original narrow ones
    audio = librosa.griffinlim(new_mag, n_iter=griffinlim_iters, hop_length=hop_length, n_fft=n_fft)
    return audio.astype(np.float32), sr


def apply_stretch(waveform: np.ndarray, sr: int, mode: str) -> tuple[np.ndarray, int]:
    assert mode in STRETCH_MODES, f"unknown spectrogram-stretch mode {mode!r}, choose from {STRETCH_MODES}"
    if mode == "none": return waveform, sr
    if mode == "linear": return stretch_linear(waveform, sr)
    return stretch_log(waveform, sr)
