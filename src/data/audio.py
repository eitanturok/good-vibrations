"""Generate a logarithmic chirp, then compute its FFT and spectrogram.

Usage:
    python src/data/audio.py [--T_sec 3.0] [--T_start 0.1] [--T_end 0.1] [--fs 44100] [--f_start 50] [--f_end 1000] [--out_dir DIR]

Defaults to data/audio/chirp_{f_start}_{f_end}_{T_sec}sec/ if --out_dir is not given.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

import numpy as np
from scipy.signal import chirp, spectrogram

from utils.io_utils import save, load
from utils.viz import plot_fft, plot_spectrogram, make_spectrogram_video

T_SEC, T_START, T_END, FS, F_START, F_END = 3.0, 0.1, 0.1, 44100, 50, 1000

#***** 0 generate chirp *****

def generate_chirp(T_sec: float, T_start: float, T_end: float, fs: int, f_start: float, f_end: float) -> np.ndarray:
    """Logarithmic chirp sweeping f_start -> f_end over T_sec, padded with T_start/T_end seconds of silence."""
    t = np.linspace(0, T_sec, int(T_sec * fs), endpoint=False)
    chirp_signal = chirp(t, f0=f_start, t1=T_sec, f1=f_end, method='logarithmic')
    silence_start = np.zeros(int(T_start * fs))
    silence_end = np.zeros(int(T_end * fs))
    return np.concatenate([silence_start, chirp_signal, silence_end])

def make_chirp(T_sec: float, T_start: float, T_end: float, fs: int, f_start: float, f_end: float) -> np.ndarray:
    """Generate a chirp and return it as int16 PCM audio."""
    signal = generate_chirp(T_sec, T_start, T_end, fs, f_start, f_end)
    return np.int16(signal / (np.max(np.abs(signal)) + 1e-8) * 32767)

#***** 1 fft *****

def compute_fft(audio: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FFT of audio. Returns (freqs, fft)."""
    audio = audio.astype(np.float32)
    fft = np.fft.rfft(audio).astype(np.complex64)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / fs)
    return freqs, fft

#***** 2 spectrogram *****

def get_spectrogram(audio: np.ndarray, fs: int, nperseg:int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the spectrogram of audio. Returns (freqs, times, Sxx)."""
    freqs, times, Sxx = spectrogram(audio.astype(np.float32), fs=fs, nperseg=nperseg, noverlap = nperseg // 3, nfft=nperseg*4)
    return freqs, times, Sxx

#***** 3 load precomputed artifacts *****

_AUDIO_ARTIFACTS_CACHE: dict = {}

def load_audio_artifacts(audio_dir: Path) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load the precomputed audio artifacts saved by main(). Returns
    (samples, fs, freqs, fft, spec_freqs, spec_times, Sxx, max_freq) with no recomputation.
    max_freq is the audio's highest frequency (f_end from metadata.jsonl), used to cap
    fft/spectrogram plot axes. Cached per directory — the same audio is reused for every
    image in an experiment, so it's only read from disk once per session."""
    audio_dir = Path(audio_dir)
    key = str(audio_dir.resolve())
    if key in _AUDIO_ARTIFACTS_CACHE: return _AUDIO_ARTIFACTS_CACHE[key]
    samples, fs = load(audio_dir / 'audio.wav')
    freqs, fft = load(audio_dir / 'fft.npz', keys=['freqs', 'fft'])
    spec_freqs, spec_times, Sxx = load(audio_dir / 'spectrogram.npz', keys=['freqs', 'times', 'Sxx'])
    metadata = {k: v for d in load(audio_dir / 'metadata.jsonl') for k, v in d.items()}
    out = (samples, fs, freqs, fft, spec_freqs, spec_times, Sxx, metadata['f_end'])
    _AUDIO_ARTIFACTS_CACHE[key] = out
    return out

#***** 4 cli *****

def main(args):

    out_dir = args.out_dir or REPO / 'data' / 'audio' / f'chirp_{int(args.f_start)}_{int(args.f_end)}_{args.T_sec}sec'
    if out_dir.exists() and (out_dir / 'audio.wav').exists():
        print(f'Already computed audio in {out_dir}')
        return load(out_dir / 'audio.wav')

    audio = make_chirp(args.T_sec, args.T_start, args.T_end, args.fs, args.f_start, args.f_end)
    metadata = [{"T_sec": args.T_sec}, {"T_start": args.T_start}, {"T_end": args.T_end}, {"fs": args.fs}, {"f_start": args.f_start}, {"f_end": args.f_end}]
    save((audio, args.fs), out_dir / 'audio.wav')
    save(metadata, out_dir / 'metadata.jsonl')

    freqs_fft, fft = compute_fft(audio, args.fs)
    save({'freqs': freqs_fft, 'fft': fft}, out_dir / 'fft.npz')
    plot_fft(freqs_fft, fft, out_dir / 'fft.png', max_freq=args.f_end)

    freqs_spec, times, Sxx = get_spectrogram(audio, args.fs)
    save({'freqs': freqs_spec, 'times': times, 'Sxx': Sxx.astype(np.float32)}, out_dir / 'spectrogram.npz')
    spec_label = f'Original {out_dir.name} Spectrogram: {{duration}}s'
    plot_spectrogram(freqs_spec, times, Sxx, out_dir / 'spectrogram.png', label=spec_label, max_freq=args.f_end)
    make_spectrogram_video(freqs_spec, times, Sxx, audio, args.fs, out_dir / 'spectrogram.mp4', label=spec_label, max_freq=args.f_end)

    print(f"saved audio.wav, metadata.jsonl, fft.npz, fft.png, spectrogram.npz, spectrogram.png, spectrogram.mp4 to {out_dir}")
    return audio

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--T_sec', type=float, default=T_SEC)
    p.add_argument('--T_start', type=float, default=T_START)
    p.add_argument('--T_end', type=float, default=T_END)
    p.add_argument('--fs', type=int, default=FS)
    p.add_argument('--f_start', type=float, default=F_START)
    p.add_argument('--f_end', type=float, default=F_END)
    p.add_argument('--out_dir', type=Path, default=None)
    args = p.parse_args()
    main(args)
