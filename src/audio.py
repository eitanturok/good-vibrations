"""Generate a logarithmic chirp, then compute its FFT and spectrogram.

Usage:
    python src/audio.py [--T_sec 3.0] [--T_start 0.1] [--T_end 0.1] [--fs 44100] [--f_start 50] [--f_end 1000] [--out_dir DIR]

Defaults to data/audio/chirp_{f_start}_{f_end}_{T_sec}sec/ if --out_dir is not given.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

import numpy as np
from scipy.signal import chirp, spectrogram
from scipy.io.wavfile import read as wav_read

from utils.io_utils import save
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

def make_chirp(out_dir: Path, T_sec: float = T_SEC, T_start: float = T_START, T_end: float = T_END, fs: int = FS,
               f_start: float = F_START, f_end: float = F_END, do_save: bool = True) -> np.ndarray:
    """Generate a chirp and save it as audio.wav + metadata.jsonl in out_dir. Returns the int16 PCM audio."""
    out_dir = Path(out_dir)
    signal = generate_chirp(T_sec, T_start, T_end, fs, f_start, f_end)
    audio = np.int16(signal / (np.max(np.abs(signal)) + 1e-8) * 32767)

    metadata = [{"T_sec": T_sec}, {"T_start": T_start}, {"T_end": T_end}, {"fs": fs}, {"f_start": f_start}, {"f_end": f_end}]
    save((audio, fs), out_dir / 'audio.wav', do_save)
    save(metadata, out_dir / 'metadata.jsonl', do_save)
    return audio

#***** 1 fft *****

def compute_fft(audio_path: Path, out_dir: Path | None = None, do_save: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FFT of a wav file, saving fft.npy + fft.png to out_dir (defaults to audio_path's parent)."""
    audio_path = Path(audio_path)
    out_dir = Path(out_dir) if out_dir is not None else audio_path.parent
    fs, raw_audio = wav_read(audio_path)
    audio = raw_audio.astype(np.float32)

    fft = np.fft.rfft(audio).astype(np.complex64)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / fs)
    save(fft, out_dir / 'fft.npy', do_save)
    plot_fft(freqs, fft, out_dir / 'fft.png', do_save)
    return freqs, fft

#***** 2 spectrogram *****

def compute_spectrogram(audio_path: Path, out_dir: Path | None = None, do_save: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the spectrogram of a wav file, saving spectrogram.npy/png/mp4 to out_dir (defaults to audio_path's parent)."""
    audio_path = Path(audio_path)
    out_dir = Path(out_dir) if out_dir is not None else audio_path.parent
    fs, raw_audio = wav_read(audio_path)
    audio = raw_audio.astype(np.float32)

    freqs, times, Sxx = spectrogram(audio, fs=fs)
    save(Sxx.astype(np.float32), out_dir / 'spectrogram.npy', do_save)
    plot_spectrogram(freqs, times, Sxx, out_dir / 'spectrogram.png', do_save)
    make_spectrogram_video(freqs, times, Sxx, raw_audio, fs, out_dir / 'spectrogram.mp4', enabled=do_save)
    return freqs, times, Sxx

#***** 3 cli *****

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--T_sec', type=float, default=T_SEC)
    p.add_argument('--T_start', type=float, default=T_START)
    p.add_argument('--T_end', type=float, default=T_END)
    p.add_argument('--fs', type=int, default=FS)
    p.add_argument('--f_start', type=float, default=F_START)
    p.add_argument('--f_end', type=float, default=F_END)
    p.add_argument('--out_dir', type=Path, default=None)
    args = p.parse_args()

    out_dir = args.out_dir or REPO / 'data' / 'audio' / f'chirp_{int(args.f_start)}_{int(args.f_end)}_{args.T_sec}sec'
    make_chirp(out_dir, args.T_sec, args.T_start, args.T_end, args.fs, args.f_start, args.f_end)
    compute_fft(out_dir / 'audio.wav', out_dir)
    compute_spectrogram(out_dir / 'audio.wav', out_dir)
    print(f"saved audio.wav, metadata.jsonl, fft.npy, fft.png, spectrogram.npy, spectrogram.png, spectrogram.mp4 to {out_dir}")

if __name__ == "__main__":
    main()
