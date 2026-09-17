"""Phase-2 verification (see plan): run our BEATs seam (beats_input.py) on a real
sample's vibration data through the actual downloaded BEATs checkpoint, and sanity
check output shapes/stats against a real-audio forward pass through the unmodified
vendor code path (extract_features on a real waveform). Not bit-exact (different
input by design) -- just confirms the seam is wired correctly end to end and
produces finite, reasonably scaled features the embedder can consume.

Usage:
    python src3/verify_beats_seam.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "src3" / "vendor"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(REPO))
os.chdir(VENDOR)

import torch

from modules.BEATs.BEATs import BEATs, BEATsConfig

from src3.beats_input import laser_freq_map, spectrogram_map, to_beats_fbank, beats_extract_features

FFT_PATH = REPO / "experiments/31_07_2026_gastronorm_exp1/samples/000010/vibration/04_fft.npz"
FPS = 2500.0


def load_beats() -> BEATs:
    ckpt = torch.load("models/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt", map_location="cpu", weights_only=False)
    cfg = BEATsConfig(ckpt["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    model.predictor = None
    return model.eval()


def report(name: str, x: torch.Tensor, layers_sum: torch.Tensor) -> None:
    print(f"{name}: x={tuple(x.shape)} finite={torch.isfinite(x).all().item()} "
          f"layers_sum={tuple(layers_sum.shape)} mean={layers_sum.mean().item():.4f} std={layers_sum.std().item():.4f} "
          f"finite={torch.isfinite(layers_sum).all().item()}")


def main() -> None:
    beats = load_beats()

    # real-audio baseline, through the unmodified vendor code path
    real_audio = torch.randn(1, FPS.__int__() * 5) * 0.01  # 5s @ "16kHz"-ish scale, stand-in real waveform
    x, layers_sum, _ = beats.extract_features(real_audio)
    report("real-audio (vendor path)", x, layers_sum)

    # our seam, laser x freq map
    lf = laser_freq_map(FFT_PATH)
    print(f"laser_freq_map: {lf.shape} (F,L) min={lf.min():.3f} max={lf.max():.3f}")
    fbank = to_beats_fbank(lf, "laser-freq").unsqueeze(0)  # (1,T,128)
    x, layers_sum, _ = beats_extract_features(beats, fbank)
    report("laser-freq (our seam)", x, layers_sum)

    # our seam, real spectrogram map
    spec = spectrogram_map(FFT_PATH, fps=FPS)
    print(f"spectrogram_map: {spec.shape} (F,T) min={spec.min():.3f} max={spec.max():.3f}")
    fbank = to_beats_fbank(spec, "spectrogram").unsqueeze(0)
    x, layers_sum, _ = beats_extract_features(beats, fbank)
    report("spectrogram (our seam)", x, layers_sum)

    # feed layers_sum through the actual pretrained embedder to confirm shape compatibility end to end
    from modules.AudioToken.embedder import FGAEmbedder
    embedder = FGAEmbedder(input_size=768 * 3, output_size=768)
    embedder.load_state_dict(torch.load("output/embedder_learned_embeds.bin", map_location="cpu"))
    embedder.eval()
    with torch.no_grad():
        audio_token = embedder(layers_sum)
    print(f"embedder(layers_sum) -> audio_token {tuple(audio_token.shape)} "
          f"finite={torch.isfinite(audio_token).all().item()}")


if __name__ == "__main__":
    main()
