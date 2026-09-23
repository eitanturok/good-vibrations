"""Minimal, audio-only port of Microsoft CLAP (https://arxiv.org/abs/2211.06687,
`msclap`) -- only the Cnn14 audio tower + its projection head, since SonicDiffusion
only ever calls `clap.audio_encoder`, never the text/caption side. Trimmed from the
original CLAP/src/{models/audio.py,models/clap.py,CLAPWrapper.py} (which also builds a
frozen BERT caption encoder we'd never use) down to what's actually needed, with the
config.yml values (audioenc_name/sample_rate/etc.) inlined as constants instead of a
yaml + importlib_resources namespace-package lookup.

Frontend: takes a raw 44.1kHz waveform, computes its own log-mel spectrogram
internally (Spectrogram + LogmelFilterBank below) -- fmin=50Hz, fmax=14000Hz, 64 mel
bins, 1024-sample window, 320-sample hop. There is no way to hand this encoder a
precomputed spectrogram image directly."""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

SAMPLE_RATE = 44100
DURATION_S = 10
FMIN, FMAX = 50, 14000
WINDOW_SIZE, HOP_SIZE, MEL_BINS = 1024, 320, 64
OUT_EMB = 2048  # Cnn14's embedding dim before projection
D_PROJ = 1024  # projected audio embedding dim (== Adapter's audio_dim in audio_projector.py)
CLASSES_NUM = 527  # AudioSet classes Cnn14's classifier head was trained on (unused, kept only so checkpoint shapes match)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size: tuple[int, int] = (2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool_size)


class Cnn14(nn.Module):
    """PANNs Cnn14 (https://arxiv.org/abs/1912.10211), as used by CLAP's audio tower."""

    def __init__(self):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, win_length=WINDOW_SIZE,
                                                   window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=SAMPLE_RATE, n_fft=WINDOW_SIZE, n_mels=MEL_BINS, fmin=FMIN,
                                                   fmax=FMAX, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(MEL_BINS)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, OUT_EMB)
        self.fc_audioset = nn.Linear(OUT_EMB, CLASSES_NUM)  # unused past load; kept for checkpoint shape compat

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, n_samples) -> (B, OUT_EMB) audio embedding."""
        x = self.spectrogram_extractor(waveform)  # (B,1,T,freq_bins)
        x = self.logmel_extractor(x)  # (B,1,T,mel_bins)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)

        x = self.conv_block1(x); x = F.dropout(x, 0.2, self.training)
        x = self.conv_block2(x); x = F.dropout(x, 0.2, self.training)
        x = self.conv_block3(x); x = F.dropout(x, 0.2, self.training)
        x = self.conv_block4(x); x = F.dropout(x, 0.2, self.training)
        x = self.conv_block5(x); x = F.dropout(x, 0.2, self.training)
        x = self.conv_block6(x, pool_size=(1, 1)); x = F.dropout(x, 0.2, self.training)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = F.dropout(x1 + x2, 0.5, self.training)
        return F.dropout(F.relu_(self.fc1(x)), 0.5, self.training)


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float = 0.5):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        return self.layer_norm(embed1 + embed2)


class ClapAudioEncoder(nn.Module):
    """Cnn14 + projection head -> D_PROJ-dim audio embedding, matching CLAP's
    `audio_encoder.<...>` checkpoint key prefix exactly (so the pretrained weights
    load directly, no key remapping needed)."""

    def __init__(self):
        super().__init__()
        self.base = Cnn14()
        self.projection = Projection(OUT_EMB, D_PROJ)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.projection(self.base(waveform))


def load_clap_audio_encoder(weights_path: Path) -> ClapAudioEncoder:
    """Loads only the `audio_encoder.*`-prefixed weights from CLAP's full checkpoint
    (which also has a frozen BERT caption encoder we never instantiate)."""
    model = ClapAudioEncoder()
    full_state_dict = torch.load(weights_path, map_location="cpu")["model"]
    audio_state_dict = {k.removeprefix("audio_encoder."): v for k, v in full_state_dict.items()
                         if k.startswith("audio_encoder.")}
    model.load_state_dict(audio_state_dict, strict=True)
    return model


def preprocess_waveform(waveform: torch.Tensor, sample_rate: int, duration_s: float = DURATION_S) -> torch.Tensor:
    """(n_samples,) at `sample_rate` -> (SAMPLE_RATE * duration_s,) at CLAP's expected
    44.1kHz/10s, resampled and repeat-padded/cropped to the fixed duration Cnn14's
    BatchNorm statistics were trained on (mirrors CLAPWrapper.load_audio_into_tensor)."""
    if sample_rate != SAMPLE_RATE:
        waveform = torch.from_numpy(resample(waveform.numpy(), int(SAMPLE_RATE * len(waveform) / sample_rate))).float()
    target_len = int(SAMPLE_RATE * duration_s)
    if len(waveform) < target_len:
        waveform = waveform.repeat(-(-target_len // len(waveform)))[:target_len]
    else:
        waveform = waveform[:target_len]
    return waveform
