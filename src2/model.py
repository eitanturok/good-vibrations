import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanSquaredError

from src2.dataset import DATA_INFO

def getenv(key: str, default=0): return type(default)(os.getenv(key, default))
SPEAKER_EMBD = getenv('SPEAKER_EMBD', 0)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim))
    freqs = torch.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

def precompute_freqs_cis_2d(dim: int, h: int, w: int, theta: float = 10000.0) -> torch.Tensor:
    freqs_h, freqs_w = precompute_freqs_cis(dim // 2, h, theta), precompute_freqs_cis(dim // 2, w, theta),
    freqs_h, freqs_w = freqs_h.reshape(h, 1, -1).repeat(1, w, 1), freqs_w.reshape(1, w, -1).repeat(h, 1, 1)
    return torch.cat([freqs_h, freqs_w], dim=-1).reshape(h * w, dim)

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    shp = [1] * (x.ndim - 2) + [x.shape[1], -1]  # works with 1D + 2D rope
    cos, sin = freqs_cis.reshape(*shp).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class MLPDecoder(nn.Module):
    def __init__(self, d_model, out_h, out_w):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.net = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, out_h * out_w))
    def forward(self, cls): return self.net(cls).view(-1, self.out_h, self.out_w)

class LaserEncoder(nn.Module):
    def __init__(self, patch_size, d_model, num_heads, num_layers, signal_length, signal):
        super().__init__()
        self.embed = nn.Linear(2 * patch_size, d_model)
        self.layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True), num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("freqs_cis", precompute_freqs_cis(d_model, signal_length // patch_size))
        self.signal = signal

    def _raw_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.signal == "magnitude": return x.abs()
        if self.signal == "complex": return torch.cat([x.real, x.imag], dim=-1)
        if self.signal == "mag_phase": return torch.cat([x.abs(), x.angle()], dim=-1)
        raise ValueError(f"Unknown signal mode: {self.signal}")

    def forward(self, x):
        # x.shape = (B_L,P,C,_PS) = (batch_size * n_lasers, n_patches, n_coords, patch_size)
        B_L, P, _, _ = x.shape
        x = self._raw_to_tokens(x).float()      # (B_L,P,C,_PS) -> (B_L,P,C,PS) where PS=_PS or 2*_PS
        x = self.embed(x.reshape(B_L, P, -1))   # (B_L,P,C,PS) -> (B_L,P,D)
        x = apply_rope(x, self.freqs_cis)       # (B_L,P,D) -> (B_L,P,D)
        x = torch.cat((self.cls_token.expand(B_L, -1, -1), x), dim=1)   # (B_L,P,D) -> (B_L,P+1,D)
        output = self.layers(x)  # (B_L,P+1,D) -> (B_L,P+1,D)
        return output[:, 0, :]  # (B_L,P+1,D) -> (B_L,D)

def create_metrics(): return {"mse": MeanSquaredError()}

class VibrationTransformer(ComposerModel):
    def __init__(self, d_model:int=128, pnt_num_heads:int=2, pnt_num_layers:int=2, seq_num_heads:int=2, seq_num_layers:int=2, data_info=DATA_INFO, signal='magnitude'):
        super().__init__()

        # encoder
        self.laser_encoder = LaserEncoder(data_info['patch_size'], d_model, pnt_num_heads, pnt_num_layers, data_info['n_freqs'], signal)
        self.box_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=seq_num_heads, batch_first=True), num_layers=seq_num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # Initialize to small random values
        self.register_buffer("freqs_laser", precompute_freqs_cis_2d(d_model, data_info['n_laser_rows'], data_info['n_laser_cols'])) # for laser grid

        # decoder
        self.speaker_embed = nn.Embedding(4, d_model)
        self.decoder = MLPDecoder(d_model, data_info['out_h'], data_info['out_w'])

        # metrics
        self.train_metrics, self.val_metrics = create_metrics(), create_metrics()

    def forward(self, batch):
        # B=batch size, L=n_lasers, C=n_coordinates=2, PS=patch_size, D=d_model
        x = batch['fft']
        B, L, _, _, _ = x.shape

        # LaserEncoder learns patterns between all frequencies from a single laser
        # flatten so LaserEncoder processes all lasers AND all batches in parallel
        x = self.laser_encoder(x.flatten(0, 1)).reshape(B, L, -1)  # (B,L,P,C,PS) -> (B,L,D)

        # BoxEncoder learns patterns between ALL the lasers shining on the box
        x = apply_rope(x, self.freqs_laser) # (B,L,D) -> (B,L,D)
        if SPEAKER_EMBD: x += self.speaker_embed(batch['info']['speakers_encoded']).unsqueeze(1)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)  # (B,L,D) (1,1,D) -> (B,L+1,D)
        output = self.box_encoder(x)  # (B,L+1,D) -> (B,L+1,D)
        cls_embedding = output[:, 0, :]  # (B,L+1,D) -> (B,D)

        # Predict segmentation mask
        mask_logits = self.decoder(cls_embedding) # (B,D) -> (B,H,W)
        mask_pred = mask_logits.sigmoid()
        return dict(mask_pred=mask_pred)

    def loss(self, outputs, batch):
        # mse is averaged over (B,H,W) so the error is independent of the height and width, making it stable across different out_h / out_w
        return F.mse_loss(outputs['mask_pred'], batch['mask_true'])

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, MeanSquaredError):
            metric.update(outputs['mask_pred'], batch['mask_true'])

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)
