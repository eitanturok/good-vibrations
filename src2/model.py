import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanSquaredError, Metric

from src2.dataset import DATA_INFO

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
    def __init__(self, patch_size:int, d_model:int, num_heads:int, num_layers:int, signal_length:int, signal_mode:str, normalize_mode:str):
        super().__init__()
        self.embed = nn.Linear(2 * patch_size, d_model)
        self.speakers_embed = nn.Embedding(4, d_model)
        self.layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True), num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("freqs_cis", precompute_freqs_cis(d_model, signal_length // patch_size))
        self.signal_mode, self.normalize_mode = signal_mode, normalize_mode

    def raw_to_tokens(self, x:torch.Tensor) -> torch.Tensor:
        if self.signal_mode == "magnitude": return x.abs()
        if self.signal_mode == "complex": return torch.cat([x.real, x.imag], dim=-1)
        if self.signal_mode == "mag_phase": return torch.cat([x.abs(), x.angle()], dim=-1)
        raise ValueError(f"Unknown signal mode: {self.signal_mode}")

    def normalize(self, x:torch.Tensor) -> torch.Tensor:
        if self.normalize_mode is None: return x
        if self.normalize_mode == 'z': return (x - x.mean()) / x.std()
        raise ValueError(f"Unknown normalize mode: {self.normalize_mode}")

    def forward(self, x, speaker=None):
        # x.shape = (B_L,P,C,_PS) = (batch_size * n_lasers, n_patches, n_coords, patch_size)
        B_L, P, _, _ = x.shape
        x = self.raw_to_tokens(x).float()       # (B_L,P,C,_PS) -> (B_L,P,C,PS) where PS=_PS or 2*_PS
        x = self.normalize(x)                   # (B_L,P,C,PS) -> (B_L,P,C,PS)
        x = self.embed(x.reshape(B_L, P, -1))   # (B_L,P,C,PS) -> (B_L,P,D)
        x = apply_rope(x, self.freqs_cis)       # (B_L,P,D) -> (B_L,P,D)
        if speaker is not None: x += self.speakers_embed(speaker).unsqueeze(1)  # (B_L,P,D), (B_L,1,D) -> (B_L,P,D)
        x = torch.cat((self.cls_token.expand(B_L, -1, -1), x), dim=1)           # (B_L,P,D) -> (B_L,P+1,D)
        output = self.layers(x) # (B_L,P+1,D) -> (B_L,P+1,D)
        return output[:, 0, :]  # (B_L,P+1,D) -> (B_L,D)

class CenterOfMassDistance(Metric):
    def __init__(self, out_h:int, out_w:int, norm:int=2, epsilon:float=1e-6):
        super().__init__()
        self.p, self.epsilon = norm, epsilon
        self.register_buffer("xs", torch.arange(out_w).float())
        self.register_buffer("ys", torch.arange(out_h).float())
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def _com(self, mask):
        total = mask.sum((-2, -1), keepdim=True).clamp(min=self.epsilon)
        cx = (self.xs * mask.sum(-2)).sum(-1) / total.squeeze()
        cy = (self.ys * mask.sum(-1)).sum(-1) / total.squeeze()
        return torch.stack([cx, cy], dim=-1)

    def update(self, mask_pred, mask_true):
        valid = mask_true.sum((-2, -1)) > 0  # skip empty ground-truth masks
        if not valid.any(): return
        com_distances = torch.linalg.norm(self._com(mask_pred[valid]) - self._com(mask_true[valid]), ord=self.p, dim=-1)
        self.total, self.count = self.total + com_distances.sum(), self.count + valid.sum()

    def compute(self): return self.total / self.count

def create_metrics(data_info): return {"mse": MeanSquaredError(), 'com-distance': CenterOfMassDistance(data_info['out_h'], data_info['out_w'])}

class VibrationTransformer(ComposerModel):
    def __init__(self, d_model:int=128, pnt_num_heads:int=2, pnt_num_layers:int=2, seq_num_heads:int=2, seq_num_layers:int=2, data_info=DATA_INFO, signal_mode:str='magnitude', normalize_mode:str='z'):
        super().__init__()

        # encoder
        self.laser_encoder = LaserEncoder(data_info['patch_size'], d_model, pnt_num_heads, pnt_num_layers, data_info['n_freqs'], signal_mode, normalize_mode)
        self.box_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=seq_num_heads, batch_first=True), num_layers=seq_num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # Initialize to small random values
        self.register_buffer("freqs_laser", precompute_freqs_cis_2d(d_model, data_info['n_laser_rows'], data_info['n_laser_cols'])) # for laser grid

        # decoder
        self.decoder = MLPDecoder(d_model, data_info['out_h'], data_info['out_w'])

        # metrics
        self.train_metrics, self.val_metrics = create_metrics(data_info), create_metrics(data_info)

    def forward(self, batch):
        # B=batch size, L=n_lasers, C=n_coordinates=2, PS=patch_size, D=d_model
        x = batch['fft'] # (B,L,P,2,PS)
        B, L, _, _, _ = x.shape

        # LaserEncoder learns patterns between all frequencies from a single laser
        # flatten so LaserEncoder processes all lasers AND all batches in parallel
        speaker = speaker.repeat_interleave(L) if (speaker := batch.get('speakers_encoded', None)) is not None else None # (BL,D)
        x = self.laser_encoder(x.flatten(0, 1), speaker).reshape(B, L, -1)  # (B,L,P,C,PS) -> (B,L,D)

        # BoxEncoder learns patterns between ALL the lasers shining on the box
        x = apply_rope(x, self.freqs_laser) # (B,L,D) -> (B,L,D)
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
        metric.update(outputs['mask_pred'], batch['mask_true'])

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)
