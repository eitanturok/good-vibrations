"""transformer: src's VibrationTransformer (a freq transformer per laser, then a laser transformer) with hybrid's recipe; encoder = transformers only.
per-box input standardization -> each laser's spectrum cut into 19 patches of 64 bins (~49 Hz), a token each, + band / laser (learned + 2D wall
position) / box embeddings -> 2 freq blocks over each laser's bands -> gated bands, concatenated -> one token per laser -> 2 laser blocks -> flatten
-> one embedding -> conv decoder (mask + outline), aux + orientation heads (as hybrid). Train-time augs in the model: speaker gain, laser jitter.
loss: mask bce + edge (bce-dice on the outline) + per-instance moments + orientation + aux heads.
batch: fft (B,L,F,C), box (B,), mask_true (B,H,W), inst (B,K,H,W), object, n_objects (B,), coms (B,K,2)"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanMetric

from utils import speaker_gain, laser_jitter, grid_coords, aux_losses, bce_dice, instance_moment_loss, orient_loss, outline, METRICS

#*** 1 blocks

class TwoBranchUp(nn.Module):
    """2x upsampling: a transposed conv, and a conv then a transposed conv, concatenated"""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up_a = nn.ConvTranspose2d(c_in, c_out // 2, kernel_size=4, stride=2, padding=1)
        self.up_b = nn.ConvTranspose2d(c_in, c_out // 2, kernel_size=4, stride=2, padding=1)
        self.conv_b = nn.Conv2d(c_in, c_in, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x): return F.relu(self.bn(torch.cat([self.up_a(x), self.up_b(F.relu(self.conv_b(x)))], 1)))  # (B,c_out,2H,2W)

class Block(nn.Module):
    """pre-norm: self-attention, then an MLP"""
    def __init__(self, d, heads, mlp=2):
        super().__init__()
        self.h, self.n1, self.qkv, self.o = heads, nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.mlp = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x):
        N, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(N, T, 3, self.h, -1).permute(2, 0, 3, 1, 4)  # 3 x (N,H,T,D/H)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, T, D))
        return x + self.mlp(self.n2(x))

#*** 2 encoder

class Encoder(nn.Module):
    """(B,L,F,C) spectrum + (B,) box -> (B,d_model)"""
    def __init__(self, n_channels, freqs, rows, cols, n_boxes, d_model, patch=64, d=128, heads=4, freq_layers=2, laser_layers=2):
        super().__init__()
        L, F_ = rows * cols, len(freqs)
        P = math.ceil(F_ / patch)                                                     # 1170 bins -> 19 bands
        self.patch, self.heads = patch, heads
        self.register_buffer("in_mean", torch.zeros(n_boxes, L, F_, n_channels))
        self.register_buffer("in_sq", torch.ones(n_boxes, L, F_, n_channels))
        self.register_buffer("in_n", torch.zeros(n_boxes))
        logf = torch.log10(torch.as_tensor(freqs, dtype=torch.float32)) - 2.5         # 100-1000 Hz -> -.5...5
        self.register_buffer("band_logf", logf[(torch.arange(P) * patch + patch // 2).clamp(max=F_ - 1)], persistent=False)  # (P,) band centres
        a = grid_coords(rows, cols)[..., None] * (2 ** torch.arange(8) * math.pi)    # laser (row,col) -> sin/cos at 2^i pi
        self.register_buffer("laser_fourier", torch.cat([a.sin(), a.cos()], -1).flatten(-2), persistent=False)  # (L,32)
        self.embed = nn.Linear(patch * n_channels, d)                                 # raw bins in: keeps exact peak positions
        self.band_emb = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.laser_emb = nn.Parameter(nn.init.trunc_normal_(torch.zeros(L, d), std=0.02))
        self.laser_pos = nn.Linear(4 * 8, d)
        self.box_emb = nn.Embedding(n_boxes, d)
        self.freqs = nn.ModuleList([Block(d, heads, mlp=4) for _ in range(freq_layers)])
        self.pool_q = nn.Parameter(nn.init.trunc_normal_(torch.zeros(heads, d // heads), std=0.02))
        self.pool_norm, self.pool_k, self.pool_o = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(P * d, d)
        self.lasers = nn.ModuleList([Block(d, heads, mlp=4) for _ in range(laser_layers)])
        self.to_emb = nn.Sequential(nn.LayerNorm(d), nn.Flatten(1), nn.Linear(L * d, d_model))

    def normalize(self, x, box):
        """per-box standardization; in training, update the running per-box mean / mean square first (cumulative, then EMA .01)"""
        if self.training:
            with torch.no_grad():
                oh = F.one_hot(box, len(self.in_n)).float()                          # (B,nb)
                cnt, xf = oh.sum(0), x.float()
                m1 = torch.einsum("bn,blfc->nlfc", oh, xf) / cnt.clamp_min(1)[:, None, None, None]
                m2 = torch.einsum("bn,blfc->nlfc", oh, xf * xf) / cnt.clamp_min(1)[:, None, None, None]
                w = torch.where(cnt > 0, (cnt / (self.in_n + cnt).clamp_min(1)).clamp_min(0.01), torch.zeros_like(cnt))[:, None, None, None]
                self.in_mean.lerp_(m1, w); self.in_sq.lerp_(m2, w); self.in_n.add_(cnt)
        mean, sq = self.in_mean[box], self.in_sq[box]
        return ((x - mean) * (sq - mean ** 2).clamp_min(1e-6).rsqrt()).to(x.dtype)

    def forward(self, x, box):
        B, L, F_, C = x.shape
        P = len(self.band_logf)
        x = F.pad(self.normalize(x, box), (0, 0, 0, P * self.patch - F_)).view(B, L, P, self.patch * C)  # (B,L,P,patch*C)
        laser = self.laser_emb + self.laser_pos(self.laser_fourier)                   # (L,d)
        x = self.embed(x) + self.band_emb(self.band_logf[:, None])[None, None] + laser[None, :, None] + self.box_emb(box)[:, None, None]
        x = x.flatten(0, 1)                                                           # (B*L,P,d)
        for block in self.freqs: x = block(x)                                         # over each laser's bands
        x = x.view(B, L, P, -1)
        k = self.pool_k(self.pool_norm(x)).view(B, L, P, self.heads, -1)             # (B,L,P,H,d/H)
        a = (torch.einsum("blphe,he->blhp", k, self.pool_q) / math.sqrt(k.shape[-1])).softmax(-1)  # (B,L,H,P) which bands each laser reads
        x = self.pool_o((x * a.mean(2)[..., None] * P).reshape(B, L, -1)) + laser     # gated bands, concatenated -> (B,L,d)
        for block in self.lasers: x = block(x)                                        # over the lasers
        return self.to_emb(x)                                                          # (B,L,d) -> (B,d_model)

#*** 3 decoder

class Decoder(nn.Module):
    """(B,d_model) -> (B,out_c,64,64) logits: a 512x4x4 seed, 4 -> 8 -> 16 -> 32 -> 64 with channels 512 -> 256 -> 128 -> 64 -> 32, a 3-conv head"""
    def __init__(self, d_model, out_c=1):
        super().__init__()
        self.project = nn.Linear(d_model, 512 * 4 * 4)
        self.ups = nn.ModuleList([TwoBranchUp(c, c // 2) for c in (512, 256, 128, 64)])
        self.conv1, self.conv2, self.conv3 = nn.Conv2d(32, 32, 3, padding=1), nn.Conv2d(32, 32, 3, padding=1), nn.Conv2d(32, out_c, 3, padding=1)

    def forward(self, emb):
        x = self.project(emb).view(-1, 512, 4, 4)
        for up in self.ups: x = up(x)                      # (B,32,64,64)
        return self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))

#*** 4 aux heads

# the heads (object / count / com / area / box + orient) are built in Transformer.__init__: their state_dict names are flat
def mlp(d, n_out): return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_out))

#*** 5 losses

WEIGHTS = dict(object=0.1, n_objects=0.1, com=1.0, area=0.1, box=0.1, edge=1.0, imoment=1.0, orient=1.0)  # on top of the mask bce

def total_loss(out, batch, n_known):
    mask = batch["mask_true"]
    losses = dict(mask=F.binary_cross_entropy_with_logits(out["mask_logits"], mask),
                  **aux_losses(out, batch, n_known),
                  edge=bce_dice(out["edge_logits"], outline(mask)),
                  imoment=instance_moment_loss(out["mask_pred"], (batch["inst"] > 0.5).float()),
                  orient=orient_loss(out["orient"], mask, batch["n_objects"]))
    return dict(total=losses["mask"] + sum(w * losses[k] for k, w in WEIGHTS.items()), **losses)

#*** 6 model

class Transformer(ComposerModel):
    def __init__(self, data_info, d_model=1024, speaker_gain=0.35, speaker_gain_hz=20.0, laser_jitter=0.5):
        super().__init__()
        self.freq_hz, self.grid_rc = data_info["freq_range"], (data_info["n_laser_rows"], data_info["n_laser_cols"])
        freqs = torch.linspace(*self.freq_hz, data_info["n_freqs"])                    # (F,) Hz, evenly spaced bins
        self.encoder = Encoder(data_info["n_channels"], freqs, *self.grid_rc, len(data_info["boxes"]), d_model)
        self.decoder = Decoder(d_model, out_c=2)                                       # channels: mask, outline
        self.n_known, K = len(data_info["objects"]) - 1, max(data_info["n_objects"])  # objects[-1] = "unseen", no logit
        self.object_head, self.n_objects_head, self.com_head = mlp(d_model, self.n_known), mlp(d_model, K + 1), mlp(d_model, 2 * K)
        self.area_head, self.box_head = mlp(d_model, 1), mlp(d_model, len(data_info["boxes"]))
        self.orient_head = mlp(d_model, 2)                                             # ecc * (cos 2θ, sin 2θ) of a single object
        self.speaker_gain, self.speaker_gain_hz, self.laser_jitter = speaker_gain, speaker_gain_hz, laser_jitter  # train-time augs
        self.train_metrics, self.test_metrics = {k: MeanMetric() for k in METRICS}, {k: MeanMetric() for k in METRICS}
        for k, m in [*self.train_metrics.items(), *self.test_metrics.items()]: m.key = k

    def forward(self, batch):
        x = batch["fft"]
        if self.training and self.speaker_gain > 0: x = speaker_gain(x, self.freq_hz, self.speaker_gain, self.speaker_gain_hz)
        if self.training and self.laser_jitter > 0: x = laser_jitter(x, *self.grid_rc, self.laser_jitter)
        emb = self.encoder(x, batch["box"])                                            # (B,d_model)
        y = self.decoder(emb)                                                          # (B,2,H,W)
        return dict(mask_logits=y[:, 0], mask_pred=y[:, 0].sigmoid(), edge_logits=y[:, 1], orient=self.orient_head(emb).float(),
                    object_logits=self.object_head(emb), n_objects_logits=self.n_objects_head(emb), com=self.com_head(emb).sigmoid().view(len(emb), -1, 2),
                    log_area=self.area_head(emb).squeeze(-1), box_logits=self.box_head(emb))

    def loss(self, out, batch): return total_loss(out, batch, self.n_known)
    def get_metrics(self, is_train=False): return self.train_metrics if is_train else self.test_metrics
    def update_metric(self, batch, out, metric): metric.update(METRICS[metric.key](out, batch))
