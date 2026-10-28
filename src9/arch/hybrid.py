"""hybrid: the model of run s8-mag-nopose-imom-region-500ep (src8 --encoder modal --decoder conv, no pose branch).
per-box input standardization -> 3 stride-4 convs over frequency (log10 f as an extra channel) -> 19 band tokens (~49 Hz) per laser, + laser /
band / box embeddings -> gated bands, concatenated -> one token per laser -> 2 transformer blocks over the 100 lasers -> flatten -> one embedding
-> conv decoder (mask + outline), aux + orientation heads. Train-time augs in the model: speaker gain, laser jitter (utils).
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

class ConvBlock(nn.Module):
    """conv -> batchnorm -> leaky relu"""
    def __init__(self, c_in, c_out, kernel, stride, padding):
        super().__init__()
        self.conv, self.bn = nn.Conv2d(c_in, c_out, kernel, stride, padding), nn.BatchNorm2d(c_out)

    def forward(self, x): return F.leaky_relu(self.bn(self.conv(x)), 0.2)

class TwoBranchUp(nn.Module):
    """2x upsampling: a transposed conv, and a conv then a transposed conv, concatenated"""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up_a = nn.ConvTranspose2d(c_in, c_out // 2, kernel_size=4, stride=2, padding=1)
        self.up_b = nn.ConvTranspose2d(c_in, c_out // 2, kernel_size=4, stride=2, padding=1)
        self.conv_b = nn.Conv2d(c_in, c_in, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x): return F.relu(self.bn(torch.cat([self.up_a(x), self.up_b(F.relu(self.conv_b(x)))], 1)))  # (B,c_out,2H,2W)

class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.q, self.kv, self.o = heads, nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, x):
        N, T, D = x.shape
        q = self.q(x).view(N, T, self.h, -1).transpose(1, 2)                       # (N,H,T,D/H)
        k, v = self.kv(x).view(N, T, 2, self.h, -1).permute(2, 0, 3, 1, 4)         # 2 x (N,H,T,D/H)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(N, T, D))

class Block(nn.Module):
    """pre-norm: self-attention, then an MLP"""
    def __init__(self, d, heads, mlp=2):
        super().__init__()
        self.n1, self.attn, self.n2 = nn.LayerNorm(d), Attention(d, heads), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))

#*** 2 encoder

class ModalEncoder(nn.Module):
    """(B,L,F,C) spectrum + (B,) box -> (B,d_model)"""
    def __init__(self, n_channels, freqs, rows, cols, n_boxes, d_model, d=128, heads=4, layers=2):
        super().__init__()
        L, F_ = rows * cols, len(freqs)
        self.register_buffer("in_mean", torch.zeros(n_boxes, L, F_, n_channels))
        self.register_buffer("in_sq", torch.ones(n_boxes, L, F_, n_channels))
        self.register_buffer("in_n", torch.zeros(n_boxes))
        self.register_buffer("logf", torch.log10(torch.as_tensor(freqs, dtype=torch.float32)) - 2.5, persistent=False)  # (F,) 100-1000 Hz -> -.5...5
        self.conv1 = ConvBlock(n_channels + 1, 32, kernel=(1, 7), stride=(1, 4), padding=(0, 3))
        self.conv2 = ConvBlock(32, 64, kernel=(1, 5), stride=(1, 4), padding=(0, 2))
        self.conv3 = ConvBlock(64, 128, kernel=(1, 5), stride=(1, 4), padding=(0, 2))
        P = math.ceil(math.ceil(math.ceil(F_ / 4) / 4) / 4)                          # 1170 -> 293 -> 74 -> 19 bands of 64 bins
        self.register_buffer("band_logf", self.logf[(torch.arange(P) * 64 + 32).clamp(max=F_ - 1)], persistent=False)  # (P,) band centres
        a = grid_coords(rows, cols)[..., None] * (2 ** torch.arange(8) * math.pi)    # laser (row,col) -> sin/cos at 2^i pi
        self.register_buffer("laser_fourier", torch.cat([a.sin(), a.cos()], -1).flatten(-2), persistent=False)  # (L,32)
        self.heads = heads
        self.embed = nn.Linear(128, d)
        self.laser_emb = nn.Parameter(nn.init.trunc_normal_(torch.zeros(L, d), std=0.02))
        self.laser_pos = nn.Linear(4 * 8, d)
        self.band_emb = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.box_emb = nn.Embedding(n_boxes, d)
        self.pool_q = nn.Parameter(nn.init.trunc_normal_(torch.zeros(heads, d // heads), std=0.02))
        self.pool_norm, self.pool_k, self.pool_o = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(P * d, d)
        self.lasers = nn.ModuleList([Block(d, heads) for _ in range(layers)])
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
        x = self.normalize(x, box)
        x = torch.cat([x, self.logf[:, None].expand(B, L, F_, 1)], -1).permute(0, 3, 1, 2)  # (B,C+1,L,F)
        x = self.conv3(self.conv2(self.conv1(x)))                                     # (B,128,L,P)
        P = x.shape[-1]
        x = self.embed(x.permute(0, 2, 3, 1))                                         # (B,L,P,d)
        x = x + (self.laser_emb + self.laser_pos(self.laser_fourier))[None, :, None] + self.band_emb(self.band_logf[:, None])[None, None] + self.box_emb(box)[:, None, None]
        k = self.pool_k(self.pool_norm(x)).view(B, L, P, self.heads, -1)             # (B,L,P,H,d/H)
        a = (torch.einsum("blphe,he->blhp", k, self.pool_q) / math.sqrt(k.shape[-1])).softmax(-1)  # (B,L,H,P) which bands each laser reads
        x = self.pool_o((x * a.mean(2)[..., None] * P).reshape(B, L, -1))            # gated bands, concatenated -> (B,L,d)
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

def mlp(d, n_out): return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_out))

class AuxHeads(nn.Module):
    """heads on the embedding: which objects, how many, each one's com, log area, which box. objects[-1] = "unseen", no logit"""
    def __init__(self, d, data_info):
        super().__init__()
        self.n_known, K = len(data_info["objects"]) - 1, max(data_info["n_objects"])
        self.object_head, self.n_objects_head, self.com_head = mlp(d, self.n_known), mlp(d, K + 1), mlp(d, 2 * K)
        self.area_head, self.box_head = mlp(d, 1), mlp(d, len(data_info["boxes"]))

    def forward(self, emb):
        return dict(object_logits=self.object_head(emb), n_objects_logits=self.n_objects_head(emb), com=self.com_head(emb).sigmoid().view(len(emb), -1, 2),
                    log_area=self.area_head(emb).squeeze(-1), box_logits=self.box_head(emb))

#*** 5 losses

WEIGHTS = dict(object=0.1, n_objects=0.1, com=1.0, area=0.1, box=0.1, edge=1.0, imoment=1.0, orient=1.0)  # on top of the mask bce

def total_loss(out, batch, n_known):
    losses = dict(mask=F.binary_cross_entropy_with_logits(out["mask_logits"], batch["mask_true"]), **aux_losses(out, batch, n_known),
                  edge=bce_dice(out["edge_logits"], outline(batch["mask_true"])),
                  imoment=instance_moment_loss(out["mask_pred"], (batch["inst"] > 0.5).float()),
                  orient=orient_loss(out["orient"], batch["mask_true"], batch["n_objects"]))
    return dict(total=losses["mask"] + sum(w * losses[k] for k, w in WEIGHTS.items()), **losses)

#*** 6 model

class Hybrid(ComposerModel):
    def __init__(self, data_info, d_model=1024, speaker_gain=0.35, speaker_gain_hz=20.0, laser_jitter=0.5):
        super().__init__()
        self.freq_hz, self.grid_rc = data_info["freq_range"], (data_info["n_laser_rows"], data_info["n_laser_cols"])
        freqs = torch.linspace(*self.freq_hz, data_info["n_freqs"])                    # (F,) Hz, evenly spaced bins
        self.encoder = ModalEncoder(data_info["n_channels"], freqs, *self.grid_rc, len(data_info["boxes"]), d_model)
        self.decoder = Decoder(d_model, out_c=2)                                       # channels: mask, outline
        self.aux = AuxHeads(d_model, data_info)
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
        return dict(mask_logits=y[:, 0], mask_pred=y[:, 0].sigmoid(), edge_logits=y[:, 1], orient=self.orient_head(emb).float(), **self.aux(emb))

    def loss(self, out, batch): return total_loss(out, batch, self.aux.n_known)
    def get_metrics(self, is_train=False): return self.train_metrics if is_train else self.test_metrics
    def update_metric(self, batch, out, metric): metric.update(METRICS[metric.key](out, batch))
