"""transformer3 (= std-samplez-v1: ID .656 / OOD .367 / mean miss .155 at ep200, 13,387,216 params).
|X| (B,L,n_bins,C) -> per-sample z over (L,n_bins,C) -> 18 bands of 65 bins, a token each -> shared Linear to d -> 4 freq blocks per laser (rope over bands)
-> f = sum_b m_b W_b LN(h_b) -> 4 laser blocks (2D rope over the grid) -> u = LN(sum_l m_l W_l LN(t_l)) + W_mask [m_l, m_b] (latent_dim 256)
-> Linear(u -> 512x4x4) -> 4 doublings -> 64x64 mask logits (bce + dice). Heads on u: object class (unseen ignored) and box class, CE x 0.1 each.
train only: speaker gain, laser jitter, token dropout (each laser / band dropped iid with prob laser_drop / band_drop, >= 1 of each kept)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanMetric

from utils import speaker_gain, laser_jitter, bce_dice, METRICS

#*** 1 blocks

def rope(pos, hd, base=100.):
    """(T,k) integer positions -> (2,T,hd) cos, sin; the hd/2 pairs split evenly over the k axes, pair (i, i + hd/2) turns by pos * base^(-j/n)"""
    n = hd // 2 // pos.shape[1]
    a = (pos[..., None] * base ** (-torch.arange(n) / n)).flatten(1)               # (T,hd/2)
    a = torch.cat([a, a], -1)
    return torch.stack([a.cos(), a.sin()])
def rotate(x, cs): x1, x2 = x.chunk(2, -1); return x * cs[0].to(x.dtype) + torch.cat([-x2, x1], -1) * cs[1].to(x.dtype)  # (N,H,T,hd)

class Block(nn.Module):
    """pre-norm self-attention (rope on q, k; keep = key mask) + MLP"""
    def __init__(self, d, heads, mlp=4):
        super().__init__()
        self.h, self.n1, self.qkv, self.o = heads, nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.mlp = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x, rope, keep):                                                # (N,T,D), (2,T,D/H), (N,T) 0 = masked key
        N, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(N, T, 3, self.h, -1).permute(2, 0, 3, 1, 4)  # 3 x (N,H,T,D/H)
        a = F.scaled_dot_product_attention(rotate(q, rope), rotate(k, rope), v, attn_mask=keep[:, None, None].bool())
        x = x + self.o(a.transpose(1, 2).reshape(N, T, D))
        return x + self.mlp(self.n2(x))

def per_token(n, d_in, d_out):
    """n (d_in -> d_out) matrices, nn.Linear init each, one shared bias -> (n,d_in,d_out), (d_out,)"""
    lins = [nn.Linear(d_in, d_out) for _ in range(n)]
    return nn.Parameter(torch.stack([l.weight.T for l in lins])), nn.Parameter(lins[0].bias.detach().clone())

class TwoBranchUp(nn.Module):
    """2x upsampling: a transposed conv, and a conv then a transposed conv, concatenated"""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up_a = nn.ConvTranspose2d(c_in, c_out // 2, kernel_size=4, stride=2, padding=1)
        self.up_b = nn.ConvTranspose2d(c_in, c_out // 2, kernel_size=4, stride=2, padding=1)
        self.conv_b = nn.Conv2d(c_in, c_in, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x): return F.relu(self.bn(torch.cat([self.up_a(x), self.up_b(F.relu(self.conv_b(x)))], 1)))  # (B,c_out,2H,2W)

def mlp(d, n_out): return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_out))

#*** 2 encoder

class FreqEncoder(nn.Module):
    """(B,L,n_bands,d) band tokens -> (B,L,d): blocks over each laser's bands, then f = sum_b m_b W_b LN(h_b) + b"""
    def __init__(self, n_bands, d, heads, layers):
        super().__init__()
        self.freqs = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.read_w, self.read_b = per_token(n_bands, d, d)                           # (n_bands,d,d), (d,)
        self.read_norm = nn.LayerNorm(d)
        self.register_buffer("rope", rope(torch.arange(n_bands)[:, None], d // heads), persistent=False)  # 1D: band index

    def forward(self, x, m_b):                                                       # (B,L,n_bands,d), (B,n_bands)
        B, L, n_bands, _ = x.shape
        x, keep = x.flatten(0, 1), m_b.repeat_interleave(L, 0)                        # (B*L,n_bands,d), (B*L,n_bands)
        for blk in self.freqs: x = blk(x, self.rope, keep)
        x = self.read_norm(x.view(B, L, n_bands, -1)) * m_b[:, None, :, None].to(x.dtype)
        return torch.einsum("blpd,pde->ble", x, self.read_w) + self.read_b             # (B,L,d)

class LaserEncoder(nn.Module):
    """(B,L,d) laser tokens -> u (B,latent_dim): blocks over the lasers, then u = LN(sum_l m_l W_l LN(t_l) + b) + W_mask [m_l, m_b]"""
    def __init__(self, rows, cols, n_bands, d, heads, layers, latent_dim):
        super().__init__()
        L = rows * cols
        self.lasers = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.read_w, self.read_b = per_token(L, d, latent_dim)                        # (L,d,latent_dim), (latent_dim,)
        self.read_norm, self.u_norm = nn.LayerNorm(d), nn.LayerNorm(latent_dim)
        self.w_mask = nn.Linear(L + n_bands, latent_dim)
        rc = torch.stack(torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij"), -1).view(L, 2)
        self.register_buffer("rope", rope(rc, d // heads), persistent=False)         # 2D: laser (row, col), row-major

    def forward(self, x, m_l, m_b):                                                  # (B,L,d), (B,L), (B,n_bands)
        for blk in self.lasers: x = blk(x, self.rope, m_l)
        u = torch.einsum("bld,lde->be", self.read_norm(x) * m_l[..., None].to(x.dtype), self.read_w) + self.read_b
        return self.u_norm(u) + self.w_mask(torch.cat([m_l, m_b], -1).to(u.dtype))   # (B,latent_dim) the bottleneck

class Encoder(nn.Module):
    """(B,L,n_bins,C) |X| -> u (B,latent_dim): per-sample z -> cut into bands -> shared Linear to d -> freq enc -> laser enc"""
    def __init__(self, n_channels, n_bins, rows, cols, bins_per_band, d, heads, freq_layers, laser_layers, latent_dim):
        super().__init__()
        assert n_bins % bins_per_band == 0, (n_bins, bins_per_band)
        self.L, self.n_bands, self.bins_per_band = rows * cols, n_bins // bins_per_band, bins_per_band  # 1170 bins -> 18 bands
        self.embed = nn.Linear(bins_per_band * n_channels, d)                         # shared by every band
        self.freq_enc = FreqEncoder(self.n_bands, d, heads, freq_layers)
        self.laser_enc = LaserEncoder(rows, cols, self.n_bands, d, heads, laser_layers, latent_dim)

    def forward(self, x, m_l, m_b):
        x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-6)  # z-score each sample
        x = x.reshape(*x.shape[:2], self.n_bands, self.bins_per_band, -1).flatten(-2)  # tokenize (B,L,n_bands,bins*C)
        return self.laser_enc(self.freq_enc(self.embed(x), m_b), m_l, m_b)            # (B,L,n_bands,d) -> (B,L,d) -> (B,latent_dim)

#*** 3 decoder

class Decoder(nn.Module):
    """u (B,latent_dim) -> (B,H,W) logits: Linear -> 512 x H/16 x W/16 seed, 4 doublings 512 -> 256 -> 128 -> 64 -> 32, a 3-conv head"""
    def __init__(self, latent_dim, height=64, width=64):
        super().__init__()
        assert height % 16 == 0 and width % 16 == 0, (height, width)
        self.h0, self.w0 = height // 16, width // 16
        self.project = nn.Linear(latent_dim, 512 * self.h0 * self.w0)
        self.ups = nn.ModuleList([TwoBranchUp(c, c // 2) for c in (512, 256, 128, 64)])
        self.conv1, self.conv2, self.conv3 = nn.Conv2d(32, 32, 3, padding=1), nn.Conv2d(32, 32, 3, padding=1), nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, u):
        x = self.project(u).view(-1, 512, self.h0, self.w0)
        for up in self.ups: x = up(x)                                                  # (B,32,H,W)
        return self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))).squeeze(1)

#*** 4 model

def drop(B, n, p, device):
    """token dropout -> (B,n) keep mask, 1 = keep; each token dropped iid with prob p, >= 1 kept"""
    keep = torch.rand(B, n, device=device) >= p
    keep[torch.arange(B, device=device), torch.randint(0, n, (B,), device=device)] |= ~keep.any(1)  # none kept -> keep a random one
    return keep.float()

class Transformer3(ComposerModel):
    def __init__(self, data_info, speaker_gain=0.35, speaker_gain_hz=20.0, laser_jitter=0.5, height=64, width=64, bins_per_band=65,
                 d=128, heads=4, freq_layers=4, laser_layers=4, latent_dim=256, laser_drop=0.05, band_drop=0.05, dice_w=0.5):
        super().__init__()
        self.freq_hz, self.grid_rc = data_info["freq_range"], (data_info["n_laser_rows"], data_info["n_laser_cols"])
        self.encoder = Encoder(data_info["n_channels"], data_info["n_freqs"], *self.grid_rc, bins_per_band, d, heads, freq_layers, laser_layers, latent_dim)
        self.decoder = Decoder(latent_dim, height, width)
        self.n_known = len(data_info["objects"]) - 1                                  # objects[-1] = "unseen": no logit, ignored by the CE
        self.object_head, self.box_head = mlp(latent_dim, self.n_known), mlp(latent_dim, len(data_info["boxes"]))
        self.speaker_gain, self.speaker_gain_hz, self.laser_jitter, self.laser_drop, self.band_drop, self.dice_w = speaker_gain, speaker_gain_hz, laser_jitter, laser_drop, band_drop, dice_w
        keys = [k for k in METRICS if k not in ("n_objects_err", "com_err")]          # no n_objects / com heads
        self.train_metrics, self.test_metrics = {k: MeanMetric() for k in keys}, {k: MeanMetric() for k in keys}
        for k, m in [*self.train_metrics.items(), *self.test_metrics.items()]: m.key = k

    def forward(self, batch):
        x, B, L, n_bands = batch["fft"], len(batch["fft"]), self.encoder.L, self.encoder.n_bands
        m_l, m_b = x.new_ones(B, L), x.new_ones(B, n_bands)
        if self.training:                                                              # augs + token dropout
            if self.speaker_gain > 0: x = speaker_gain(x, self.freq_hz, self.speaker_gain, self.speaker_gain_hz)
            if self.laser_jitter > 0: x = laser_jitter(x, *self.grid_rc, self.laser_jitter)
            if self.laser_drop or self.band_drop: m_l, m_b = drop(B, L, self.laser_drop, x.device), drop(B, n_bands, self.band_drop, x.device)
        u = self.encoder(x, m_l, m_b)                                                  # (B,latent_dim)
        y = self.decoder(u)                                                            # (B,H,W)
        return dict(mask_logits=y, mask_pred=y.sigmoid(), object_logits=self.object_head(u), box_logits=self.box_head(u), latent=u)

    def loss(self, out, batch):
        losses = dict(mask=bce_dice(out["mask_logits"], batch["mask_true"], self.dice_w),
                      object=F.cross_entropy(out["object_logits"], batch["object"], ignore_index=self.n_known),
                      box=F.cross_entropy(out["box_logits"], batch["box"]))
        return dict(total=losses["mask"] + 0.1 * losses["object"] + 0.1 * losses["box"], **losses)

    def get_metrics(self, is_train=False): return self.train_metrics if is_train else self.test_metrics
    def update_metric(self, batch, out, metric): metric.update(METRICS[metric.key](out, batch))
