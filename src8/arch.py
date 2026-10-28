import itertools, math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanMetric

#*** 1 layers

class ConvBlock(nn.Module):
    """conv -> batchnorm -> leaky relu"""
    def __init__(self, c_in, c_out, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), 0.2)

class TwoBranchUp(nn.Module):
    """The paper's 2x upsampling layer: a transposed conv, and a conv then a transposed conv, concatenated.
    Branch b's extra conv is what stops the two branches collapsing to the same function."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up_a = nn.ConvTranspose2d(c_in, c_out // 2, kernel_size=4, stride=2, padding=1)
        self.up_b = nn.ConvTranspose2d(c_in, c_out // 2, kernel_size=4, stride=2, padding=1)  # before conv_b: src/ draws init in this order
        self.conv_b = nn.Conv2d(c_in, c_in, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        a = self.up_a(x)                                   # (B,c_in,H,W) -> (B,c_out/2,2H,2W)
        b = F.relu(self.conv_b(x))                         # (B,c_in,H,W) -> (B,c_in,H,W)
        b = self.up_b(b)                                   # (B,c_in,H,W) -> (B,c_out/2,2H,2W)
        x = torch.cat([a, b], dim=1)                       # 2 x (B,c_out/2,2H,2W) -> (B,c_out,2H,2W)
        return F.relu(self.bn(x))                          # (B,c_out,2H,2W) -> (B,c_out,2H,2W)

#*** 2 encoder

class FreqEncoder(nn.Module):
    """(B,L,F,C) spectrum -> (B,256,L): a 256-dim feature per laser.
    Convolves frequency only (kernel height 1), so lasers don't mix here: the laser grid samples global standing waves,
    not local texture. Four stride-4 convs shrink frequency, then whatever is left of it is averaged out.
    The convs share weights over frequency, so on their own they can't tell 300Hz from 600Hz. Three opt-in fixes, all defined on
    frequency in Hz (not bin index), so they work for any audio band / any number of bins:
      gain:  x * exp(mlp(log10 f)) -- a learned per-frequency gain (a learned deconv), smooth in f; starts at 1
      coord: log10(f) - 2.5 as an extra input channel (CoordConv) -- every layer can act differently per band
      pool="flatten": keep 5 frequency positions and mix them with a linear layer, instead of averaging them away"""
    def __init__(self, n_channels, freqs, gain=False, coord=False, pool="mean", n_pool=5):
        super().__init__()
        self.register_buffer("logf", torch.log10(torch.as_tensor(freqs, dtype=torch.float32)) - 2.5, persistent=False)  # (F,); 100-1000Hz -> -0.5..0.5; not saved: old checkpoints still load
        self.gain = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1)) if gain else None
        if gain: nn.init.zeros_(self.gain[-1].weight); nn.init.zeros_(self.gain[-1].bias)              # exp(0) = 1 at init
        self.coord, self.pool, self.n_pool = coord, pool, n_pool
        self.conv1 = ConvBlock(n_channels + coord, 32, kernel=(1, 7), stride=(1, 4), padding=(0, 3))
        self.conv2 = ConvBlock(32, 64, kernel=(1, 5), stride=(1, 4), padding=(0, 2))
        self.conv3 = ConvBlock(64, 128, kernel=(1, 5), stride=(1, 4), padding=(0, 2))
        self.conv4 = ConvBlock(128, 256, kernel=(1, 5), stride=(1, 4), padding=(0, 2))
        self.mix = nn.Linear(256 * n_pool, 256) if pool == "flatten" else None

    def forward(self, x):
        if self.gain is not None: x = x * self.gain(self.logf[:, None]).exp()                         # (B,L,F,C) * (F,1)
        if self.coord: x = torch.cat([x, self.logf[:, None].expand(*x.shape[:3], 1)], dim=-1)          # (B,L,F,C) -> (B,L,F,C+1)
        x = x.permute(0, 3, 1, 2)                          # (B,L,F,C) -> (B,C,L,F)
        x = self.conv1(x)                                  # (B,C,L,F) -> (B,32,L,F/4)        e.g. F 1170 -> 293
        x = self.conv2(x)                                  # (B,32,L,F/4) -> (B,64,L,F/16)    293 -> 74
        x = self.conv3(x)                                  # (B,64,L,F/16) -> (B,128,L,F/64)  74 -> 19
        x = self.conv4(x)                                  # (B,128,L,F/64) -> (B,256,L,F/256) 19 -> 5
        if self.pool == "mean": return x.mean(dim=-1)      # (B,256,L,F/256) -> (B,256,L)
        x = F.adaptive_avg_pool2d(x, (x.shape[2], self.n_pool))           # (B,256,L,F/256) -> (B,256,L,n_pool); identity at F=1170
        return self.mix(x.permute(0, 2, 1, 3).flatten(2)).transpose(1, 2)  # (B,L,256*n_pool) -> (B,L,256) -> (B,256,L)

class LaserGridEncoder(nn.Module):
    """(B,256,L) per-laser features -> (B,d_model).
    Lays the lasers out on their (rows,cols) grid, then two stride-2 convs (10x10 -> 5x5 -> 3x3) and one conv the size of
    what's left (3x3 -> 1x1) mix the whole grid into one embedding."""
    def __init__(self, d_model, n_laser_rows, n_laser_cols):
        super().__init__()
        self.grid_shape = (n_laser_rows, n_laser_cols)
        h, w = math.ceil(n_laser_rows / 4), math.ceil(n_laser_cols / 4)
        self.conv1 = ConvBlock(256, 512, kernel=3, stride=2, padding=1)
        self.conv2 = ConvBlock(512, 1024, kernel=3, stride=2, padding=1)
        self.conv3 = ConvBlock(1024, d_model, kernel=(h, w), stride=1, padding=0)

    def forward(self, x):
        x = x.view(len(x), 256, *self.grid_shape)          # (B,256,L) -> (B,256,rows,cols)  e.g. 10x10
        x = self.conv1(x)                                  # (B,256,10,10) -> (B,512,5,5)
        x = self.conv2(x)                                  # (B,512,5,5) -> (B,1024,3,3)
        x = self.conv3(x)                                  # (B,1024,3,3) -> (B,d_model,1,1)
        return x.flatten(1)                                # (B,d_model,1,1) -> (B,d_model)

#*** 2b transformer encoder: src's VibrationTransformer (src/model/arch.py), its defaults

def rope_1d(dim, n, theta=10000.0):
    """(n,dim) [cos | sin] of position i at dim/2 rates"""
    freqs = torch.arange(n)[:, None] / theta ** (torch.arange(0, dim, 2)[:dim // 2] / dim)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

def rope_2d(dim, h, w, theta=10000.0):
    """(h*w,dim) [cos | sin] of grid cell (row, col): rows take every other rate, cols the rest (Pixtral)"""
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2).float() / dim)                              # (dim/2,)
    rows, cols = torch.outer(torch.arange(h).float(), freqs[::2]), torch.outer(torch.arange(w).float(), freqs[1::2])  # (h,dim/4), (w,dim/4)
    angles = torch.cat([rows[:, None].expand(h, w, -1), cols[None].expand(h, w, -1)], -1).reshape(h * w, dim // 2)
    return torch.cat([angles.cos(), angles.sin()], dim=-1)

def apply_rope(x, freqs):
    """(B,N,D) tokens, (N,D) [cos | sin] -> (B,N,D), channel i rotated with channel i + D/2"""
    cos, sin = freqs.chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

def transformer(d_model, heads, layers, ffn_dim):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, heads, ffn_dim or 4 * d_model, dropout=0.0, batch_first=True), layers)

class TransformerEncoder(nn.Module):
    """(B,L,P,PS,C) patched spectrum -> (B,d_model). A freq transformer per laser over its P patches of PS bins (1D RoPE), its cls
    token = that laser's feature; then a laser transformer over the L lasers (2D RoPE on the laser grid), its cls = the embedding"""
    def __init__(self, d_model, n_channels, patch_size, n_patches, n_laser_rows, n_laser_cols, heads=2, freq_layers=2, laser_layers=2, ffn_dim=None):
        super().__init__()
        self.embed = nn.Linear(patch_size * n_channels, d_model)
        self.freq_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.laser_cls = nn.Parameter(nn.init.trunc_normal_(torch.zeros(1, 1, d_model), std=0.02))
        self.freq_layers, self.laser_layers = transformer(d_model, heads, freq_layers, ffn_dim), transformer(d_model, heads, laser_layers, ffn_dim)
        self.register_buffer("rope_freq", rope_1d(d_model, n_patches), persistent=False)
        self.register_buffer("rope_laser", rope_2d(d_model, n_laser_rows, n_laser_cols), persistent=False)

    def forward(self, x):
        B, L, P, PS, C = x.shape
        x = apply_rope(self.embed(x.reshape(B * L, P, PS * C)), self.rope_freq)                  # (B,L,P,PS,C) -> (B*L,P,D)
        x = self.freq_layers(torch.cat([self.freq_cls.expand(B * L, -1, -1), x], 1))[:, 0]       # (B*L,1+P,D) -> (B*L,D) each laser's cls
        x = apply_rope(x.view(B, L, -1), self.rope_laser)                                        # (B,L,D)
        return self.laser_layers(torch.cat([self.laser_cls.expand(B, -1, -1), x], 1))[:, 0]      # (B,1+L,D) -> (B,D) the box's cls

#*** 2c ModalFormer (src8): an axial laser x band transformer, then pixel queries on the box floor

def chunked_sdpa(q, k, v, bias=None, max_batch=65535):
    """scaled_dot_product_attention on (N,H,T,D) in chunks of N, so that N*H stays under the CUDA grid limit (65535)"""
    n = max(1, max_batch // q.shape[1])
    if bias is not None: bias = bias.to(q.dtype)
    if len(q) <= n: return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    pick = lambda i: None if bias is None else bias[i:i + n] if bias.shape[0] > 1 else bias
    return torch.cat([F.scaled_dot_product_attention(q[i:i + n], k[i:i + n], v[i:i + n], attn_mask=pick(i)) for i in range(0, len(q), n)])

class Attention(nn.Module):
    """multi-head attention of x (N,T,D) over ctx (N,S,D) (x itself if None), + an additive bias (N or 1,H or 1,T,S).
    weights=True also returns the (N,H,T,S) attention map (explain)"""
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.q, self.kv, self.o = heads, nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, x, ctx=None, bias=None, weights=False):
        ctx = x if ctx is None else ctx
        N, T, D = x.shape
        q = self.q(x).view(N, T, self.h, -1).transpose(1, 2)                                    # (N,H,T,D/H)
        k, v = self.kv(ctx).view(N, ctx.shape[1], 2, self.h, -1).permute(2, 0, 3, 1, 4)        # 2 x (N,H,S,D/H)
        if not weights: return self.o(chunked_sdpa(q, k, v, bias).transpose(1, 2).reshape(N, T, D)), None
        a = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]) + (0 if bias is None else bias.to(q.dtype))).softmax(-1)
        return self.o((a @ v).transpose(1, 2).reshape(N, T, D)), a

class Block(nn.Module):
    """pre-norm transformer block: attention (self, or cross to ctx), then an MLP"""
    def __init__(self, d, heads, mlp=2, cross=False):
        super().__init__()
        self.n1, self.attn, self.n2 = nn.LayerNorm(d), Attention(d, heads), nn.LayerNorm(d)
        self.nc = nn.LayerNorm(d) if cross else None
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x, ctx=None, bias=None, weights=False):
        a, w = self.attn(self.n1(x), None if ctx is None else self.nc(ctx), bias, weights)
        x = x + a
        return x + self.mlp(self.n2(x)), w

def fourier(u, n=8):
    """(...,k) coords in [0,1] -> (...,k*2n) sin/cos at 2^i pi, i < n"""
    a = u[..., None] * (2 ** torch.arange(n, device=u.device) * math.pi)                       # (...,k,n)
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)

def grid_coords(h, w, device=None):
    """(h*w,2) cell-center (row,col) in [0,1]"""
    r, c = torch.meshgrid((torch.arange(h, device=device) + 0.5) / h, (torch.arange(w, device=device) + 0.5) / w, indexing="ij")
    return torch.stack([r, c], -1).view(-1, 2)

class ModalFormer(nn.Module):
    """(B,L,F,C) spectrum + (B,) box -> (B,out_c+extra,out_h,out_w) logits, (B,d_model) embedding, maps for explain().
    1 stem: FreqEncoder's first three convs (frequency only, log10 f as an input channel) -> P band tokens per laser, 64 bins (~49 Hz)
      each; 128 channels keep where in its band a peak sits (a few Hz is the signal: probe.py, roll 9 Hz -> iou .30 -> .06)
    2 + laser embedding (learned per laser + fourier of its grid (row,col)), band embedding (mlp of the band's log10 Hz), box embedding
    3 axial blocks in `order`: L = attention over the lasers within one band (a mode shape), F = over the bands within one laser
    4 band pooling: per laser, `heads` learned queries over its bands -> one laser token; then `laser_layers` blocks over the lasers
    5 pixel queries on a GxG grid of the floor (fourier of (row,col) + box) cross-attend to the laser tokens, then self-attend; with
      geo: a learnable floor<->wall prior, head h adds -alpha_h |q - p_box(r,c)|^2 with p_box an affine map of the laser grid onto the
      floor (init: identity) and alpha_h >= 0 learned (init 0.1 .. 30: some heads start global)
    6 two 2x upsamplings (TwoBranchUp) G -> 4G, a 3-conv head -> out_c channels (+ edge, sdf): the head is the only task-specific part
    laser_drop > 0: in training each sample drops a U(0, laser_drop) fraction of its lasers (masked as keys wherever lasers are read)
    band_drop > 0: likewise for the P bands: the band's input bins are set to the box mean (0 after box_norm, so neighbouring tokens'
      receptive fields see nothing there either), its token is zeroed and masked out of the band gate. A model trained this way can have
      any laser / band truly removed at eval (remove_lasers: (L,), remove_bands: (P,) or (L,P) per laser; bool, True = removed) and stay in distribution"""
    def __init__(self, n_channels, freqs, n_rows, n_cols, n_boxes, d_model, out_h, out_w, out_c, n_extra, d=128, heads=4, order="LFLF",
                 laser_layers=2, dec_layers=2, grid=16, geo=True, laser_drop=0.0, band_drop=0.0, checkpoint=True, box_norm=True, pool="gate", queries=True, lin_skip=0, box_film=False):
        super().__init__()
        # box_norm: x' = (x - mean_box) / std_box per (laser, bin, axis), running stats of the training batches. The object is a small
        # perturbation of the box's own spectrum: without this the shared part swamps it and LayerNorm (per token) never removes it,
        # so the model fit a constant mask (fitdiag.py: train bce stuck at .075 for every lr; the conv model's BatchNorm does this job)
        self.box_norm = box_norm
        if box_norm:
            self.register_buffer("in_mean", torch.zeros(n_boxes, n_rows * n_cols, len(freqs), n_channels))
            self.register_buffer("in_sq", torch.ones(n_boxes, n_rows * n_cols, len(freqs), n_channels))
            self.register_buffer("in_n", torch.zeros(n_boxes))
        self.register_buffer("logf", torch.log10(torch.as_tensor(freqs, dtype=torch.float32)) - 2.5, persistent=False)  # (F,)
        self.conv1 = ConvBlock(n_channels + 1, 32, kernel=(1, 7), stride=(1, 4), padding=(0, 3))
        self.conv2 = ConvBlock(32, 64, kernel=(1, 5), stride=(1, 4), padding=(0, 2))
        self.conv3 = ConvBlock(64, 128, kernel=(1, 5), stride=(1, 4), padding=(0, 2))
        # box_film: per-box scale/shift of every stem layer's channels (FiLM, Perez et al. 2018). Finding a resonance's shift is a matched
        # filter whose best shape depends on the linewidth, and the boxes differ there (cardboard is heavily damped, nb99): one filter
        # bank for all three boxes is a compromise. Starts as the identity (gamma = beta = 0)
        self.film = nn.ParameterList([nn.Parameter(torch.zeros(n_boxes, 2, c)) for c in (32, 64, 128)]) if box_film else None
        n_bands = -(-(-(-(-(-len(freqs) // 4)) // 4)) // 4)
        self.register_buffer("band_logf", self.logf[(torch.arange(n_bands) * 64 + 32).clamp(max=len(freqs) - 1)], persistent=False)  # (P,)
        self.register_buffer("laser_rc", grid_coords(n_rows, n_cols), persistent=False)       # (L,2)
        self.rows, self.cols, self.grid, self.order, self.laser_drop, self.checkpoint, self.heads = n_rows, n_cols, grid, order, laser_drop, checkpoint, heads
        self.band_drop, self.remove_lasers, self.remove_bands = band_drop, None, None
        self.embed = nn.Linear(128, d)
        self.laser_emb = nn.Parameter(nn.init.trunc_normal_(torch.zeros(n_rows * n_cols, d), std=0.02))
        self.laser_pos = nn.Linear(4 * 8, d)
        self.band_emb = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.box_emb = nn.Embedding(n_boxes, d)
        self.axial = nn.ModuleList([Block(d, heads) for _ in order])
        self.pool_q = nn.Parameter(nn.init.trunc_normal_(torch.zeros(heads, d // heads), std=0.02))
        self.pool_norm, self.pool_kv, self.pool_o = nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        # pool="gate": each band token times its gate (softmax over bands x P, mean of the heads), all P concatenated -> Linear. "attn" =
        # attention-pooling (a weighted MEAN over bands): it washed the per-band content out (across-sample std .86 -> .10 of the total;
        # the laser embedding, the same in every band, survives a mean intact) and the model fit a constant mask (fitdiag.py)
        self.pool = pool
        if pool == "gate": self.pool_o = nn.Linear(n_bands * d, d)
        self.lasers = nn.ModuleList([Block(d, heads) for _ in range(laser_layers)])
        self.to_emb = nn.Sequential(nn.LayerNorm(d), nn.Flatten(1), nn.Linear(n_rows * n_cols * d, d_model))  # flatten, not a mean: keeps which laser
        self.queries = queries  # False: no pixel-query decoder, the embedding goes to Boombox's conv decoder (hybrid)
        # lin_skip k > 0: a rank-k LINEAR map from the input (box-normalized, averaged to ~7.7 Hz bands) straight into the embedding. A ridge
        # probe on the input reads rotated-soap orientation at 17 deg; the deep path gets ~28-36: does it lose what is linearly there?
        self.lin_bins = len(freqs) // 10
        self.lin_skip = nn.Sequential(nn.Linear(n_rows * n_cols * self.lin_bins * n_channels, lin_skip), nn.Linear(lin_skip, d_model)) if lin_skip else None
        self.register_buffer("query_rc", grid_coords(grid, grid), persistent=False)          # (G*G,2)
        self.query_pos = nn.Linear(4 * 8, d)
        self.cross = nn.ModuleList([Block(d, heads, cross=True) for _ in range(dec_layers)])
        self.self_ = nn.ModuleList([Block(d, heads) for _ in range(dec_layers)])
        self.geo = geo
        if geo:  # p_box(r,c) = offset + A @ (r,c); alpha_h = softplus(raw), per decoder layer and head
            self.geo_offset = nn.Parameter(torch.zeros(n_boxes, 2))
            self.geo_A = nn.Parameter(torch.eye(2).repeat(n_boxes, 1, 1))
            alpha = torch.logspace(-1, math.log10(30), heads).repeat(dec_layers, 1)
            self.geo_alpha = nn.Parameter(alpha.expm1().log())
        self.up = nn.Sequential(TwoBranchUp(d, 128), TwoBranchUp(128, 64))
        self.out_h, self.out_w = out_h, out_w
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, out_c + n_extra, 3, padding=1))

    def run(self, block, explain, x, ctx=None, bias=None):
        """block(x, ctx, bias) -> (x, attention map or None); activation-checkpointed in training (the (B,L,P,d) tokens are ~0.5 GB each)"""
        if explain: return block(x, ctx, bias, True)
        if self.checkpoint and self.training: return torch.utils.checkpoint.checkpoint(block, x, ctx, bias, use_reentrant=False)[0], None
        return block(x, ctx, bias)[0], None

    def anchors(self, box):
        """(B,) -> (B,L,2) each laser's anchor on the floor, (row,col) in box fractions"""
        return self.geo_offset[box][:, None] + torch.einsum("bij,lj->bli", self.geo_A[box], self.laser_rc)

    def normalize(self, x, box):
        """(B,L,F,C) -> per-box standardized; in training, update the running per-box mean / mean square first (cumulative, then EMA 0.01)"""
        if self.training:
            with torch.no_grad():
                oh = F.one_hot(box, len(self.in_n)).float()                                  # (B,nb)
                cnt = oh.sum(0)                                                            # (nb,)
                xf = x.float()
                m1 = torch.einsum("bn,blfc->nlfc", oh, xf) / cnt.clamp_min(1)[:, None, None, None]
                m2 = torch.einsum("bn,blfc->nlfc", oh, xf * xf) / cnt.clamp_min(1)[:, None, None, None]
                w = torch.where(cnt > 0, (cnt / (self.in_n + cnt).clamp_min(1)).clamp_min(0.01), torch.zeros_like(cnt))[:, None, None, None]
                self.in_mean.lerp_(m1, w); self.in_sq.lerp_(m2, w); self.in_n.add_(cnt)
        mean, sq = self.in_mean[box], self.in_sq[box]
        return ((x - mean) * (sq - mean ** 2).clamp_min(1e-6).rsqrt()).to(x.dtype)

    def forward(self, x, box, explain=False):
        B, L, F_, C = x.shape
        maps = {}
        if self.box_norm: x = self.normalize(x, box)
        n_bands = len(self.band_logf)
        bkeep = torch.ones(B, L, n_bands, dtype=torch.bool, device=x.device)                  # (B,L,P)
        if self.training and self.band_drop > 0:                                              # the same bands for all lasers of a sample
            bkeep = (torch.rand(B, 1, n_bands, device=x.device) >= torch.rand(B, 1, 1, device=x.device) * self.band_drop).expand(B, L, n_bands)
        if self.remove_bands is not None: bkeep = bkeep & ~self.remove_bands.to(x.device).expand(L, n_bands)[None]
        x = x * bkeep.repeat_interleave(64, 2)[:, :, :F_, None].to(x.dtype)                 # removed bins -> box mean (always: a data-dependent if breaks the compiled graph)
        x_in = x                                                                               # (B,L,F,C), for lin_skip
        x = torch.cat([x, self.logf[:, None].expand(B, L, F_, 1)], -1).permute(0, 3, 1, 2)    # (B,L,F,C) -> (B,C+1,L,F)
        if self.film is None: x = self.conv3(self.conv2(self.conv1(x)))                       # -> (B,128,L,P)
        else:
            for conv, fp in zip((self.conv1, self.conv2, self.conv3), self.film):
                g = fp[box]                                                                    # (B,2,c)
                x = conv(x) * (1 + g[:, 0, :, None, None]) + g[:, 1, :, None, None]
        P = x.shape[-1]
        x = self.embed(x.permute(0, 2, 3, 1))                                                  # (B,L,P,d)
        box_e = self.box_emb(box)                                                              # (B,d)
        x = x + (self.laser_emb + self.laser_pos(fourier(self.laser_rc, 8)))[None, :, None] + self.band_emb(self.band_logf[:, None])[None, None] + box_e[:, None, None]
        d = x.shape[-1]
        keep = torch.ones(B, L, dtype=torch.bool, device=x.device)
        if self.training and self.laser_drop > 0:                                             # drop a U(0, laser_drop) fraction per sample
            keep = torch.rand(B, L, device=x.device) >= torch.rand(B, 1, device=x.device) * self.laser_drop
        if self.remove_lasers is not None: keep = keep & ~self.remove_lasers.to(x.device)[None]
        drop = torch.zeros(B, 1, 1, L, device=x.device).masked_fill(~keep[:, None, None], -1e4)  # (B,1,1,L) additive key mask
        for i, (kind, block) in enumerate(zip(self.order, self.axial)):
            if kind == "L":                                                                    # over lasers, per band
                y, w = self.run(block, explain, x.transpose(1, 2).reshape(B * P, L, d), None, drop.repeat_interleave(P, 0))
                x = y.view(B, P, L, d).transpose(1, 2)
                if explain: maps[f"axial{i}_L"] = w.view(B, P, *w.shape[1:])                  # (B,P,H,L,L)
            else:                                                                              # over bands, per laser
                y, w = self.run(block, explain, x.reshape(B * L, P, d))
                x = y.view(B, L, P, d)
                if explain: maps[f"axial{i}_F"] = w.view(B, L, *w.shape[1:])                  # (B,L,H,P,P)
        k, v = self.pool_kv(self.pool_norm(x)).view(B, L, P, 2, self.heads, -1).unbind(3)     # each (B,L,P,H,d/H)
        a = torch.einsum("blphe,he->blhp", k, self.pool_q) / math.sqrt(k.shape[-1])
        a = a.masked_fill(~bkeep[:, :, None], -1e4).softmax(-1)                                # (B,L,H,P) which bands each laser reads
        x = x * bkeep[..., None].to(x.dtype)                                                  # removed band tokens -> 0
        maps["band_pool"] = a
        if self.pool == "gate": x = self.pool_o((x * a.mean(2)[..., None] * P).reshape(B, L, P * d))  # (B,L,P,d) gated -> (B,L,P*d) -> (B,L,d)
        else: x = self.pool_o(torch.einsum("blhp,blphe->blhe", a, v).reshape(B, L, d))         # (B,L,d) one token per laser
        for i, block in enumerate(self.lasers):
            x, w = self.run(block, explain, x, None, drop)
            if explain: maps[f"laser{i}"] = w
        emb = self.to_emb(x * keep.float()[..., None])                                        # (B,L,d) -> (B,d_model), dropped lasers zeroed
        maps["tokens"], maps["keep"] = x, keep                                              # (B,L,d), (B,L): for --obj-decoder queries
        if self.lin_skip is not None: emb = emb + self.lin_skip(x_in[:, :, :self.lin_bins * 10].unflatten(2, (self.lin_bins, 10)).mean(3).flatten(1))
        if not self.queries: return None, emb, maps
        q = (self.query_pos(fourier(self.query_rc, 8))[None] + box_e[:, None]).expand(B, -1, -1)  # (B,G*G,d)
        bias = drop
        if self.geo:
            dist = (self.query_rc[None, :, None] - self.anchors(box)[:, None]).pow(2).sum(-1)  # (B,G*G,L)
        for i, (cross, self_) in enumerate(zip(self.cross, self.self_)):
            if self.geo: bias = drop - F.softplus(self.geo_alpha[i])[None, :, None, None] * dist[:, None]  # (B,H,G*G,L)
            q, w = self.run(cross, explain, q, x, bias)
            if explain: maps[f"cross{i}"] = w                                                  # (B,H,G*G,L) which lasers each floor cell reads
            q = self.run(self_, explain, q)[0]
        y = q.transpose(1, 2).reshape(B, d, self.grid, self.grid)                              # (B,d,G,G)
        y = self.up(y)                                                                         # (B,64,4G,4G)
        if y.shape[-2:] != (self.out_h, self.out_w): y = F.interpolate(y, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        return self.head(y), emb, maps

#*** 3a' field decoders (src8, 2026-10-22): the mask as a field over the floor, one small MLP SHARED by every cell

class FieldDecoder(nn.Module):
    """(B,d_model) embedding, (B,L,d) laser tokens, (B,) box -> (B,H,W) mask logits, (B,H,W) outline logits or None.
    Every floor cell r runs the same small MLP on its own (row, col) and the box, conditioned on the measurement. The position enters
    as low-frequency Fourier features (n_freq octaves, the finest 2^(n_freq-2) cycles per box), so the field is smooth in r: an unseen
    cell is drawn like its neighbours, not from a memorized template (the conv decoder paints trained positions next to an unseen
    region, never in it: gray-region UNSEEN-POS iou .055). A separate MLP per cell would memorize each cell the same way.
      field: c = W emb                       the global embedding, the same for every cell (v0, an implicit / CPPN decoder)
      modes: m_k = mean_l v_k(w_l) z_l       mode k's amplitude, read from the laser tokens z_l through its WALL shape v_k at each
                                             laser's (row, col) on the wall
             c(r) = sum_k u_k(r) m_k         and spread over the floor by its FLOOR shape u_k (v1, the modal picture: an object at r
                                             couples to mode k by u_k(r), laser l reads mode k by v_k(w_l)); u, v: small MLPs of
                                             Fourier features + the box, so both are smooth and can be plotted
      logit(r) = MLP(P fourier(r) + box + c(r))  -> mask (+ the outline channel)"""
    def __init__(self, d_model, d_tok, n_boxes, laser_rc, out_h, out_w, kind="field", width=128, n_freq=5, n_modes=32, edge=False):
        super().__init__()
        self.kind, self.out_h, self.out_w, self.n_freq, self.edge = kind, out_h, out_w, n_freq, edge
        self.register_buffer("cell_rc", grid_coords(out_h, out_w), persistent=False)          # (R,2) floor cell centres
        self.register_buffer("laser_rc", laser_rc, persistent=False)                          # (L,2) laser spots on the wall grid
        pos = lambda: nn.Linear(4 * n_freq, width)
        self.cell_pos, self.box = pos(), nn.Embedding(n_boxes, width)
        if kind == "field": self.cond = nn.Linear(d_model, width)
        else:
            shape = lambda: nn.Sequential(nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_modes))
            self.u_pos, self.u, self.v_pos, self.v = pos(), shape(), pos(), shape()              # floor shapes u_k(r), wall shapes v_k(w)
            self.cond = nn.Linear(d_tok, width)                                                # per mode: token space -> the cell MLP's
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1 + edge))

    def shapes(self, box):
        """(B,) -> u (B,R,K) floor shapes, v (B,L,K) wall shapes"""
        b = self.box(box)[:, None]
        return (self.u(self.u_pos(fourier(self.cell_rc, self.n_freq))[None] + b), self.v(self.v_pos(fourier(self.laser_rc, self.n_freq))[None] + b))

    def forward(self, emb, tokens, keep, box):
        B = len(emb)
        h = self.cell_pos(fourier(self.cell_rc, self.n_freq))[None] + self.box(box)[:, None]  # (B,R,w)
        if self.kind == "field": h = h + self.cond(emb)[:, None]
        else:
            u, v = self.shapes(box)
            v = v * keep.float()[..., None]                                                    # dropped lasers read nothing
            m = torch.einsum("blk,bld->bkd", v, tokens) / keep.float().sum(1).clamp_min(1)[:, None, None]  # (B,K,d) mode amplitudes
            h = h + torch.einsum("brk,bkw->brw", u, self.cond(m))                              # (B,R,w): sum_k u_k(r) W m_k
        y = self.mlp(h).float().view(B, self.out_h, self.out_w, -1)
        return y[..., 0], (y[..., 1] if self.edge else None)

#*** 3b pose-factored branch (src8): shape in an object-centred frame + a pose, warped to the floor (STN, Jaderberg et al. 2015)

class PoseBranch(nn.Module):
    """(B,d_model) embedding + (B,K,2) com (row, col in [0,1]) -> (B,H,W) mask logits, and (B,K,2) doubled-angle vectors.
    Objects are rigid: the floor mask = shape (+) pose. Each slot k < K decodes a canonical mask (object-centred, its long axis along
    the rows) on a res x res canvas spanning `window` of the box, and is warped to its pose: centre = the com head's slot k, angle
    θ_k = 1/2 atan2(v_k) from a doubled-angle vector v_k (slot 0 supervised by orient_loss on single objects). Rotation is one shared
    parameter, learned from every rotated object; the direct decoder alone has to learn each pose of each object as its own template
    (the hybrid: corr(true, pred cos 2θ) .27 on rotated soap, sideways soaps drawn upright). Slots merge with logsumexp (a soft max)"""
    def __init__(self, d_model, K, out_h, out_w, window=0.35, merge="or"):
        super().__init__()
        self.K, self.window, self.out_h, self.out_w, self.merge = K, window, out_h, out_w, merge
        self.project = nn.Linear(d_model, K * 128 * 4 * 4)
        self.ups = nn.Sequential(TwoBranchUp(128, 64), TwoBranchUp(64, 32), TwoBranchUp(32, 16))   # 4 -> 8 -> 16 -> 32
        self.head = nn.Conv2d(16, 1, 3, padding=1)
        if merge == "or": nn.init.constant_(self.head.bias, -6.0)                              # starts off: logaddexp(direct, -6) ~ direct
        elif merge in ("only", "slots"): nn.init.constant_(self.head.bias, -2.0)               # only / slots: the mask itself, starts faint inside the window
        else: nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)                 # sum: starts at 0, direct + 0 = direct
        self.angle = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2 * K))

    def forward(self, emb, com):
        B, K = len(emb), self.K
        canon = self.head(self.ups(self.project(emb).view(B * K, 128, 4, 4))).float()          # (B*K,1,32,32)
        v = self.angle(emb).float().view(B, K, 2)
        th = 0.5 * torch.atan2(v[..., 1], v[..., 0]).reshape(-1)                              # (B*K,) row-axis convention, as orientation()
        c = (2 * com.float() - 1).reshape(-1, 2)                                              # (B*K,2) normalized (row, col)
        cos, sin = th.cos(), th.sin()
        # canonical q = M^T (p - c) / w, p = (x=col, y=row); M^T = [[cos, -sin], [sin, cos]]
        A = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], 1) / self.window   # (B*K,2,2)
        t = -(A @ c.flip(-1)[..., None])[..., 0]                                              # c as (x, y)
        # affine_grid by hand: F.affine_grid needs concrete sizes, which torch.compile's dynamic batch doesn't give it
        ys, xs = (torch.arange(self.out_h, device=emb.device) * 2 + 1) / self.out_h - 1, (torch.arange(self.out_w, device=emb.device) * 2 + 1) / self.out_w - 1
        base = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).float()               # (H,W,2) as (x, y), align_corners=False
        grid = torch.einsum("nij,hwj->nhwi", A, base) + t[:, None, None]                    # (B*K,H,W,2)
        if self.merge == "sum":                                                                # residual: outside the canvas adds 0
            warped = F.grid_sample(canon, grid, padding_mode="zeros", align_corners=False)
            return warped.view(B, K, self.out_h, self.out_w).sum(1), v
        warped = F.grid_sample(canon + 10, grid, padding_mode="zeros", align_corners=False) - 10  # outside the canvas: logit -10 (or, only)
        if self.merge == "slots": return warped.view(B, K, self.out_h, self.out_w), v       # (B,K,H,W): Boombox merges them with exists
        return warped.view(B, K, self.out_h, self.out_w).logsumexp(1), v

#*** 3c multi-object decoders (src8): any number of objects, each drawn once, on the conv trunk's pixel features

def match_sets(masks, exists, inst, real):
    """set matching (DETR / Mask2Former): (B,K,H,W) mask logits, (B,K) exists logits, (B,N,H,W) instance masks, (B,N) bool real ->
    (B,N) the query matched to each true object. Cost per (query, object) = 5 bce + 5 dice - 2 p(exists) (Mask2Former's weights);
    every ordered choice of N of the K queries is tried (K!/(K-N)!: 20 for K=5, N=2), the cheapest over the real objects kept"""
    B, K, H, W = masks.shape
    N = inst.shape[1]
    x, t = masks.detach().float().flatten(2), inst.float().flatten(2)                       # (B,K,HW), (B,N,HW)
    bce = F.softplus(x).mean(-1)[:, :, None] - x @ t.transpose(1, 2) / (H * W)             # (B,K,N) mean bce of query k vs object j
    p = x.sigmoid()
    dice = 1 - (2 * p @ t.transpose(1, 2) + 1) / (p.sum(-1)[:, :, None] + t.sum(-1)[:, None] + 1)
    cost = 5 * bce + 5 * dice - 2 * exists.detach().float().sigmoid()[:, :, None]          # (B,K,N)
    orders = torch.tensor(list(itertools.permutations(range(K), N)), device=masks.device)  # (P,N)
    tot = (cost[:, orders, torch.arange(N, device=masks.device)] * real[:, None]).sum(-1)  # (B,P,N) -> (B,P)
    return orders[tot.argmin(1)]                                                           # (B,N)

class ObjectQueries(nn.Module):
    """Mask2Former-style set decoder (Cheng et al. 2022): K learned object queries; per layer they cross-attend to the ModalFormer's
    laser tokens (the signal), then MASKED cross-attention to the pixel features (each query only where its previous mask is on:
    Mask2Former's localization trick), then self-attend (two queries learn not to claim one object). Before the first layer and after
    every one, each query predicts mask_k = <mlp(q_k), pixel embedding> and an exists logit (deep supervision). Any count up to K"""
    def __init__(self, d_tok, c_pix, K=5, d=256, heads=8, layers=3, pose=False, out_hw=(32, 32), window=0.35, pix_head=True, init="learned", pose_merge="sum", shape="dot"):
        super().__init__()
        self.K, self.init, self.pose_merge, self.shape = K, init, pose_merge, shape
        # shape "gauss" (--obj-shape gauss): each query IS an object = (centroid, principal angle, two std devs) + a shape residual.
        # Its mask = the uniform ellipse with those moments (logit 4 - |x'|^2 in the object's own frame x' = diag(1/s) R^T (p - mu):
        # a uniform ellipse with std s has its edge at |x'| = 2) + a 16x16 canvas over |x'| <= 3 decoded from the query. The canvas
        # lives in the object's frame, so a shape is learned once for every rotation, and the angle is read straight off the query.
        # The moments are what the plate's modes see (centroid to first order, mass covariance to second); obj_gauss supervises them
        if shape == "gauss":
            self.gauss, self.canvas = nn.Linear(d, 6), nn.Linear(d, 16 * 16)
            nn.init.zeros_(self.canvas.weight); nn.init.zeros_(self.canvas.bias)
            with torch.no_grad(): self.gauss.weight.mul_(0.1); self.gauss.bias.copy_(torch.tensor([0.0, 0.0, 0.1, 0.0, math.log(0.05), math.log(0.05)]))
        # init="peaks" (two-stage, as Deformable DETR / DINO): each query starts at one of the K highest local maxima of the direct head's
        # map -- its content = a learned embedding + the pixel features there, plus the fourier code of where it is. Learned queries all
        # start identical and hedge at the mean position for a long time (s8-mag-qpose-v3 at ep50: all 5 at the middle of the two cubes)
        if init == "peaks": self.peak_feat, self.peak_pos = nn.Linear(128 if pix_head else c_pix, d), nn.Linear(4 * 8, d)
        # pix_head: the pixel embedding is a small conv head (3x3, 3x3, as the direct head) on the trunk's features + the direct head's
        # logits, not a 1x1 linear readout: at 64x64 the trunk has 32 channels, and a linear readout of them could not localize --
        # s8-mag-qpose v2 drew one input-independent blob at the image centre (std over samples .018 vs the direct head's .089)
        self.pix_head = nn.Sequential(nn.Conv2d(c_pix + 1, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()) if pix_head else None
        c_pix = 128 if pix_head else c_pix
        # pose: each query also decodes a canonical (object-centred, long axis along the rows) mask + an angle from its embedding (the
        # pose branch, once per query), warped to the soft centroid of the query's own mask and added to its last mask (residual,
        # zero-init): rotation as one shared parameter, any count
        # pose_merge "only": the query's last mask IS its warped canonical mask (rotation must go through the angle: qpose-v3's residual
        # let the free-form part memorize rotated templates -- rotated soap 25.8 deg vs 17.1 for the single pose slot); its free-form mask
        # stays supervised as one more layer and still places it (the soft-argmax centre)
        self.pose = PoseBranch(d, 1, *out_hw, window, "only" if pose_merge == "only" else "sum") if pose else None
        self.q = nn.Parameter(nn.init.trunc_normal_(torch.zeros(K, d), std=0.02))
        self.tok_in, self.tok_pos = nn.Linear(d_tok, d), nn.Linear(4 * 8, d)
        self.pix_in, self.pix_pos, self.pix_emb = nn.Conv2d(c_pix, d, 1), nn.Linear(4 * 8, d), nn.Conv2d(c_pix, d, 1)
        self.layers = nn.ModuleList([nn.ModuleList([Block(d, heads, cross=True), Block(d, heads, cross=True), Block(d, heads)]) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.mask_mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.exists = nn.Linear(d, 1)
        self.mask_bias = nn.Parameter(torch.tensor(-4.0))                                     # masks start faint: K queries at 0 would paint the union everywhere

    def predict(self, q, emb):
        qn = self.norm(q)
        if self.shape == "gauss": return *self.render(qn, *emb.shape[-2:]), self.exists(qn)[..., 0]
        return torch.einsum("bkd,bdhw->bkhw", self.mask_mlp(qn), emb) + self.mask_bias, None, self.exists(qn)[..., 0]  # (B,K,H,W), -, (B,K)

    def render(self, qn, H, W):
        """(B,K,d) -> (B,K,H,W) mask logits of each query's object (shape "gauss"), and its moments for the loss: (B,K,2) centroid,
        (B,K,2) doubled-angle vector, (B,K,3) covariance (rr, cc, rc, box fractions, as moments())"""
        B, K = qn.shape[:2]
        g = self.gauss(qn).float()
        mu, v, s = g[..., :2].sigmoid(), g[..., 2:4], g[..., 4:].exp().clamp(0.004, 0.5)
        th = 0.5 * torch.atan2(v[..., 1], v[..., 0])                                          # row-axis angle, as orientation()
        cos, sin = th.cos(), th.sin()
        d = grid_coords(H, W, qn.device)[None, None] - mu[:, :, None]                         # (B,K,HW,2) (row, col)
        a = (d[..., 0] * cos[..., None] + d[..., 1] * sin[..., None]) / s[..., :1]           # along the long axis, in std devs
        b = (-d[..., 0] * sin[..., None] + d[..., 1] * cos[..., None]) / s[..., 1:]
        prior = (4 - a ** 2 - b ** 2).clamp_min(-20)
        canvas = self.canvas(qn).float().view(B * K, 1, 16, 16)                               # rows = the long axis
        res = F.grid_sample(canvas, torch.stack([b, a], -1).view(B * K, H * W, 1, 2) / 3, padding_mode="zeros", align_corners=False)
        s2 = s ** 2
        cov = torch.stack([s2[..., 0] * cos ** 2 + s2[..., 1] * sin ** 2, s2[..., 0] * sin ** 2 + s2[..., 1] * cos ** 2, (s2[..., 0] - s2[..., 1]) * sin * cos], -1)
        return (prior + res.view(B, K, H * W)).view(B, K, H, W), dict(mu=mu, v=v, cov=cov)

    def forward(self, tokens, laser_rc, keep, feats, direct=None):
        """(B,L,d_tok) laser tokens, (L,2) their grid coords, (B,L) kept lasers, (B,c,H,W) pixel features ->
        (B,layers+1,K,H,W) mask logits, (B,layers+1,K) exists logits"""
        B, _, H, W = feats.shape
        if self.pix_head is not None: feats = self.pix_head(torch.cat([feats, direct[:, None].to(feats.dtype)], 1))
        t = self.tok_in(tokens) + self.tok_pos(fourier(laser_rc, 8))[None]                   # (B,L,d)
        tb = torch.zeros(B, 1, 1, t.shape[1], device=t.device).masked_fill(~keep[:, None, None], -1e4)
        pix = self.pix_in(feats).flatten(2).transpose(1, 2) + self.pix_pos(fourier(grid_coords(H, W, feats.device), 8))[None]  # (B,HW,d)
        emb = self.pix_emb(feats)                                                             # (B,d,H,W)
        q = self.q[None].expand(B, -1, -1)
        if self.init == "peaks":                                                              # the direct map's top-K local maxima (no grad)
            with torch.no_grad():
                h = direct.float()[:, None]
                h = h.masked_fill(F.max_pool2d(h, 3, 1, 1) != h, -1e4).flatten(1)
                idx = h.topk(self.K, 1).indices                                               # (B,K)
                rc = torch.stack([(idx // W + 0.5) / H, (idx % W + 0.5) / W], -1)            # (B,K,2) row, col in [0,1]
            f = feats.flatten(2).gather(2, idx[:, None].expand(-1, feats.shape[1], -1)).transpose(1, 2)  # (B,K,c)
            q = q + self.peak_feat(f.to(q.dtype)) + self.peak_pos(fourier(rc, 8)).to(q.dtype)
        m, g, e = self.predict(q, emb)
        ms, es = [m], [e]
        for cross_t, cross_p, self_ in self.layers:
            q = cross_t(q, t, tb)[0]
            off = m.detach().flatten(2) < 0                                                   # (B,K,HW) outside the query's own mask
            off = off & ~off.all(-1, keepdim=True)                                            # a query that drew nothing looks everywhere
            q = cross_p(q, pix, off[:, None].to(q.dtype) * -1e4)[0]
            q = self_(q)[0]
            m, g, e = self.predict(q, emb)
            ms.append(m); es.append(e)
        ang = g["v"] if g is not None else None
        if self.pose is not None:
            # each query's centre = the soft-argmax (spatial softmax) of its mask logits. Not the sigmoid^2-weighted centroid: a faint mask's
            # diffuse background (logit ~-4 on 4096 px) outweighs the object and drags it to the image centre (qpose v1 / v2: one central blob;
            # toy: both found .16 vs 1.00 without the pose head)
            w = m.detach().float().flatten(2).softmax(-1).view_as(m)                          # (B,K,H,W); detached: the shape loss must not move the centre
            r, c = (torch.arange(H, device=w.device) + 0.5) / H, (torch.arange(W, device=w.device) + 0.5) / W
            centre = torch.stack([(w.sum(3) * r).sum(2), (w.sum(2) * c).sum(2)], -1)          # (B,K,2)
            warped, v = self.pose(self.norm(q).reshape(B * self.K, -1), centre.reshape(B * self.K, 1, 2))
            if self.pose_merge == "only": ms.append(warped.view(B, self.K, H, W).to(m.dtype)); es.append(e)
            else: ms[-1] = m + warped.view(B, self.K, H, W).to(m.dtype)
            ang = v.view(B, self.K, 2)
        return torch.stack(ms, 1), torch.stack(es, 1), ang, g                                 # g: the last layer's moments (gauss) or None

class CenterObjects(nn.Module):
    """CenterNet (Zhou et al. 2019) with the pose branch per object: a class-agnostic centre heatmap on the pixel features (its peaks
    are the objects, any number), and at each centre a vector (the pixel features there + the embedding) that decodes the object's
    canonical mask and angle (PoseBranch, one slot), warped there. Training: at the true centres (teacher forcing, as CenterNet: heatmap
    and shapes learn separately); eval: the top-N peaks above thresh (3x3 max-pool NMS), refined by the 3x3 soft centroid"""
    def __init__(self, c_pix, d_model, N, out_h, out_w, window=0.35, thresh=0.3, d=256):
        super().__init__()
        self.N, self.thresh = N, thresh
        self.heat = nn.Sequential(nn.Conv2d(c_pix, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))
        nn.init.constant_(self.heat[-1].bias, -2.19)                                           # CenterNet's prior: p = .1
        self.z = nn.Sequential(nn.Linear(c_pix + d_model, d), nn.ReLU(), nn.Linear(d, d))
        self.pose = PoseBranch(d, 1, out_h, out_w, window, "slots")

    def peaks(self, heat):
        """(B,H,W) logits -> (B,N,2) centres (row,col in [0,1]), (B,N) scores"""
        B, H, W = heat.shape
        h = heat.float().sigmoid()[:, None]
        h = h * (F.max_pool2d(h, 3, 1, 1) == h)
        score, idx = h.flatten(1).topk(self.N, 1)                                             # (B,N)
        nb = F.unfold(heat.float().sigmoid()[:, None], 3, padding=1)                          # (B,9,HW) each pixel's 3x3 neighbourhood
        w = nb.gather(2, idx[:, None].expand(-1, 9, -1))                                      # (B,9,N)
        d = torch.tensor([-1.0, 0.0, 1.0], device=heat.device)
        dr, dc = d.repeat_interleave(3), d.repeat(3)                                          # (9,) unfold's order: row-major
        a = w.sum(1).clamp_min(1e-6)
        r = (idx // W + 0.5 + (w * dr[None, :, None]).sum(1) / a) / H
        c = (idx % W + 0.5 + (w * dc[None, :, None]).sum(1) / a) / W
        return torch.stack([r, c], -1), score

    def forward(self, feats, emb, centre=None, on=None, n_min=None):
        """(B,c,H,W), (B,d_model), training: (B,N',2) true centres + (B,N') real -> heat (B,H,W), masks (B,N,H,W), angles (B,N,2), on (B,N).
        eval: the peaks above thresh, and at least the n_min (B,) best ones (--obj-count-floor: the count head's argmax)"""
        B = len(feats)
        heat = self.heat(feats)[:, 0]
        if centre is None:
            centre, score = self.peaks(heat)
            on = score > self.thresh
            if n_min is not None: on = on | (torch.arange(self.N, device=on.device)[None] < n_min[:, None])  # topk: best first
        n = centre.shape[1]
        f = F.grid_sample(feats.float(), (2 * centre.float() - 1).flip(-1)[:, :, None], align_corners=False)[..., 0].transpose(1, 2)  # (B,n,c)
        z = self.z(torch.cat([f.to(emb.dtype), emb[:, None].expand(-1, n, -1)], -1))
        masks, v = self.pose(z.reshape(B * n, -1), centre.reshape(B * n, 1, 2))              # (B*n,1,H,W), (B*n,1,2)
        return heat, masks.view(B, n, *masks.shape[-2:]), v.view(B, n, 2), on

def soft_or(masks, gates):
    """(B,K,H,W) mask logits, (B,K) gate logits -> (B,H,W) logits of p = 1 - prod_k (1 - sigmoid(mask_k) sigmoid(gate_k)): an object
    query / slot that says "no object" adds at most its own p(exists). Adding log p(exists) to the logits and taking a logsumexp
    (the first version) let a confident blob through at any p(exists): s8-mag-qpose v1 at ep50 drew 125-px phantoms with p .01"""
    q = (masks.float().sigmoid() * gates.float().sigmoid()[..., None, None]).clamp(max=1 - 1e-6)
    S = torch.log1p(-q).sum(1).clamp(max=-1e-7)                                              # log(1 - p)
    return torch.log(-torch.expm1(S)) - S                                                  # logit(p)

def centre_heat_loss(logits, centre, real, sigma=1.5, cov=None):
    """CornerNet / CenterNet penalty-reduced focal loss (alpha 2, beta 4) vs gaussians (sigma px) at the true centres, per object.
    cov (B,N,3) (--obj-heat-sigma obj): each object's gaussian is its own mask covariance x 1/4 (+ 1 px^2): sized and oriented as
    the object, so an unsure centre is not punished for a peak a few px off inside the object (CenterNet sizes its gaussians too)"""
    B, H, W = logits.shape
    r, c = torch.arange(H, device=logits.device) + 0.5, torch.arange(W, device=logits.device) + 0.5
    dr, dc = r[None, None, :, None] - centre[..., 0, None, None] * H, c[None, None, None] - centre[..., 1, None, None] * W  # (B,N,H,W) px
    if cov is None: d2 = (dr ** 2 + dc ** 2) / sigma ** 2
    else:
        srr, scc, src = cov[..., 0] * H * H / 4 + 1, cov[..., 1] * W * W / 4 + 1, cov[..., 2] * H * W / 4          # (B,N) px^2
        det = srr * scc - src ** 2
        d2 = (scc[..., None, None] * dr ** 2 - 2 * src[..., None, None] * dr * dc + srr[..., None, None] * dc ** 2) / det[..., None, None]
    g = (torch.exp(-d2 / 2) * real[..., None, None]).amax(1)                                 # (B,H,W)
    px = (centre[..., 0] * H).long().clamp(0, H - 1) * W + (centre[..., 1] * W).long().clamp(0, W - 1)  # (B,N) each centre's pixel
    pos = torch.zeros(B, H * W, device=logits.device).scatter_add(1, px, real.float()).view(B, H, W) > 0
    p = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    lp = -(1 - p) ** 2 * p.log() * pos
    ln = -(1 - g) ** 4 * p ** 2 * (1 - p).log() * ~pos
    return (lp.sum() + ln.sum()) / real.sum().clamp_min(1)

#*** 3 decoder

class MLPDecoder(nn.Module):
    """src's default decoder: (B,d_model) -> (B,out_h,out_w) mask logits (or (B,out_h,out_w,3) rgb), one hidden layer to every pixel"""
    def __init__(self, d_model, out_h, out_w, rgb, hidden=256):
        super().__init__()
        self.out_h, self.out_w, self.out_c = out_h, out_w, (3 if rgb else 1)
        self.net = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, out_h * out_w * self.out_c))

    def forward(self, emb): return self.net(emb).view(-1, self.out_h, self.out_w, self.out_c).squeeze(-1), None  # no outline channel

class Decoder(nn.Module):
    """(B,d_model) -> (B,out_h,out_w) mask logits, or (B,out_h,out_w,3) rgb logits.
    Seed a 512x4x4 map and double it until it covers the output (4 -> 8 -> 16 -> 32 -> 64), halving channels each time,
    resize to exactly (out_h,out_w) if that isn't a power of 2, then a 3-conv head emits the output."""
    def __init__(self, d_model, out_h, out_w, rgb, edge=False, edge_width=0, head=True, n_heat=0):
        super().__init__()
        self.out_h, self.out_w, self.out_c, self.edge = out_h, out_w, (3 if rgb else 1), edge and not edge_width
        self.n_heat = n_heat                               # extra output channels (--pose-merge slots: centre heatmaps of slots 1..K-1)
        # each upsampling stage doubles the 4x4 seed and halves the channels (floored at 16), until the map covers the output:
        # out 64 -> 4 stages, 4 -> 8 -> 16 -> 32 -> 64 with channels 512 -> 256 -> 128 -> 64 -> 32
        n_ups = math.ceil(math.log2(max(out_h, out_w) / 4))
        widths = [max(512 // 2 ** i, 16) for i in range(n_ups + 1)]
        self.project = nn.Linear(d_model, 512 * 4 * 4)
        self.ups = nn.ModuleList([TwoBranchUp(c_in, c_out) for c_in, c_out in zip(widths, widths[1:])])  # how many depends on the output size
        self.width = widths[-1]                            # channels of features(): what a head on top of the trunk gets
        if not head: return                                # head=False: just the trunk, the polar head sits on features()
        self.conv1 = nn.Conv2d(widths[-1], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, self.out_c + self.edge + n_heat, kernel_size=3, padding=1)  # + 1 outline channel (+ heatmaps)
        c = edge_width  # edge_width > 0: a separate, deeper outline head on the upsampled features instead
        self.edge_head = nn.Sequential(nn.Conv2d(widths[-1], c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, 1, 3, padding=1)) if edge and c else None

    def features(self, emb):
        x = self.project(emb)                              # (B,d_model) -> (B,512*4*4)
        x = x.view(-1, 512, 4, 4)                          # (B,512*4*4) -> (B,512,4,4)
        for up in self.ups:
            x = up(x)                                      # (B,c,S,S) -> (B,c/2,2S,2S): 4 -> 8 -> 16 -> 32 -> 64 at out 64x64
        if x.shape[-2:] != (self.out_h, self.out_w):       # out_h/out_w not powers of 2, e.g. 21x30
            x = F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)  # (B,c,S,S) -> (B,c,out_h,out_w)
        return x

    def forward(self, emb): return self.from_features(self.features(emb))

    def from_features(self, feats):
        x = feats                                          # (B,c,out_h,out_w)
        x = F.relu(self.conv1(x))                          # (B,c,out_h,out_w) -> (B,32,out_h,out_w)
        x = F.relu(self.conv2(x))                          # (B,32,out_h,out_w) -> (B,32,out_h,out_w)
        x = self.conv3(x)                                  # (B,32,out_h,out_w) -> (B,out_c+edge+n_heat,out_h,out_w)
        heat = (x[:, -self.n_heat:],) if self.n_heat else ()  # a 3rd output only with n_heat
        if self.n_heat: x = x[:, :-self.n_heat]
        edge, x = (x[:, -1] if self.edge else None), x[:, :self.out_c]
        if self.edge_head is not None: edge = self.edge_head(feats)[:, 0]
        if self.out_c == 1: return x.squeeze(1), edge, *heat  # (B,1,out_h,out_w) -> (B,out_h,out_w)
        return x.permute(0, 2, 3, 1), edge                 # (B,3,out_h,out_w) -> (B,out_h,out_w,3), the rgb target's layout

#*** 4 polar head: PolarMask (Xie et al. 2020), class-agnostic, so it works on unseen objects

# every pixel predicts: is it an object's center (one channel, no classes), its centerness, and n ray lengths from it to the object's
# edge, ray j at angle 2πj/n (row += sin, col += cos). Masks = the polygons of the score peaks, drawn with a soft polar fill.

def ray_angles(n, device=None): return torch.arange(n, device=device) * 2 * math.pi / n  # (n,)

RAY_PARAMS = {"free": lambda n, k: n, "fourier": lambda n, k: 2 * k + 1, "superellipse": lambda n, k: 4}  # head channels per pixel

def log_rays(p, shape, n):
    """(B,P,H,W) ray params -> (B,n,H,W) log ray lengths, px. The shape is the prior on the outline:
    free: P = n, every ray its own (the paper)
    fourier: P = 2k+1, log r(θ) = a0 + Σ a_i cos iθ + b_i sin iθ, i <= k: smooth, circle = a0 alone, rectangles and squares ~ i = 2, 4
    superellipse: P = 4, |x/a|^m + |y/b|^m = 1 rotated by t: circle, ellipse, square, rectangle, all with rounded corners"""
    th = ray_angles(n, p.device)
    if shape == "free": return p
    if shape == "fourier":
        i = torch.arange(1, (p.shape[1] - 1) // 2 + 1, device=p.device)[:, None]          # (k,1)
        basis = torch.cat([torch.ones(1, n, device=p.device), (i * th).cos(), (i * th).sin()])  # (2k+1,n)
        return torch.einsum("bphw,pn->bnhw", p, basis)                                   # (B,2k+1,H,W) -> (B,n,H,W)
    log_a, log_b, m, t = p[:, 0:1], p[:, 1:2], 1 + F.softplus(p[:, 2:3]), p[:, 3:4]       # each (B,1,H,W); m >= 1
    th = th[:, None, None] - t                                                           # (n,1,1) - (B,1,H,W) -> (B,n,H,W)
    c, s = th.cos().abs().clamp_min(1e-6).log(), th.sin().abs().clamp_min(1e-6).log()
    return -torch.logsumexp(torch.stack([m * (c - log_a), m * (s - log_b)]), 0) / m     # r = ((|cos|/a)^m + (|sin|/b)^m)^(-1/m), in logs

def rays_at(params, idx, shape, n):
    """(B,P,H,W) ray params, (B,N) flat pixel indices -> (B,N,n) those pixels' ray lengths, px. Only where they're needed (training
    points, peaks): every pixel's rays at batch 1024 is a (1024,n,H,W) map per intermediate, GBs for the superellipse"""
    p = params.flatten(2).gather(2, idx[:, None].expand(-1, params.shape[1], -1))[..., None]  # (B,P,H*W) -> (B,P,N,1)
    return log_rays(p.float(), shape, n)[..., 0].transpose(1, 2).clamp(-3, 10).exp()      # (B,n,N,1) -> (B,N,n), 0.05 to 22000 px: never 0

def init_ray_bias(bias, shape, n_params):
    """start every outline as a circle of radius 4 px (a superellipse: a rounded square, m = 4)"""
    bias.zero_()
    bias[:2 if shape == "superellipse" else n_params if shape == "free" else 1] = math.log(4)
    if shape == "superellipse": bias[2] = math.log(math.e ** 3 - 1)  # m = 1 + softplus = 4

class PolarHead(nn.Module):
    """(B,c,H,W) decoder features -> (B,H,W) center logits, (B,H,W) centerness logits, (B,P,H,W) ray params.
    Two conv towers as in the paper: one for center + centerness, one for the rays, GroupNorm after each conv as in the paper (without
    it the center tower's gradients blew up to NaN by epoch 33)"""
    def __init__(self, c_in, shape, n_rays, n_harmonics, width=32):
        super().__init__()
        n_params = RAY_PARAMS[shape](n_rays, n_harmonics)
        tower = lambda c_out: nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(),
                                            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(), nn.Conv2d(width, c_out, 3, padding=1))
        self.cls, self.reg = tower(2), tower(n_params)
        with torch.no_grad():
            for t in (self.cls, self.reg): nn.init.normal_(t[-1].weight, std=0.01)  # as in the paper: outputs start at their biases
            self.cls[-1].bias.copy_(torch.tensor([-math.log(99), 0.0]))        # focal-loss prior: 1% of pixels are centers
            init_ray_bias(self.reg[-1].bias, shape, n_params)

    def forward(self, x):
        c = self.cls(x)                                    # (B,c,H,W) -> (B,2,H,W)
        return c[:, 0], c[:, 1], self.reg(x)               # (B,H,W), (B,H,W), (B,P,H,W)

def mass_centers(inst):
    """(B,K,H,W) each object's own mask -> (B,K) area px, (B,K) row and (B,K) col of its mass center, px (pixel i spans [i, i+1])"""
    H, W = inst.shape[-2:]
    m = (inst > 0.5).float()
    area = m.flatten(2).sum(-1)
    rows, cols = torch.arange(H, device=m.device) + 0.5, torch.arange(W, device=m.device) + 0.5
    return area, (m.sum(-1) * rows).sum(-1) / area.clamp_min(1), (m.sum(-2) * cols).sum(-1) / area.clamp_min(1)

def march(inst, oy, ox, n_rays, reach):
    """(B,K,H,W) each object's own mask, (B,K,Q) ray origins (row, col px) -> (B,K,Q,n) the distance from each origin to its object's
    farthest point along each ray, px (>= 0.5). March in 1/2 px steps out to reach px and find where the (bilinear) mask last drops
    through 0.5: the edge, interpolated between the last step inside and the next one"""
    B, K, H, W = inst.shape
    r = torch.arange(0, reach, 0.5, device=inst.device)                                  # (S,)
    th = ray_angles(n_rays, inst.device)
    y = oy[..., None, None] + r * th.sin()[:, None]                                      # (B,K,Q,1,1) + (n,S) -> (B,K,Q,n,S), px
    x = ox[..., None, None] + r * th.cos()[:, None]
    grid = torch.stack([x / W * 2 - 1, y / H * 2 - 1], -1).flatten(2, 4)[:, :, :, None]  # (B,K,Q*n*S,1,2) in grid_sample's [-1,1]
    v = F.grid_sample(inst.flatten(0, 1)[:, None], grid.flatten(0, 1), align_corners=False).view(B, K, -1, n_rays, len(r))  # (B,K,Q,n,S)
    v = F.pad(v, (0, 1))                                                                 # a last step outside: every ray leaves
    last = ((v[..., :-1] > 0.5) * torch.arange(1, len(r) + 1, device=inst.device)).argmax(-1, keepdim=True)  # (B,K,Q,n,1) last step inside
    v0, v1 = v.gather(-1, last), v.gather(-1, last + 1)                                  # mask at that step and the next
    rays = (r[last] + 0.5 * (v0 - 0.5) / (v0 - v1).clamp_min(1e-6)).squeeze(-1)          # (B,K,Q,n) where it crosses 0.5
    return torch.where(v0.squeeze(-1) > 0.5, rays, 0).clamp_min(0.5)                     # origin outside the object: half a pixel

def reach(area): return 2 * math.sqrt(area.max().item()) + 2  # px a ray can need: a square of that area has a diagonal of 1.4 sqrt(area)

@torch.no_grad()
def polar_targets(inst, n_rays, radius, chunk=128):
    """(B,K,H,W) each object's own mask -> each object's training points and their targets, all (B,K,Q,...):
    idx: flat pixel index of the Q = (2*ceil(radius)+1)^2 pixels around the object's mass center (static shapes)
    w: 1 for the training points: inside the object and within radius px of its mass center (its center pixel always), else 0
    rays: (B,K,Q,n) the distance from the pixel's center to the object's farthest point along each ray, px (>= 0.5, half a pixel)
    Done chunk samples at a time: the ray march is (B,K,Q,n,steps), several GB at batch 1024"""
    if len(inst) > chunk: return [torch.cat(t) for t in zip(*(polar_targets(c, n_rays, radius, chunk) for c in inst.split(chunk)))]
    B, K, H, W = inst.shape
    m = (inst > 0.5).float()
    area, cy, cx = mass_centers(inst)                                                    # (B,K) each
    o = torch.arange(-math.ceil(radius), math.ceil(radius) + 1, device=m.device)
    qy = (cy.floor()[..., None, None] + o[:, None]).expand(B, K, len(o), len(o)).flatten(2)  # (B,K,Q) candidate pixel rows
    qx = (cx.floor()[..., None, None] + o[None, :]).expand(B, K, len(o), len(o)).flatten(2)  # (B,K,Q) and cols
    in_img = (qy >= 0) & (qy < H) & (qx >= 0) & (qx < W)
    idx = (qy.clamp(0, H - 1) * W + qx.clamp(0, W - 1)).long()                           # (B,K,Q)
    inside = m.flatten(2).gather(2, idx) > 0                                             # (B,K,Q)
    near = (qy + 0.5 - cy[..., None]) ** 2 + (qx + 0.5 - cx[..., None]) ** 2 <= radius ** 2
    center = (o[:, None] == 0) & (o[None, :] == 0)                                       # (2P+1,2P+1), the pixel holding the mass center
    w = ((inside & near) | center.flatten()) & in_img & (area[..., None] > 0)            # (B,K,Q)
    return idx, w.float(), march(inst, qy + 0.5, qx + 0.5, n_rays, reach(area))          # rays from each pixel's center

def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    """sigmoid focal loss (Lin et al. 2017) summed over pixels, / the number of positives"""
    p, ce = logits.sigmoid(), F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt, a = p * target + (1 - p) * (1 - target), alpha * target + (1 - alpha) * (1 - target)
    return (a * (1 - pt) ** gamma * ce).sum() / target.sum().clamp_min(1)

def polar_iou(pred, true): return (torch.maximum(pred, true).sum(-1) / torch.minimum(pred, true).sum(-1)).log()  # (...,n) -> (...,), the paper's loss

def polar_losses(out, inst, shape, n_rays, radius):
    """center: focal on the center map; rays: polar iou at the training points, weighted by their centerness (as in the paper);
    centerness: bce to sqrt(min ray / max ray) at the training points"""
    idx, w, rays = polar_targets(inst.float(), n_rays, radius)                          # (B,K,Q), (B,K,Q), (B,K,Q,n)
    B, K, Q = idx.shape
    at = lambda v: v.flatten(1, 2).gather(1, idx.flatten(1))                            # (B,H,W) -> (B,K*Q)
    target = torch.zeros_like(out["center_logits"].flatten(1), dtype=torch.float).scatter_reduce(1, idx.flatten(1), w.flatten(1), "amax").view_as(out["center_logits"])
    pred_rays = rays_at(out["ray_params"], idx.flatten(1), shape, n_rays).view(B, K, Q, n_rays)  # (B,K*Q,n) -> (B,K,Q,n)
    ctr = (rays.amin(-1) / rays.amax(-1)).sqrt()                                         # (B,K,Q) centerness target
    bce = F.binary_cross_entropy_with_logits(at(out["centerness_logits"].float()).view(B, K, Q), ctr, reduction="none")
    return dict(center=focal_loss(out["center_logits"].float(), target),
                rays=(polar_iou(pred_rays, rays) * ctr * w).sum() / (ctr * w).sum().clamp_min(1e-6),
                centerness=(bce * w).sum() / w.sum().clamp_min(1))

def detect(center_logits, centerness_logits, n_peaks):
    """(B,H,W) -> the n_peaks best local maxima of score = p(center) * centerness: (B,N) scores and flat pixel indices, best first"""
    score = center_logits.float().sigmoid() * centerness_logits.float().sigmoid()       # (B,H,W)
    peak = score * (score == F.max_pool2d(score[:, None], 3, 1, 1)[:, 0])                # (B,H,W), 0 off the 3x3 maxima
    return peak.flatten(1).topk(n_peaks)                                                 # (B,N), (B,N)

def render(r, rc, keep, H, W, tau=0.2):
    """(B,N,n) outline rays (px), (B,N,2) their centers (row, col px), (B,N) which are objects (0..1) -> (B,H,W) soft union of the outlines.
    Pixel p is inside an outline by how far it is inside the ray length at p's angle (interpolated between the two nearest rays):
    sigmoid((r(θ) - ρ) / tau), tau in px. Differentiable in the rays and, through ρ, in the centers (the angle is detached: its
    gradient is 0/0 at the center)"""
    n = r.shape[-1]
    cy, cx = rc[..., 0, None, None], rc[..., 1, None, None]                              # (B,N,1,1) centers
    dy = torch.arange(H, device=rc.device)[:, None] + 0.5 - cy                           # (B,N,H,1)
    dx = torch.arange(W, device=rc.device)[None, :] + 0.5 - cx                           # (B,N,1,W)
    rho = (dy ** 2 + dx ** 2 + 1e-6).sqrt()                                              # (B,N,H,W) distance, never 0 (hypot's grad is 0/0 there)
    k = torch.atan2(dy, dx).detach() % (2 * math.pi) / (2 * math.pi / n)                 # (B,N,H,W) angle in rays
    i0, f = k.floor().long() % n, k - k.floor()
    ray = lambda i: r.gather(2, i.flatten(2)).view_as(rho)                               # (B,N,H,W) ray i's length at every pixel
    inside = torch.sigmoid(((1 - f) * ray(i0) + f * ray((i0 + 1) % n) - rho) / tau)      # (B,N,H,W)
    # max, not amax: amax's backward splits the grad over the entries equal to the max, and under torch.compile it recomputes the
    # interpolated `inside` slightly differently, so some pixels have no entry equal to it -> 0/0 = NaN in every grad, even with the
    # mask out of the loss (compile backprops every output, with a zero grad). Starts once a peak is kept; eager is fine.
    # max's backward routes the grad to the saved argmax instead
    return (inside * keep[..., None, None]).max(1).values                                # (B,N,H,W) -> (B,H,W)

#*** 4b slots head: per object a center of mass + one outline, from the embedding (no per-pixel maps)

# K = max_objects slots, each: exists logit, com (row, col) in [0,1], ray params. One shape per object, wherever it sits: every
# sample of an object trains the same few numbers. Slots are matched to the true objects by com (every order of K, as com_err)

def slot_targets(inst, n_rays):
    """(B,K,H,W) each object's own mask -> (B,K) real objects, (B,K,2) their mass centers px, (B,K,n) rays from them, px"""
    with torch.no_grad():
        area, cy, cx = mass_centers(inst)
        rays = march(inst, cy[..., None], cx[..., None], n_rays, reach(area))[:, :, 0]   # (B,K,1,n) -> (B,K,n)
    return area > 0, torch.stack([cy, cx], -1), rays

def match_slots(com_px, real, true_com):
    """(B,K,2) predicted slot coms px, (B,K) real objects, (B,K,2) their coms -> (B,K) which slot each true object gets: the order
    of slots with the least total com distance over the real objects"""
    K = com_px.shape[1]
    orders = torch.tensor(list(itertools.permutations(range(K))), device=com_px.device)  # (P,K)
    d = (com_px.detach()[:, orders] - true_com[:, None]).abs().sum(-1)                   # (B,P,K,2) - (B,1,K,2) -> (B,P,K)
    return orders[(d * real[:, None]).sum(-1).argmin(1)]                                 # (B,P) -> (B,) -> (B,K)

def slot_losses(out, inst, n_rays):
    """exists: bce, the matched slots on; com: L1 px of the matched slots; rays: polar iou of the matched slots, from the true com"""
    real, true_com, true_rays = slot_targets(inst.float(), n_rays)
    sel = match_slots(out["slot_com_px"], real, true_com)                                # (B,K)
    pick = lambda v: v.gather(1, sel[..., None].expand(-1, -1, v.shape[-1]))             # (B,K,...) slots -> in true-object order
    on = torch.zeros_like(out["slot_logits"], dtype=torch.float).scatter(1, sel, real.float())  # (B,K) slot is a real object
    n = real.sum().clamp_min(1)
    return dict(exists=F.binary_cross_entropy_with_logits(out["slot_logits"].float(), on),
                slot_com=((pick(out["slot_com_px"]) - true_com).abs().sum(-1) * real).sum() / n,
                rays=(polar_iou(pick(out["slot_rays"]), true_rays) * real).sum() / n)

def slot_ray_iou(out, batch):
    """slots head: sum(min)/sum(max) of the matched slot's rays vs the true rays from the true com, per real object"""
    real, true_com, true_rays = slot_targets(batch["inst"].float(), out["slot_rays"].shape[-1])
    sel = match_slots(out["slot_com_px"], real, true_com)
    pred = out["slot_rays"].gather(1, sel[..., None].expand(-1, -1, true_rays.shape[-1])).float()
    return (torch.minimum(pred, true_rays).sum(-1) / torch.maximum(pred, true_rays).sum(-1))[real]

#*** 5 helpers

def mask_area(mask): return mask.flatten(1).sum(1)  # (B,H,W) -> (B,)
def dilate(m, k=3): return F.max_pool2d(m, k, 1, k // 2)
def erode(m, k=3): return -dilate(-m, k)
def outline(mask): return (dilate(mask[:, None]) - erode(mask[:, None]))[:, 0]  # (B,H,W) -> (B,H,W), 1 px each side of the edge
def band(mask): return mask - erode(mask, 5)                                       # (B,1,H,W) -> pixels within 2 px of the edge, inside

def signed_dist(mask):
    """(B,H,W) -> (B,H,W) chessboard distance to the edge in image heights, < 0 inside. Empty mask -> 1 everywhere"""
    m = (mask[:, None] > 0.5).float()
    d, grow, shrink = torch.zeros_like(m), m, 1 - m
    for _ in range(max(m.shape[-2:])): d, grow, shrink = d + shrink - grow, dilate(grow), dilate(shrink)
    return d[:, 0] / m.shape[-2]

def com_err(pred, true, n_objects):
    """(B,K,2) predicted and true per-object (row,col) coms -> the distance of each real object's com.
    A sample with n objects uses the first n predicted slots. The model can't know which object is 'first', so try every
    order of those slots (up to K! orders, small for a handful of objects) and keep the closest. Padding slots are dropped."""
    B, K, _ = pred.shape
    orders = torch.tensor(list(itertools.permutations(range(K))), device=pred.device)  # (P,K), P = K!
    real = torch.arange(K, device=pred.device) < n_objects[:, None]                     # (B,K)
    allowed = ((orders < n_objects[:, None, None]) | ~real[:, None]).all(-1)            # (B,P): real objects only take real slots
    d = (pred[:, orders] - true[:, None]).norm(dim=-1)     # (B,P,K,2) - (B,1,K,2) -> (B,P,K)
    cost = (d * real[:, None]).sum(-1).masked_fill(~allowed, float("inf"))  # (B,P,K) -> (B,P), total distance of each order
    best = cost.argmin(1)                                  # (B,P) -> (B,), the closest allowed order
    d = d[torch.arange(B), best]                           # (B,P,K) -> (B,K)
    return d[real]                                         # (B,K) -> (n_real_objects,)

#*** 6 losses

# mask losses: (mask logits, true mask or rgb image)
def bce_loss(logits, true): return F.binary_cross_entropy_with_logits(logits, true)
def mse_loss(logits, true): return F.mse_loss(logits.sigmoid(), true)
def soft_dice_loss(pred, true): return 1 - (2 * (pred * true).flatten(1).sum(1) + 1e-6).div((pred + true).flatten(1).sum(1) + 1e-6).mean()  # pred in [0,1], product form
def dice_loss(logits, true): return soft_dice_loss(logits.sigmoid(), true)
def bce_dice_loss(logits, true, alpha=0.5): return alpha * bce_loss(logits, true) + (1 - alpha) * dice_loss(logits, true)

def tversky_loss(logits, true, a=0.3, b=0.7):
    """Tversky (Salehi et al. 2017): 1 - TP / (TP + a FP + b FN) per sample; b > a weighs a missed object pixel over a false one --
    for the misses and under-painting of small / weak objects (cardboard cube: area ratio .6-.8, 37-44% missed on random poses)"""
    p, t = logits.float().sigmoid().flatten(1), true.float().flatten(1)
    tp = (p * t).sum(1)
    return (1 - (tp + 1e-6) / (tp + a * (p * (1 - t)).sum(1) + b * ((1 - p) * t).sum(1) + 1e-6)).mean()
def bce_tversky_loss(logits, true, alpha=0.5): return alpha * bce_loss(logits, true) + (1 - alpha) * tversky_loss(logits, true)
LOSS_FN = {"bce": bce_loss, "mse": mse_loss, "dice": dice_loss, "bce-dice": bce_dice_loss, "bce-tversky": bce_tversky_loss}

def boundary_loss(logits, true): return (signed_dist(true) * logits.sigmoid()).mean()  # Kervadec et al. 2019
def edge_loss(logits, true): return bce_dice_loss(logits, outline(true))
SDF_CLIP = 0.1  # image heights: ~6 px at 64
def sdf_loss(pred, true): return F.l1_loss(pred, (signed_dist(true.float()) / SDF_CLIP).clamp(-1, 1))  # pred = tanh, in units of SDF_CLIP

def moments(m):
    """(B,H,W) soft mask -> (B,) area px, (B,2) centroid (row,col) and (B,3) covariance (rr, cc, rc), in box fractions"""
    m = m.float()
    H, W = m.shape[-2:]
    r, c = (torch.arange(H, device=m.device) + 0.5) / H, (torch.arange(W, device=m.device) + 0.5) / W
    a = m.flatten(1).sum(1).clamp_min(1e-3)
    mr, mc = (m.sum(2) * r).sum(1) / a, (m.sum(1) * c).sum(1) / a
    dr, dc = r[None] - mr[:, None], c[None] - mc[:, None]                              # (B,H), (B,W)
    cov = torch.stack([(m.sum(2) * dr ** 2).sum(1), (m.sum(1) * dc ** 2).sum(1), (m * dr[:, :, None] * dc[:, None]).flatten(1).sum(1)], -1) / a[:, None]
    return a, torch.stack([mr, mc], -1), cov

def moment_loss(pred, true, n_objects, terms="acs"):
    """single-object samples: L1 on log area / 10 (a), centroid (c), and 100 x covariance (s: size + orientation) of the soft masks.
    terms picks which: s8-posesum-contour (acs) made better shapes but worse placement, so 'as' leaves placement to the pose branch"""
    one = n_objects == 1
    if not one.any(): return pred.sum() * 0
    (ap, mp, cp), (at, mt, ct) = moments(pred[one]), moments(true[one])
    l = 0
    if "a" in terms: l = l + 0.1 * (ap.log() - at.log()).abs()
    if "c" in terms: l = l + (mp - mt).abs().sum(-1)
    if "s" in terms: l = l + 100 * (cp - ct).abs().sum(-1)
    return l.mean()

def instance_moment_loss(pred, inst, real, terms="acs"):
    """moment_loss for every object, any count: each pixel goes to the true object nearest in Mahalanobis distance (its centroid and
    covariance, + log det: a gaussian mixture's assignment, so an elongated object keeps its own cell), and the predicted mask inside
    each object's cell is matched to that object's area, centroid and covariance (size + orientation). One object: the whole image
    is its cell, = moment_loss. Physics: to second order an object's mode shifts read its centroid and its mass covariance"""
    B, N, H, W = inst.shape
    if not real.any(): return pred.sum() * 0
    _, mt, ct = moments(inst.flatten(0, 1))
    mt, ct = mt.view(B, N, 2), ct.view(B, N, 3)
    S = torch.stack([torch.stack([ct[..., 0] + 1e-3, ct[..., 2]], -1), torch.stack([ct[..., 2], ct[..., 1] + 1e-3], -1)], -2)  # (B,N,2,2)
    d = grid_coords(H, W, pred.device)[None, None] - mt[:, :, None]                       # (B,N,HW,2)
    md = torch.einsum("bnpi,bnij,bnpj->bnp", d, torch.linalg.inv(S), d) + torch.logdet(S)[..., None]
    cell = F.one_hot(md.masked_fill(~real[..., None], float("inf")).argmin(1), N).permute(0, 2, 1).float().view(B, N, H, W)
    (ap, mp, cp), (at, mt_, ct_) = moments((pred.float()[:, None] * cell)[real]), moments(inst[real])
    l = 0
    if "a" in terms: l = l + 0.1 * (ap.log() - at.log()).abs()
    if "c" in terms: l = l + (mp - mt_).abs().sum(-1)
    if "s" in terms: l = l + 100 * (cp - ct_).abs().sum(-1)
    return l.mean()

def orient_target(true):
    """(B,H,W) -> (B,2) ecc * (cos 2θ, sin 2θ) of the mask's principal axis: ~0 for round objects, so no angle to fight over"""
    _, _, cov = moments(true)
    th, e = orientation(cov), eccentricity(cov)
    return torch.stack([(2 * th).cos(), (2 * th).sin()], -1) * e[:, None]
def orient_loss(pred, true, n_objects):
    one = n_objects == 1
    return F.mse_loss(pred[one], orient_target(true[one])) if one.any() else pred.sum() * 0

def orientation(cov): return 0.5 * torch.atan2(2 * cov[:, 2], cov[:, 0] - cov[:, 1])  # (B,3) -> (B,) principal axis angle
def eccentricity(cov):
    tr, det = cov[:, 0] + cov[:, 1], cov[:, 0] * cov[:, 1] - cov[:, 2] ** 2
    return ((tr ** 2 - 4 * det).clamp_min(0).sqrt() / tr.clamp_min(1e-9))              # (l1 - l2) / (l1 + l2), 0 = round

# aux losses
def object_loss(logits, obj, unseen): return F.cross_entropy(logits, obj, ignore_index=unseen)  # objects never seen in train are ignored
def n_objects_loss(logits, n_objects): return F.cross_entropy(logits, n_objects)
def box_loss(logits, box): return F.cross_entropy(logits, box)
def heatmap_loss(logits, true):
    """--pose-merge only: the centre heatmap's softmax over pixels vs the true mask as a distribution (CE); empty boxes are skipped"""
    t = (true > 0.5).float().flatten(1)
    n = t.sum(1)
    ce = -(t / n.clamp_min(1)[:, None] * logits.float().flatten(1).log_softmax(1)).sum(1)
    return (ce * (n > 0)).sum() / (n > 0).sum().clamp_min(1)
def com_loss(pred, true, n_objects): return com_err(pred, true, n_objects).sum() / n_objects.sum().clamp_min(1)  # mean over real objects
def area_loss(log_area, true): return F.l1_loss(log_area, mask_area(true).log1p())

#*** 7 metrics

def iou(out, batch): return torch.minimum(out["mask_pred"], batch["mask_true"]).flatten(1).sum(1) / torch.maximum(out["mask_pred"], batch["mask_true"]).flatten(1).sum(1).clamp_min(1e-6)  # soft, sum(min) / sum(max)
def object_acc(out, batch): return (out["object_logits"].argmax(-1) == batch["object"])[batch["object"] < out["object_logits"].shape[-1]].float()  # unseen objects (no logit) are left out
def n_objects_err(out, batch): return (out["n_objects_logits"].argmax(-1) - batch["n_objects"]).abs().float()  # counts are ordered: off by 2 is worse than off by 1
def box_acc(out, batch): return (out["box_logits"].argmax(-1) == batch["box"]).float()
def com_error(out, batch): return com_err(out["com"], batch["coms"], batch["n_objects"])  # as a fraction of the box
def biou(pred, true):
    """(B,H,W) -> (B,) boundary iou (Cheng et al. 2021), NaN on an empty box"""
    p, t = band((pred[:, None] > 0.5).float()), band((true[:, None] > 0.5).float())
    return ((p * t).flatten(1).sum(1) / torch.maximum(p, t).flatten(1).sum(1).clamp_min(1)).where(t.flatten(1).sum(1) > 0, torch.nan)
def boundary_iou(out, batch): b = biou(out["mask_pred"], batch["mask_true"]); return b[~b.isnan()]
def area_rel_err(out, batch): return ((out["log_area"].expm1() - mask_area(batch["mask_true"])).abs() / mask_area(batch["mask_true"]))[mask_area(batch["mask_true"]) > 0]

def ray_iou(out, batch, shape, n_rays):
    """polar head: sum(min)/sum(max) of the predicted vs true rays at each object's center pixel, 1 = the right shape and size wherever
    the peak lands; the shape half of iou, the center half is com_err. NaN-free: only real objects"""
    idx, w, rays = polar_targets(batch["inst"].float(), n_rays, 0)                         # radius 0: the center pixel only, (B,K,1,...)
    pred = rays_at(out["ray_params"], idx.flatten(1), shape, n_rays)                       # (B,K,n)
    return (torch.minimum(pred, rays[:, :, 0]).sum(-1) / torch.maximum(pred, rays[:, :, 0]).sum(-1))[w[:, :, 0] > 0]

THR = 0.08  # src8: the threshold the goal is scored at (score.py)
def hard_iou(out, batch):
    b, o = out["mask_pred"] > THR, batch["mask_true"] > 0.5
    keep = o.flatten(1).any(1)
    return ((b & o).flatten(1).sum(1) / (b | o).flatten(1).sum(1).clamp_min(1)).float()[keep]
def miss(out, batch):
    b, o = out["mask_pred"] > THR, batch["mask_true"] > 0.5
    return (~(b & o).flatten(1).any(1)).float()[o.flatten(1).any(1)]
def orient_err(out, batch):
    """degrees between the predicted and true masks' principal axes, single objects with eccentricity > 0.2 only (0..90)"""
    one = batch["n_objects"] == 1
    if not one.any(): return out["mask_pred"].new_zeros(0)
    _, _, cp = moments(out["mask_pred"][one]); _, _, ct = moments(batch["mask_true"][one])
    d = (orientation(cp) - orientation(ct)).abs() % math.pi
    return torch.minimum(d, math.pi - d)[eccentricity(ct) > 0.2] * 180 / math.pi

def _keep(batch): return (batch["mask_true"] > 0.5).flatten(1).any(1)
def area_ratio(out, batch):
    """predicted area over thr / true area: < 1 = under-painting"""
    k = _keep(batch)
    return ((out["mask_pred"] > THR).flatten(1).sum(1) / (batch["mask_true"] > 0.5).flatten(1).sum(1).clamp_min(1)).float()[k]
def biou08(out, batch): b = biou(out["mask_pred"] - (THR - 0.5), batch["mask_true"]); return b[~b.isnan()]  # boundary iou of pred > THR
def iou_auc(out, batch):
    """mean hard iou over 13 thresholds 0.02..0.5: threshold-free, so calibration alone can't win"""
    p, o = out["mask_pred"], batch["mask_true"] > 0.5
    v = torch.stack([((p > t) & o).flatten(1).sum(1) / ((p > t) | o).flatten(1).sum(1).clamp_min(1) for t in torch.linspace(0.02, 0.5, 13).tolist()]).mean(0)
    return v.float()[_keep(batch)]
def far_fp(out, batch):
    """fraction of the predicted area over THR more than 4 px from any object: false positives that aren't just a fat edge"""
    b, near = out["mask_pred"] > THR, dilate((batch["mask_true"] > 0.5).float()[:, None], 9)[:, 0] > 0
    return ((b & ~near).flatten(1).sum(1) / b.flatten(1).sum(1).clamp_min(1)).float()

def gray_losses(out, batch):
    """the object is 2-5% of the image, so a whole-image L1 is mostly the floor. Three terms:
    gray       L1 of the composite over the whole image: the floor (bg) and the final picture
    gray_obj   L1 of the appearance head vs the photo on the TRUE object pixels, per-object-area mean (a small object weighs as much as the
               floor; teacher-forced by the true mask, so appearance learns even where the predicted mask is off)
    gray_grad  L1 of the composite's image gradients vs the photo's within 2 px of the object: its contour and inner detail (label, lid)
               stay sharp, not averaged into a flat blob"""
    t, o = batch["gray"].float(), (batch["mask_true"] > 0.5).float()
    has = o.flatten(1).sum(1) > 0
    obj = ((out["gray_app"] - t).abs() * o).flatten(1).sum(1) / o.flatten(1).sum(1).clamp_min(1)
    near = dilate(o[:, None], 5)[:, 0] > 0
    dr = lambda x: x[:, 1:] - x[:, :-1]; dc = lambda x: x[:, :, 1:] - x[:, :, :-1]
    p = out["gray_pred"]
    gr = ((dr(p) - dr(t)).abs() * (near[:, 1:] & near[:, :-1])).flatten(1).sum(1) + ((dc(p) - dc(t)).abs() * (near[:, :, 1:] & near[:, :, :-1])).flatten(1).sum(1)
    gr = gr / near.flatten(1).sum(1).clamp_min(1)
    z = p.sum() * 0
    return dict(gray=(p - t).abs().mean(), gray_obj=obj[has].mean() if has.any() else z, gray_grad=gr[has].mean() if has.any() else z)

def gray_l1(out, batch): return (out["gray_pred"].float() - batch["gray"]).abs().flatten(1).mean(1)
def _obj_l1(pred, batch):
    """mean |pred - photo| over the true object's pixels, per sample with an object"""
    o = (batch["mask_true"] > 0.5).float()
    return ((pred.float() - batch["gray"]).abs() * o).flatten(1).sum(1)[o.flatten(1).sum(1) > 0] / o.flatten(1).sum(1)[o.flatten(1).sum(1) > 0]
def gray_obj_l1(out, batch): return _obj_l1(out["gray_pred"], batch)
def gray_bg_l1(out, batch): return _obj_l1(out["gray_bg"], batch)  # the empty floor drawn over the object: what appearance must beat
GRAY = ("gray_l1", "gray_obj_l1", "gray_bg_l1")

METRICS = {"iou08": hard_iou, "miss08": miss, "orient_err": orient_err, "area_ratio08": area_ratio, "biou08": biou08, "iou_auc": iou_auc, "far_fp08": far_fp, "iou": iou, "ray_iou": ray_iou, "boundary_iou": boundary_iou, "object_acc": object_acc, "n_objects_err": n_objects_err, "box_acc": box_acc, "com_err": com_error, "area_rel_err": area_rel_err,
           "gray_l1": gray_l1, "gray_obj_l1": gray_obj_l1, "gray_bg_l1": gray_bg_l1}

#*** 8 model

class Boombox(ComposerModel):
    def __init__(self, d_model: int, data_info: dict, out_h: int, out_w: int, rgb: bool = False,
                 loss_fn: str = "bce", alpha: float = 0.5, loss_weights: dict | None = None, edge_width: int = 0,
                 freq_gain: bool = False, freq_coord: bool = False, freq_pool: str = "mean",
                 head: str = "mask", n_rays: int = 36, ray_shape: str = "free", n_harmonics: int = 4, center_radius: float = 1.5, peak_thresh: float = 0.3,
                 slots_aux: bool = False, encoder: str = "conv", decoder: str = "conv", patch_size: int | None = None, pose_branch: bool = False, pose_window: float = 0.35, pose_merge: str = "or", pose_center: str = "com", moment_terms: str = "acs", color_ell: float = 0.0, laser_jitter: float = 0.0, laser_jitter_p: float = 1.0, freq_jitter: float = 0.0,
                 obj_decoder: str = "none", n_queries: int = 5, obj_layers: int = 3, obj_thresh: float = 0.3, obj_pose: bool = False, obj_pix: bool = True, obj_init: str = "learned", obj_pose_merge: str = "sum",
                 obj_shape: str = "dot", obj_heat_sigma: str = "px", obj_count_floor: bool = False, obj_jitter: float = 0.0, gray: bool = False, gray_joint: bool = False, gray_layers: int = 2, gray_detach: bool = False,
                 heads: int = 2, freq_layers: int = 2, laser_layers: int = 2, ffn_dim: int | None = None, modal: dict | None = None, color_aug: float = 0.0,
                 field_width: int = 128, field_freqs: int = 5, field_modes: int = 32):
        super().__init__()
        self.loss_weights = loss_weights or dict(object=0.1, n_objects=0.1, com=1.0, area=0.1, box=0.1)
        # head="polar": PolarHead on the decoder's trunk instead of its mask convs, trained on center + rays + centerness (+ render)
        # head="slots": per object exists + com + one outline from the embedding (slot_losses: exists, slot_com, rays, + render)
        self.polar, self.slots = head == "polar", head == "slots"
        # slots_aux: the mask head draws the mask; the slots head on the same embedding trains alongside it (exists + com + rays), as
        # an extra signal only. Its losses are "aux_*", summed into "slots" at weight loss_weights["slots"]
        self.slots_aux = slots_aux
        assert head == "mask" or not slots_aux, "slots_aux: the mask head's"
        # encoder="transformer" / decoder="mlp": src's VibrationTransformer (freq + laser transformers, then an MLP to the pixels)
        assert decoder in ("conv", "query", "field", "modes") or (head == "mask" and not self.loss_weights.get("edge")), "the polar head and the outline sit on the conv decoder"
        assert (decoder == "query") <= (encoder == "modal"), "pixel queries are ModalFormer's"
        assert (encoder == "transformer") == (patch_size is not None), "the transformer reads patches (--patch-size), the conv encoder the raw spectrum"
        assert head == "mask" or not (rgb or self.loss_weights.get("boundary") or self.loss_weights.get("edge")), f"{head} head: no rgb, boundary or edge"
        assert head == "mask" or not (self.loss_weights.get("com") or self.loss_weights.get("n_objects")), f"{head} head: com and count come from it"
        self.n_rays, self.ray_shape, self.center_radius, self.peak_thresh = n_rays, ray_shape, center_radius, peak_thresh
        freqs = torch.linspace(*data_info.get("freq_range", (100.0, 1000.0)), data_info["n_freqs"])  # (F,) Hz; bins are evenly spaced
        if encoder == "conv":
            self.freq_encoder = FreqEncoder(data_info["n_channels"], freqs, gain=freq_gain, coord=freq_coord, pool=freq_pool)
            self.grid_encoder = LaserGridEncoder(d_model, data_info["n_laser_rows"], data_info["n_laser_cols"])
        elif encoder == "modal":  # src8: ModalFormer, its own pixel-query decoder; extra out channels: edge, sdf
            self.n_extra = int(bool(self.loss_weights.get("edge"))) + int(bool(self.loss_weights.get("sdf")))
            self.modal = ModalFormer(data_info["n_channels"], freqs, data_info["n_laser_rows"], data_info["n_laser_cols"], len(data_info["boxes"]), d_model,
                                     out_h, out_w, 3 if rgb else 1, self.n_extra, **(modal or {}), queries=decoder == "query")
        else:
            self.transformer = TransformerEncoder(d_model, data_info["n_channels"], patch_size, -(-data_info["n_freqs"] // patch_size),
                                                  data_info["n_laser_rows"], data_info["n_laser_cols"], heads, freq_layers, laser_layers, ffn_dim)
        self.encoder, self.rgb, self.color_aug, self.signal = encoder, rgb, color_aug, data_info.get("signal", "magnitude")
        self.color_ell, self.laser_jitter, self.laser_jitter_p, self.grid_rc = color_ell, laser_jitter, laser_jitter_p, (data_info["n_laser_rows"], data_info["n_laser_cols"])
        self.freq_hz = data_info.get("freq_range", (100.0, 1000.0))
        self.freq_jitter = freq_jitter
        self.phase, self.n_freqs = data_info.get("phase", "none"), data_info["n_freqs"]  # old checkpoints' data_info has no "phase"
        self.out_h, self.out_w = out_h, out_w
        # decoder "field" / "modes" (FieldDecoder): one MLP shared by every floor cell, on the ModalFormer's embedding / laser tokens
        self.field = decoder in ("field", "modes")
        assert not self.field or (encoder == "modal" and head == "mask" and not (rgb or pose_branch or obj_decoder != "none" or gray)), "--decoder field/modes: the ModalFormer's plain mask head"
        self.decoder = FieldDecoder(d_model, self.modal.laser_emb.shape[1], len(data_info["boxes"]), grid_coords(data_info["n_laser_rows"], data_info["n_laser_cols"]),
                                    out_h, out_w, decoder, field_width, field_freqs, field_modes, edge=bool(self.loss_weights.get("edge"))) if self.field else (None if self.slots or (encoder == "modal" and decoder != "conv") else MLPDecoder(d_model, out_h, out_w, rgb) if decoder == "mlp" else
                        Decoder(d_model, out_h, out_w, rgb, edge=bool(self.loss_weights.get("edge")), edge_width=edge_width, head=not self.polar,
                                n_heat=max(data_info["n_objects"]) - 1 if pose_branch and pose_merge == "slots" else 0))
        self.polar_head = PolarHead(self.decoder.width, ray_shape, n_rays, n_harmonics) if self.polar else None

        # aux heads on the embedding. objects[-1] is "unseen": never a train target, so it gets no logit
        self.n_known = len(data_info["objects"]) - 1
        self.max_objects = max_objects = max(data_info["n_objects"])
        mlp = lambda n_out: nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_out))
        self.object_head = mlp(self.n_known)       # which objects
        self.n_objects_head = mlp(max_objects + 1) # how many objects, 0..max_objects
        self.com_head = mlp(2 * max_objects)       # (row,col) com of each object
        self.area_head = mlp(1)                    # log(1 + mask area)
        self.box_head = mlp(len(data_info["boxes"])) # which box type
        self.orient_head = mlp(2) if self.loss_weights.get("orient") and not pose_branch else None  # src8: ecc * (cos 2θ, sin 2θ) of a single object's mask
        assert not (pose_branch and rgb), "the pose branch draws masks"
        # pose_center "com": slots at the com head (2x worse than the mask's own centroid: CORE .043 vs .024 -- global coordinate regression
        # vs a dense map, as in pose estimation); "mask": one slot at the soft-argmax of the direct decoder's mask (integral regression, Sun 2018)
        self.pose_center, self.moment_terms = pose_center, moment_terms
        self.pose = PoseBranch(d_model, max_objects if pose_merge == "slots" or pose_center != "mask" else 1, out_h, out_w, pose_window, pose_merge) if pose_branch else None  # its slot-0 angle is the orient output
        # pose_merge "slots" (Mask2Former-style, one pose branch per object slot): each slot k has its own centre heatmap (decoder channel k),
        # canonical mask + angle, and an exists logit; slots are matched to the true objects by centre, supervised per object
        self.exists_head = mlp(max_objects) if pose_branch and pose_merge == "slots" else None
        # obj_decoder (multi-object, any count): the mask is the union of per-object masks drawn on the conv trunk's pixel features; the
        # direct head stays as an auxiliary (loss "direct"). queries = Mask2Former object queries (set-matched to the instances); centernet =
        # a centre heatmap whose peaks each get a pose-branch canonical mask + angle
        self.obj_decoder, self.obj_heat_sigma, self.obj_count_floor, self.obj_jitter = obj_decoder, obj_heat_sigma, obj_count_floor, obj_jitter
        assert obj_decoder == "none" or (encoder == "modal" and decoder == "conv" and not pose_branch and not rgb), "--obj-decoder: on the hybrid's conv trunk, instead of the pose branch"
        if obj_decoder == "queries": self.objq = ObjectQueries(self.modal.laser_emb.shape[1], self.decoder.width, n_queries, layers=obj_layers, pose=obj_pose, out_hw=(out_h, out_w), window=pose_window, pix_head=obj_pix, init=obj_init, pose_merge=obj_pose_merge, shape=obj_shape)
        if obj_decoder == "centernet": self.objc = CenterObjects(self.decoder.width, d_model, n_queries, out_h, out_w, pose_window, obj_thresh)
        if self.slots or slots_aux:                  # per slot: exists logit, com (row, col), ray params
            n_params = RAY_PARAMS[ray_shape](n_rays, n_harmonics)
            self.slot_head = mlp(max_objects * (3 + n_params))
            with torch.no_grad():
                nn.init.normal_(self.slot_head[-1].weight, std=0.01)  # outputs start at their biases: off-center 50/50, a 4 px circle at the box center
                b = self.slot_head[-1].bias.view(max_objects, 3 + n_params); b.zero_()
                for k in range(max_objects): init_ray_bias(b[k, 3:], ray_shape, n_params)

        # gray: the overhead photo in grayscale as an alpha composite, gray = (1 - m) bg[box] + m app, m the predicted mask DETACHED -- the
        # photo loss teaches only what the object looks like, never where it is, so every mask metric means what it did. Where the mask is
        # off it draws the empty box. Both are differences from the empty box, in logits (dataset.gray_prior):
        #   bg  = sigmoid(floor[box] + d_bg[box])      the box's median train photo + a learned correction (starts at 0)
        #   app = sigmoid(floor[box] + head(features))  the object's contrast with the floor under it; head's bias starts at the mean contrast
        # v1 learned both from 0 (a 0.5 gray): at lr 1e-4 Adam moves a logit ~1e-4 per step, so at ep50 the floor was still 0.5 everywhere
        # gray_joint: the mask is NOT detached, so the photo (sharp object edges) also trains where the object is
        self.gray, self.gray_joint, self.gray_detach = gray, gray_joint, gray_detach
        assert not gray or (encoder == "modal" and decoder == "conv" and obj_decoder == "none" and pose_merge != "slots" and not rgb), "--gray: the hybrid's direct head (+ one pose slot)"
        if gray:
            self.register_buffer("gray_floor", data_info.get("gray_floor", torch.zeros(len(data_info["boxes"]), out_h, out_w)).clone())
            self.gray_bg = nn.Parameter(torch.zeros(len(data_info["boxes"]), out_h, out_w))
            # gray_layers > 2: dilated middle convs (2, 4, ..) so each pixel sees its whole object (where the soap's pump is, which end is up)
            dil = [1] + [2 ** k for k in range(1, gray_layers - 1)]
            convs = [l for i, d in enumerate(dil) for l in (nn.Conv2d(self.decoder.width if i == 0 else 32, 32, 3, padding=d, dilation=d), nn.ReLU())]
            self.gray_head = nn.Sequential(*convs, nn.Conv2d(32, 1, 3, padding=1))
            nn.init.constant_(self.gray_head[-1].bias, data_info.get("gray_contrast", 0.0))

        # loss = mask loss + weighted aux losses (a weight of 0 means that head doesn't train)
        self.mask_loss = partial(LOSS_FN[loss_fn], alpha=alpha) if loss_fn in ("bce-dice", "bce-tversky") else LOSS_FN[loss_fn]

        # metrics, the same ones on train and test: tag each with its key so update_metric knows which one it was handed
        keys = [key for key in METRICS if (head != "mask" or slots_aux or key != "ray_iou") and (gray or key not in GRAY)]  # ray_iou needs rays
        self.train_metrics = {key: MeanMetric() for key in keys}
        self.test_metrics = {key: MeanMetric() for key in keys}
        for key, metric in [*self.train_metrics.items(), *self.test_metrics.items()]: metric.key = key

    def phase_features(self, x):
        """--phase cossin: the dataset stores [mag, phi] (2C channels) to keep batches small in RAM; the model reads
        [mag, cos phi, sin phi] (3C). Zero-padded bins past n_freqs (patching) stay 0, not cos 0 = 1."""
        if self.phase != "cossin": return x
        C = x.shape[-1] // 2
        mag, phi = x[..., :C], x[..., C:]                  # (B,L,F,C) each, or (B,L,P,PS,C)
        cs = torch.cat([phi.cos(), phi.sin()], dim=-1)     # (...,2C)
        if x.dim() == 5:                                   # patched: zero the padding past the last real bin
            real = torch.arange(x.shape[2] * x.shape[3], device=x.device).view(x.shape[2], x.shape[3], 1) < self.n_freqs
            cs = cs * real
        return torch.cat([mag, cs], dim=-1)                # (...,3C)

    def speaker_gain(self, x):
        """--color-aug with --color-ell > 0: a random log-gain varying on the scale of a resonance, shared by every laser and axis -- an
        unseen speaker. Measured (speaker 4 vs 1/3/5/7 at the same positions): 64-67% of the difference is common to all lasers, its
        autocorrelation halves at 8-28 Hz, rms 0.3-0.4 input units. Knots every color_ell Hz ~ N(0, color_aug), linearly interpolated"""
        B, F_ = len(x), x.shape[2]
        n = max(2, int((self.freq_hz[1] - self.freq_hz[0]) / self.color_ell) + 2)
        g = F.interpolate(torch.randn(B, 1, n, device=x.device) * self.color_aug, size=F_, mode="linear", align_corners=True)[:, 0]  # (B,F)
        g = g[:, None, :, None]
        return x + g if self.signal.startswith("log") else x * g.exp()

    def jitter_lasers(self, x):
        """--laser-jitter s: resample the wall's 10x10 field at the grid shifted by U(-s, s) cells per axis, per sample (bilinear, border
        clamp): the model can't memorize exact laser placement (probe: grid roll by 1 cell -> iou .30 -> .12). Neighbouring lasers' spectra
        correlate .87 (distance 9: .67-.77), so sub-cell resampling is a fair stand-in for a slightly moved grid"""
        B, L, F_, C = x.shape
        R, Cc = self.grid_rc
        f = x.permute(0, 2, 3, 1).reshape(B, F_ * C, R, Cc).float()                        # (B,F*C,R,Cc)
        sh = (torch.rand(B, 2, device=x.device) * 2 - 1) * self.laser_jitter                # (B,2) cells, (row, col)
        # --laser-jitter-p: only this fraction of samples is shifted, the rest keep the sharp grid (a sub-cell bilinear shift also
        # averages neighbours, a spatial low-pass that cost the two-object splits .33 -> .25)
        sh = sh * (torch.rand(B, 1, device=x.device) < self.laser_jitter_p)
        theta = torch.zeros(B, 2, 3, device=x.device); theta[:, 0, 0] = theta[:, 1, 1] = 1
        theta[:, 0, 2], theta[:, 1, 2] = sh[:, 1] * 2 / Cc, sh[:, 0] * 2 / R                 # normalized: x = col, y = row
        grid = F.affine_grid(theta, (B, 1, R, Cc), align_corners=False)
        f = F.grid_sample(f, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return f.reshape(B, F_, C, L).permute(0, 3, 1, 2).to(x.dtype)

    def jitter_freqs(self, x):
        """--freq-jitter s: the frequency counterpart of --laser-jitter. Resample every spectrum on a stretched axis, f -> f (1 + eps),
        eps ~ U(-s, s) per sample, shared by all lasers and axes (linear interpolation, edges clamped): every resonance moves by the same
        fraction, as when the box's stiffness or temperature shifts slightly (f ~ sqrt(k/m)). s = .002 is +-0.2 Hz at 100 Hz and +-2 Hz at
        1 kHz, under the ~9 Hz roll that destroys the signal (probe.py: iou .30 -> .06)"""
        B, L, F_, C = x.shape
        lo, hi = self.freq_hz
        f = torch.linspace(lo, hi, F_, device=x.device)
        eps = (torch.rand(B, 1, device=x.device) * 2 - 1) * self.freq_jitter
        src = ((f[None] / (1 + eps) - lo) / (hi - lo) * (F_ - 1)).clamp(0, F_ - 1)        # (B,F) the bin each output bin reads from
        i0 = src.floor().long().clamp(max=F_ - 2); w = (src - i0)[:, None, :, None]
        idx = lambda i: i[:, None, :, None].expand(B, L, F_, C)
        xf = x.float()
        return (xf.gather(2, idx(i0)) * (1 - w) + xf.gather(2, idx(i0 + 1)) * w).to(x.dtype)

    def colorize(self, x):
        """--color-aug: a random smooth spectral tilt per sample, shared by every laser and channel (an unseen speaker's S(f)):
        g(f) = color_aug * sum_j N(0,1)/j cos(pi j u + phase), u = log f in [0,1], j <= 4; added to log signals, exp(g) times the others"""
        F_ = x.shape[2]
        u = torch.linspace(0, 1, F_, device=x.device)                                      # bins are linear in f; close enough in log over a decade
        j = torch.arange(1, 5, device=x.device)
        a = torch.randn(len(x), 4, device=x.device) / j * self.color_aug
        ph = torch.rand(len(x), 4, device=x.device) * 2 * math.pi
        g = (a[:, :, None] * torch.cos(math.pi * j[None, :, None] * u[None, None] + ph[:, :, None])).sum(1)  # (B,F)
        g = g[:, None, :, None]
        return x + g if self.signal.startswith("log") else x * g.exp()

    def forward(self, batch):
        x = self.phase_features(batch["fft"])
        if self.training and self.color_aug > 0: x = self.speaker_gain(x) if self.color_ell > 0 else self.colorize(x)
        if self.training and self.laser_jitter > 0: x = self.jitter_lasers(x)
        if self.training and self.freq_jitter > 0: x = self.jitter_freqs(x)
        if self.encoder == "modal": return self.forward_modal(x, batch)
        if self.encoder == "conv": emb = self.grid_encoder(self.freq_encoder(x))  # (B,L,F,C) -> (B,256,L) -> (B,d_model)
        else: emb = self.transformer(x)                    # (B,L,P,PS,C) -> (B,d_model)
        if self.polar: return self.forward_polar(emb)
        if self.slots: return self.forward_slots(emb)
        logits, edge_logits, *heat = self.decoder(emb)     # (B,d_model) -> (B,H,W), or (B,H,W,3) if rgb; (B,H,W) or None; slots: (B,K-1,H,W)
        if self.pose is not None: return self.with_pose(dict(mask_logits=logits, **self.heads(emb)) | ({"edge_logits": edge_logits} if edge_logits is not None else {}), emb, *heat)
        out = dict(
            mask_logits=logits,                                       # (B,H,W)
            mask_pred=logits.sigmoid(),                               # (B,H,W) -> (B,H,W) in [0,1]
            object_logits=self.object_head(emb),                      # (B,d_model) -> (B,n_known)
            n_objects_logits=self.n_objects_head(emb),                # (B,d_model) -> (B,max_objects+1)
            com=self.com_head(emb).sigmoid().view(len(emb), -1, 2),   # (B,d_model) -> (B,max_objects,2) in [0,1]
            log_area=self.area_head(emb).squeeze(-1),                 # (B,d_model) -> (B,)
            box_logits=self.box_head(emb),                            # (B,d_model) -> (B,n_boxes)
        )
        if edge_logits is not None: out["edge_logits"] = edge_logits
        if self.slots_aux: out.update({k: v for k, v in self.forward_slots(emb, draw=False).items() if k.startswith(("slot_", "peak_"))})  # mask_pred stays the mask head's
        return out

    def with_pose(self, out, emb, heat=None):
        """merge the pose branch's warped canonical masks into the direct decoder's: logaddexp (a soft OR)"""
        if self.pose.merge == "slots": return self.with_slots(out, emb, heat)
        if self.pose.merge == "only":                                                      # the direct map is a centre heatmap, not a mask:
            w = out["mask_logits"].float().flatten(1).softmax(1).view_as(out["mask_logits"])  # soft-argmax of its softmax (integral regression)
            H, W = w.shape[-2:]
            r, c = (torch.arange(H, device=w.device) + 0.5) / H, (torch.arange(W, device=w.device) + 0.5) / W
            centre = torch.stack([(w.sum(2) * r).sum(1), (w.sum(1) * c).sum(1)], -1)[:, None]
            out["center_logits"] = out["mask_logits"]
        elif self.pose_center == "mask":                                                   # soft centroid of sigma(direct)^2, (B,1,2)
            w = out["mask_logits"].float().sigmoid() ** 2
            H, W = w.shape[-2:]
            r, c = (torch.arange(H, device=w.device) + 0.5) / H, (torch.arange(W, device=w.device) + 0.5) / W
            a = w.flatten(1).sum(1).clamp_min(1e-6)
            centre = torch.stack([(w.sum(2) * r).sum(1) / a, (w.sum(1) * c).sum(1) / a], -1)[:, None]
        else: centre = out["com"]
        pl, v = self.pose(emb, centre)
        # or: a soft OR -- the pose path can only ADD pixels (s8-hybrid-pose: area x1.5, phantoms .40 vs .22, iou -.07, orientation no
        # better); sum: a residual -- it can add and erase, as rotating a shape needs, and the direct path can cancel its phantoms
        # only: the warped canonical mask IS the mask, drawn once (s8-best-poseonly)
        merged = {"or": lambda d: torch.logaddexp(d, pl), "sum": lambda d: d + pl, "only": lambda d: pl}[self.pose.merge]
        out["mask_logits"] = merged(out["mask_logits"].float())
        out["mask_pred"], out["orient"], out["pose_logits"] = out["mask_logits"].sigmoid(), v[:, 0], pl
        return out

    def with_slots(self, out, emb, heat):
        """--pose-merge slots: K centre heatmaps (decoder channel 0 + the K-1 extra) -> soft-argmax centres; slot k's canonical mask
        warped there by its angle, gated by sigmoid(exists_k); the mask = logsumexp over slots (a soft max: each object drawn once)"""
        hm = torch.cat([out["mask_logits"][:, None], heat], 1).float()                     # (B,K,H,W)
        B, K, H, W = hm.shape
        w = hm.flatten(2).softmax(-1).view_as(hm)
        r, c = (torch.arange(H, device=w.device) + 0.5) / H, (torch.arange(W, device=w.device) + 0.5) / W
        centre = torch.stack([(w.sum(3) * r).sum(2), (w.sum(2) * c).sum(2)], -1)            # (B,K,2) row, col in [0,1]
        warped, v = self.pose(emb, centre)                                                 # (B,K,H,W), (B,K,2)
        ex = self.exists_head(emb).float()                                                 # (B,K)
        out["mask_logits"] = soft_or(warped, ex)
        out.update(mask_pred=out["mask_logits"].sigmoid(), slot_heat=hm, slot_centre=centre, slot_masks=warped, slot_exists=ex, slot_angle=v,
                   orient=v[:, 0])                                                         # orient: replaced by the matched slot's in the loss
        return out

    def loss_slots_pose(self, out, batch):
        """per object (slots matched to the true objects by centre, as match_slots): exists bce, its warped mask vs its own instance
        mask (the mask loss), its heatmap vs its instance (softmax CE); the orient loss reads the slot matched to object 0"""
        inst = (batch["inst"] > 0.5).float()                                               # (B,K,H,W)
        real = inst.flatten(2).sum(-1) > 0                                                 # (B,K)
        # match by centre AND exists, as DETR's cost (5 x L1 box - p(class)): centre alone ties when both slots sit on one object (a
        # single-object scene), the match flips at random and exists can't learn (s8-mag-slots v1: exists bce stuck at .63)
        K = inst.shape[1]
        orders = torch.tensor(list(itertools.permutations(range(K))), device=inst.device)  # (P,K)
        cost = 5 * (out["slot_centre"].detach()[:, orders] - batch["coms"][:, None]).abs().sum(-1) - out["slot_exists"].detach().sigmoid()[:, orders]
        sel = orders[(cost * real[:, None]).sum(-1).argmin(1)]                              # (B,K) slot of true object j
        g = lambda v: v.gather(1, sel.view(*sel.shape, *[1] * (v.ndim - 2)).expand(-1, -1, *v.shape[2:]))  # slots -> true-object order
        on = torch.zeros_like(out["slot_exists"]).scatter(1, sel, real.float())
        m, h = g(out["slot_masks"])[real], g(out["slot_heat"])[real]                        # (N,H,W) the matched slots of real objects
        t = inst[real]
        out["orient"] = g(out["slot_angle"])[:, 0]
        return dict(slot_exists=F.binary_cross_entropy_with_logits(out["slot_exists"], on),
                    slot_mask=self.mask_loss(m, t) if len(t) else m.sum() * 0,
                    heatmap=heatmap_loss(h, t) if len(t) else h.sum() * 0)

    def heads(self, emb):
        out = dict(object_logits=self.object_head(emb), n_objects_logits=self.n_objects_head(emb), com=self.com_head(emb).sigmoid().view(len(emb), -1, 2),
                   log_area=self.area_head(emb).squeeze(-1), box_logits=self.box_head(emb))
        if self.orient_head is not None: out["orient"] = self.orient_head(emb).float()
        return out

    def forward_modal(self, x, batch, explain=False):
        y, emb, maps = self.modal(x, batch["box"], explain)                               # (B,out_c+extra,H,W), (B,d_model)
        if y is None and self.obj_decoder != "none": return self.forward_objects(emb, maps, batch, explain)
        if y is None and self.field:                                                       # the field decoder: one MLP per floor cell
            logits, edge = self.decoder(emb, maps["tokens"], maps["keep"], batch["box"])
            out = dict(mask_logits=logits, mask_pred=logits.sigmoid(), **self.heads(emb))
            if edge is not None: out["edge_logits"] = edge
            if explain: out["maps"] = maps
            return out
        if y is None:                                                                      # hybrid: the conv decoder draws from the embedding
            feats = self.decoder.features(emb)
            logits, edge, *heat = self.decoder.from_features(feats)
            out = dict(mask_logits=logits, mask_pred=logits.sigmoid(), **self.heads(emb))
            if self.pose is not None: out = self.with_pose(out, emb, *heat)
            if self.gray:                                                                  # after the pose merge: m = the final mask
                floor = self.gray_floor[batch["box"]]                                                # (B,H,W) logits
                m, bg = (out["mask_pred"] if self.gray_joint else out["mask_pred"].detach()).float(), (floor + self.gray_bg[batch["box"]]).sigmoid()
                app = (floor + self.gray_head(feats.detach() if self.gray_detach else feats)[:, 0].float()).sigmoid()
                out.update(gray_pred=(1 - m) * bg + m * app, gray_bg=bg, gray_app=app)
            if edge is not None: out["edge_logits"] = edge
            if explain: out["maps"] = maps
            return out
        c = 3 if self.rgb else 1
        logits = y[:, :c].permute(0, 2, 3, 1) if self.rgb else y[:, 0]
        out = dict(mask_logits=logits, mask_pred=logits.sigmoid(), **self.heads(emb))
        k = c
        if self.loss_weights.get("edge"): out["edge_logits"] = y[:, k]; k += 1
        if self.loss_weights.get("sdf"): out["sdf_pred"] = y[:, k].float().tanh()
        if explain: out["maps"] = maps
        return out

    def forward_objects(self, emb, maps, batch, explain=False):
        """--obj-decoder: per-object masks on the trunk's pixel features, mask = their soft union (logsumexp of logit + log p(on))"""
        feats = self.decoder.features(emb)                                                 # (B,c,H,W)
        direct, edge = self.decoder.from_features(feats)
        out = dict(direct_logits=direct, **self.heads(emb))
        if edge is not None: out["edge_logits"] = edge
        if self.obj_decoder == "queries":
            ms, es, ang, g = self.objq(maps["tokens"], self.modal.laser_rc, maps["keep"], feats, direct)  # (B,Ly,K,H,W), (B,Ly,K), (B,K,2) or None
            if ang is not None: out["set_angle"] = ang
            if g is not None: out.update(set_mu=g["mu"], set_cov=g["cov"])
            m, e = ms[:, -1].float(), es[:, -1].float()
            union = soft_or(m, e)
            out.update(set_masks=ms, set_exists=es, obj_masks=m.sigmoid(), obj_scores=e.sigmoid())
        else:
            if self.training:                                                              # teacher forcing at the true centres
                inst = (batch["inst"] > 0.5).float()                                       # (B,N,H,W); eval needs no targets (explain callback)
                real = inst.flatten(2).sum(-1) > 0
                centre = moments(inst.flatten(0, 1))[1].view(*inst.shape[:2], 2)
                if self.obj_jitter:                                                        # --obj-jitter px: the masks learn to draw from an imperfect peak
                    centre = centre + torch.randn_like(centre) * self.obj_jitter / torch.tensor(inst.shape[-2:], device=centre.device)
                heat, m, v, on = self.objc(feats, emb, centre, real)
                out["obj_centre_true"] = centre
            else: heat, m, v, on = self.objc(feats, emb, n_min=out["n_objects_logits"].argmax(-1) if self.obj_count_floor else None)
            m = m.float()
            union = m.masked_fill(~on[..., None, None], -30.0).logsumexp(1)                  # no object: ~0, a finite loss
            out.update(center_heat=heat, obj_masks_logits=m, obj_on=on, obj_masks=m.sigmoid(), obj_scores=on.float(), orient=v[:, 0], obj_angle=v)
        out.update(mask_logits=union, mask_pred=union.sigmoid())
        if explain: out["maps"] = maps
        return out

    def loss_objects(self, out, batch):
        """queries: per decoder layer, set matching then bce-dice of each matched query vs its instance + exists bce (unmatched queries'
        weight .1, DETR's eos_coef), averaged over layers. centernet: focal heatmap loss + each teacher-forced mask vs its instance"""
        inst = (batch["inst"] > 0.5).float()                                               # (B,N,H,W)
        real = inst.flatten(2).sum(-1) > 0                                                 # (B,N)
        losses = dict(direct=self.mask_loss(out["direct_logits"], batch["mask_true"]))
        if "set_masks" in out:
            lm, le = [], []
            for l in range(out["set_masks"].shape[1]):
                m, e = out["set_masks"][:, l].float(), out["set_exists"][:, l].float()
                sel = match_sets(m, e, inst, real)                                         # (B,N)
                on = torch.zeros_like(e).scatter(1, sel, real.float())
                le.append(F.binary_cross_entropy_with_logits(e, on, weight=0.1 + 0.9 * on))
                mm = m.gather(1, sel[..., None, None].expand(-1, -1, *m.shape[-2:]))[real]
                lm.append(self.mask_loss(mm, inst[real]) if real.any() else m.sum() * 0)
            losses.update(obj_mask=torch.stack(lm).mean(), obj_exists=torch.stack(le).mean())
            if "set_angle" in out:                                                         # the last layer's match: each object's own angle
                a = out["set_angle"].float().gather(1, sel[..., None].expand(-1, -1, 2))[real]
                losses["obj_orient"] = F.mse_loss(a, orient_target(inst[real])) if real.any() else a.sum() * 0
            if "set_mu" in out:                                                            # gauss: each object's moments straight from its query
                _, mt, ct = moments(inst[real])
                g = lambda k, n: out[k].float().gather(1, sel[..., None].expand(-1, -1, n))[real]
                losses["obj_gauss"] = ((g("set_mu", 2) - mt).abs().sum(-1) + 100 * (g("set_cov", 3) - ct).abs().sum(-1)).mean() if real.any() else out["set_mu"].sum() * 0
        else:                                                                              # the masks only at the true centres (training)
            if "obj_centre_true" in out:
                m = out["obj_masks_logits"]
                losses["obj_mask"] = self.mask_loss(m[real], inst[real]) if real.any() else m.sum() * 0
                a = out["obj_angle"].float()[real]
                losses["obj_orient"] = F.mse_loss(a, orient_target(inst[real])) if real.any() else a.sum() * 0
            _, mt, ct = moments(inst.flatten(0, 1))
            losses["obj_heat"] = centre_heat_loss(out["center_heat"], mt.view(*inst.shape[:2], 2), real, cov=ct.view(*inst.shape[:2], 3) if self.obj_heat_sigma == "obj" else None)
        return losses

    def forward_polar(self, emb):
        center_logits, centerness_logits, params = self.polar_head(self.decoder.features(emb))  # (B,d_model) -> (B,H,W), (B,H,W), (B,P,H,W)
        B, H, W = center_logits.shape
        scores, idx = detect(center_logits, centerness_logits, self.max_objects + 2)         # (B,N) best peaks first: room to count false positives
        keep = (scores > self.peak_thresh).float()                                          # (B,N) peaks taken as objects
        n_found = keep.sum(1).long().clamp(max=self.max_objects)                            # (B,)
        peak_rc = torch.stack([idx // W + 0.5, idx % W + 0.5], -1)                          # (B,N,2) peak pixel centers, px
        peak_rays = rays_at(params, idx, self.ray_shape, self.n_rays)                       # (B,N,n) px
        return dict(
            center_logits=center_logits, centerness_logits=centerness_logits, ray_params=params,
            peak_scores=scores, peak_rc=peak_rc, peak_keep=keep,                            # (B,N), (B,N,2), (B,N)
            peak_rays=peak_rays,
            mask_pred=render(peak_rays, peak_rc, keep, H, W),                               # (B,H,W) in [0,1]
            com=peak_rc[:, :self.max_objects] / torch.tensor([H, W], device=emb.device),    # (B,max_objects,2) best peaks as coms, in [0,1]
            n_objects_logits=F.one_hot(n_found, self.max_objects + 1).float(),              # (B,max_objects+1): argmax = peaks above peak_thresh
            object_logits=self.object_head(emb), log_area=self.area_head(emb).squeeze(-1), box_logits=self.box_head(emb),
        )

    def forward_slots(self, emb, draw=True):
        B, K, H, W = len(emb), self.max_objects, self.out_h, self.out_w
        o = self.slot_head(emb).float().view(B, K, -1)                                      # (B,d_model) -> (B,K,3+P)
        logits, com = o[..., 0], o[..., 1:3].sigmoid()                                      # (B,K), (B,K,2) in [0,1]
        rays = log_rays(o[..., 3:].transpose(1, 2)[..., None], self.ray_shape, self.n_rays)[..., 0].transpose(1, 2).clamp(-3, 10).exp()  # (B,P,K,1) -> (B,K,n) px
        rc, p = com * torch.tensor([H, W], device=emb.device), logits.sigmoid()             # (B,K,2) px, (B,K)
        keep = (p > 0.5).float()
        if not draw: return dict(slot_logits=logits, slot_com_px=rc, slot_rays=rays, peak_scores=p, peak_rc=rc, peak_rays=rays, peak_keep=keep)  # slots_aux: no masks
        best = p.detach().argsort(1, descending=True)                                       # (B,K) slots most-likely-on first: com_err gives
        first = lambda v: v.gather(1, best.view(B, K, *[1] * (v.ndim - 2)).expand_as(v))    # n objects the first n slots (as the polar peaks)
        return dict(
            slot_logits=logits, slot_com_px=rc, slot_rays=rays,                             # in slot order, for the loss
            peak_scores=first(p), peak_rc=first(rc), peak_rays=first(rays), peak_keep=first(keep),  # the polar head's names: callbacks + viz read these
            mask_pred=render(rays, rc, keep, H, W),                                         # (B,H,W) the slots that are on
            mask_soft=render(rays, rc, p, H, W),                                            # (B,H,W) each slot weighted by p(on), for the render loss
            com=first(com), n_objects_logits=F.one_hot(keep.sum(1).long(), K + 1).float(),
            object_logits=self.object_head(emb), log_area=self.area_head(emb).squeeze(-1), box_logits=self.box_head(emb),
        )

    def loss_slots(self, out, batch):
        losses = slot_losses(out, batch["inst"], self.n_rays)                              # exists, slot_com, rays
        losses["render"] = soft_dice_loss(out["mask_soft"], batch["mask_true"])            # the drawn masks vs the true union
        losses.update(object=object_loss(out["object_logits"], batch["object"], self.n_known), area=area_loss(out["log_area"], batch["mask_true"]),
                      box=box_loss(out["box_logits"], batch["box"]))
        total = losses["exists"] + sum(weight * losses[key] for key, weight in self.loss_weights.items() if weight and key in losses)
        return dict(total=total, **losses)

    def loss(self, out, batch):
        if self.polar: return self.loss_polar(out, batch)
        if self.slots: return self.loss_slots(out, batch)
        losses = dict(
            mask=self.mask_loss(out["mask_logits"], batch["mask_true"]),
            object=object_loss(out["object_logits"], batch["object"], self.n_known),
            n_objects=n_objects_loss(out["n_objects_logits"], batch["n_objects"]),
            com=com_loss(out["com"], batch["coms"], batch["n_objects"]),
            area=area_loss(out["log_area"], batch["mask_true"]),
            box=box_loss(out["box_logits"], batch["box"]),
        )
        if self.loss_weights.get("boundary"): losses["boundary"] = boundary_loss(out["mask_logits"], batch["mask_true"])
        if self.loss_weights.get("edge"): losses["edge"] = edge_loss(out["edge_logits"], batch["mask_true"])
        if self.loss_weights.get("sdf"): losses["sdf"] = sdf_loss(out["sdf_pred"], batch["mask_true"])
        if self.gray: losses.update(gray_losses(out, batch))
        if "slot_exists" in out: losses.update(self.loss_slots_pose(out, batch))
        if "direct_logits" in out: losses.update(self.loss_objects(out, batch))  # before orient: it picks the matched slot's angle
        if self.loss_weights.get("imoment"):
            inst = (batch["inst"] > 0.5).float()
            losses["imoment"] = instance_moment_loss(out["mask_pred"], inst, inst.flatten(2).sum(-1) > 0, self.moment_terms)
        if self.loss_weights.get("moment"): losses["moment"] = moment_loss(out["mask_pred"], batch["mask_true"], batch["n_objects"], self.moment_terms)
        if self.loss_weights.get("orient"): losses["orient"] = orient_loss(out["orient"], batch["mask_true"], batch["n_objects"])
        if "center_logits" in out: losses["heatmap"] = heatmap_loss(out["center_logits"], batch["mask_true"])
        if self.slots_aux:
            aux = slot_losses(out, batch["inst"], self.n_rays)
            losses.update({f"aux_{k}": v for k, v in aux.items()})
            losses["slots"] = aux["exists"] + self.loss_weights["slot_com"] * aux["slot_com"] + self.loss_weights["rays"] * aux["rays"]
        total = losses["mask"] + sum(weight * losses[key] for key, weight in self.loss_weights.items() if weight and key in losses)  # rays etc: polar/slots only
        return dict(total=total, **losses)  # composer backprops 'total' and logs every key

    def loss_polar(self, out, batch):
        losses = polar_losses(out, batch["inst"], self.ray_shape, self.n_rays, self.center_radius)         # center, rays, centerness
        losses["render"] = soft_dice_loss(out["mask_pred"], batch["mask_true"])            # the drawn masks vs the true union
        losses.update(object=object_loss(out["object_logits"], batch["object"], self.n_known), area=area_loss(out["log_area"], batch["mask_true"]),
                      box=box_loss(out["box_logits"], batch["box"]))
        total = losses["center"] + sum(weight * losses[key] for key, weight in self.loss_weights.items() if weight and key in losses)
        if not total.isfinite():  # say which term went first and whether the outputs (so the weights) already had, before NaNMonitor stops the run
            bad = lambda v: int((~v.isfinite()).sum())
            print("non-finite polar loss:", {k: v.item() for k, v in losses.items()}, "| non-finite outputs:",
                  {k: bad(out[k]) for k in ("center_logits", "centerness_logits", "ray_params")},
                  "| non-finite weights:", [n for n, w in self.named_parameters() if bad(w)][:5], flush=True)
        return dict(total=total, **losses)

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.test_metrics

    def update_metric(self, batch, out, metric):
        if metric.key == "ray_iou": return metric.update(slot_ray_iou(out, batch) if self.slots or self.slots_aux else ray_iou(out, batch, self.ray_shape, self.n_rays))
        metric.update(METRICS[metric.key](out, batch))
