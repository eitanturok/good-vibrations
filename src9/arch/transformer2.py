"""transformer: src's VibrationTransformer (a freq transformer per laser, then a laser transformer) with hybrid's recipe; encoder = transformers only.
per-box input standardization -> each laser's spectrum cut into 18 freq bands of 65 bins (~50 Hz), a token each, + freq band / laser (learned + 2D wall
position) / box embeddings -> 2 freq blocks over each laser's freq bands -> gated bands, concatenated -> one token per laser -> 2 laser blocks -> flatten
-> one embedding -> conv decoder (mask + outline), aux + orientation heads (as hybrid). Train-time augs in the model: speaker gain, laser jitter.
norm: "box" per-box stats | "box_z" per-box stats, then (x-mean)/std over the sample's (L,F,C) | "sample_std" x/std | "sample_z" (x-mean)/std over the sample's (L,F,C) | "laser_z" per (f,c) z-score across lasers. box_emb=0: no box embedding.
pos="rope": no additive band/laser embeddings, rope only (1D over bands, 2D over the laser grid).
readout="archA": no gate, no flatten: f = sum_b m_b W_b LN(h_b) per laser, u = LN(sum_l m_l W_l LN(t_l)) + W_mask m (latent_dim), emb = W_up u; m = band/laser keep masks (all ones by default).
token dropout (archA, train only): each laser dropped iid with prob laser_drop, each band with prob band_drop, >= 1 of each kept. eval: all ones.
loss="full": mask bce + edge (bce-dice on the outline) + per-instance moments + orientation + aux heads. "bce_dice": mask bce-dice + edge (if outline) + aux;
"bce_dice_imom": + per-instance moments. dice_w: mask loss (1-dice_w) bce + dice_w dice; area_w: + |log(Σp+1) - log(Σtrue+1)| (px). tversky: soft tversky (tv_alpha FP, tv_beta FN) instead of dice; ms_dice: dice/tversky averaged over /1,/2,/4,/8. edge_w: bce weight 1+edge_w on the true outline pixels. outline=0: mask channel only, no edge. aux="box_nobj": only the box and n_objects heads (no object/com/area/orient).
batch: fft (B,L,n_bins,C), box (B,), mask_true (B,H,W), inst (B,K,H,W), object, n_objects (B,), coms (B,K,2)"""
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

# RoPE: rotate q, k pairs (i, i + hd/2) by pos * base^(-j/n). base 100, not 10000: positions only reach 17 (bands) / 9 (grid), so with 10000 most pairs
# would barely turn (<0.01 rad); with 100 the slowest pair still turns ~0.2-0.5 rad over the sequence. a [cls] first gets angle 0 = identity
def _rope(a, cls): a = torch.cat([F.pad(a, (0, 0, cls, 0))] * 2, -1); return torch.stack([a.cos(), a.sin()])  # (T,hd/2) angles -> (2,cls+T,hd) cos, sin
def rope_1d(T, hd, cls=0, base=100.): return _rope(torch.arange(T)[:, None] * base ** (-torch.arange(hd // 2) / (hd // 2)), cls)  # 1D: band index
def rope_2d(rows, cols, hd, cls=0, base=100.):                                  # 2D: half the pairs turn by laser row, half by col (row-major lasers)
    r, c = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    f = base ** (-torch.arange(hd // 4) / (hd // 4))
    return _rope(torch.cat([r.reshape(-1, 1) * f, c.reshape(-1, 1) * f], -1), cls)
def rotate(x, cs): x1, x2 = x.chunk(2, -1); return x * cs[0].to(x.dtype) + torch.cat([-x2, x1], -1) * cs[1].to(x.dtype)  # (N,H,T,hd) rotated

class Block(nn.Module):
    """pre-norm: self-attention, then an MLP. n_v > 0: W_V^t, one (d -> d) value matrix per position t (T must be n_v), Q, K, O, MLP shared"""
    def __init__(self, d, heads, mlp=2, n_v=0):
        super().__init__()
        self.h, self.n1, self.qkv, self.o = heads, nn.LayerNorm(d), nn.Linear(d, (2 if n_v else 3) * d), nn.Linear(d, d)
        self.n2, self.mlp = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))
        self.n_v = n_v
        if n_v:  # nn.Linear init per position
            lins = [nn.Linear(d, d) for _ in range(n_v)]
            self.v_w = nn.Parameter(torch.stack([l.weight.T for l in lins]))         # (n_v,d,d)
            self.v_b = nn.Parameter(torch.stack([l.bias for l in lins]))             # (n_v,d)

    def forward(self, x, rope=None, keep=None):                                  # rope: (2,T,D/H) cos, sin or None; keep: (N,T) key padding, 0 = masked key, or None
        N, T, D = x.shape
        h = self.n1(x)
        if self.n_v:
            q, k = self.qkv(h).view(N, T, 2, self.h, -1).permute(2, 0, 3, 1, 4)      # 2 x (N,H,T,D/H)
            v = (torch.einsum("ntd,tde->nte", h, self.v_w) + self.v_b).view(N, T, self.h, -1).transpose(1, 2)  # (N,T,D) -> (N,H,T,D/H) per-position W_V^t
        else: q, k, v = self.qkv(h).view(N, T, 3, self.h, -1).permute(2, 0, 3, 1, 4)  # 3 x (N,H,T,D/H)
        if rope is not None: q, k = rotate(q, rope), rotate(k, rope)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v, attn_mask=None if keep is None else keep[:, None, None].bool()).transpose(1, 2).reshape(N, T, D))  # masked keys -> -inf
        return x + self.mlp(self.n2(x))

#*** 2 encoder

class FreqEncoder(nn.Module):
    """normalized (B,L,n_bins,C) spectrum + (B,) box + (n_bands,) freq bands + (L,d) laser -> (B,L,d) per-laser freq-band tokens"""
    def __init__(self, n_channels, n_boxes, n_bands, bins_per_band, d, heads, layers, cls=0, band_proj=0, band_v=0, rope=None, readout="ref", box_emb=1, pos="add"):
        super().__init__()
        self.register_buffer("rope", rope, persistent=False)                         # (2,cls+n_bands,d/heads) or None
        self.bins_per_band, self.heads, self.cls, self.band_proj, self.readout = bins_per_band, heads, cls, band_proj, readout
        self.use_box, self.pos = box_emb, pos
        if band_proj:  # W_x^(b): one (bins_per_band*C -> d) matrix per band, so each band lands in its own subspace (nn.Linear init per band)
            lins = [nn.Linear(bins_per_band * n_channels, d) for _ in range(n_bands)]
            self.embed_w = nn.Parameter(torch.stack([l.weight.T for l in lins]))     # (n_bands,bins_per_band*C,d)
            self.embed_b = nn.Parameter(torch.stack([l.bias for l in lins]))         # (n_bands,d)
        else:          # one W_x shared by every band
            self.embed = nn.Linear(bins_per_band * n_channels, d)                     # raw bins in: keeps exact peak positions
        if pos == "add": self.freq_band_emb = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))  # pos="rope": none, rope only
        if box_emb: self.box_emb = nn.Embedding(n_boxes, d)
        self.freqs = nn.ModuleList([Block(d, heads, mlp=4, n_v=band_v * (cls + n_bands)) for _ in range(layers)])  # band_v: W_V^b per band (+ [cls])
        if cls:  # a learned [cls] token per laser's band sequence; its output is the laser's token
            self.freq_cls = nn.Parameter(nn.init.trunc_normal_(torch.zeros(d), std=0.02))
        elif readout == "archA":  # no gate: f = sum_b m_b W_b LN(h_b) + bias, one (d -> d) W_b per band (nn.Linear init per band)
            lins = [nn.Linear(d, d) for _ in range(n_bands)]
            self.read_norm = nn.LayerNorm(d)
            self.read_w = nn.Parameter(torch.stack([l.weight.T for l in lins]))      # (n_bands,d,d)
            self.read_b = nn.Parameter(lins[0].bias.detach().clone())                 # (d,) one bias
        else:    # gated bands, concatenated
            self.pool_q = nn.Parameter(nn.init.trunc_normal_(torch.zeros(heads, d // heads), std=0.02))
            self.pool_norm, self.pool_k, self.pool_o = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(n_bands * d, d)

    def forward(self, x, box, freq_band, laser, mask_b=None):                         # (B,L,n_bins,C) + (B,) + (n_bands,) + (L,d) + (B,n_bands) or None
        B, L = x.shape[:2]
        n_bands = len(freq_band)
        x = x.reshape(B, L, n_bands, self.bins_per_band, -1).flatten(-2)  # (B,L,n_bins,C) -> (B,L,n_bands,bins_per_band*C)

        # learn meaningful representation for each freq band by replacing channel dim with learned 128 dim embedding for every band across all lasers, freqs, samples in batch
        x = torch.einsum("blpi,pid->blpd", x, self.embed_w) + self.embed_b if self.band_proj else self.embed(x)  # (B,L,n_bands,bins_per_band*C) -> (B,L,n_bands,d)
        if self.pos == "add": x = x + self.freq_band_emb(freq_band[:, None])[None, None] + laser[None, :, None]  # + band, laser identity/position
        if self.use_box: x = x + self.box_emb(box)[:, None, None]                     # + box
        x = x.flatten(0, 1)                                                           # (B,L,n_bands,d) -> (B*L,n_bands,d)
        if self.cls: x = torch.cat([self.freq_cls.expand(len(x), 1, -1).to(x.dtype), x], 1)  # (B*L,n_bands,d) -> (B*L,1+n_bands,d), [cls] first, no embeddings

        keep = None if mask_b is None else F.pad(mask_b, (self.cls, 0), value=1).repeat_interleave(L, 0)  # (B,n_bands) -> (B*L,cls+n_bands) [cls] always kept

        # learn how to combine the freq bands of each laser
        for block in self.freqs: x = block(x, self.rope, keep)                                   # (B*L,n_bands,d) -> (B*L,n_bands,d), over each laser's bands
        if self.cls: return x[:, 0].view(B, L, -1)                                   # (B*L,1+n_bands,d) -> (B,L,d) the [cls] outputs (no norm)
        x = x.view(B, L, n_bands, -1)
        if self.readout == "archA": return torch.einsum("blpd,pde->ble", self.read_norm(x) * mask_b[:, None, :, None].to(x.dtype), self.read_w) + self.read_b  # (B,L,n_bands,d) -> (B,L,d) masked sum of W_b LN(h_b)

        # now we average the freq bands to get one token per laser of d dimensions
        # to do this smartly, we use the attenion mechanisim
        # pool_q asks "which bands should I read for each laser?"
        # pool_k is the representation of each band
        # pool_o is the final linear projection to get back to d dimensions
        # a.mean(2) is the average over the bands,
        # the laser is added again in Encoder.forward, before the laser stage
        k = self.pool_k(self.pool_norm(x)).view(B, L, n_bands, self.heads, -1)       # (B,L,n_bands,d) -> (B,L,n_bands,H,d/H)
        a = (torch.einsum("blphe,he->blhp", k, self.pool_q) / math.sqrt(k.shape[-1])).softmax(-1)  # (B,L,n_bands,H,d/H)+(H,d/H) -> (B,L,H,n_bands) which bands each laser reads
        return self.pool_o((x * a.mean(2)[..., None] * n_bands).reshape(B, L, -1))  # (B,L,n_bands,d) -> (B,L,n_bands*d) -> (B,L,d)

class LaserEncoder(nn.Module):
    """(B,L,d) tokens -> (B,d_model)"""
    def __init__(self, L, d, d_model, heads, layers, cls=0, laser_v=0, rope=None, readout="ref", latent_dim=128, n_bands=18):
        super().__init__()
        self.register_buffer("rope", rope, persistent=False)                         # (2,cls+L,d/heads) or None
        self.cls, self.readout = cls, readout
        self.lasers = nn.ModuleList([Block(d, heads, mlp=4, n_v=laser_v * (cls + L)) for _ in range(layers)])  # laser_v: W_V^l per laser (+ [cls])
        if cls:  # a learned [cls] token over all lasers; its output is the embedding
            self.laser_cls = nn.Parameter(nn.init.trunc_normal_(torch.zeros(d), std=0.02))
            self.to_emb = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_model))
        elif readout == "archA":  # u = LN(sum_l m_l W_l LN(t_l) + bias) + W_mask m, one (d -> latent_dim) W_l per laser (nn.Linear init per laser); z = W_up u
            lins = [nn.Linear(d, latent_dim) for _ in range(L)]
            self.read_norm, self.u_norm = nn.LayerNorm(d), nn.LayerNorm(latent_dim)
            self.read_w = nn.Parameter(torch.stack([l.weight.T for l in lins]))      # (L,d,latent_dim)
            self.read_b = nn.Parameter(lins[0].bias.detach().clone())                 # (latent_dim,) one bias
            self.w_mask, self.w_up = nn.Linear(L + n_bands, latent_dim), nn.Linear(latent_dim, d_model)
        else:    # flatten every laser token
            self.to_emb = nn.Sequential(nn.LayerNorm(d), nn.Flatten(1), nn.Linear(L * d, d_model))

    def forward(self, x, mask_l=None, mask_b=None):                                  # (B,L,d) + (B,L), (B,n_bands) or None
        if self.cls: x = torch.cat([self.laser_cls.expand(len(x), 1, -1).to(x.dtype), x], 1)  # (B,L,d) -> (B,1+L,d), [cls] first
        keep = None if mask_l is None else F.pad(mask_l, (self.cls, 0), value=1)     # (B,L) -> (B,cls+L) [cls] always kept
        for block in self.lasers: x = block(x, self.rope, keep)                                  # (B,L,d) -> (B,L,d), over the lasers
        if self.readout == "archA":
            u = torch.einsum("bld,lde->be", self.read_norm(x) * mask_l[..., None].to(x.dtype), self.read_w) + self.read_b  # (B,L,d) -> (B,latent_dim) masked sum of W_l LN(t_l)
            self.last_u = u = self.u_norm(u) + self.w_mask(torch.cat([mask_l, mask_b], -1).to(u.dtype))  # (B,latent_dim) the bottleneck, kept for analysis
            return self.w_up(u)                                                        # (B,latent_dim) -> (B,d_model)
        if self.cls: return self.to_emb(x[:, 0])                                       # (B,1+L,d) -> (B,d) the [cls] output -> (B,d_model)
        return self.to_emb(x)                                                          # (B,L,d) -> (B,L*d) -> (B,d_model)

LOG_EPS = 1e-4  # log_z floor. x = |X| / sample std: per-sample median ~.2, mean ~.5, 1% ~.01, min ~1e-3, 2e-6 of bins < 1e-4 (8 train samples): 1e-4 (~5e-4 x median) only touches near-zero bins

class Encoder(nn.Module):
    """(B,L,n_bins,C) spectrum + (B,) box -> (B,d_model)"""
    def __init__(self, n_channels, freqs, rows, cols, n_boxes, d_model, cum_mean, cum_var, bins_per_band=65, d=128, heads=4, freq_layers=2, laser_layers=2, freq_cls=0, laser_cls=0,
                 band_proj=0, laser_proj=0, band_v=0, laser_v=0, rope=0, readout="ref", latent_dim=128, norm="box", box_emb=1, pos="add", empty5=None):
        super().__init__()
        assert readout in ("ref", "archA"), readout
        assert norm in ("box", "box_z", "sample_std", "sample_z", "laser_z", "log_z", "empty5", "log_empty5") and pos in ("add", "rope"), (norm, pos)
        rope = rope or pos == "rope"                                                  # pos="rope" implies rope
        assert readout == "ref" or not (freq_cls or laser_cls), "archA needs freq_cls=0 and laser_cls=0"
        L, n_bins = rows * cols, len(freqs)
        assert n_bins % bins_per_band == 0, f"n_bins {n_bins} must be divisible by bins_per_band {bins_per_band}"
        n_bands = n_bins // bins_per_band                                             # 1170 bins -> 18 freq bands
        edges = [(float(freqs[i]), float(freqs[i + bins_per_band - 1])) for i in range(0, n_bins, bins_per_band)]
        print(f"tokenize freqs [{freqs[0]:.0f}, {freqs[-1]:.0f}] Hz of {n_bins} bins to {n_bands} bands of {bins_per_band} bins: {[(round(lo), round(hi)) for lo, hi in edges]}")
        self.register_buffer("cum_mean", cum_mean)                                    # (n_boxes,L,n_bins,C) precomputed train stats
        self.register_buffer("cum_std", cum_var.rsqrt())
        if norm in ("empty5", "log_empty5"): self.register_buffer("empty5_mean", empty5[0]); self.register_buffer("empty5_std", empty5[1])  # (n_boxes,L,n_bins,C) from 5 empty train scans per box; only then, so other checkpoints keep their keys
        logf = torch.log10(torch.as_tensor(freqs, dtype=torch.float32)) - 2.5         # 100-1000 Hz -> -.5....5
        self.register_buffer("freq_band_logf", logf[torch.arange(n_bands) * bins_per_band + bins_per_band // 2], persistent=False)  # (n_bands,) freq band centres
        # 2^0..2^3 only: on a 10-cell grid 2^4..2^7 pi alias back onto 2^2, 2^3 pi (same sin, -cos)
        a = grid_coords(rows, cols)[..., None] * (2 ** torch.arange(4) * math.pi)    # laser (row,col) -> sin/cos at 2^i pi
        self.register_buffer("laser_fourier", torch.cat([a.sin(), a.cos()], -1).flatten(-2), persistent=False)  # (L,2 coords * 2 sin/cos * 4) = (L,16)
        if pos == "add":
            self.laser_emb = nn.Parameter(nn.init.trunc_normal_(torch.zeros(L, d), std=0.02))
            self.laser_pos = nn.Linear(2 * 2 * 4, d)
        self.freq_enc = FreqEncoder(n_channels, n_boxes, n_bands, bins_per_band, d, heads, freq_layers, freq_cls, band_proj, band_v, rope_1d(n_bands, d // heads, freq_cls) if rope else None, readout, box_emb, pos)
        self.laser_enc = LaserEncoder(L, d, d_model, heads, laser_layers, laser_cls, laser_v, rope_2d(rows, cols, d // heads, laser_cls) if rope else None, readout, latent_dim, n_bands)
        self.readout, self.L, self.n_bands, self.norm, self.pos = readout, L, n_bands, norm, pos
        self.laser_proj = laser_proj
        if laser_proj:  # P_l: one (d -> d) matrix per laser on its token before the laser stage, so each laser lands in its own subspace (nn.Linear init per laser)
            lins = [nn.Linear(d, d) for _ in range(L)]
            self.proj_w = nn.Parameter(torch.stack([l.weight.T for l in lins]))      # (L,d,d)
            self.proj_b = nn.Parameter(torch.stack([l.bias for l in lins]))          # (L,d)

    def normalize(self, x, box, eps=1e-6):
        """box: per-box standardization with precomputed train-set stats; empty5: the same with 5 empty train scans per box; log_empty5: log(x+eps) - log(empty5_mean+eps); log_z: log(x + LOG_EPS) then z over the sample; sample_*: over each sample's (L,F,C); laser_z: per (f,c) across the L lasers"""
        if self.norm in ("box", "box_z"): x = ((x - self.cum_mean[box]) * self.cum_std[box]).to(x.dtype)
        if self.norm == "box": return x
        if self.norm == "log_empty5": return ((x + LOG_EPS).log() - (self.empty5_mean[box] + LOG_EPS).log()).to(x.dtype)  # log ratio to the 5-scan empty mean, no std
        if self.norm == "empty5": return ((x - self.empty5_mean[box]) / self.empty5_std[box]).to(x.dtype)  # "box" with 5-empty-scan stats
        if self.norm == "box_z": return (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + eps)
        if self.norm == "sample_std": return x / (x.std((1, 2, 3), keepdim=True) + eps)
        if self.norm == "log_z": x = (x + LOG_EPS).log(); return (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + eps)  # box-free: log, then per-sample z
        if self.norm == "sample_z": return (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + eps)
        return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + eps)      # laser_z: removes the speaker/box spectrum per sample

    def forward(self, x, box, mask_l=None, mask_b=None):                          # masks: (B,L), (B,n_bands) 1 = keep, 0 = drop; archA: None = all ones
        if self.readout == "archA":                                                   # always passed to the blocks, so explicit ones == default
            if mask_l is None: mask_l = x.new_ones(len(x), self.L)
            if mask_b is None: mask_b = x.new_ones(len(x), self.n_bands)
        x = self.normalize(x, box)                                                    # (B,L,n_bins,C) -> (B,L,n_bins,C) per-box standardized
        laser = self.laser_emb + self.laser_pos(self.laser_fourier) if self.pos == "add" else None  # (L,16) -> (L,d) identity + 2D position; rope: none
        t = self.freq_enc(x, box, self.freq_band_logf, laser, mask_b)                 # (B,L,n_bins,C) -> (B,L,d) one token per laser
        if self.laser_proj: t = torch.einsum("bld,lde->ble", t, self.proj_w) + self.proj_b  # (B,L,d) -> (B,L,d) per-laser P_l
        return self.laser_enc(t + laser if self.pos == "add" else t, mask_l, mask_b)  # + laser identity/position -> (B,d_model)

#*** 3 decoder

class Decoder(nn.Module):
    """(B,d_model) -> (B,out_c,H,W) logits: a 512 x H/s x W/s seed, s = 2^(n ups), doublings with channels 512 -> 256 -> 128 -> 64 -> 32, a 3-conv head"""
    def __init__(self, d_model, out_c=1, height=64, width=64):
        super().__init__()
        self.ups = nn.ModuleList([TwoBranchUp(c, c // 2) for c in (512, 256, 128, 64)])
        s = 2 ** len(self.ups)                                                         # each up doubles H and W: 4 ups -> 16
        assert height % s == 0 and width % s == 0, f"height {height} and width {width} must be divisible by {s} ({len(self.ups)} doublings)"
        self.h0, self.w0 = height // s, width // s                                    # 64x64 -> 4x4 seed
        self.project = nn.Linear(d_model, 512 * self.h0 * self.w0)
        self.conv1, self.conv2, self.conv3 = nn.Conv2d(32, 32, 3, padding=1), nn.Conv2d(32, 32, 3, padding=1), nn.Conv2d(32, out_c, 3, padding=1)

    def forward(self, emb):
        x = self.project(emb).view(-1, 512, self.h0, self.w0)
        for up in self.ups: x = up(x)                      # (B,32,H,W)
        return self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))

#*** 4 aux heads

# the heads (object / count / com / area / box + orient) are built in Transformer.__init__: their state_dict names are flat
def drop_masks(B, rows, cols, n_bands, laser_drop, band_drop, device):
    """train-time token dropout -> (B,L) laser, (B,n_bands) band keep masks, 1 = keep"""
    L = rows * cols
    def drop(n, p):
        keep = torch.rand(B, n, device=device) >= p                                   # (B,n) each token kept iid with prob 1-p
        keep[torch.arange(B, device=device), torch.randint(0, n, (B,), device=device)] |= ~keep.any(1)  # none kept -> keep a random one
        return keep.float()
    return drop(L, laser_drop), drop(n_bands, band_drop)

def mlp(d, n_out): return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, n_out))

#*** 5 losses

WEIGHTS = dict(object=0.1, n_objects=0.1, com=1.0, area=0.1, box=0.1, edge=1.0, imoment=1.0, orient=1.0)  # on top of the mask bce

def overlap_loss(p, t, tversky=0, tv_alpha=0.3, tv_beta=0.7):
    """1 - dice, or 1 - soft tversky TP / (TP + a FP + b FN), per sample then averaged. p, t: (B,H,W)"""
    if not tversky: return 1 - (2 * (p * t).flatten(1).sum(1) + 1e-6).div((p + t).flatten(1).sum(1) + 1e-6).mean()
    tp, fp, fn = (p * t).flatten(1).sum(1), (p * (1 - t)).flatten(1).sum(1), ((1 - p) * t).flatten(1).sum(1)
    return 1 - ((tp + 1e-6) / (tp + tv_alpha * fp + tv_beta * fn + 1e-6)).mean()

def mask_loss(logits, t, dice_w=0.5, tversky=0, tv_alpha=0.3, tv_beta=0.7, ms_dice=0, edge_w=0.0):
    """(1 - dice_w) bce + dice_w overlap; ms_dice: overlap averaged over full res and avg-pooled p, t at /2, /4, /8.
    edge_w: bce pixel weight 1 + edge_w on the true outline (1 px each side of the edge). defaults: utils.bce_dice exactly"""
    if not (tversky or ms_dice or edge_w): return bce_dice(logits, t, dice_w)
    p = logits.sigmoid()
    d = sum(overlap_loss(F.avg_pool2d(p[:, None], s)[:, 0], F.avg_pool2d(t[:, None], s)[:, 0], tversky, tv_alpha, tv_beta) for s in ((1, 2, 4, 8) if ms_dice else (1,))) / (4 if ms_dice else 1)
    return (1 - dice_w) * F.binary_cross_entropy_with_logits(logits, t, weight=1 + edge_w * outline(t) if edge_w else None) + dice_w * d

def total_loss(out, batch, n_known, loss="full", aux="all", dice_w=0.5, area_w=0.0, tversky=0, tv_alpha=0.3, tv_beta=0.7, ms_dice=0, edge_w=0.0):
    mask = batch["mask_true"]
    losses = dict(mask=F.binary_cross_entropy_with_logits(out["mask_logits"], mask) if loss == "full" else mask_loss(out["mask_logits"], mask, dice_w, tversky, tv_alpha, tv_beta, ms_dice, edge_w))
    if aux == "all":
        losses |= aux_losses(out, batch, n_known)
        # log area as a fraction of the image, so it means the same at any H,W. + 1/4096 (1 px at 64x64) keeps empty masks finite;
        # at 64x64 this is exactly utils' log1p(px) - log(4096), a constant the head's bias absorbs
        losses["area"] = F.l1_loss(out["log_area"], (mask.flatten(1).mean(1) + 1 / 4096).log())
    else: losses |= dict(n_objects=F.cross_entropy(out["n_objects_logits"], batch["n_objects"]), box=F.cross_entropy(out["box_logits"], batch["box"]))  # box_nobj
    if "edge_logits" in out: losses["edge"] = bce_dice(out["edge_logits"], outline(mask))
    if loss != "bce_dice": losses["imoment"] = instance_moment_loss(out["mask_pred"], (batch["inst"] > 0.5).float())
    if aux == "all": losses["orient"] = orient_loss(out["orient"], mask, batch["n_objects"])
    if area_w > 0: losses["area_mask"] = ((out["mask_pred"].flatten(1).sum(1) + 1).log() - (mask.flatten(1).sum(1) + 1).log()).abs().mean()  # global log-area in px, +1 px keeps empty finite
    total = losses["mask"] + sum(w * losses[k] for k, w in WEIGHTS.items() if k in losses)
    return dict(total=total + area_w * losses["area_mask"] if area_w > 0 else total, **losses)

#*** 6 model

class Transformer2(ComposerModel):
    def __init__(self, data_info, d_model=1024, speaker_gain=0.35, speaker_gain_hz=20.0, laser_jitter=0.5, height=64, width=64, freq_cls=0, laser_cls=0, freq_layers=2,
                 band_proj=0, laser_proj=0, band_v=0, laser_v=0, laser_layers=2, rope=0, heads=4, readout="ref", latent_dim=128, d=128, norm="box", box_emb=1, pos="add", laser_drop=0.0, band_drop=0.0,
                 outline=1, aux="all", loss="full", dice_w=0.5, area_w=0.0, tversky=0, tv_alpha=0.3, tv_beta=0.7, ms_dice=0, edge_w=0.0):
        super().__init__()
        assert aux in ("all", "box_nobj") and loss in ("full", "bce_dice", "bce_dice_imom"), (aux, loss)
        self.outline, self.aux, self.loss_type, self.dice_w, self.area_w = outline, aux, loss, dice_w, area_w
        self.tversky, self.tv_alpha, self.tv_beta, self.ms_dice, self.edge_w = tversky, tv_alpha, tv_beta, ms_dice, edge_w
        assert readout == "archA" or not (laser_drop or band_drop), "laser_drop/band_drop need readout archA"
        self.laser_drop, self.band_drop = laser_drop, band_drop
        self.freq_hz, self.grid_rc = data_info["freq_range"], (data_info["n_laser_rows"], data_info["n_laser_cols"])
        freqs = torch.linspace(*self.freq_hz, data_info["n_freqs"])                    # (F,) Hz, evenly spaced bins
        self.encoder = Encoder(data_info["n_channels"], freqs, *self.grid_rc, len(data_info["boxes"]), d_model, data_info["cum_mean"], data_info["cum_var"],
                               d=d, heads=heads, freq_layers=freq_layers, laser_layers=laser_layers, rope=rope, freq_cls=freq_cls, laser_cls=laser_cls, band_proj=band_proj, laser_proj=laser_proj, band_v=band_v, laser_v=laser_v,
                               readout=readout, latent_dim=latent_dim, norm=norm, box_emb=box_emb, pos=pos, empty5=(data_info["empty5_mean"], data_info["empty5_std"]) if norm in ("empty5", "log_empty5") else None)
        self.decoder = Decoder(d_model, out_c=2 if outline else 1, height=height, width=width)                     # channels: mask, outline (if outline)
        self.n_known, K = len(data_info["objects"]) - 1, max(data_info["n_objects"])  # objects[-1] = "unseen", no logit
        if aux == "all":
            self.object_head, self.n_objects_head, self.com_head = mlp(d_model, self.n_known), mlp(d_model, K + 1), mlp(d_model, 2 * K)
            self.area_head, self.box_head = mlp(d_model, 1), mlp(d_model, len(data_info["boxes"]))
            self.orient_head = mlp(d_model, 2)                                         # ecc * (cos 2θ, sin 2θ) of a single object
        else: self.n_objects_head, self.box_head = mlp(d_model, K + 1), mlp(d_model, len(data_info["boxes"]))  # box_nobj
        self.speaker_gain, self.speaker_gain_hz, self.laser_jitter = speaker_gain, speaker_gain_hz, laser_jitter  # train-time augs
        keys = [k for k in METRICS if aux == "all" or k not in ("object_acc", "com_err")]  # box_nobj: no object / com heads to score
        self.train_metrics, self.test_metrics = {k: MeanMetric() for k in keys}, {k: MeanMetric() for k in keys}
        for k, m in [*self.train_metrics.items(), *self.test_metrics.items()]: m.key = k

    def forward(self, batch):
        x = batch["fft"]
        if self.training and self.speaker_gain > 0: x = speaker_gain(x, self.freq_hz, self.speaker_gain, self.speaker_gain_hz)
        if self.training and self.laser_jitter > 0: x = laser_jitter(x, *self.grid_rc, self.laser_jitter)
        masks = drop_masks(len(x), *self.grid_rc, self.encoder.n_bands, self.laser_drop, self.band_drop, x.device) if self.training and (self.laser_drop or self.band_drop) else (None, None)
        emb = self.encoder(x, batch["box"], *masks)                                          # (B,d_model)
        y = self.decoder(emb)                                                          # (B,2 or 1,H,W)
        if self.aux == "all": heads = dict(orient=self.orient_head(emb).float(), object_logits=self.object_head(emb), n_objects_logits=self.n_objects_head(emb),
                                           com=self.com_head(emb).sigmoid().view(len(emb), -1, 2), log_area=self.area_head(emb).squeeze(-1), box_logits=self.box_head(emb))
        else: heads = dict(n_objects_logits=self.n_objects_head(emb), box_logits=self.box_head(emb))
        return dict(mask_logits=y[:, 0], mask_pred=y[:, 0].sigmoid(), **(dict(edge_logits=y[:, 1]) if self.outline else {}), **heads,
                    **(dict(latent=self.encoder.laser_enc.last_u) if self.encoder.readout == "archA" else {}))  # archA: u (B,latent_dim)

    def loss(self, out, batch): return total_loss(out, batch, self.n_known, self.loss_type, self.aux, self.dice_w, self.area_w, self.tversky, self.tv_alpha, self.tv_beta, self.ms_dice, self.edge_w)
    def get_metrics(self, is_train=False): return self.train_metrics if is_train else self.test_metrics
    def update_metric(self, batch, out, metric): metric.update(METRICS[metric.key](out, batch))
