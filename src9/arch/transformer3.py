"""transformer3 (defaults = big-drop20-v1's model: d 256, 8 heads, 4 freq + 8 laser blocks, latent 512, 20% laser + band dropout, 300 ep;
the old std-samplez-v1 = d 128, 4 heads, 4 + 4 blocks, latent 256, 5% dropout, 200 ep: ID .656 / OOD .367 / mean miss .155, 13,387,216 params).
|X| (B,L,n_bins,C) -> per-sample z over (L,n_bins,C) -> 18 bands of 65 bins, a token each -> shared Linear to d -> freq blocks per laser (rope over bands)
-> f = sum_b m_b W_b LN(h_b) -> laser blocks (2D rope over the grid) -> u = LN(sum_l m_l W_l LN(t_l)) + W_mask [m_l, m_b] (latent_dim)
-> Linear(u -> 512x4x4) -> 4 doublings -> 64x64 mask logits (bce + dice). Heads on u: object class (unseen ignored) and box class, CE x 0.1 each.
optional (0 = off): objrecall_w x mean over true objects of 1 - (prob mass on it / its area); boundary_w x mean(p * signed distance) (Kervadec), ramped 0 -> w over epochs 0-50.
mask_loss: bce_dice; sdf = the 64x64 output is a signed distance s (< 0 inside, units of 4 px), L1(tanh s, tanh(true px / 4)) + bce(sigmoid(-s)), mask logits = -s;
lovasz = (1 - dice_w) bce + Lovasz hinge (Berman 2018) instead of the dice term.
heatmap_w (0 = off): a 2nd decoder output channel = object-centre heatmap (Gaussian sigma 2 px at each true object's centroid, rounded to a pixel),
heatmap_w x CenterNet penalty-reduced focal loss (alpha 2, beta 4, / n objects); its bias starts at -2.19 (p = 0.1) as in CenterNet.
maskae (a maskae.pt, src9/maskae.py): Linear(u -> z) -> the frozen mask-AE decoder replaces the decoder, + 0.1 x |z - AE encoder(true mask)|^2.
readout cls (default per): a [cls] token first in each laser's band sequence (its LN'd output = the laser token) and first in the laser sequence
(Linear(LN(its output)) -> u); [cls] sits at rope position 0 (no rotation) and is never dropped.
readout pool: masked mean over the kept tokens of LN(h) -> Linear (freq: d -> d per laser; laser: d -> latent). readout query (DETR / Perceiver IO / Set-Transformer PMA_k):
n_queries learned queries per stage, query_layers decoder layers (self-attn among queries, cross-attn with a learned key-only position embedding, MLP),
concat the n_queries LN'd outputs -> Linear (freq: -> d per laser; laser: -> latent). abspos 1: learned absolute position embeddings N(0, 0.02) added to the token
content before each stage (one per band / per laser), any readout. freeze_readout 1 (per only): W_b, W_l (+ biases) fixed at init. drop_rescale 1 (per only): the per readout sums x n / n_kept, inverted dropout (pool / query / cls / attention renormalize already). All default off = the old model.
band_overlap K (0 = off): each band token also sees K bins on each side (window 65 + 2K, stride 65, still 18 tokens), the frequency axis zero-padded by K (= the mean, after the norm).
tokenizer (default linear): mlp = Linear(in -> 2d) -> GELU -> Linear(2d -> d) instead of the shared Linear, shared by every band.
gan_w (0 = off): a PatchGAN disc on masks (real = true, fake = sigmoid), hinge loss, stepped by its own Adam inside loss() on every microbatch;
+ gan_w x -D(fake), off until epoch 50, ramped 0 -> gan_w over epochs 50-100.
norm (default sample): z-score each sample over (L,n_bins,C); laser = per (sample, bin, channel) over the lasers; freq = per (sample, laser, channel) over the bins
(laser / freq: std clamped >= 1e-3 x the sample's median std). log_mag 1: x = log(x + 1e-3 x the sample's mean |X|) before the norm (default 0 = linear |X|).
imom_w (0 = off): imom_w x utils.instance_moment_loss(p, true instances) (hybrid / s8-mag-nopose-imom's imoment term, weight 1 there).
imom_norm "scale": utils.instance_moment_loss(scale=True), the scale-invariant moments (log area ratio + centroid / s + covariance / s^2, s = sqrt(true area)).
imom_start (0 = off): imom_w and outline_w are 0 before epoch imom_start, then ramped linearly to full over the next 50 epochs.
outline_w (0 = off): hybrid's edge term without a 2nd decoder channel: the predicted outline is utils.outline (3x3 dilation - erosion) of the mask probs p,
outline_w x [(1 - dice_w) bce(outline(p), outline(true)) + dice_w (1 - dice)] (hybrid: bce_dice(edge_logits, outline(true)), dice_w 0.5).
train only: speaker gain, laser jitter (each on a random fraction speaker_gain_p / laser_jitter_p of the samples), token dropout (each laser / band dropped iid with prob laser_drop / band_drop, >= 1 of each kept)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanMetric

from maskae import load as maskae_load
from utils import speaker_gain, laser_jitter, bce_dice, lovasz_hinge, instance_moment_loss, outline, METRICS

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

def heads_attn(q, k, v, h, keep=None):
    """(N,Tq,D), (N,Tk,D), (N,Tk,D) -> (N,Tq,D): h-head softmax attention, no rope; keep (N,Tk) 0 = masked key"""
    split = lambda t: t.view(*t.shape[:2], h, -1).transpose(1, 2)                   # (N,H,T,D/H)
    a = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=None if keep is None else keep[:, None, None].bool())
    return a.transpose(1, 2).flatten(2)

class QueryLayer(nn.Module):
    """DETR decoder layer (pre-norm): self-attention among the queries, cross-attention to the memory (key = W_k(mem + kpos), value = W_v mem), MLP"""
    def __init__(self, d, heads, mlp=4):
        super().__init__()
        self.h, self.n1, self.qkv, self.o1 = heads, nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n2, self.q, self.k, self.v, self.o2 = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.n3, self.mlp = nn.LayerNorm(d), nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x, mem, kpos, keep):                                          # (N,K,D) queries, (N,T,D) LN'd memory, (T,D), (N,T)
        x = x + self.o1(heads_attn(*self.qkv(self.n1(x)).chunk(3, -1), self.h))
        x = x + self.o2(heads_attn(self.q(self.n2(x)), self.k(mem + kpos), self.v(mem), self.h, keep))
        return x + self.mlp(self.n3(x))

class QueryReadout(nn.Module):
    """(N,T,D) LN'd tokens -> (N,d_out): n_q learned queries (DETR / Perceiver IO / PMA_k), `layers` QueryLayers, concat the n_q LN'd outputs -> Linear.
    keys get a learned key-only position embedding kpos (one per token, as DETR adds its spatial encoding to the keys at every layer); values don't."""
    def __init__(self, n_tokens, d, heads, n_q, layers, d_out):
        super().__init__()
        self.queries, self.kpos = nn.Parameter(torch.randn(n_q, d) * 0.02), nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.layers, self.norm, self.out = nn.ModuleList([QueryLayer(d, heads) for _ in range(layers)]), nn.LayerNorm(d), nn.Linear(n_q * d, d_out)

    def forward(self, mem, keep):
        x = self.queries.to(mem.dtype).expand(len(mem), -1, -1)
        for l in self.layers: x = l(x, mem, self.kpos.to(mem.dtype), keep)
        return self.out(self.norm(x).flatten(1))                                       # (N,d_out)

def masked_mean(x, m): return (x * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)  # (N,T,D), (N,T) -> (N,D)

#*** 2 encoder

READOUTS = ("per", "cls", "pool", "query")

def stage_readout(self, readout, n, d, heads, d_out, n_queries, query_layers, freeze_readout, cls_proj):
    """the readout params of a stage (n tokens -> d_out): per = n (d -> d_out) matrices; cls = a [cls] token (+ Linear (d -> d_out) if cls_proj: the laser stage);
    pool = Linear(masked mean); query = QueryReadout. freeze_readout (per only): W, b fixed at init"""
    assert readout in READOUTS, readout
    self.cls = nn.Parameter(torch.randn(d) * 0.02) if readout == "cls" else None
    if readout == "per":
        self.read_w, self.read_b = per_token(n, d, d_out)                              # (n,d,d_out), (d_out,)
        if freeze_readout: self.read_w.requires_grad_(False); self.read_b.requires_grad_(False)
    elif readout == "cls" and cls_proj or readout == "pool": self.read = nn.Linear(d, d_out)
    elif readout == "query": self.query = QueryReadout(n, d, heads, n_queries, query_layers, d_out)
    assert not freeze_readout or readout == "per", "--freeze-readout is for --readout per"

class FreqEncoder(nn.Module):
    """(B,L,n_bands,d) band tokens -> (B,L,d): (+ abspos) blocks over each laser's bands, then f = sum_b m_b W_b LN(h_b) + b (per), LN(h_cls) (cls),
    W mean_b(m_b LN(h_b)) + b (pool), or QueryReadout(LN(h)) (query)"""
    def __init__(self, n_bands, d, heads, layers, readout="per", abspos=0, n_queries=16, query_layers=1, freeze_readout=0, drop_rescale=0):
        super().__init__()
        self.drop_rescale = drop_rescale
        self.readout, self.freqs = readout, nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.read_norm, pos = nn.LayerNorm(d), torch.arange(n_bands)[:, None]
        stage_readout(self, readout, n_bands, d, heads, d, n_queries, query_layers, freeze_readout, cls_proj=False)
        if readout == "cls": pos = torch.cat([torch.zeros(1, 1), pos])                # [cls] at position 0
        self.abspos = nn.Parameter(torch.randn(n_bands, d) * 0.02) if abspos else None  # learned, added to each band token's content
        self.register_buffer("rope", rope(pos, d // heads), persistent=False)        # 1D: band index

    def forward(self, x, m_b):                                                       # (B,L,n_bands,d), (B,n_bands)
        B, L, n_bands, _ = x.shape
        if self.abspos is not None: x = x + self.abspos.to(x.dtype)
        x, keep = x.flatten(0, 1), m_b.repeat_interleave(L, 0)                        # (B*L,n_bands,d), (B*L,n_bands)
        if self.cls is not None: x, keep = torch.cat([self.cls.to(x.dtype).expand(len(x), 1, -1), x], 1), F.pad(keep, (1, 0), value=1)
        for blk in self.freqs: x = blk(x, self.rope, keep)
        if self.cls is not None: return self.read_norm(x[:, 0]).view(B, L, -1)        # (B,L,d)
        if self.readout == "pool": return self.read(masked_mean(self.read_norm(x), keep.to(x.dtype))).view(B, L, -1)
        if self.readout == "query": return self.query(self.read_norm(x), keep).view(B, L, -1)
        x = self.read_norm(x.view(B, L, n_bands, -1)) * m_b[:, None, :, None].to(x.dtype)
        if self.drop_rescale: x = x * (n_bands / m_b.sum(1)).to(x.dtype)[:, None, None, None]  # inverted dropout: sum over kept bands x n / n_kept (= 1 at eval)
        return torch.einsum("blpd,pde->ble", x, self.read_w) + self.read_b             # (B,L,d)

class LaserEncoder(nn.Module):
    """(B,L,d) laser tokens -> u (B,latent_dim): (+ abspos) blocks over the lasers, then u = LN(r) + W_mask [m_l, m_b], r = sum_l m_l W_l LN(t_l) + b (per),
    W LN(t_cls) + b (cls), W mean_l(m_l LN(t_l)) + b (pool), or QueryReadout(LN(t)) (query)"""
    def __init__(self, rows, cols, n_bands, d, heads, layers, latent_dim, readout="per", abspos=0, n_queries=16, query_layers=1, freeze_readout=0, drop_rescale=0):
        super().__init__()
        L, self.drop_rescale = rows * cols, drop_rescale
        self.readout, self.lasers = readout, nn.ModuleList([Block(d, heads) for _ in range(layers)])
        rc = torch.stack(torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij"), -1).view(L, 2)
        stage_readout(self, readout, L, d, heads, latent_dim, n_queries, query_layers, freeze_readout, cls_proj=True)
        if readout == "cls": rc = torch.cat([torch.zeros(1, 2, dtype=rc.dtype), rc])  # [cls] at (0, 0) rotation = none
        self.read_norm, self.u_norm = nn.LayerNorm(d), nn.LayerNorm(latent_dim)
        self.w_mask = nn.Linear(L + n_bands, latent_dim)
        self.abspos = nn.Parameter(torch.randn(L, d) * 0.02) if abspos else None       # learned, added to each laser token's content
        self.register_buffer("rope", rope(rc, d // heads), persistent=False)         # 2D: laser (row, col), row-major

    def forward(self, x, m_l, m_b):                                                  # (B,L,d), (B,L), (B,n_bands)
        if self.abspos is not None: x = x + self.abspos.to(x.dtype)
        if self.cls is not None:
            x = torch.cat([self.cls.to(x.dtype).expand(len(x), 1, -1), x], 1)
            for blk in self.lasers: x = blk(x, self.rope, F.pad(m_l, (1, 0), value=1))
            u = self.read(self.read_norm(x[:, 0]))
        else:
            for blk in self.lasers: x = blk(x, self.rope, m_l)
            if self.readout == "pool": u = self.read(masked_mean(self.read_norm(x), m_l.to(x.dtype)))
            elif self.readout == "query": u = self.query(self.read_norm(x), m_l)
            else:
                x = self.read_norm(x) * m_l[..., None].to(x.dtype)
                if self.drop_rescale: x = x * (m_l.shape[1] / m_l.sum(1)).to(x.dtype)[:, None, None]  # inverted dropout over the kept lasers
                u = torch.einsum("bld,lde->be", x, self.read_w) + self.read_b
        return self.u_norm(u) + self.w_mask(torch.cat([m_l, m_b], -1).to(u.dtype))   # (B,latent_dim) the bottleneck

class Encoder(nn.Module):
    """(B,L,n_bins,C) |X| -> u (B,latent_dim): per-sample z -> cut into bands -> shared Linear to d -> freq enc -> laser enc"""
    def __init__(self, n_channels, n_bins, rows, cols, bins_per_band, d, heads, freq_layers, laser_layers, latent_dim, readout="per", norm="sample", log_mag=0,
                 abspos=0, n_queries=16, query_layers=1, freeze_readout=0, drop_rescale=0, band_overlap=0, tokenizer="linear"):
        super().__init__()
        assert norm in ("sample", "laser", "freq"), norm
        self.norm, self.log_mag = norm, log_mag
        assert n_bins % bins_per_band == 0, (n_bins, bins_per_band)
        self.L, self.n_bands, self.bins_per_band = rows * cols, n_bins // bins_per_band, bins_per_band  # 1170 bins -> 18 bands
        assert tokenizer in ("linear", "mlp"), tokenizer
        self.overlap, n_in = band_overlap, (bins_per_band + 2 * band_overlap) * n_channels  # window = 65 + 2K bins
        self.embed = nn.Linear(n_in, d) if tokenizer == "linear" else nn.Sequential(nn.Linear(n_in, 2 * d), nn.GELU(), nn.Linear(2 * d, d))  # shared by every band
        ro = dict(abspos=abspos, n_queries=n_queries, query_layers=query_layers, freeze_readout=freeze_readout, drop_rescale=drop_rescale)
        self.freq_enc = FreqEncoder(self.n_bands, d, heads, freq_layers, readout, **ro)
        self.laser_enc = LaserEncoder(rows, cols, self.n_bands, d, heads, laser_layers, latent_dim, readout, **ro)

    def forward(self, x, m_l, m_b):
        if self.log_mag: x = torch.log(x.clamp_min(0) + 1e-3 * x.mean((1, 2, 3), keepdim=True))  # eps = 1e-3 x the sample's mean |X|
        if self.norm == "sample": x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-6)  # z-score each sample
        else:                                                                          # z over lasers (dim 1) or freq bins (dim 2)
            dim = 1 if self.norm == "laser" else 2
            std = x.std(dim, keepdim=True)
            floor = 1e-3 * std.flatten(1).median(1).values.view(-1, 1, 1, 1)          # >= 1e-3 x the sample's median std: near-silent rows don't blow up
            x = (x - x.mean(dim, keepdim=True)) / (torch.maximum(std, floor) + 1e-6)
        if self.overlap: x = F.pad(x, (0, 0, self.overlap, self.overlap)).unfold(2, self.bins_per_band + 2 * self.overlap, self.bins_per_band).transpose(-1, -2).flatten(-2)  # (B,L,n_bands,(65+2K)*C)
        else: x = x.reshape(*x.shape[:2], self.n_bands, self.bins_per_band, -1).flatten(-2)  # tokenize (B,L,n_bands,bins*C)
        return self.laser_enc(self.freq_enc(self.embed(x), m_b), m_l, m_b)            # (B,L,n_bands,d) -> (B,L,d) -> (B,latent_dim)

#*** 3 decoder

class Decoder(nn.Module):
    """u (B,latent_dim) -> (B,H,W) logits: Linear -> 512 x H/16 x W/16 seed, 4 doublings 512 -> 256 -> 128 -> 64 -> 32, a 3-conv head"""
    def __init__(self, latent_dim, height=64, width=64, n_out=1):
        super().__init__()
        assert height % 16 == 0 and width % 16 == 0, (height, width)
        self.h0, self.w0 = height // 16, width // 16
        self.project = nn.Linear(latent_dim, 512 * self.h0 * self.w0)
        self.ups = nn.ModuleList([TwoBranchUp(c, c // 2) for c in (512, 256, 128, 64)])
        self.conv1, self.conv2, self.conv3 = nn.Conv2d(32, 32, 3, padding=1), nn.Conv2d(32, 32, 3, padding=1), nn.Conv2d(32, n_out, 3, padding=1)

    def forward(self, u):
        x = self.project(u).view(-1, 512, self.h0, self.w0)
        for up in self.ups: x = up(x)                                                  # (B,32,H,W)
        return self.conv3(F.relu(self.conv2(F.relu(self.conv1(x))))).squeeze(1)     # (B,H,W), or (B,n_out,H,W)

#*** 4 model

def centre_heatmap(objs, sigma=2.0):
    """(B,K,H,W) object masks -> (B,H,W): max over objects of a peak-1 Gaussian at each one's centroid (rounded to a pixel); 0 = none"""
    H, W = objs.shape[-2:]
    r, c = torch.arange(H, device=objs.device, dtype=torch.float32), torch.arange(W, device=objs.device, dtype=torch.float32)
    a = objs.flatten(2).sum(-1)                                                       # (B,K)
    cr, cc = ((objs.sum(-1) * r).sum(-1) / a.clamp_min(1)).round(), ((objs.sum(-2) * c).sum(-1) / a.clamp_min(1)).round()
    g = (-((r[:, None] - cr[..., None, None]) ** 2 + (c - cc[..., None, None]) ** 2) / (2 * sigma ** 2)).exp() * (a > 0)[..., None, None]
    return g.amax(1)

def focal(logits, t):
    """CenterNet penalty-reduced focal loss (alpha 2, beta 4), summed and / the number of peaks"""
    logits, pos = logits.float(), t == 1
    p = logits.sigmoid()
    l = torch.where(pos, (1 - p) ** 2 * F.logsigmoid(logits), (1 - t) ** 4 * p ** 2 * F.logsigmoid(-logits))
    return -l.sum() / pos.sum().clamp_min(1)

def patch_disc():
    """PatchGAN on (B,1,64,64) masks in [0,1] -> (B,1,8,8) real / fake logits: 3 stride-2 4x4 convs 1 -> 32 -> 64 -> 128 (spectral norm, leaky relu), a 3x3 conv to 1"""
    sn = nn.utils.parametrizations.spectral_norm
    return nn.Sequential(sn(nn.Conv2d(1, 32, 4, 2, 1)), nn.LeakyReLU(0.2), sn(nn.Conv2d(32, 64, 4, 2, 1)), nn.LeakyReLU(0.2),
                         sn(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2), sn(nn.Conv2d(128, 1, 3, padding=1)))

def drop(B, n, p, device):
    """token dropout -> (B,n) keep mask, 1 = keep; each token dropped iid with prob p, >= 1 kept"""
    keep = torch.rand(B, n, device=device) >= p
    keep[torch.arange(B, device=device), torch.randint(0, n, (B,), device=device)] |= ~keep.any(1)  # none kept -> keep a random one
    return keep.float()

class Transformer3(ComposerModel):
    def __init__(self, data_info, speaker_gain=0.35, speaker_gain_hz=20.0, laser_jitter=0.5, height=64, width=64, bins_per_band=65,
                 d=256, heads=8, freq_layers=4, laser_layers=8, latent_dim=512, laser_drop=0.2, band_drop=0.2, dice_w=0.5, objrecall_w=0.0, boundary_w=0.0,
                 mask_loss="bce_dice", maskae="", readout="per", heatmap_w=0.0, laser_jitter_p=1.0, speaker_gain_p=1.0, gan_w=0.0, norm="sample", log_mag=0,
                 abspos=0, n_queries=16, query_layers=1, freeze_readout=0, drop_rescale=0, band_overlap=0, tokenizer="linear", imom_w=0.0, outline_w=0.0, imom_norm="", imom_start=0):
        super().__init__()
        assert imom_norm in ("", "scale"), imom_norm
        self.imom_w, self.outline_w, self.imom_norm, self.imom_start = imom_w, outline_w, imom_norm, imom_start
        self.freq_hz, self.grid_rc = data_info["freq_range"], (data_info["n_laser_rows"], data_info["n_laser_cols"])
        self.encoder = Encoder(data_info["n_channels"], data_info["n_freqs"], *self.grid_rc, bins_per_band, d, heads, freq_layers, laser_layers, latent_dim, readout, norm, log_mag,
                               abspos, n_queries, query_layers, freeze_readout, drop_rescale, band_overlap, tokenizer)
        if maskae:                                                                     # u -> z -> the frozen mask-AE decoder
            assert (height, width) == (64, 64), (height, width)
            self.ae = maskae_load(maskae)
            self.decoder = nn.Sequential(nn.Linear(latent_dim, self.ae.z_dim), self.ae.dec)
        else: self.ae, self.decoder = None, Decoder(latent_dim, height, width, 2 if heatmap_w else 1)  # 2: + the centre heatmap
        assert not (maskae and heatmap_w), "the mask-AE decoder has no heatmap channel"
        if heatmap_w: nn.init.constant_(self.decoder.conv3.bias[1:], -2.19)
        assert mask_loss in ("bce_dice", "sdf", "lovasz"), mask_loss
        self.mask_loss, self.heatmap_w, self.laser_jitter_p, self.speaker_gain_p = mask_loss, heatmap_w, laser_jitter_p, speaker_gain_p
        self.n_known = len(data_info["objects"]) - 1                                  # objects[-1] = "unseen": no logit, ignored by the CE
        self.object_head, self.box_head = mlp(latent_dim, self.n_known), mlp(latent_dim, len(data_info["boxes"]))
        self.speaker_gain, self.speaker_gain_hz, self.laser_jitter, self.laser_drop, self.band_drop, self.dice_w = speaker_gain, speaker_gain_hz, laser_jitter, laser_drop, band_drop, dice_w
        self.objrecall_w, self.boundary_w, self.epoch = objrecall_w, boundary_w, 0                       # epoch: set by callbacks.SetEpoch, for the boundary ramp
        self.gan_w, self.disc, self.d_opt = gan_w, patch_disc() if gan_w else None, None  # disc: its own Adam (made in loss()), not in the main optimizer (run.py)
        keys = [k for k in METRICS if k not in ("n_objects_err", "com_err")]          # no n_objects / com heads
        self.train_metrics, self.test_metrics = {k: MeanMetric() for k in keys}, {k: MeanMetric() for k in keys}
        for k, m in [*self.train_metrics.items(), *self.test_metrics.items()]: m.key = k

    def forward(self, batch):
        x, B, L, n_bands = batch["fft"], len(batch["fft"]), self.encoder.L, self.encoder.n_bands
        m_l, m_b = x.new_ones(B, L), x.new_ones(B, n_bands)
        if self.training:                                                              # augs + token dropout
            if self.speaker_gain > 0: x = speaker_gain(x, self.freq_hz, self.speaker_gain, self.speaker_gain_hz, self.speaker_gain_p)
            if self.laser_jitter > 0: x = laser_jitter(x, *self.grid_rc, self.laser_jitter, self.laser_jitter_p)
            if self.laser_drop or self.band_drop: m_l, m_b = drop(B, L, self.laser_drop, x.device), drop(B, n_bands, self.band_drop, x.device)
        return self.forward_tokens(x, m_l, m_b)

    def forward_tokens(self, x, m_l, m_b):
        """(B,L,n_bins,C) post-aug |X|, (B,L) / (B,n_bands) keep masks -> outputs; debug_nan (set by run.py --debug-nan 1): + the encoder's inputs, for callbacks.NaNDebug"""
        u = self.encoder(x, m_l, m_b)                                                  # (B,latent_dim)
        out = dict(object_logits=self.object_head(u), box_logits=self.box_head(u), latent=u)
        if getattr(self, "debug_nan", 0): out |= dict(x_in=x, m_l=m_l, m_b=m_b)
        if self.ae is not None: out["z"] = self.decoder[0](u); y = self.ae.decode(out["z"])  # (B,z_dim) -> (B,H,W)
        else: y = self.decoder(u)                                                      # (B,H,W)
        if self.heatmap_w: y, out["heatmap_logits"] = y[:, 0], y[:, 1]
        if self.mask_loss == "sdf": out["sdf"], y = y, -y                              # the output is the signed distance; mask logits = -sdf
        return dict(mask_logits=y, mask_pred=y.sigmoid(), **out)

    def loss(self, out, batch):
        y, true = out["mask_logits"], batch["mask_true"]
        if self.mask_loss == "sdf":
            t = (batch["sdf"] * true.shape[-2] / 4).tanh().masked_fill(~(true > 0.5).flatten(1).any(1)[:, None, None], 1.0)  # image heights -> units of 4 px; empty = all outside
            mask = F.l1_loss(out["sdf"].float().tanh(), t) + F.binary_cross_entropy_with_logits(y, true)
        elif self.mask_loss == "lovasz": mask = (1 - self.dice_w) * F.binary_cross_entropy_with_logits(y, true) + lovasz_hinge(y, true)
        else: mask = bce_dice(y, true, self.dice_w)
        losses = dict(mask=mask,
                      object=F.cross_entropy(out["object_logits"], batch["object"], ignore_index=self.n_known),
                      box=F.cross_entropy(out["box_logits"], batch["box"]))
        p, total = out["mask_logits"].float().sigmoid(), losses["mask"] + 0.1 * losses["object"] + 0.1 * losses["box"]
        if self.objrecall_w:
            a = batch["objs"].flatten(2).sum(-1)                                       # (B,K) object areas, 0 = no object
            losses["objrecall"] = (1 - (p[:, None] * batch["objs"]).flatten(2).sum(-1) / a.clamp_min(1))[a > 0].mean() if (a > 0).any() else p.sum() * 0
            total = total + self.objrecall_w * losses["objrecall"]
        if self.ae is not None:
            losses["latent"] = (out["z"].float() - self.ae.encode(true.float())).pow(2).sum(1).mean()
            total = total + 0.1 * losses["latent"]
        if self.heatmap_w:
            losses["heatmap"] = focal(out["heatmap_logits"], centre_heatmap(batch["objs"].float()))
            total = total + self.heatmap_w * losses["heatmap"]
        if self.boundary_w:
            losses["boundary"] = (p * batch["sdf"]).mean()
            total = total + self.boundary_w * min(1.0, self.epoch / 50) * losses["boundary"]
        ramp = min(1.0, max(0.0, (self.epoch - self.imom_start) / 50)) if self.imom_start else 1.0  # imom + outline: off until imom_start, full 50 epochs later
        if self.imom_w:
            losses["imoment"] = instance_moment_loss(p, (batch["inst"] > 0.5).float(), scale=self.imom_norm == "scale")
            total = total + self.imom_w * ramp * losses["imoment"]
        if self.outline_w:                                                             # bce_dice on probabilities: the outline of p is no sigmoid of a logit
            o, t = outline(p), outline(true.float())
            with torch.autocast(o.device.type, enabled=False): bce = F.binary_cross_entropy(o, t)  # fp32; autocast refuses bce on probabilities
            dice = 1 - (2 * (o * t).flatten(1).sum(1) + 1e-6).div((o + t).flatten(1).sum(1) + 1e-6).mean()
            losses["outline"] = (1 - self.dice_w) * bce + self.dice_w * dice
            total = total + self.outline_w * ramp * losses["outline"]
        if self.gan_w:                                                                 # 1 disc step (hinge) on this microbatch, then the generator's adversarial term
            real, fake, ps = true.float()[:, None], p[:, None], list(self.disc.parameters())
            if self.d_opt is None: self.d_opt = torch.optim.Adam(ps, 1e-4, betas=(0.5, 0.999))
            losses["disc"] = F.relu(1 - self.disc(real)).mean() + F.relu(1 + self.disc(fake.detach())).mean()
            for q, g in zip(ps, torch.autograd.grad(losses["disc"], ps)): q.grad = g  # overwrites whatever the last generator backward left there
            self.d_opt.step()
            losses["disc"], losses["adv"] = losses["disc"].detach(), -self.disc(fake).mean()
            total = total + self.gan_w * min(1.0, max(0.0, (self.epoch - 50) / 50)) * losses["adv"]  # off until epoch 50, ramped to gan_w by 100
        return dict(total=total, **losses)

    def get_metrics(self, is_train=False): return self.train_metrics if is_train else self.test_metrics
    def update_metric(self, batch, out, metric): metric.update(METRICS[metric.key](out, batch))
