"""transformer4 = transformer3 with base300-v1's settings, minus W_mask (ablation t4-nowmask-v1 tied t4-v1).
x (B, L=100 lasers on a 10x10 grid, 1170 bins, C=2) linear |X|
-> z-score each sample over (L, bins, C) -> 18 bands x 65 bins, a token each (65*2 = 130 -> Linear -> D=256, shared by every band)
-> 4 freq blocks per laser (1D rope over the 18 bands; --freq-rope linear = band index | log = log band-center Hz on the same 0..17 span) -> flatten readout f = Linear(flatten(m_b LN(h_b)))  (B, L, D)
-> 8 laser blocks (2D rope over the laser grid) -> flatten readout u = LN(Linear(flatten(m_l LN(t_l))))  (B, 512)
   (--latent-norm rescale: no LN on u, u = (L / sum_l m_l) sum_l m_l W_l LN(t_l) + b, the token-dropout mean over kept lasers; factor 1 at eval)
-> Linear(u -> 512x4x4) -> 4 doublings (--decoder: two = two-branch Up, one = UpOne, res = pre-act UpRes, res-coord = UpRes + CoordConv,
   res-xattn-freq / res-xattn-laser = res + a cross-attention block onto the 100 laser tokens after the 8x8 and 16x16 doublings) -> 3 convs
-> 64x64 mask logits (0.5 bce + 0.5 dice)
-> heads on u: object class (CE, "unseen" ignored) and box class (CE), x 0.1 each.
train only: speaker gain (sd 0.35, knots every 20 Hz), laser jitter (+-0.5 cells), each laser / band token dropped iid with prob 0.2 (>= 1 of each kept)."""
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanMetric

from utils import speaker_gain, laser_jitter, bce_dice, METRICS

H, W = 64, 64
BINS, D, HEADS, FREQ_LAYERS, LASER_LAYERS, LATENT = 65, 256, 8, 4, 8, 512
LASER_DROP, BAND_DROP, DICE_W, AUX_W = 0.2, 0.2, 0.5, 0.1
SPEAKER_GAIN, SPEAKER_GAIN_HZ, LASER_JITTER = 0.35, 20.0, 0.5

#*** 0 helper functions

def drop(B, n, p, device, p_sample=1.0):
    """(B,n) keep mask: each token dropped with prob p, >= 1 kept; p_sample < 1: only a random fraction p_sample of the samples drop any"""
    keep = torch.rand(B, n, device=device) >= p
    if p_sample < 1: keep |= torch.rand(B, 1, device=device) >= p_sample
    keep[torch.arange(B, device=device), torch.randint(0, n, (B,), device=device)] |= ~keep.any(1)
    return keep

def aux_head(n_out): return nn.Sequential(nn.Linear(LATENT, LATENT), nn.ReLU(), nn.Linear(LATENT, n_out))

#*** 1 attention

def rope(pos, hd, base=100.):
    """(T,k) positions -> (2,T,hd) cos, sin; the hd/2 pairs split evenly over the k axes"""
    n = hd // 2 // pos.shape[1]
    a = (pos[..., None] * base ** (-torch.arange(n) / n)).flatten(1)
    a = torch.cat([a, a], -1)
    return torch.stack([a.cos(), a.sin()])

def rotate(x, cs): x1, x2 = x.chunk(2, -1); return x * cs[0].to(x.dtype) + torch.cat([-x2, x1], -1) * cs[1].to(x.dtype)

class AttnBlock(nn.Module):
    """pre-norm rope self-attention (keep = key mask) + MLP"""
    def __init__(self):
        super().__init__()
        self.n1, self.qkv, self.o = nn.LayerNorm(D), nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.n2, self.mlp = nn.LayerNorm(D), nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, cs, keep):                                                  # (N,T,D), (2,T,D/H), (N,T)
        N, T, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(N, T, 3, HEADS, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rotate(q, cs), rotate(k, cs), v, attn_mask=keep[:, None, None])
        x = x + self.o(a.transpose(1, 2).reshape(N, T, D))
        return x + self.mlp(self.n2(x))

#*** 2 MergeBlock

class MergeBlock(nn.Module):
    """(..., n, D) tokens, (..., n) keep mask -> (..., d_out): LN each token, zero the dropped ones, flatten, Linear(n*D -> d_out).
    (= sum_i m_i W_i LN(x_i) + b with W_i the i-th D-column block of the Linear: a flatten head, as in PatchTST / TNT)
    rescale: the matmul x n / sum_i m_i (before + bias). Init as n separate nn.Linear(D, d_out): U(+-1/sqrt(D)), not 1/sqrt(n*D)."""
    def __init__(self, n, d_out, rescale=False):
        super().__init__()
        self.norm, self.rescale = nn.LayerNorm(D), rescale
        lins = [nn.Linear(D, d_out) for _ in range(n)]                                # same RNG draws as the old per-token init -> same init for every module
        self.lin = nn.Linear(n * D, d_out, device="meta")                             # (meta: draws no RNG)
        self.lin.weight, self.lin.bias = nn.Parameter(torch.cat([l.weight.detach() for l in lins], 1)), nn.Parameter(lins[0].bias.detach().clone())

    def forward(self, x, m):
        x = self.norm(x).masked_fill(~m[..., None], 0).flatten(-2)                   # (..., n*D)
        s = F.linear(x, self.lin.weight)
        if self.rescale: s = s * (m.shape[-1] / m.sum(-1, keepdim=True)).to(s.dtype)
        return s + self.lin.bias

    def _load_from_state_dict(self, sd, prefix, *args, **kw):                        # old checkpoints: w (n,D,d_out), b -> lin.weight (d_out, n*D), lin.bias
        if prefix + "w" in sd:
            w = sd.pop(prefix + "w"); sd[prefix + "lin.weight"] = w.permute(2, 0, 1).reshape(w.shape[2], -1); sd[prefix + "lin.bias"] = sd.pop(prefix + "b")
        super()._load_from_state_dict(sd, prefix, *args, **kw)

#**** 3 freq encoder

FREQ_ROPES = ("linear", "log")

class FreqEncoder(nn.Module):
    """freq_rope: linear = rope positions are the band index 0..n_bands-1 | log = (n_bands-1) log(fc_b / fc_0) / log(fc_last / fc_0), fc_b the band's center Hz"""
    def __init__(self, n_bands, freq_rope="linear", freq_hz=None, n_bins=None):
        super().__init__()
        self.freqs, self.read = nn.ModuleList([AttnBlock() for _ in range(FREQ_LAYERS)]), MergeBlock(n_bands, D)
        pos = torch.arange(n_bands)
        if freq_rope == "log":  # mass loading shifts mode frequencies proportionally (df ~ f): equal steps in log f = equal relative shifts
            fc = freq_hz[0] + (freq_hz[1] - freq_hz[0]) * (pos * BINS + (BINS - 1) / 2) / (n_bins - 1)  # band centers: n_bins linear bins over freq_range
            pos = ((n_bands - 1) * (fc / fc[0]).log() / (fc[-1] / fc[0]).log()).float()
        self.register_buffer("rope", rope(pos[:, None], D // HEADS), persistent=False)

    def forward(self, x, m_b):                                                       # (B,L,n_bands,D), (B,n_bands) -> (B,L,D)
        B, L, n_bands, _ = x.shape
        x, keep = x.flatten(0, 1), m_b.repeat_interleave(L, 0)
        for blk in self.freqs: x = blk(x, self.rope, keep)
        return self.read(x.view(B, L, n_bands, -1), m_b[:, None])

#**** 4 laser encoder

LATENT_NORMS = ("ln", "rescale")

class LaserEncoder(nn.Module):
    """latent_norm: ln = LN(readout) | rescale = readout with the kept-laser rescale, no LN (no u_norm module)"""
    def __init__(self, rows, cols, latent_norm="ln"):
        super().__init__()
        L = rows * cols
        self.lasers = nn.ModuleList([AttnBlock() for _ in range(LASER_LAYERS)])
        self.read = MergeBlock(L, LATENT, rescale=latent_norm == "rescale")
        self.u_norm = nn.LayerNorm(LATENT) if latent_norm == "ln" else None
        rc = torch.stack(torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij"), -1).view(L, 2)
        self.register_buffer("rope", rope(rc, D // HEADS), persistent=False)

    def forward(self, x, m_l, tokens=False):                                         # (B,L,D), (B,L) -> (B,LATENT) [, the (B,L,D) tokens before the readout]
        for blk in self.lasers: x = blk(x, self.rope, m_l)
        u = self.read(x, m_l)
        if self.u_norm is not None: u = self.u_norm(u)
        return (u, x) if tokens else u

#*** 5 encoder

class Encoder(nn.Module):
    def __init__(self, n_channels, n_bins, rows, cols, latent_norm="ln", freq_rope="linear", freq_hz=None):
        super().__init__()
        self.L, self.n_bands = rows * cols, n_bins // BINS                             # 100 lasers, 1170 / 65 = 18 bands
        self.embed = nn.Linear(BINS * n_channels, D)
        self.freq_enc, self.laser_enc = FreqEncoder(self.n_bands, freq_rope, freq_hz, n_bins), LaserEncoder(rows, cols, latent_norm)

    def forward(self, x, m_l, m_b, tokens=None):                                     # (B,L,n_bins,C) -> (B,LATENT); tokens "freq" | "laser": also return those (B,L,D) tokens
        x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-6)
        x = x.reshape(*x.shape[:2], self.n_bands, BINS, -1).flatten(-2)               # (B,L,n_bands,BINS*C)
        f = self.freq_enc(self.embed(x), m_b)
        if tokens is None: return self.laser_enc(f, m_l)
        u, t = self.laser_enc(f, m_l, tokens=True)
        return u, (f if tokens == "freq" else t)

#*** 6 decoder

class Up(nn.Module):
    """2x: a transposed conv || conv -> relu -> transposed conv, concat, BN, relu"""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up_a, self.up_b = nn.ConvTranspose2d(c_in, c_out // 2, 4, 2, 1), nn.ConvTranspose2d(c_in, c_out // 2, 4, 2, 1)
        self.conv_b, self.bn = nn.Conv2d(c_in, c_in, 3, padding=1), nn.BatchNorm2d(c_out)

    def forward(self, x): return F.relu(self.bn(torch.cat([self.up_a(x), self.up_b(F.relu(self.conv_b(x)))], 1)))

class UpOne(nn.Module):
    """2x, one branch: conv -> BN -> relu -> transposed conv -> BN -> relu (~17 c_in^2, = Up)"""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c_in, c_in, 3, padding=1), nn.BatchNorm2d(c_in), nn.ReLU(),
                                 nn.ConvTranspose2d(c_in, c_out, 4, 2, 1), nn.BatchNorm2d(c_out), nn.ReLU())

    def forward(self, x): return self.net(x)

def gn(c): return nn.GroupNorm(math.gcd(32, c), c)                                  # 32 groups, fewer when they don't divide c

def add_coords(x):
    """(B,C,h,w) -> (B,C+2,h,w): + row, col channels linearly spaced in [-1, 1] (CoordConv)"""
    B, _, h, w = x.shape
    r, c = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype), torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
    return torch.cat([x, torch.stack(torch.meshgrid(r, c, indexing="ij")).expand(B, 2, h, w)], 1)

class UpRes(nn.Module):
    """2x, pre-act residual: h = nearest up(x); GN -> silu -> conv -> GN -> silu -> conv (zero-init) + 1x1(h).
    hidden width m = 11 c_in / 9 (to a multiple of 16) so 9 c_in m + 9 m c_out + c_in c_out = 17 c_in^2 = Up's; m_scale shrinks it (xattn param match). coord: + row/col channels into both convs on h"""
    def __init__(self, c_in, c_out, coord=0, m_scale=1.0):
        super().__init__()
        m, k = 16 * round(m_scale * 11 * c_in / 144), 2 * coord
        self.cat = add_coords if coord else (lambda x: x)
        self.n1, self.conv1 = gn(c_in), nn.Conv2d(c_in + k, m, 3, padding=1)
        self.n2, self.conv2 = gn(m), nn.Conv2d(m, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in + k, c_out, 1)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)          # starts as upsample + 1x1

    def forward(self, x):
        h = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.skip(self.cat(h)) + self.conv2(F.silu(self.n2(self.conv1(self.cat(F.silu(self.n1(h)))))))

def sincos2d(rows, cols, dim):
    """(rows*cols, dim) fixed 2-D sin-cos encoding of the cell centers in [0,1]^2 (one frame for the pixel and laser grids): per axis dim/4 freqs pi .. 16 pi, sin | cos"""
    rc = torch.stack(torch.meshgrid((torch.arange(rows) + 0.5) / rows, (torch.arange(cols) + 0.5) / cols, indexing="ij"), -1).view(-1, 2)
    n = dim // 4
    a = (rc[..., None] * math.pi * 16 ** (torch.arange(n) / (n - 1))).flatten(1)     # (T, 2n)
    return torch.cat([a.sin(), a.cos()], -1)

class XAttn(nn.Module):
    """pre-norm cross-attention of the (B,c,s,s) pixel grid (queries, + sincos) onto the (B,L,D) laser tokens (keys/values, + sincos, D -> c; keep = key mask) + MLP.
    c/64 heads; Out and the last MLP Linear zero-init, so the block starts as the identity"""
    def __init__(self, c, s, rows, cols, mlp=4):
        super().__init__()
        self.heads = max(1, c // 64)
        self.nq, self.q, self.nkv, self.kv, self.o = nn.LayerNorm(c), nn.Linear(c, c), nn.LayerNorm(D), nn.Linear(D, 2 * c), nn.Linear(c, c)
        self.n2, self.mlp = nn.LayerNorm(c), nn.Sequential(nn.Linear(c, mlp * c), nn.GELU(), nn.Linear(mlp * c, c))
        for lin in (self.o, self.mlp[2]): nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.register_buffer("pe_q", sincos2d(s, s, c), persistent=False)
        self.register_buffer("pe_kv", sincos2d(rows, cols, D), persistent=False)

    def forward(self, h, t, keep):                                                   # (B,c,s,s), (B,L,D), (B,L) -> (B,c,s,s)
        B, c, s, _ = h.shape
        x = h.flatten(2).transpose(1, 2)                                             # (B,s*s,c)
        q = self.q(self.nq(x + self.pe_q)).view(B, s * s, self.heads, -1).transpose(1, 2)
        k, v = self.kv(self.nkv(t + self.pe_kv)).view(B, -1, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None])
        x = x + self.o(a.transpose(1, 2).reshape(B, s * s, c))
        x = x + self.mlp(self.n2(x))
        return x.transpose(1, 2).reshape(B, c, s, s)

XATTN, XATTN_M = {"res-xattn-freq": "freq", "res-xattn-laser": "laser"}, 9 / 11   # kind -> Encoder tokens; UpRes m_scale 9/11: m = c_in, decoder -0.19% vs two's params
DECODERS = ("two", "one", "res", "res-coord", *XATTN)

class Decoder(nn.Module):
    """kind: two / one (BN-relu blocks, plain tail) | res / res-coord (GN-silu pre-act blocks and tail; coord: + row/col channels into every tail conv)
    | res-xattn-freq / res-xattn-laser (res, smaller UpRes, + XAttn onto the FreqEncoder / LaserEncoder tokens after the 8x8 and 16x16 doublings)"""
    def __init__(self, kind="two", grid_rc=(10, 10)):
        super().__init__()
        res, k, self.tokens = kind.startswith("res"), 2 * (kind == "res-coord"), XATTN.get(kind)
        up = {"two": Up, "one": UpOne, "res": UpRes, "res-coord": lambda c_in, c_out: UpRes(c_in, c_out, coord=1),
              "res-xattn-freq": partial(UpRes, m_scale=XATTN_M), "res-xattn-laser": partial(UpRes, m_scale=XATTN_M)}[kind]
        self.project = nn.Linear(LATENT, 512 * (H // 16) * (W // 16))
        self.ups = nn.ModuleList([up(c, c // 2) for c in (512, 256, 128, 64)])
        self.conv1, self.conv2, self.conv3 = nn.Conv2d(32 + k, 32, 3, padding=1), nn.Conv2d(32 + k, 32, 3, padding=1), nn.Conv2d(32 + k, 1, 3, padding=1)
        self.acts = nn.ModuleList([nn.Sequential(gn(32), nn.SiLU()) for _ in range(3)] if res else [nn.Identity(), nn.ReLU(), nn.ReLU()])
        self.cat = add_coords if k else (lambda x: x)
        self.xattn = nn.ModuleList([XAttn(c, s, *grid_rc) for c, s in ((256, H // 8), (128, H // 4))] if self.tokens else [])  # none for the other kinds: same state_dict

    def forward(self, u, t=None, keep=None):                                         # (B,LATENT) [, (B,L,D) tokens, (B,L) keep] -> (B,H,W)
        x = self.project(u).view(-1, 512, H // 16, W // 16)
        for i, up in enumerate(self.ups):
            x = up(x)
            if i < len(self.xattn): x = self.xattn[i](x, t, keep)
        for act, conv in zip(self.acts, (self.conv1, self.conv2, self.conv3)): x = conv(self.cat(act(x)))
        return x.squeeze(1)

#*** 7 model

class Transformer4(ComposerModel):
    def __init__(self, info, speaker_gain=SPEAKER_GAIN, speaker_gain_hz=SPEAKER_GAIN_HZ, laser_jitter=LASER_JITTER, laser_jitter_p=1.0, speaker_gain_p=1.0, laser_drop_p=1.0, band_drop_p=1.0, decoder="two", latent_norm="ln", freq_rope="linear"):
        super().__init__()
        self.sg, self.sg_hz, self.jitter, self.jitter_p, self.sg_p, self.ld_p, self.bd_p = speaker_gain, speaker_gain_hz, laser_jitter, laser_jitter_p, speaker_gain_p, laser_drop_p, band_drop_p
        self.freq_hz, self.grid_rc = info["freq_range"], (info["n_laser_rows"], info["n_laser_cols"])
        self.encoder = Encoder(info["n_channels"], info["n_freqs"], *self.grid_rc, latent_norm, freq_rope, self.freq_hz)
        self.decoder = Decoder(decoder, self.grid_rc)
        self.n_known = len(info["objects"]) - 1                                       # objects[-1] = "unseen": no logit, ignored by the CE
        self.object_head, self.box_head = aux_head(self.n_known), aux_head(len(info["boxes"]))
        keys = [k for k in METRICS if k not in ("n_objects_err", "com_err")]
        self.train_metrics, self.test_metrics = {k: MeanMetric() for k in keys}, {k: MeanMetric() for k in keys}
        for k, m in [*self.train_metrics.items(), *self.test_metrics.items()]: m.key = k

    def forward(self, batch):
        x, B, L, n_bands = batch["fft"], len(batch["fft"]), self.encoder.L, self.encoder.n_bands
        m_l, m_b = x.new_ones(B, L, dtype=torch.bool), x.new_ones(B, n_bands, dtype=torch.bool)  # keep masks (token dropout)
        if self.training:
            x = speaker_gain(x, self.freq_hz, self.sg, self.sg_hz, self.sg_p)
            x = laser_jitter(x, *self.grid_rc, self.jitter, self.jitter_p)
            m_l, m_b = drop(B, L, LASER_DROP, x.device, self.ld_p), drop(B, n_bands, BAND_DROP, x.device, self.bd_p)
        src = self.decoder.tokens                                                     # None, or the Encoder tokens the xattn decoder reads ("freq" | "laser")
        u, t = self.encoder(x, m_l, m_b, src) if src else (self.encoder(x, m_l, m_b), None)
        obj, box = self.object_head(u), self.box_head(u)                              # heads before the decoder: u's grads sum in transformer3's order (bit-equal)
        y = self.decoder(u, t, m_l)
        return dict(mask_logits=y, mask_pred=y.sigmoid(), object_logits=obj, box_logits=box, latent=u)

    def loss(self, out, batch):
        mask = bce_dice(out["mask_logits"], batch["mask_true"], DICE_W)
        obj = F.cross_entropy(out["object_logits"], batch["object"], ignore_index=self.n_known)
        box = F.cross_entropy(out["box_logits"], batch["box"])
        return dict(total=mask + AUX_W * obj + AUX_W * box, mask=mask, object=obj, box=box)

    def get_metrics(self, is_train=False): return self.train_metrics if is_train else self.test_metrics
    def update_metric(self, batch, out, metric): metric.update(METRICS[metric.key](out, batch))
