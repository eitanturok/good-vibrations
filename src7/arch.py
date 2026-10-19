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
    def __init__(self, d_model, out_h, out_w, rgb, edge=False, edge_width=0, head=True):
        super().__init__()
        self.out_h, self.out_w, self.out_c, self.edge = out_h, out_w, (3 if rgb else 1), edge and not edge_width
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
        self.conv3 = nn.Conv2d(32, self.out_c + self.edge, kernel_size=3, padding=1)  # + 1 outline channel
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

    def forward(self, emb):
        x = feats = self.features(emb)                     # (B,d_model) -> (B,c,out_h,out_w)
        x = F.relu(self.conv1(x))                          # (B,c,out_h,out_w) -> (B,32,out_h,out_w)
        x = F.relu(self.conv2(x))                          # (B,32,out_h,out_w) -> (B,32,out_h,out_w)
        x = self.conv3(x)                                  # (B,32,out_h,out_w) -> (B,out_c+edge,out_h,out_w)
        edge, x = (x[:, -1] if self.edge else None), x[:, :self.out_c]
        if self.edge_head is not None: edge = self.edge_head(feats)[:, 0]
        if self.out_c == 1: return x.squeeze(1), edge      # (B,1,out_h,out_w) -> (B,out_h,out_w)
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

LOSS_FN = {"bce": bce_loss, "mse": mse_loss, "dice": dice_loss, "bce-dice": bce_dice_loss}

def boundary_loss(logits, true): return (signed_dist(true) * logits.sigmoid()).mean()  # Kervadec et al. 2019
def edge_loss(logits, true): return bce_dice_loss(logits, outline(true))

# aux losses
def object_loss(logits, obj, unseen): return F.cross_entropy(logits, obj, ignore_index=unseen)  # objects never seen in train are ignored
def n_objects_loss(logits, n_objects): return F.cross_entropy(logits, n_objects)
def box_loss(logits, box): return F.cross_entropy(logits, box)
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

METRICS = {"iou": iou, "ray_iou": ray_iou, "boundary_iou": boundary_iou, "object_acc": object_acc, "n_objects_err": n_objects_err, "box_acc": box_acc, "com_err": com_error, "area_rel_err": area_rel_err}

#*** 8 model

class Boombox(ComposerModel):
    def __init__(self, d_model: int, data_info: dict, out_h: int, out_w: int, rgb: bool = False,
                 loss_fn: str = "bce", alpha: float = 0.5, loss_weights: dict | None = None, edge_width: int = 0,
                 freq_gain: bool = False, freq_coord: bool = False, freq_pool: str = "mean",
                 head: str = "mask", n_rays: int = 36, ray_shape: str = "free", n_harmonics: int = 4, center_radius: float = 1.5, peak_thresh: float = 0.3,
                 slots_aux: bool = False, encoder: str = "conv", decoder: str = "conv", patch_size: int | None = None,
                 heads: int = 2, freq_layers: int = 2, laser_layers: int = 2, ffn_dim: int | None = None):
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
        assert decoder == "conv" or (head == "mask" and not self.loss_weights.get("edge")), "the polar head and the outline sit on the conv decoder"
        assert (encoder == "transformer") == (patch_size is not None), "the transformer reads patches (--patch-size), the conv encoder the raw spectrum"
        assert head == "mask" or not (rgb or self.loss_weights.get("boundary") or self.loss_weights.get("edge")), f"{head} head: no rgb, boundary or edge"
        assert head == "mask" or not (self.loss_weights.get("com") or self.loss_weights.get("n_objects")), f"{head} head: com and count come from it"
        self.n_rays, self.ray_shape, self.center_radius, self.peak_thresh = n_rays, ray_shape, center_radius, peak_thresh
        freqs = torch.linspace(*data_info.get("freq_range", (100.0, 1000.0)), data_info["n_freqs"])  # (F,) Hz; bins are evenly spaced
        if encoder == "conv":
            self.freq_encoder = FreqEncoder(data_info["n_channels"], freqs, gain=freq_gain, coord=freq_coord, pool=freq_pool)
            self.grid_encoder = LaserGridEncoder(d_model, data_info["n_laser_rows"], data_info["n_laser_cols"])
        else:
            self.transformer = TransformerEncoder(d_model, data_info["n_channels"], patch_size, -(-data_info["n_freqs"] // patch_size),
                                                  data_info["n_laser_rows"], data_info["n_laser_cols"], heads, freq_layers, laser_layers, ffn_dim)
        self.encoder = encoder
        self.phase, self.n_freqs = data_info.get("phase", "none"), data_info["n_freqs"]  # old checkpoints' data_info has no "phase"
        self.out_h, self.out_w = out_h, out_w
        self.decoder = (None if self.slots else MLPDecoder(d_model, out_h, out_w, rgb) if decoder == "mlp" else
                        Decoder(d_model, out_h, out_w, rgb, edge=bool(self.loss_weights.get("edge")), edge_width=edge_width, head=not self.polar))
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
        if self.slots or slots_aux:                  # per slot: exists logit, com (row, col), ray params
            n_params = RAY_PARAMS[ray_shape](n_rays, n_harmonics)
            self.slot_head = mlp(max_objects * (3 + n_params))
            with torch.no_grad():
                nn.init.normal_(self.slot_head[-1].weight, std=0.01)  # outputs start at their biases: off-center 50/50, a 4 px circle at the box center
                b = self.slot_head[-1].bias.view(max_objects, 3 + n_params); b.zero_()
                for k in range(max_objects): init_ray_bias(b[k, 3:], ray_shape, n_params)

        # loss = mask loss + weighted aux losses (a weight of 0 means that head doesn't train)
        self.mask_loss = partial(bce_dice_loss, alpha=alpha) if loss_fn == "bce-dice" else LOSS_FN[loss_fn]

        # metrics, the same ones on train and test: tag each with its key so update_metric knows which one it was handed
        keys = [key for key in METRICS if head != "mask" or slots_aux or key != "ray_iou"]  # ray_iou needs rays
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

    def forward(self, batch):
        x = self.phase_features(batch["fft"])
        if self.encoder == "conv": emb = self.grid_encoder(self.freq_encoder(x))  # (B,L,F,C) -> (B,256,L) -> (B,d_model)
        else: emb = self.transformer(x)                    # (B,L,P,PS,C) -> (B,d_model)
        if self.polar: return self.forward_polar(emb)
        if self.slots: return self.forward_slots(emb)
        logits, edge_logits = self.decoder(emb)            # (B,d_model) -> (B,H,W), or (B,H,W,3) if rgb; (B,H,W) or None
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
