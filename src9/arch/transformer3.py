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
gan_w (0 = off): a PatchGAN disc on masks (real = true, fake = sigmoid), hinge loss, stepped by its own Adam inside loss() on every microbatch;
+ gan_w x -D(fake), off until epoch 50, ramped 0 -> gan_w over epochs 50-100.
train only: speaker gain, laser jitter (each on a random fraction speaker_gain_p / laser_jitter_p of the samples), token dropout (each laser / band dropped iid with prob laser_drop / band_drop, >= 1 of each kept)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanMetric

from maskae import load as maskae_load
from utils import speaker_gain, laser_jitter, bce_dice, lovasz_hinge, METRICS

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
    """(B,L,n_bands,d) band tokens -> (B,L,d): blocks over each laser's bands, then f = sum_b m_b W_b LN(h_b) + b (per) or LN(h_cls) (cls)"""
    def __init__(self, n_bands, d, heads, layers, readout="per"):
        super().__init__()
        self.freqs = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.read_norm, pos = nn.LayerNorm(d), torch.arange(n_bands)[:, None]
        if readout == "per": self.cls, (self.read_w, self.read_b) = None, per_token(n_bands, d, d)  # (n_bands,d,d), (d,)
        else: self.cls, pos = nn.Parameter(torch.randn(d) * 0.02), torch.cat([torch.zeros(1, 1), pos])  # [cls] at position 0
        self.register_buffer("rope", rope(pos, d // heads), persistent=False)        # 1D: band index

    def forward(self, x, m_b):                                                       # (B,L,n_bands,d), (B,n_bands)
        B, L, n_bands, _ = x.shape
        x, keep = x.flatten(0, 1), m_b.repeat_interleave(L, 0)                        # (B*L,n_bands,d), (B*L,n_bands)
        if self.cls is not None: x, keep = torch.cat([self.cls.to(x.dtype).expand(len(x), 1, -1), x], 1), F.pad(keep, (1, 0), value=1)
        for blk in self.freqs: x = blk(x, self.rope, keep)
        if self.cls is not None: return self.read_norm(x[:, 0]).view(B, L, -1)        # (B,L,d)
        x = self.read_norm(x.view(B, L, n_bands, -1)) * m_b[:, None, :, None].to(x.dtype)
        return torch.einsum("blpd,pde->ble", x, self.read_w) + self.read_b             # (B,L,d)

class LaserEncoder(nn.Module):
    """(B,L,d) laser tokens -> u (B,latent_dim): blocks over the lasers, then u = LN(sum_l m_l W_l LN(t_l) + b (per) or W LN(t_cls) + b (cls)) + W_mask [m_l, m_b]"""
    def __init__(self, rows, cols, n_bands, d, heads, layers, latent_dim, readout="per"):
        super().__init__()
        L = rows * cols
        self.lasers = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        rc = torch.stack(torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij"), -1).view(L, 2)
        if readout == "per": self.cls, (self.read_w, self.read_b) = None, per_token(L, d, latent_dim)  # (L,d,latent_dim), (latent_dim,)
        else: self.cls, self.read, rc = nn.Parameter(torch.randn(d) * 0.02), nn.Linear(d, latent_dim), torch.cat([torch.zeros(1, 2, dtype=rc.dtype), rc])  # [cls] at (0, 0) rotation = none
        self.read_norm, self.u_norm = nn.LayerNorm(d), nn.LayerNorm(latent_dim)
        self.w_mask = nn.Linear(L + n_bands, latent_dim)
        self.register_buffer("rope", rope(rc, d // heads), persistent=False)         # 2D: laser (row, col), row-major

    def forward(self, x, m_l, m_b):                                                  # (B,L,d), (B,L), (B,n_bands)
        if self.cls is not None:
            x = torch.cat([self.cls.to(x.dtype).expand(len(x), 1, -1), x], 1)
            for blk in self.lasers: x = blk(x, self.rope, F.pad(m_l, (1, 0), value=1))
            u = self.read(self.read_norm(x[:, 0]))
        else:
            for blk in self.lasers: x = blk(x, self.rope, m_l)
            u = torch.einsum("bld,lde->be", self.read_norm(x) * m_l[..., None].to(x.dtype), self.read_w) + self.read_b
        return self.u_norm(u) + self.w_mask(torch.cat([m_l, m_b], -1).to(u.dtype))   # (B,latent_dim) the bottleneck

class Encoder(nn.Module):
    """(B,L,n_bins,C) |X| -> u (B,latent_dim): per-sample z -> cut into bands -> shared Linear to d -> freq enc -> laser enc"""
    def __init__(self, n_channels, n_bins, rows, cols, bins_per_band, d, heads, freq_layers, laser_layers, latent_dim, readout="per"):
        super().__init__()
        assert n_bins % bins_per_band == 0, (n_bins, bins_per_band)
        self.L, self.n_bands, self.bins_per_band = rows * cols, n_bins // bins_per_band, bins_per_band  # 1170 bins -> 18 bands
        self.embed = nn.Linear(bins_per_band * n_channels, d)                         # shared by every band
        self.freq_enc = FreqEncoder(self.n_bands, d, heads, freq_layers, readout)
        self.laser_enc = LaserEncoder(rows, cols, self.n_bands, d, heads, laser_layers, latent_dim, readout)

    def forward(self, x, m_l, m_b):
        x = (x - x.mean((1, 2, 3), keepdim=True)) / (x.std((1, 2, 3), keepdim=True) + 1e-6)  # z-score each sample
        x = x.reshape(*x.shape[:2], self.n_bands, self.bins_per_band, -1).flatten(-2)  # tokenize (B,L,n_bands,bins*C)
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
                 mask_loss="bce_dice", maskae="", readout="per", heatmap_w=0.0, laser_jitter_p=1.0, speaker_gain_p=1.0, gan_w=0.0):
        super().__init__()
        self.freq_hz, self.grid_rc = data_info["freq_range"], (data_info["n_laser_rows"], data_info["n_laser_cols"])
        self.encoder = Encoder(data_info["n_channels"], data_info["n_freqs"], *self.grid_rc, bins_per_band, d, heads, freq_layers, laser_layers, latent_dim, readout)
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
        u = self.encoder(x, m_l, m_b)                                                  # (B,latent_dim)
        out = dict(object_logits=self.object_head(u), box_logits=self.box_head(u), latent=u)
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
