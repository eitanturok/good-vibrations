"""transformer_calib: transformer.py with every box dependency coming from the box's empty-box scans (a calibration set: scan a new box empty, then use it).
empty-box standardization ((x - per-laser empty mean) / empty std pooled over lasers) -> each laser's spectrum cut into 19 freq bands of 64 bins (~49 Hz),
a token each, + freq band / laser (learned + 2D wall position) / box embeddings (box = MLP of its laser-averaged empty spectrum) -> 2 freq blocks over each
laser's freq bands -> gated bands, concatenated -> one token per laser -> 2 laser blocks -> flatten -> one embedding -> conv decoder (mask only), aux +
orientation heads. Train-time augs in the model: speaker gain, laser jitter.
loss: mask bce + mask dice + orientation + aux heads (box head kept: meaningless on a new box).
batch: fft (B,L,n_bins,C), box (B,), mask_true (B,H,W), object, n_objects (B,), coms (B,K,2)"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanMetric

from utils import speaker_gain, laser_jitter, grid_coords, aux_losses, orient_loss, METRICS

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

class FreqEncoder(nn.Module):
    """normalized (B,L,n_bins,C) spectrum + (B,d) box + (n_bands,) freq bands + (L,d) laser -> (B,L,d) per-laser freq-band tokens"""
    def __init__(self, n_channels, n_bands, bins_per_band, d, heads, layers):
        super().__init__()
        self.bins_per_band, self.heads = bins_per_band, heads
        self.embed = nn.Linear(bins_per_band * n_channels, d)                                 # raw bins in: keeps exact peak positions
        self.freq_band_emb = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.freqs = nn.ModuleList([Block(d, heads, mlp=4) for _ in range(layers)])
        self.pool_q = nn.Parameter(nn.init.trunc_normal_(torch.zeros(heads, d // heads), std=0.02))
        self.pool_norm, self.pool_k, self.pool_o = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(n_bands * d, d)

    def forward(self, x, box, freq_band, laser):                                      # (B,L,n_bins,C) + (B,d) + (n_bands,) + (L,d)
        B, L = x.shape[:2]
        n_bands = len(freq_band)
        x = F.pad(x, (0, 0, 0, n_bands * self.bins_per_band - x.shape[2])).view(B, L, n_bands, -1)  # (B,L,n_bins,C) -> (B,L,n_bands,bins_per_band*C)

        # learn meaningful representation for each freq band by replacing channel dim with learned 128 dim embedding for every band across all lasers, freqs, samples in batch
        x = self.embed(x) + self.freq_band_emb(freq_band[:, None])[None, None] + laser[None, :, None] + box[:, None, None]  # (B,L,n_bands,bins_per_band*C) -> (B,L,n_bands,d)
        x = x.flatten(0, 1)                                                           # (B,L,n_bands,d) -> (B*L,n_bands,d)

        # learn how to combine the freq bands of each laser
        for block in self.freqs: x = block(x)                                         # (B*L,n_bands,d) -> (B*L,n_bands,d), over each laser's bands
        x = x.view(B, L, n_bands, -1)

        # now we average the freq bands to get one token per laser of d dimensions
        # to do this smartly, we use the attenion mechanisim
        # pool_q asks "which bands should I read for each laser?"
        # pool_k is the representation of each band
        # pool_o is the final linear projection to get back to d dimensions
        # a.mean(2) is the average over the bands, 
        # we add the laser again as a residual at the end
        k = self.pool_k(self.pool_norm(x)).view(B, L, n_bands, self.heads, -1)       # (B,L,n_bands,d) -> (B,L,n_bands,H,d/H)
        a = (torch.einsum("blphe,he->blhp", k, self.pool_q) / math.sqrt(k.shape[-1])).softmax(-1)  # (B,L,n_bands,H,d/H)+(H,d/H) -> (B,L,H,n_bands) which bands each laser reads
        return self.pool_o((x * a.mean(2)[..., None] * n_bands).reshape(B, L, -1)) + laser  # (B,L,n_bands,d) -> (B,L,n_bands*d) -> (B,L,d), + laser resize

class LaserEncoder(nn.Module):
    """(B,L,d) tokens + (L,d) laser -> (B,d_model)"""
    def __init__(self, L, d, d_model, layers):
        super().__init__()
        self.lasers = nn.ModuleList([Block(d, heads=4, mlp=4) for _ in range(layers)])
        self.to_emb = nn.Sequential(nn.LayerNorm(d), nn.Flatten(1), nn.Linear(L * d, d_model))

    def forward(self, x, laser):
        for block in self.lasers: x = block(x)                                        # (B,L,d) -> (B,L,d), over the lasers
        return self.to_emb(x)                                                          # (B,L,d) -> (B,L*d) -> (B,d_model)

class Encoder(nn.Module):
    """(B,L,n_bins,C) spectrum + (B,) box -> (B,d_model)"""
    def __init__(self, n_channels, freqs, rows, cols, d_model, empty_mean, empty_std, bins_per_band=64, d=128, heads=4, freq_layers=2, laser_layers=2):
        super().__init__()
        L, n_bins = rows * cols, len(freqs)
        n_bands = math.ceil(n_bins / bins_per_band)                                   # 1170 bins -> 19 freq bands
        self.register_buffer("empty_mean", empty_mean)                                # (n_boxes,L,n_bins,C) per-laser mean of the box's empty scans
        self.register_buffer("empty_std", empty_std)                                  # (n_boxes,n_bins,C) their spread, pooled over lasers
        self.register_buffer("box_ref", empty_mean.mean(1).flatten(1))                # (n_boxes,n_bins*C) laser-averaged empty spectrum
        logf = torch.log10(torch.as_tensor(freqs, dtype=torch.float32)) - 2.5         # 100-1000 Hz -> -.5...5
        self.register_buffer("freq_band_logf", logf[(torch.arange(n_bands) * bins_per_band + bins_per_band // 2).clamp(max=n_bins - 1)], persistent=False)  # (n_bands,) freq band centres
        a = grid_coords(rows, cols)[..., None] * (2 ** torch.arange(8) * math.pi)    # laser (row,col) -> sin/cos at 2^i pi
        self.register_buffer("laser_fourier", torch.cat([a.sin(), a.cos()], -1).flatten(-2), persistent=False)  # (L,32)
        self.laser_emb = nn.Parameter(nn.init.trunc_normal_(torch.zeros(L, d), std=0.02))
        self.laser_pos = nn.Linear(4 * 8, d)
        self.box_emb = nn.Sequential(nn.LayerNorm(n_bins * n_channels), nn.Linear(n_bins * n_channels, d), nn.GELU(), nn.Linear(d, d))
        self.freq_enc = FreqEncoder(n_channels, n_bands, bins_per_band, d, heads, freq_layers)
        self.laser_enc = LaserEncoder(L, d, d_model, laser_layers)

    def normalize(self, x, box):
        """empty-box standardization: subtract each laser's empty mean, divide by the empty spread (same at every laser)"""
        return ((x - self.empty_mean[box]) / self.empty_std[box][:, None]).to(x.dtype)

    def forward(self, x, box):
        x = self.normalize(x, box)                                                    # (B,L,n_bins,C) -> (B,L,n_bins,C) empty-box standardized
        box = self.box_emb(self.box_ref[box])                                         # (B,n_bins*C) -> (B,d)
        laser = self.laser_emb + self.laser_pos(self.laser_fourier)                   # (L,32) -> (L,d) identity + 2D position
        return self.laser_enc(self.freq_enc(x, box, self.freq_band_logf, laser), laser)  # -> (B,L,n_bands,d) -> (B,L,d) -> (B,d_model)

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

WEIGHTS = dict(object=0.1, n_objects=0.1, com=1.0, area=0.1, box=0.1, orient=1.0)  # on top of the mask bce + dice

def dice_loss(logits, true):
    p = logits.sigmoid()
    return 1 - ((2 * (p * true).flatten(1).sum(1) + 1e-6) / ((p + true).flatten(1).sum(1) + 1e-6)).mean()

def total_loss(out, batch, n_known):
    mask = batch["mask_true"]
    losses = dict(mask=F.binary_cross_entropy_with_logits(out["mask_logits"], mask),
                  dice=dice_loss(out["mask_logits"], mask),
                  **aux_losses(out, batch, n_known),
                  orient=orient_loss(out["orient"], mask, batch["n_objects"]))
    return dict(total=losses["mask"] + losses["dice"] + sum(w * losses[k] for k, w in WEIGHTS.items()), **losses)

#*** 6 model

class TransformerCalib(ComposerModel):
    def __init__(self, data_info, d_model=1024, speaker_gain=0.35, speaker_gain_hz=20.0, laser_jitter=0.5):
        super().__init__()
        self.freq_hz, self.grid_rc = data_info["freq_range"], (data_info["n_laser_rows"], data_info["n_laser_cols"])
        freqs = torch.linspace(*self.freq_hz, data_info["n_freqs"])                    # (F,) Hz, evenly spaced bins
        self.encoder = Encoder(data_info["n_channels"], freqs, *self.grid_rc, d_model, data_info["empty_mean"], data_info["empty_std"])
        self.decoder = Decoder(d_model)                                                # mask only
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
        y = self.decoder(emb)[:, 0]                                                    # (B,H,W)
        return dict(mask_logits=y, mask_pred=y.sigmoid(), orient=self.orient_head(emb).float(),
                    object_logits=self.object_head(emb), n_objects_logits=self.n_objects_head(emb), com=self.com_head(emb).sigmoid().view(len(emb), -1, 2),
                    log_area=self.area_head(emb).squeeze(-1), box_logits=self.box_head(emb))

    def loss(self, out, batch): return total_loss(out, batch, self.n_known)
    def get_metrics(self, is_train=False): return self.train_metrics if is_train else self.test_metrics
    def update_metric(self, batch, out, metric): metric.update(METRICS[metric.key](out, batch))
