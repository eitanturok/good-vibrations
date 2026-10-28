"""boombox: the model of run s6-gastro-plastic-cardboard-3k (src6's Boombox). Per-laser frequency convs -> laser-grid convs -> one
embedding -> upsampling conv decoder for the mask, + aux heads. Loss: mask bce + .1 object + .1 count + 1 com + .1 area + .1 box.
batch: fft (B,L,F,C), mask_true (B,H,W), object, n_objects, box (B,), coms (B,K,2)"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanMetric

from utils import aux_losses, METRICS

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

#*** 2 encoder

class FreqEncoder(nn.Module):
    """(B,L,F,C) -> (B,256,L): four stride-4 convs over frequency only (lasers don't mix), then a mean over what's left"""
    def __init__(self, n_channels):
        super().__init__()
        self.conv1 = ConvBlock(n_channels, 32, kernel=(1, 7), stride=(1, 4), padding=(0, 3))
        self.conv2 = ConvBlock(32, 64, kernel=(1, 5), stride=(1, 4), padding=(0, 2))
        self.conv3 = ConvBlock(64, 128, kernel=(1, 5), stride=(1, 4), padding=(0, 2))
        self.conv4 = ConvBlock(128, 256, kernel=(1, 5), stride=(1, 4), padding=(0, 2))

    def forward(self, x):
        x = self.conv4(self.conv3(self.conv2(self.conv1(x.permute(0, 3, 1, 2)))))  # (B,C,L,F) -> F 1170 -> 293 -> 74 -> 19 -> 5
        return x.mean(-1)                                                          # (B,256,L)

class LaserGridEncoder(nn.Module):
    """(B,256,L) -> (B,d_model): lasers on their grid, 10x10 -> 5x5 -> 3x3 -> 1x1"""
    def __init__(self, d_model, rows, cols):
        super().__init__()
        self.rows, self.cols = rows, cols
        self.conv1 = ConvBlock(256, 512, kernel=3, stride=2, padding=1)
        self.conv2 = ConvBlock(512, 1024, kernel=3, stride=2, padding=1)
        self.conv3 = ConvBlock(1024, d_model, kernel=(math.ceil(rows / 4), math.ceil(cols / 4)), stride=1, padding=0)

    def forward(self, x): return self.conv3(self.conv2(self.conv1(x.view(len(x), 256, self.rows, self.cols)))).flatten(1)

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

WEIGHTS = dict(object=0.1, n_objects=0.1, com=1.0, area=0.1, box=0.1)  # on top of the mask bce

def total_loss(out, batch, n_known):
    losses = dict(mask=F.binary_cross_entropy_with_logits(out["mask_logits"], batch["mask_true"]), **aux_losses(out, batch, n_known))
    return dict(total=losses["mask"] + sum(w * losses[k] for k, w in WEIGHTS.items()), **losses)

#*** 6 model

class Boombox(ComposerModel):
    def __init__(self, data_info, d_model=1024):
        super().__init__()
        self.freq_encoder = FreqEncoder(data_info["n_channels"])
        self.grid_encoder = LaserGridEncoder(d_model, data_info["n_laser_rows"], data_info["n_laser_cols"])
        self.decoder = Decoder(d_model)
        self.aux = AuxHeads(d_model, data_info)
        self.train_metrics, self.test_metrics = {k: MeanMetric() for k in METRICS}, {k: MeanMetric() for k in METRICS}
        for k, m in [*self.train_metrics.items(), *self.test_metrics.items()]: m.key = k

    def forward(self, batch):
        emb = self.grid_encoder(self.freq_encoder(batch["fft"]))  # (B,d_model)
        logits = self.decoder(emb)[:, 0]                          # (B,64,64)
        return dict(mask_logits=logits, mask_pred=logits.sigmoid(), **self.aux(emb))

    def loss(self, out, batch): return total_loss(out, batch, self.aux.n_known)
    def get_metrics(self, is_train=False): return self.train_metrics if is_train else self.test_metrics
    def update_metric(self, batch, out, metric): metric.update(METRICS[metric.key](out, batch))
