"""Boombox (arXiv 2105.08052) adapted to the gastronorm capture.

The paper: 4 contact mics -> 128x128x4 mel-spectrogram -> conv encoder -> 1x1
embedding -> two-branch transposed-conv decoder -> RGB + depth.

What changes here:
* The 100 lasers are the mics, not the 8 speakers. Lasers share one camera
  capture so they're already synchronized; speakers are separate recordings, so
  fusing them uses a permutation-invariant pool.
* No spectrogram: Y(f)=H(f)X(f) holds however the chirp distributes energy in
  time, so we feed the global rFFT (docs/physics_and_architecture.md sec 4).
* No depth head: this dataset has no depth target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel

from model.arch import LOSSES, count_loss, create_metrics, CountAccuracy, N_COUNT_CLASSES

def _drop(x, p, dim, training):
    """Zero whole slices along `dim` of (B,C,L,F), rescaling survivors so the mean is
    preserved. Structured rather than per-element: dropping a scattered 30% of bins is
    nearly a no-op for a conv that pools over neighbours, while dropping a whole laser
    removes a real measurement."""
    if not training or p == 0.0: return x
    shape = [x.shape[0] if d == 0 else (x.shape[d] if d == dim else 1) for d in range(x.ndim)]
    keep = (torch.rand(shape, device=x.device) >= p).to(x.dtype)
    return x * keep / (1 - p)

def conv_block(c_in, c_out, kernel, stride, padding=0):
    return nn.Sequential(nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=padding),
                         nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, inplace=True))

class Encoder(nn.Module):
    """(B,C,L,F) -> (B,D). Convolves frequency only, then mixes the laser grid.

    Kernel height is 1 throughout the frequency stack, so lasers never mix early:
    the 10x10 grid isn't translation-covariant (adjacent lasers are ~100px apart
    and sample global standing waves, not local texture). Frequency is the axis
    with real locality, so it's the one we convolve.

    The two dropouts are structured, not per-element: whole frequency bands and whole
    lasers are zeroed, matching the transformer's freq/laser dropout. Only ~230 distinct
    scenes exist (the 8 speakers at a position re-measure one scene), so an unregularized
    model memorizes them -- dropping whole lasers forces it to use the grid rather than
    latch onto a few.
    """
    def __init__(self, n_channels, d_model, n_laser_rows, n_laser_cols,
                 freq_dropout=0.0, laser_dropout=0.0):
        super().__init__()
        self.grid_shape = (n_laser_rows, n_laser_cols)
        self.freq_dropout, self.laser_dropout = freq_dropout, laser_dropout
        self.freq = nn.Sequential(
            conv_block(n_channels, 32, (1, 7), (1, 4), (0, 3)),
            conv_block(32, 64, (1, 5), (1, 4), (0, 2)),
            conv_block(64, 128, (1, 5), (1, 4), (0, 2)),
            conv_block(128, 256, (1, 5), (1, 4), (0, 2)),
            nn.AdaptiveAvgPool2d((None, 1)),  # collapse whatever width survives
            )
        self.grid = nn.Sequential(
            conv_block(256, 512, 3, 2, 1),       # 10x10 -> 5x5
            conv_block(512, 1024, 3, 2, 1),      # 5x5 -> 3x3
            conv_block(1024, d_model, 3, 1, 0),  # 3x3 -> 1x1
            )

    def forward(self, x):
        x = _drop(x, self.laser_dropout, dim=2, training=self.training)  # whole lasers
        x = _drop(x, self.freq_dropout, dim=3, training=self.training)   # whole freq columns
        x = self.freq(x).reshape(x.shape[0], -1, *self.grid_shape)
        return self.grid(x).flatten(1)

class TwoStreamEncoder(nn.Module):
    """(B,C,L,F) -> (B,D), with magnitude and phase convolved by SEPARATE weights and fused
    only after the frequency stack has run.

    The single-stream Encoder above sums magnitude and phase inside the FIRST conv_block's
    weights, before any nonlinearity, so the model never gets to process them apart. That is
    what the v2 ladder measured (scripts/phase_ablation.sh, 8 arms at 1000ep): every phase arm
    landed below no-phase (best 0.2636 vs 0.2823), and the damage scaled with channel count --
    2-channel arms beat 4-channel arms consistently. That is the signature of interference at
    fusion, not of phase being uninformative. B2 (ungauged) being the worst arm while every
    gauged arm beat it says the phase content itself is real.

    Each stream keeps the FULL 32->256 width of the original, so phase gets its own
    full-capacity frequency stack rather than a narrowed one. Fusion yields 512 channels, so
    the grid stack widens to match: ~21.3M params vs 19.87M (+7%). The cost is small because
    the parameters live in the grid stack and decoder, not the frequency convs.

    NOTE the capacity confound: a win here could be late fusion OR the extra 1.4M params. The
    control is a single-stream Encoder widened to the same grid input; run it before claiming
    fusion depth is what mattered.

    Magnitude is always channels [:2] and phase [2:], because process_vibration concatenates
    the phase block last (dataset.py:357). n_channels is 2 (no phase), 6 (one gauge -- 2C=4
    from _phasor_cos_sin) or 10 (both gauges).
    """
    def __init__(self, n_channels, d_model, n_laser_rows, n_laser_cols,
                 freq_dropout=0.0, laser_dropout=0.0, fuse='concat'):
        super().__init__()
        self.grid_shape = (n_laser_rows, n_laser_cols)
        self.freq_dropout, self.laser_dropout = freq_dropout, laser_dropout
        self.n_mag, self.n_phase = 2, n_channels - 2
        self.fuse = fuse

        def stream(c_in):  # same (1,k) kernels and stride-4 schedule as Encoder.freq
            return nn.Sequential(
                conv_block(c_in, 32, (1, 7), (1, 4), (0, 3)),
                conv_block(32,   64, (1, 5), (1, 4), (0, 2)),
                conv_block(64,  128, (1, 5), (1, 4), (0, 2)),
                conv_block(128, 256, (1, 5), (1, 4), (0, 2)),
                nn.AdaptiveAvgPool2d((None, 1)))

        self.mag = stream(self.n_mag)
        self.phase = stream(self.n_phase) if self.n_phase > 0 else None
        if self.n_phase > 0 and fuse == 'gate':
            # Phase admitted per-channel through a sigmoid gate driven by the MAGNITUDE
            # stream. Bias init -2 (sigmoid ~ 0.12) so training starts near the no-phase
            # model at 0.2823 and has to earn its way up, rather than starting at the fused
            # optimum v2 showed is worse. The gate can always close, so this arm's floor is
            # roughly B1 instead of P2.
            self.gate = nn.Conv2d(256, 256, 1)
            nn.init.constant_(self.gate.bias, -2.0)

        # With no phase channels this degenerates to exactly the single-stream widths.
        c_fused = 256 * (2 if self.n_phase > 0 else 1)
        self.grid = nn.Sequential(
            conv_block(c_fused, 512, 3, 2, 1),   # 10x10 -> 5x5
            conv_block(512, 1024, 3, 2, 1),      # 5x5 -> 3x3
            conv_block(1024, d_model, 3, 1, 0),  # 3x3 -> 1x1
            )

    def forward(self, x):
        x = _drop(x, self.laser_dropout, dim=2, training=self.training)  # whole lasers
        x = _drop(x, self.freq_dropout, dim=3, training=self.training)   # whole freq columns
        m = self.mag(x[:, :self.n_mag])
        if self.phase is not None:
            p = self.phase(x[:, self.n_mag:])
            m = torch.cat([m, p * self.gate(m).sigmoid()], dim=1) if self.fuse == 'gate' \
                else torch.cat([m, p], dim=1)
        return self.grid(m.reshape(x.shape[0], -1, *self.grid_shape)).flatten(1)

class TwoBranchUp(nn.Module):
    """The paper's decoder layer: a transposed conv, and a conv then a transposed
    conv, concatenated. Branch (b)'s extra conv is what stops the two branches
    collapsing to the same function."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.a = nn.ConvTranspose2d(c_in, c_out // 2, 4, stride=2, padding=1)
        self.b = nn.Sequential(nn.Conv2d(c_in, c_in, 3, padding=1), nn.ReLU(inplace=True),
                               nn.ConvTranspose2d(c_in, c_out // 2, 4, stride=2, padding=1))
        self.out = nn.Sequential(nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x): return self.out(torch.cat([self.a(x), self.b(x)], dim=1))

class Decoder(nn.Module):
    """(B,D) -> (B,out_h,out_w), or (B,out_h,out_w,out_c) when out_c > 1. Seeds a 3x4
    grid, doubles it to 24x32, resizes. 21x30 isn't a power of two, so the resize is
    explicit rather than an asymmetric crop that would bias one edge."""
    SEED = (3, 4)

    def __init__(self, d_model, out_h, out_w, out_c=1):
        super().__init__()
        self.out_hw, self.out_c = (out_h, out_w), out_c
        self.project = nn.Linear(d_model, 512 * self.SEED[0] * self.SEED[1])
        self.up = nn.Sequential(TwoBranchUp(512, 256), TwoBranchUp(256, 128), TwoBranchUp(128, 64))
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, out_c, 3, padding=1))

    def forward(self, emb):
        x = self.up(self.project(emb).view(-1, 512, *self.SEED))
        x = F.interpolate(x, size=self.out_hw, mode='bilinear', align_corners=False)
        x = self.head(x)                                     # (B,out_c,H,W)
        # .contiguous(): permute leaves a non-contiguous view, and torchmetrics' MSE does
        # preds.view(-1), which requires contiguity. Only bites on the rgb path -- out_c == 1
        # takes squeeze(1), which stays contiguous.
        return x.squeeze(1) if self.out_c == 1 else x.permute(0, 2, 3, 1).contiguous()

class BoomboxModel(ComposerModel):
    """Same ComposerModel contract as VibrationTransformer, so callbacks, metrics
    and run.py work on it unchanged."""
    def __init__(self, d_model=1024, data_info=None, fuse_speakers=False,
                 loss_fn='mse', loss_alpha=0.5, count_loss_weight=0.0,
                 freq_dropout=0.0, laser_dropout=0.0, encoder='single', fuse='concat'):
        super().__init__()
        # 'single' is the v2 encoder and stays the default, so every existing command is
        # unchanged. 'two-stream' gives magnitude and phase separate frequency stacks.
        enc = TwoStreamEncoder if encoder == 'two-stream' else Encoder
        kw = dict(fuse=fuse) if encoder == 'two-stream' else {}
        self.encoder = enc(data_info.get('n_channels', 2), d_model, data_info['n_laser_rows'],
                           data_info['n_laser_cols'], freq_dropout, laser_dropout, **kw)
        self.decoder = Decoder(d_model, data_info['out_h'], data_info['out_w'], data_info.get('out_c', 1))
        self.fuse_speakers = fuse_speakers
        self.empty_head = nn.Linear(d_model, 1)
        self.count_head = nn.Linear(d_model, N_COUNT_CLASSES)
        self.count_loss_weight = count_loss_weight
        self.loss_fn = LOSSES[loss_fn]
        self.is_spatial_loss = loss_fn.startswith('ce-spatial')
        self.is_asym_loss = loss_fn.endswith('-asym')
        self.loss_alpha = loss_alpha
        self.train_metrics, self.val_metrics = create_metrics(data_info), create_metrics(data_info)

    def forward(self, batch):
        # dataset gives patched tokens (B,L,P,PS,C); a conv wants the frequency axis
        # back, so un-patch here instead of adding a path to dataset.py
        x = batch['fft']
        if self.fuse_speakers:  # (B,K,L,P,PS,C) -> shared encoder -> mean over K
            B, K = x.shape[:2]
            emb = self.encoder(self._to_conv(x.flatten(0, 1))).reshape(B, K, -1).mean(1)
        else:
            emb = self.encoder(self._to_conv(x))
        mask_logits = self.decoder(emb)
        return dict(mask_pred=mask_logits.sigmoid(), mask_logits=mask_logits,
                    empty_logit=self.empty_head(emb), count_logits=self.count_head(emb))

    @staticmethod
    def _to_conv(x):
        """(B,L,P,PS,C) -> (B,C,L,P*PS)"""
        assert x.ndim == 5, f"expected (B,L,P,PS,C), got {tuple(x.shape)}"
        return x.flatten(2, 3).permute(0, 3, 1, 2)

    def loss(self, outputs, batch):
        kw = dict(empty_logit=outputs['empty_logit']) if self.is_spatial_loss else {}
        if self.is_asym_loss: kw = dict(alpha=self.loss_alpha)
        total = self.loss_fn(outputs['mask_logits'], outputs['mask_pred'], batch['mask_true'], **kw)
        if self.count_loss_weight:
            total = total + self.count_loss_weight * count_loss(outputs['count_logits'], batch['info']['n_objects'])
        return total

    def get_metrics(self, is_train=False): return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, CountAccuracy): metric.update(outputs['count_logits'], batch['info']['n_objects'])
        else: metric.update(outputs['mask_pred'], batch['mask_true'])

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)
