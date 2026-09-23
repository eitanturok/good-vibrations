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

from model.arch import LOSSES, count_loss, create_metrics, CHEAP_SEG_KEYS, N_COUNT_CLASSES

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

def grid_stack(c_in, d_model, grid_shape):
    """The paper's laser-grid head: two stride-2 convs, then one valid conv down to 1x1.

    The final kernel is sized from the grid rather than fixed at 3, because a laser subset makes
    the grid smaller than 10x10 -- an 8x5 selection reaches 2x2 here, where a 3x3 kernel does not
    fit. At 10x10 this resolves to 3, i.e. exactly the original stack.
    """
    h, w = (-(-d // 4) for d in grid_shape)  # two stride-2, padding-1 convs: ceil(ceil(d/2)/2)
    if min(h, w) < 1: raise ValueError(f"laser grid {grid_shape} is too small for the boombox grid stack")
    return nn.Sequential(
        conv_block(c_in, 512, 3, 2, 1),           # 10x10 -> 5x5
        conv_block(512, 1024, 3, 2, 1),           # 5x5 -> 3x3
        conv_block(1024, d_model, (h, w), 1, 0),  # 3x3 -> 1x1
        )

def freq_collapse(c, width, learned):
    """Collapse the surviving frequency width to 1. `learned` swaps the uniform mean for a
    per-channel weighted sum: AdaptiveAvgPool throws away WHERE in the spectrum a filter fired,
    but band identity is exactly what carries the signal here (133-215Hz and 380-463Hz dominate
    the ablations), and it is also the only way the trailing zero-pad can be downweighted rather
    than averaged in at full strength."""
    if not learned: return nn.AdaptiveAvgPool2d((None, 1))
    return nn.Conv2d(c, c, (1, width), groups=1)  # valid-mode: (.,.,L,width) -> (.,.,L,1)

# (kernel, padding) per stride-4 stage. One list so the stack, the width arithmetic and both
# encoders can never drift apart. Wider first kernel: at full resolution you want a bigger
# receptive field on raw bins before throwing resolution away; after that each stride-4 already
# quadruples the effective field.
FREQ_STAGES = ((7, 3), (5, 2), (5, 2), (5, 2))

def freq_stack(c_in, freq_width, mult=1, depth=1):
    """The (1,k) frequency stack: 32->64->128->256 (times `mult`), stride 4 each stage.

    `mult` scales every width, so the grid stack's input scales with it too -- widening here
    drags params into the grid, which is why it is not free. `depth` inserts stride-1 (1,3)
    blocks after each downsample, adding nonlinearity at each scale WITHOUT touching the
    output width, so the grid stack is unaffected. The frequency stack is only ~216K params of
    a ~28M model at mult=1, so this is the cheapest place in the arch to buy capacity.
    """
    layers, c = [], c_in
    for i, (k, p) in enumerate(FREQ_STAGES):
        c_out = int(32 * 2 ** i * mult)
        layers.append(conv_block(c, c_out, (1, k), (1, 4), (0, p)))
        for _ in range(depth - 1):  # stride 1, 'same' padding: refines at this scale, no resize
            layers.append(conv_block(c_out, c_out, (1, 3), (1, 1), (0, 1)))
        c = c_out
    layers.append(freq_collapse(c, freq_width, freq_width is not None))
    return nn.Sequential(*layers), c

def _surviving_width(n_freqs):
    """Frequency width left after the four stride-4 stages, which is what a learned collapse
    must be sized to. Mirrors conv arithmetic rather than hardcoding: the width depends on
    patch_size and on whether the pad was trimmed (1235 -> 5, 2816 -> 11, 3072 -> 12). The
    stride-1 `depth` blocks use 'same' padding, so they do not change it."""
    w = n_freqs
    for k, p in FREQ_STAGES: w = (w + 2 * p - k) // 4 + 1
    return w

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
                 freq_dropout=0.0, laser_dropout=0.0, freq_width=None,
                 freq_mult=1, freq_depth=1):
        super().__init__()
        self.grid_shape = (n_laser_rows, n_laser_cols)
        self.freq_dropout, self.laser_dropout = freq_dropout, laser_dropout
        self.freq, c = freq_stack(n_channels, freq_width, freq_mult, freq_depth)
        self.grid = grid_stack(c, d_model, self.grid_shape)

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
                 freq_dropout=0.0, laser_dropout=0.0, fuse='concat', freq_width=None,
                 freq_mult=1, freq_depth=1):
        super().__init__()
        self.grid_shape = (n_laser_rows, n_laser_cols)
        self.freq_dropout, self.laser_dropout = freq_dropout, laser_dropout
        self.n_mag, self.n_phase = 2, n_channels - 2
        self.fuse = fuse

        def stream(c_in):  # identical schedule to Encoder.freq, separate weights
            return freq_stack(c_in, freq_width, freq_mult, freq_depth)

        self.mag, c_stream = stream(self.n_mag)
        self.phase, _ = stream(self.n_phase) if self.n_phase > 0 else (None, None)
        if self.n_phase > 0 and fuse == 'gate':
            # Phase admitted per-channel through a sigmoid gate driven by the MAGNITUDE
            # stream. Bias init -2 (sigmoid ~ 0.12) so training starts near the no-phase
            # model at 0.2823 and has to earn its way up, rather than starting at the fused
            # optimum v2 showed is worse. The gate can always close, so this arm's floor is
            # roughly B1 instead of P2.
            self.gate = nn.Conv2d(c_stream, c_stream, 1)
            nn.init.constant_(self.gate.bias, -2.0)

        # With no phase channels this degenerates to exactly the single-stream widths.
        c_fused = c_stream * (2 if self.n_phase > 0 else 1)
        self.grid = grid_stack(c_fused, d_model, self.grid_shape)

    def forward(self, x):
        x = _drop(x, self.laser_dropout, dim=2, training=self.training)  # whole lasers
        x = _drop(x, self.freq_dropout, dim=3, training=self.training)   # whole freq columns
        m = self.mag(x[:, :self.n_mag])
        if self.phase is not None:
            p = self.phase(x[:, self.n_mag:])
            m = torch.cat([m, p * self.gate(m).sigmoid()], dim=1) if self.fuse == 'gate' \
                else torch.cat([m, p], dim=1)
        return self.grid(m.reshape(x.shape[0], -1, *self.grid_shape)).flatten(1)

class CoordConv2d(nn.Module):
    """Liu et al., "An Intriguing Failing of Convolutional Neural Networks and the
    CoordConv Solution" (NeurIPS 2018, https://arxiv.org/abs/1807.03247). A plain conv
    is translation-equivariant: the same kernel produces the same output no matter
    WHERE in the feature map it fires, so the network has no direct way to represent
    "this activation belongs at row 3, column 9" -- it has to infer position
    indirectly from where a signal happens to land, which the paper shows can fail
    outright even on trivial coordinate-regression tasks. CoordConv concatenates two
    extra channels -- normalized (x,y) in [-1,1] -- onto the input before every conv,
    making position an explicit, always-available input instead of something the
    receptive field has to reconstruct. Used in the decoder (see --coordconv), which
    is exactly where this model has to turn a bottleneck embedding into precise pixel
    positions."""
    def __init__(self, c_in, c_out, kernel, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(c_in + 2, c_out, kernel, stride=stride, padding=padding)

    def forward(self, x):
        B, _, H, W = x.shape
        yy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        return self.conv(torch.cat([x, xx, yy], dim=1))

def _conv2d(c_in, c_out, kernel, stride=1, padding=0, coordconv=False):
    cls = CoordConv2d if coordconv else nn.Conv2d
    return cls(c_in, c_out, kernel, stride=stride, padding=padding)

class PixelShuffleUp(nn.Module):
    """Shi et al., "Real-Time Single Image and Video Super-Resolution Using an
    Efficient Sub-Pixel Convolutional Neural Network" (CVPR 2016,
    https://arxiv.org/abs/1609.05158). A stride-2 ConvTranspose2d (this model's
    default upsampler, in TwoBranchUp) computes each output pixel from overlapping,
    unevenly-spaced input windows, which is the textbook cause of the checkerboard
    grid artifact documented in Odena et al., "Deconvolution and Checkerboard
    Artifacts" (Distill 2016, https://distill.pub/2016/deconv-checkerboard/) --
    exactly the kind of periodic bias that corrupts precise pixel localization.
    PixelShuffle instead computes all r^2 sub-pixel positions with one ordinary
    (evenly-strided) conv, then rearranges channels into space, so there is no
    overlap pattern to alias into a grid."""
    def __init__(self, c_in, c_out, coordconv=False):
        super().__init__()
        self.conv = _conv2d(c_in, c_out * 4, 3, padding=1, coordconv=coordconv)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x): return self.shuffle(self.conv(x))

class TwoBranchUp(nn.Module):
    """The paper's decoder layer: a transposed conv, and a conv then a transposed
    conv, concatenated. Branch (b)'s extra conv is what stops the two branches
    collapsing to the same function.

    `upsample='pixelshuffle'` swaps both branches' ConvTranspose2d for PixelShuffleUp
    (see its docstring); `coordconv=True` makes every conv in this block a CoordConv2d
    (see its docstring)."""
    def __init__(self, c_in, c_out, upsample:str='transposed', coordconv:bool=False):
        super().__init__()
        if upsample == 'transposed':
            self.a = nn.ConvTranspose2d(c_in, c_out // 2, 4, stride=2, padding=1)
            up_b = nn.ConvTranspose2d(c_in, c_out // 2, 4, stride=2, padding=1)
        elif upsample == 'pixelshuffle':
            self.a = PixelShuffleUp(c_in, c_out // 2, coordconv=coordconv)
            up_b = PixelShuffleUp(c_in, c_out // 2, coordconv=coordconv)
        else:
            raise ValueError(f"unknown upsample mode: {upsample}")
        self.b = nn.Sequential(_conv2d(c_in, c_in, 3, padding=1, coordconv=coordconv), nn.ReLU(inplace=True), up_b)
        self.out = nn.Sequential(nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x): return self.out(torch.cat([self.a(x), self.b(x)], dim=1))

class NonLocalBlock2d(nn.Module):
    """Wang et al., "Non-local Neural Networks" (CVPR 2018,
    https://arxiv.org/abs/1711.07971). Every conv in this decoder only mixes a local
    kxk neighbourhood, so two spatial locations only start to influence each other
    once enough stacked layers make their receptive fields overlap -- e.g. a cube's
    near edge cannot directly inform the prediction at its far edge without real
    depth. A non-local block computes one similarity-weighted average over ALL other
    locations per query (plain self-attention over the H*W feature map, "embedded
    Gaussian" instantiation), giving every location a direct path to every other
    location in a single layer. Applied at the decoder's 8x8 stage (64 tokens) --
    cheap enough there to run as O(HW^2), and coarse enough that long-range spatial
    reasoning (how many objects, how far apart) is still cheap to learn before the
    later stages commit to fine pixel detail. `out` is zero-initialized so the block
    starts as an identity function (pure residual) and only earns a nonzero
    contribution once training rewards using it."""
    def __init__(self, c, c_inter:int|None=None):
        super().__init__()
        c_inter = c_inter or max(c // 2, 1)
        self.theta = nn.Conv2d(c, c_inter, 1)
        self.phi = nn.Conv2d(c, c_inter, 1)
        self.g = nn.Conv2d(c, c_inter, 1)
        self.out = nn.Conv2d(c_inter, c, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        theta = self.theta(x).flatten(2)                                    # (B,Ci,HW)
        phi = self.phi(x).flatten(2)                                        # (B,Ci,HW)
        g = self.g(x).flatten(2)                                            # (B,Ci,HW)
        attn = torch.softmax(theta.transpose(1, 2) @ phi / (theta.shape[1] ** 0.5), dim=-1)  # (B,HW,HW)
        y = (g @ attn.transpose(1, 2)).view(B, -1, H, W)
        return x + self.out(y)

class ResBlock(nn.Module):
    """Same-resolution refinement block: y = x + f(x), two 3x3 convs (stride 1, 'same'
    padding) with a skip connection. Unlike TwoBranchUp (which always upsamples via
    stride-2 transposed convs even at equal in/out channels), this adds depth to the
    decoder WITHOUT changing spatial size -- the "residual blocks per scale" the
    ladder script (scripts/pm_size_ladder.sh) asks for between upsampling stages."""
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                                  nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): return self.relu(x + self.net(x))

class Decoder(nn.Module):
    """(B,D) -> (B,out_h,out_w), or (B,out_h,out_w,out_c) when out_c > 1. Seeds a 4x4
    grid and triples it (stride-2 transposed convs) to a 32x32 feature map, then a 3x3
    conv head cleans up the transposed-conv checkerboard and emits the mask.

    An out_h x out_w below 32 in either axis (a box run at 21x30, say) is reconciled by a
    bilinear resample of the 32x32 feature map BEFORE the head, so the head always runs at
    the target resolution with its full 3x3 receptive field.

    `resize` is kept for backwards compatibility but is now INERT. The old 'conv' mode
    sized the head's first conv to a valid-mode kernel that ate the 32 - out margin; at the
    standardized 32x32 grid that margin is 0, so the kernel degenerated to 1x1 -- a pure
    channel projection with no spatial mixing -- which cost ~15-25% held-out IoU and much
    slower convergence (grid ablation, 2026-09-10: 32x32+conv 0.242 vs 32x32+bilinear 0.300
    vs hyb-h2 0.285, peak 1-cube/2-cubes soft-IoU). Both modes now do the same thing.

    The head is 3 stacked size-preserving 3x3 convs (padding 1): two hidden 3x3 layers of
    spatial mixing over the transposed-conv output before the projection to out_c. Changing
    the head shape breaks loading pre-32x32 boombox checkpoints -- accepted.

    Three orthogonal modifiers, all off by default (identical to the original arch when all
    are left at default):
    * `coordconv`: every conv in the upsampling stages and the head becomes a CoordConv2d
      (see its docstring) -- gives the decoder an explicit position input everywhere it turns
      the bottleneck embedding into pixels.
    * `upsample='pixelshuffle'`: swap TwoBranchUp's ConvTranspose2d for PixelShuffleUp (see
      its docstring) -- avoids the transposed-conv checkerboard pattern.
    * `nonlocal_stage`: insert a NonLocalBlock2d (see its docstring) after this many upsampling
      stages have run (e.g. 1 = after the first TwoBranchUp, at 8x8) -- gives every spatial
      location a direct, one-layer path to every other location before the later stages commit
      to fine pixel detail. None disables it.
    """
    SEED = (4, 4)

    def __init__(self, d_model, out_h, out_w, out_c=1, resize='bilinear', mult:float=1.0, num_res_blocks:int|None=None,
                 coordconv:bool=False, upsample:str='transposed', nonlocal_stage:int|None=None):
        super().__init__()
        self.out_hw, self.out_c = (out_h, out_w), out_c
        self.resize = resize  # accepted for backwards compat, no longer used (see docstring)
        self.nonlocal_stage = nonlocal_stage
        base = int(512 * mult)
        self.base = base  # store for use in forward()
        self.project = nn.Linear(d_model, base * self.SEED[0] * self.SEED[1])
        # Default: 3 upsampling stages (512->256->128->64). num_res_blocks adds that many
        # same-resolution ResBlocks after each TwoBranchUp -- refinement depth, not more
        # upsampling (TwoBranchUp itself always doubles spatial size via stride-2 transposed
        # convs, even at equal in/out channels, so it can't be reused for this). Each stage is
        # its own nn.Sequential (not one flat stack) so a subclass (MaskedConvDecoder) can hook
        # in between stages; NonLocalBlock2d is inserted the same way via nonlocal_stage.
        widths = [base, base // 2, base // 4, base // 8]
        self.stages = nn.ModuleList()
        for i in range(len(widths) - 1):
            stage = [TwoBranchUp(widths[i], widths[i + 1], upsample=upsample, coordconv=coordconv)]
            if num_res_blocks:
                stage += [ResBlock(widths[i + 1]) for _ in range(num_res_blocks)]
            self.stages.append(nn.Sequential(*stage))
        self.nonlocal_block = NonLocalBlock2d(widths[nonlocal_stage]) if nonlocal_stage is not None else None
        head_in = widths[-1]  # final upsampling layer outputs this many channels
        self.head = nn.Sequential(
            _conv2d(head_in, 32, 3, padding=1, coordconv=coordconv), nn.ReLU(inplace=True),
            _conv2d(32, 32, 3, padding=1, coordconv=coordconv), nn.ReLU(inplace=True),
            nn.Conv2d(32, out_c, 3, padding=1),
            )

    def run_stages(self, x):
        """(B,base,4,4) -> (B,head_in,32,32). Split out of forward() so MaskedConvDecoder can
        override it alone and reuse everything else (project, resize, head)."""
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.nonlocal_block is not None and i + 1 == self.nonlocal_stage:
                x = self.nonlocal_block(x)
        return x

    def forward(self, emb):
        x = self.run_stages(self.project(emb).view(-1, self.base, *self.SEED))
        if tuple(x.shape[-2:]) != self.out_hw:
            x = F.interpolate(x, size=self.out_hw, mode='bilinear', align_corners=False)
        x = self.head(x)                                     # (B,out_c,H,W)
        # .contiguous(): permute leaves a non-contiguous view, and torchmetrics' MSE does
        # preds.view(-1), which requires contiguity. Only bites on the rgb path -- out_c == 1
        # takes squeeze(1), which stays contiguous.
        return x.squeeze(1) if self.out_c == 1 else x.permute(0, 2, 3, 1).contiguous()

class MaskedConvDecoder(Decoder):
    """Conv analogue of Mask2Former's masked attention (Cheng et al., "Masked-attention Mask
    Transformer for Universal Image Segmentation", CVPR 2022, https://arxiv.org/abs/2112.01527).
    Mask2Former restricts each decoder layer's cross-attention to only the FOREGROUND predicted
    by the previous layer's mask (hard-thresholded at 0.5), instead of attending everywhere --
    every layer refines just the region the model already believes contains the object, rather
    than splitting capacity over the whole (mostly empty) frame.

    This decoder has no attention, so the conv equivalent is a spatial gate: after each
    upsampling stage except the last, a 1x1 conv predicts a coarse mask from the current
    feature map, which is thresholded at 0.5 (hard threshold, like the paper -- not a soft
    sigmoid gate, which would just be an ordinary squeeze-excite) and multiplied into the
    features before the NEXT stage runs. Background locations are zeroed and stop drawing
    gradient into the next stage's convs, which then only have to spend capacity refining the
    region already believed to be foreground. The gate is applied post-BatchNorm/ReLU (stage
    output is already non-negative there), and multiplying by a boolean mask commutes with the
    later stages' convs' receptive fields, so this doesn't change the head or run_stages
    signature -- only what flows between stages.

    A dead gate (all-zero coarse mask) would zero every later stage and stop all
    gradient to them, so this is TRAINED WITH a small auxiliary BCE loss per intermediate mask
    against the downsampled ground truth (see BoomboxModel.loss / MASKED_CONV_AUX_WEIGHT) --
    exactly as Mask2Former supervises its intermediate masks, and for the same reason: the gate
    only starts out useful once the mask predictions do.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        widths = [self.base, self.base // 2, self.base // 4, self.base // 8]
        self.mask_heads = nn.ModuleList([nn.Conv2d(widths[i + 1], 1, 1) for i in range(len(self.stages) - 1)])

    def run_stages(self, x):
        aux_logits = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.nonlocal_block is not None and i + 1 == self.nonlocal_stage:
                x = self.nonlocal_block(x)
            if i < len(self.stages) - 1:
                coarse_logits = self.mask_heads[i](x)                  # (B,1,H_i,W_i)
                aux_logits.append(coarse_logits)
                gate = (coarse_logits.sigmoid() > 0.5).to(x.dtype)     # hard threshold, like the paper
                x = x * gate
        self.last_aux_logits = aux_logits  # stashed for the auxiliary loss (see class docstring)
        return x

class BoomboxModel(ComposerModel):
    """Same ComposerModel contract as VibrationTransformer, so callbacks, metrics
    and run.py work on it unchanged."""
    def __init__(self, d_model=512, data_info=None, fuse_speakers=False,
                 loss_fn='mse', loss_alpha=0.5, count_loss_weight=0.0,
                 freq_dropout=0.0, laser_dropout=0.0, encoder='single', fuse='concat',
                 trim_pad=False, learned_collapse=False, freq_mult=1, freq_depth=1,
                 resize='conv', decoder_arch='default', coordconv=False,
                 decoder_upsample='transposed', decoder_nonlocal_stage=None,
                 masked_conv_aux_weight=0.5):
        super().__init__()
        # tokenize() zero-pads F up to a whole number of patches. That padding is justified for
        # the transformer (FreqEncoder.embed sees the zeros at FIXED positions and absorbs them),
        # but a conv slides across them and then averages them in, so here they are dead weight:
        # 45/1280 bins at patch_size=64, 211/3072 at 256. Only Boombox trims -- tokenize() and
        # the MDS cache keys are untouched, so the transformer path and every precomputed .npy
        # stay valid.
        self.n_freqs_real = data_info.get('n_freqs_real') if trim_pad else None
        n_freqs = self.n_freqs_real or data_info.get('n_freqs')
        # 'single' is the v2 encoder and stays the default, so every existing command is
        # unchanged. 'two-stream' gives magnitude and phase separate frequency stacks.
        enc = TwoStreamEncoder if encoder == 'two-stream' else Encoder
        kw = dict(fuse=fuse) if encoder == 'two-stream' else {}
        if learned_collapse:
            if n_freqs is None:
                raise ValueError("learned_collapse needs data_info['n_freqs'] to size the collapse")
            kw['freq_width'] = _surviving_width(n_freqs)
        kw.update(freq_mult=freq_mult, freq_depth=freq_depth)
        self.encoder = enc(data_info.get('n_channels', 2), d_model, data_info['n_laser_rows'],
                           data_info['n_laser_cols'], freq_dropout, laser_dropout, **kw)
        dec_cls = MaskedConvDecoder if decoder_arch == 'masked-conv' else Decoder
        self.decoder = dec_cls(d_model, data_info['out_h'], data_info['out_w'],
                               data_info.get('out_c', 1), resize=resize, coordconv=coordconv,
                               upsample=decoder_upsample, nonlocal_stage=decoder_nonlocal_stage)
        self.masked_conv_aux_weight = masked_conv_aux_weight if decoder_arch == 'masked-conv' else 0.0
        self.fuse_speakers = fuse_speakers
        self.empty_head = nn.Linear(d_model, 1)
        self.count_head = nn.Linear(d_model, N_COUNT_CLASSES)
        self.count_loss_weight = count_loss_weight
        self.loss_fn = LOSSES[loss_fn]
        self.is_spatial_loss = loss_fn.startswith('ce-spatial')
        self.is_asym_loss = loss_fn.endswith('-asym')
        self.loss_alpha = loss_alpha
        self.train_metrics = create_metrics(data_info, CHEAP_SEG_KEYS)  # cheap subset per step
        self.val_metrics = create_metrics(data_info)                   # full suite on eval loaders

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
        out = dict(mask_pred=mask_logits.sigmoid(), mask_logits=mask_logits,
                   empty_logit=self.empty_head(emb), count_logits=self.count_head(emb))
        if self.masked_conv_aux_weight: out['aux_mask_logits'] = self.decoder.last_aux_logits
        return out

    def _to_conv(self, x):
        """(B,L,P,PS,C) -> (B,C,L,P*PS), dropping tokenize()'s trailing zero-pad if enabled."""
        assert x.ndim == 5, f"expected (B,L,P,PS,C), got {tuple(x.shape)}"
        x = x.flatten(2, 3)
        if self.n_freqs_real is not None: x = x[:, :, :self.n_freqs_real]
        return x.permute(0, 3, 1, 2)

    def loss(self, outputs, batch):
        kw = dict(empty_logit=outputs['empty_logit']) if self.is_spatial_loss else {}
        if self.is_asym_loss: kw = dict(alpha=self.loss_alpha)
        total = self.loss_fn(outputs['mask_logits'], outputs['mask_pred'], batch['mask_true'], **kw)
        if self.count_loss_weight:
            total = total + self.count_loss_weight * count_loss(outputs['count_logits'], batch['info']['n_objects'])
        if self.masked_conv_aux_weight and 'aux_mask_logits' in outputs:
            # supervise MaskedConvDecoder's intermediate gates (see its docstring for why: an
            # untrained gate is all-zero and kills gradient to every later stage) -- BCE against
            # the ground-truth mask, downsampled to each stage's resolution.
            gt = batch['mask_true'].unsqueeze(1).float()  # (B,1,out_h,out_w), binary segmentation target
            aux = 0.0
            for logits in outputs['aux_mask_logits']:
                target = F.interpolate(gt, size=logits.shape[-2:], mode='bilinear', align_corners=False)
                aux = aux + F.binary_cross_entropy_with_logits(logits, target)
            total = total + self.masked_conv_aux_weight * (aux / len(outputs['aux_mask_logits']))
        return total

    def get_metrics(self, is_train=False): return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch, outputs, metric):
        metric.update(outputs['mask_logits'], outputs['mask_pred'], batch['mask_true'], batch['info']['n_objects'])

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)
