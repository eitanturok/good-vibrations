import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanSquaredError, Metric
from torchmetrics.classification import MulticlassAccuracy

from utils.metrics import center_of_mass, soft_iou, soft_dice

#***** 0 rope *****

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim))
    freqs = torch.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

def precompute_freqs_cis_2d(dim: int, h: int, w: int, theta: float = 10000.0) -> torch.Tensor:
    """2D RoPE (matches Pixtral/HF): rows take freqs[::2], cols freqs[1::2], so every channel gets a
    distinct rate. Returns [cos | sin] for apply_rope's half-split pairing of channel i with i+dim/2."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))                # (dim//2,)
    angles_h = torch.outer(torch.arange(h).float(), freqs[::2])                     # (h,dim//4) rows
    angles_w = torch.outer(torch.arange(w).float(), freqs[1::2])                    # (w,dim//4) cols
    angles = torch.cat([angles_h[:, None, :].repeat(1, w, 1),
                        angles_w[None, :, :].repeat(h, 1, 1)], dim=-1).reshape(h * w, dim // 2)
    return torch.cat([angles.cos(), angles.sin()], dim=-1)                          # (h*w,dim)

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    shp = [1] * (x.ndim - 2) + [x.shape[1], -1]  # works with 1D + 2D rope
    cos, sin = freqs_cis.reshape(*shp).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

#***** 1 metrics *****

N_COUNT_CLASSES = 4  # n_objects observed in {0,1,2,3}

def com_distances(mask_pred, mask_true, epsilon, normalize):
    com_dist = center_of_mass(mask_pred, normalize=normalize, epsilon=epsilon) - center_of_mass(mask_true, normalize=normalize, epsilon=epsilon)
    return torch.linalg.norm(com_dist, ord=2, dim=-1)

def mses(mask_pred, mask_true):
    return (mask_pred - mask_true).square().flatten(1).mean(-1)  # flatten, so an (B,H,W,3) rgb target reduces correctly too

class CenterOfMassDistance(Metric):
    def __init__(self, norm:int=2, epsilon:float=1e-6):
        super().__init__()
        self.p, self.epsilon = norm, epsilon
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mask_pred, mask_true):
        valid = mask_true.sum((-2, -1)) > 0  # skip empty ground-truth masks
        if not valid.any(): return
        com_distances_ = com_distances(mask_pred[valid], mask_true[valid], self.epsilon, normalize=True)
        self.total, self.count = self.total + com_distances_.sum(), self.count + valid.sum()

    def compute(self): return self.total / self.count

class SoftIoU(Metric):
    def __init__(self, epsilon:float=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mask_pred, mask_true):
        ious = soft_iou(mask_pred, mask_true, self.epsilon)
        self.total, self.count = self.total + ious.sum(), self.count + ious.numel()

    def compute(self): return self.total / self.count

class SoftDice(Metric):
    def __init__(self, epsilon:float=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mask_pred, mask_true):
        dices = soft_dice(mask_pred, mask_true, self.epsilon)
        self.total, self.count = self.total + dices.sum(), self.count + dices.numel()

    def compute(self): return self.total / self.count

# soft-iou fuses over- and under-prediction; these split them so a loss alpha sweep is readable.
# precision = tp/(tp+fp) punishes over-painting, recall = tp/(tp+fn) punishes missing.
class SoftPrecision(Metric):
    def __init__(self, epsilon:float=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, mask_pred, mask_true):
        self.tp = self.tp + torch.minimum(mask_pred, mask_true).sum()
        self.fp = self.fp + (mask_pred - mask_true).clamp_min(0).sum()

    def compute(self): return (self.tp + self.epsilon) / (self.tp + self.fp + self.epsilon)

class SoftRecall(Metric):
    def __init__(self, epsilon:float=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, mask_pred, mask_true):
        self.tp = self.tp + torch.minimum(mask_pred, mask_true).sum()
        self.fn = self.fn + (mask_true - mask_pred).clamp_min(0).sum()

    def compute(self): return (self.tp + self.epsilon) / (self.tp + self.fn + self.epsilon)

# scores the count head instead of the mask; update_metric dispatches on this type.
# classes are imbalanced ({0:55, 1:624, 2:2288, 3:40}) so macro is the honest read.
class CountAccuracy(MulticlassAccuracy): pass

# Boombox's two metrics (arXiv 2105.08052), on BINARIZED masks -- the paper computes
# them that way, so soft-iou is not comparable to its published numbers.
def _bbox(mask):
    """(B,H,W) bool -> (center_rc, extent_hw), both (B,2) float. NaN where mask is empty."""
    rows, cols = mask.any(-1), mask.any(-2)
    idx_r = torch.arange(mask.shape[-2], device=mask.device, dtype=torch.float32)
    idx_c = torch.arange(mask.shape[-1], device=mask.device, dtype=torch.float32)
    inf = torch.tensor(float('inf'), device=mask.device)
    lo_r, hi_r = torch.where(rows, idx_r, inf).amin(-1), torch.where(rows, idx_r, -inf).amax(-1)
    lo_c, hi_c = torch.where(cols, idx_c, inf).amin(-1), torch.where(cols, idx_c, -inf).amax(-1)
    center = torch.stack([(lo_r + hi_r) / 2, (lo_c + hi_c) / 2], -1)
    return center, torch.stack([hi_r - lo_r + 1, hi_c - lo_c + 1], -1)

class LocalizationScore(Metric):
    """Fraction of samples whose predicted box center lands within half the ground-truth
    box diagonal of the true center. Boxes come from the masks, not metadata: this capture
    has no per-object bboxes and info['x_com'/'y_com'] are the -1.0 sentinel.
    An empty prediction gives inf/NaN and counts as a miss."""
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mask_pred, mask_true):
        p, t = mask_pred > self.threshold, mask_true > self.threshold
        valid = t.flatten(1).any(-1)  # an empty ground truth has no box to localize
        if not valid.any(): return
        c_p, _ = _bbox(p[valid])
        c_t, extent = _bbox(t[valid])
        hit = torch.linalg.norm(c_p - c_t, dim=-1) <= 0.5 * torch.linalg.norm(extent, dim=-1)
        self.correct = self.correct + torch.nan_to_num(hit.float(), nan=0.0).sum()
        self.count = self.count + valid.sum()

    def compute(self): return self.correct / self.count

class HardIoU(Metric):
    """IoU on masks binarized at `threshold` -- the paper's IoU."""
    def __init__(self, threshold: float = 0.5, epsilon: float = 1e-6):
        super().__init__()
        self.threshold, self.epsilon = threshold, epsilon
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, mask_pred, mask_true):
        p, t = (mask_pred > self.threshold).flatten(1), (mask_true > self.threshold).flatten(1)
        ious = ((p & t).sum(-1) + self.epsilon) / ((p | t).sum(-1) + self.epsilon)
        self.total, self.count = self.total + ious.sum(), self.count + ious.numel()

    def compute(self): return self.total / self.count

def create_metrics(data_info):
    counts = {'count-acc': CountAccuracy(num_classes=N_COUNT_CLASSES, average='micro'),
              'count-acc-macro': CountAccuracy(num_classes=N_COUNT_CLASSES, average='macro')}
    # the mask metrics read (H,W) as occupancy, which is meaningless on an rgb target
    if data_info.get('out_c', 1) != 1: return {"mse": MeanSquaredError()} | counts
    return {
    "mse": MeanSquaredError(),
    'com-distance': CenterOfMassDistance(),
    'soft-iou': SoftIoU(),
    'soft-precision': SoftPrecision(),
    'soft-recall': SoftRecall(),
    'hard-iou': HardIoU(),
    'localization': LocalizationScore(),
    # 'soft-dice': SoftDice(),
    } | counts

#***** 2 losses *****

# mse is averaged over (B,H,W) so the error is independent of the out_h out_w we choose
def mse_loss(mask_logits, mask_pred, mask_true): return F.mse_loss(mask_pred, mask_true)
# l1 is averaged over (B,H,W) like mse. Unlike mse's 2*err, the gradient is a constant +-1
# regardless of how small the error is, so near-zero background is still pushed to EXACTLY zero
# instead of stalling in a low-cost haze. That makes it the sparse-friendly counterpart to mse.
def l1_loss(mask_logits, mask_pred, mask_true): return F.l1_loss(mask_pred, mask_true)
# ce-pixel: H*W independent binary questions, cells never compete. Reads logits for numerical stability.
def ce_pixel_loss(mask_logits, mask_pred, mask_true): return F.binary_cross_entropy_with_logits(mask_logits, mask_true)

# asym: the signal is sparse, so background dominates the gradient. alpha weighs under-prediction
# (false negatives) against over-prediction; alpha>0.5 paints more, alpha<0.5 holds back. The 2*
# keeps the scale fixed so alpha=0.5 is exactly mse / ce-pixel.
def mse_asym_loss(mask_logits, mask_pred, mask_true, alpha=0.5):
    err = mask_pred - mask_true
    fp = 2 * alpha * (-err).clamp_min(0).square() # we want more fp, then fn -> low alpha
    fn = 2 * (1 - alpha) * err.clamp_min(0).square()
    return (fn + fp).mean()
# pos_weight scales only the y*log(p) term, i.e. the false-negative half
def ce_pixel_asym_loss(mask_logits, mask_pred, mask_true, alpha=0.5):
    return F.binary_cross_entropy_with_logits(mask_logits, mask_true, pos_weight=mask_logits.new_tensor(alpha / (1 - alpha)))
def iou_loss(mask_logits, mask_pred, mask_true): return 1 - soft_iou(mask_pred, mask_true).mean()
def dice_loss(mask_logits, mask_pred, mask_true): return 1 - soft_dice(mask_pred, mask_true).mean()
def mse_iou_loss(mask_logits, mask_pred, mask_true, theta=0.5): return theta * mse_loss(mask_logits, mask_pred, mask_true) + (1 - theta) * iou_loss(mask_logits, mask_pred, mask_true)
def mse_dice_loss(mask_logits, mask_pred, mask_true, theta=0.5): return theta * mse_loss(mask_logits, mask_pred, mask_true) + (1 - theta) * dice_loss(mask_logits, mask_pred, mask_true)

# ce-spatial: ONE softmax over H*W cells + 1 "empty box" slot, so cells compete and mass sums to 1.
# The empty slot is a pure occupancy bit in both branches -- 1 iff the mask is entirely empty -- which
# avoids 0/0 on all-zero masks and lets the model say "no cube".
# The branches differ in the cell targets: normalized=True rescales cube mass to sum to 1, so every
# sample contributes the same total probability regardless of object size; normalized=False keeps the
# raw per-cell mass, so bigger/brighter objects carry proportionally more of the target distribution
# (the trailing q/q.sum() then makes it a distribution either way).
def spatial_ce_loss(mask_logits, mask_pred, mask_true, empty_logit, normalized: bool = False):
    flat = mask_true.flatten(1)                                              # (B,H*W)
    mass = flat.sum(-1, keepdim=True)
    occ = (mass > 0).float()
    if normalized:
        q = torch.cat([flat / mass.clamp_min(1e-9) * occ, 1 - occ], dim=-1)  # (B,H*W+1)
    else:
        q = torch.cat([flat, 1 - occ], dim=-1)                               # (B,H*W+1)
    q = q / q.sum(-1, keepdim=True)
    logits = torch.cat([mask_logits.flatten(1), empty_logit], dim=-1)                    # (B,H*W+1)
    return -(q * F.log_softmax(logits, dim=-1)).sum(-1).mean()

def spatial_ce_normalized_loss(mask_logits, mask_pred, mask_true, empty_logit):
    return spatial_ce_loss(mask_logits, mask_pred, mask_true, empty_logit, normalized=True)

# n_objects arrives from the dataset already validated, long, and on-device
def count_loss(count_logits, n_objects): return F.cross_entropy(count_logits, n_objects)

LOSSES = {'mse': mse_loss, 'l1': l1_loss, 'iou': iou_loss, 'dice': dice_loss, 'mse+iou': mse_iou_loss, 'mse+dice': mse_dice_loss,
          'ce-pixel': ce_pixel_loss, 'ce-spatial': spatial_ce_loss, 'ce-spatial-normalized': spatial_ce_normalized_loss,
          'mse-asym': mse_asym_loss, 'ce-pixel-asym': ce_pixel_asym_loss}

#***** 3 decoder *****

class MLPDecoder(nn.Module):
    def __init__(self, d_model, out_h, out_w, out_c=1, depth:int|None=None, hidden:int|None=None):
        super().__init__()
        self.out_h, self.out_w, self.out_c = out_h, out_w, out_c
        depth = depth or 2
        hidden = hidden or 256
        layers = []
        in_dim = d_model
        for i in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, out_h * out_w * out_c))
        self.net = nn.Sequential(*layers)
    def forward(self, cls): return self.net(cls).view(-1, self.out_h, self.out_w, self.out_c).squeeze(-1)

class MLPMidDecoder(nn.Module):
    def __init__(self, d_model, out_h, out_w, out_c=1):
        super().__init__()
        self.out_h, self.out_w, self.out_c = out_h, out_w, out_c
        self.net = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, out_h * out_w * out_c),
            )
    def forward(self, cls): return self.net(cls).view(-1, self.out_h, self.out_w, self.out_c).squeeze(-1)

class AttnDecoder(nn.Module):
    def __init__(self, d_model, out_h, out_w, num_heads:int=2, num_layers:int=2, do_rope:bool=True, out_c:int=1, ffn_dim:int|None=None):
        super().__init__()
        self.out_h, self.out_w, self.out_c = out_h, out_w, out_c
        # v1: one seed shared by every position, so RoPE alone distinguished the queries
        # self.query_seed = nn.Parameter(torch.zeros(1, 1, d_model))
        # v2: one learned query per output position (DETR-style); ties the ckpt to this resolution
        self.query_seed = nn.Parameter(torch.zeros(1, out_h * out_w, d_model))
        nn.init.trunc_normal_(self.query_seed, std=0.02)
        self.register_buffer("freqs_query", precompute_freqs_cis_2d(d_model, out_h, out_w))  # 2D RoPE over the output grid
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim or 4 * d_model, batch_first=True)
        self.layers = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_c)
        self.do_rope = do_rope

    def forward(self, memory, memory_key_padding_mask=None):
        # memory: (B,S,D) per-laser token sequence to cross-attend into (S = L+1, includes cls token)
        B = memory.shape[0]
        queries = self.query_seed.expand(B, -1, -1)  # (1,out_h*out_w,D) -> (B,out_h*out_w,D)
        if self.do_rope: queries = apply_rope(queries, self.freqs_query)                    # give each query its 2D grid position
        out = self.layers(queries, memory, memory_key_padding_mask=memory_key_padding_mask)  # (B,out_h*out_w,D)
        return self.head(out).view(B, self.out_h, self.out_w, self.out_c).squeeze(-1)

def build_decoder(decoder, d_model, out_h, out_w, decoder_num_heads:int=2, decoder_num_layers:int=2, out_c:int=1, ffn_dim:int|None=None, mlp_dec_depth:int|None=None, mlp_dec_hidden:int|None=None, conv_dec_mult:float|None=None, conv_dec_res_blocks:int|None=None):
    if decoder == 'mlp': return MLPDecoder(d_model, out_h, out_w, out_c, depth=mlp_dec_depth, hidden=mlp_dec_hidden)
    # boombox's transposed-conv stack on the transformer's cls token. Imported here, not at module
    # scope: boombox.py imports from this file, so a top-level import would be circular. Its
    # signature is already (B,D)->(B,H,W), the same contract as MLPDecoder, so nothing else changes.
    if decoder == 'conv':
        from model.boombox import Decoder as ConvDecoder
        kwargs = {'d_model': d_model, 'out_h': out_h, 'out_w': out_w, 'out_c': out_c}
        if conv_dec_mult is not None: kwargs['mult'] = conv_dec_mult
        if conv_dec_res_blocks is not None: kwargs['num_res_blocks'] = conv_dec_res_blocks
        return ConvDecoder(**kwargs)
    if decoder == 'mlp-mid': return MLPMidDecoder(d_model, out_h, out_w, out_c)
    if decoder == 'attn': return AttnDecoder(d_model, out_h, out_w, num_heads=decoder_num_heads, num_layers=decoder_num_layers, out_c=out_c, ffn_dim=ffn_dim)
    if decoder == 'attn-no-rope': return AttnDecoder(d_model, out_h, out_w, num_heads=decoder_num_heads, num_layers=decoder_num_layers, do_rope=False, out_c=out_c, ffn_dim=ffn_dim)
    raise ValueError(f"Unknown decoder: {decoder}")

#***** 4 encoder *****

def dropout(x, mask_shape, p, training):
    if not training or p == 0.0: return x, None
    keep = torch.rand(mask_shape, dtype=torch.float32, device=x.device) >= p
    return x * keep.unsqueeze(-1) / (1 - p), keep

def pad_mask(keep, B_or_BL):
    """(B,L) keep mask -> (B,L+1) key_padding_mask (True = ignore), with the prepended cls token never masked."""
    if keep is None: return None
    cls_keep = torch.ones(B_or_BL, 1, dtype=torch.bool, device=keep.device)
    return ~torch.cat([cls_keep, keep], dim=1)

class FreqEncoder(nn.Module):
    def __init__(self, patch_size:int, d_model:int, num_heads:int, num_layers:int, signal_length:int, freq_dropout:float, n_channels:int=2, ffn_dim:int|None=None):
        super().__init__()
        self.embed = nn.Linear(patch_size * n_channels, d_model)
        # no self.speakers_embed: dataset.py never produces a 'speakers_encoded' batch key, so
        # `speaker` below is always None -- a real speaker-conditioning embedding would need that
        # wired up in dataset.py first.
        self.layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim or 4 * d_model, batch_first=True), num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("freqs_cis", precompute_freqs_cis(d_model, signal_length // patch_size))
        self.freq_dropout = freq_dropout

    def forward(self, x, speaker=None):
        # x.shape = (B_L,P,PS,C) = (batch_size * n_lasers, n_patches, patch_size, n_coords)
        # canonical layout, matching dataset.tokenize: each token flattens to [f0c0,f0c1,f1c0,...]
        B_L, P, _, _ = x.shape
        x = self.embed(x.reshape(B_L, P, -1))                                   # (B_L,P,PS,C)          -> (B_L,P,D)
        # drop entire freq patches by setting them to zero, don't actually remove them
        x, keep = dropout(x, (B_L, P), self.freq_dropout, self.training)        # (B_L,P,D)             -> (B_L,P,D), (B_L,P)
        x = apply_rope(x, self.freqs_cis)                                       # (B_L,P,D)             -> (B_L,P,D)
        assert speaker is None, "speaker conditioning has no embedding to apply it with (see __init__)"
        x = torch.cat((self.cls_token.expand(B_L, -1, -1), x), dim=1)           # (B_L,P,D)             -> (B_L,P+1,D)
        output = self.layers(x, src_key_padding_mask=pad_mask(keep, B_L))       # (B_L,P+1,D)           -> (B_L,P+1,D)
        return output[:, 0, :]  # (B_L,P+1,D) -> (B_L,D)

#***** 5 model *****

class VibrationTransformer(ComposerModel):
    def __init__(self, d_model:int=128, pnt_num_heads:int=2, pnt_num_layers:int=2, seq_num_heads:int=2, seq_num_layers:int=2, data_info=None, decoder:str='mlp', decoder_num_heads:int=2, decoder_num_layers:int=2, freq_dropout:float=0.3, laser_dropout:float=0.3, loss_fn:str='mse', loss_alpha:float=0.5, count_loss_weight:float=0.0, ffn_dim:int|None=None, enc_ffn_dim:int|None=None, dec_ffn_dim:int|None=None, mlp_dec_depth:int|None=None, mlp_dec_hidden:int|None=None, conv_dec_mult:float|None=None, conv_dec_res_blocks:int|None=None):
        super().__init__()

        # ffn_dim=None keeps the 4*d_model default. It is worth setting explicitly: torch's own
        # default is a FIXED 2048, so runs from before that was pinned to 4*d_model had a 2048-wide
        # FFN at any width -- 16x d_model at d_model=128, not 4x.
        # encoder FFN dimension: enc_ffn_dim overrides ffn_dim; ffn_dim is the fallback
        _enc_ffn_dim = enc_ffn_dim if enc_ffn_dim is not None else ffn_dim
        _dec_ffn_dim = dec_ffn_dim if dec_ffn_dim is not None else ffn_dim
        # encoder
        self.freq_encoder = FreqEncoder(data_info['patch_size'], d_model, pnt_num_heads, pnt_num_layers, data_info['n_freqs'], freq_dropout, n_channels=data_info.get('n_channels', 2), ffn_dim=_enc_ffn_dim)
        self.laser_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=seq_num_heads, dim_feedforward=_enc_ffn_dim or 4 * d_model, batch_first=True), num_layers=seq_num_layers)
        self.laser_dropout = laser_dropout

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # Initialize to small random values
        self.register_buffer("freqs_laser", precompute_freqs_cis_2d(d_model, data_info['n_laser_rows'], data_info['n_laser_cols'])) # for laser grid

        # decoder
        out_c = data_info.get('out_c', 1)
        assert out_c == 1 or loss_fn in ('mse', 'ce-pixel'), f"{loss_fn} is mask-only; use mse or ce-pixel with an rgb target"
        self.decoder = build_decoder(decoder, d_model, data_info['out_h'], data_info['out_w'], decoder_num_heads, decoder_num_layers, out_c, ffn_dim=_dec_ffn_dim, mlp_dec_depth=mlp_dec_depth, mlp_dec_hidden=mlp_dec_hidden, conv_dec_mult=conv_dec_mult, conv_dec_res_blocks=conv_dec_res_blocks)

        # loss and metrics
        self.empty_head = nn.Linear(d_model, 1)  # extra "empty box" class for the spatial losses
        # a single Linear on cls, so count-acc answers exactly "is n_objects linearly decodable from cls"
        self.count_head = nn.Linear(d_model, N_COUNT_CLASSES)
        self.count_loss_weight = count_loss_weight
        self.loss_fn = LOSSES[loss_fn]
        self.is_spatial_loss = loss_fn.startswith('ce-spatial')
        self.is_asym_loss = loss_fn.endswith('-asym')
        self.loss_alpha = loss_alpha
        self.train_metrics, self.val_metrics = create_metrics(data_info), create_metrics(data_info)

    def forward(self, batch):
        # B=batch size, L=n_lasers, C=n_coordinates=2, PS=patch_size, D=d_model
        x = batch['fft'] # (B,L,P,PS,2)
        B, L, _, _, _ = x.shape

        # FreqEncoder learns patterns between all frequencies from a single laser
        # flatten so FreqEncoder processes all lasers AND all batches in parallel
        speaker = speaker.repeat_interleave(L) if (speaker := batch.get('speakers_encoded', None)) is not None else None # (BL,D)
        x = self.freq_encoder(x.flatten(0, 1), speaker).reshape(B, L, -1)  # (B,L,P,PS,C) -> (B,L,D)
        # drop entire laser positions by setting them to zero, don't actually remove them
        x, keep = dropout(x, (B, L), self.laser_dropout, self.training)
        key_padding_mask = pad_mask(keep, B)  # (B,L+1), True = ignore; None if not dropping

        # LaserEncoder learns patterns between ALL the lasers shining on the box
        x = apply_rope(x, self.freqs_laser) # (B,L,D) -> (B,L,D)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)  # (B,L,D) (1,1,D) -> (B,L+1,D)
        output = self.laser_encoder(x, src_key_padding_mask=key_padding_mask)  # (B,L+1,D) -> (B,L+1,D)

        # Predict segmentation mask
        cls = output[:, 0, :]  # (B,L+1,D) -> (B,D)
        decoder_input = output if isinstance(self.decoder, AttnDecoder) else cls
        mask_logits = self.decoder(decoder_input, key_padding_mask) if isinstance(self.decoder, AttnDecoder) else self.decoder(decoder_input) # (B,L+1,D) or (B,D) -> (B,H,W)
        mask_pred = mask_logits.sigmoid()
        empty_logit = self.empty_head(cls)  # (B,D) -> (B,1), the "no cube anywhere" class
        count_logits = self.count_head(cls)  # (B,D) -> (B,n_classes), how many objects in the box
        return dict(mask_pred=mask_pred, mask_logits=mask_logits, empty_logit=empty_logit, count_logits=count_logits)

    def loss(self, outputs, batch):
        kw = dict(empty_logit=outputs['empty_logit']) if self.is_spatial_loss else {}
        if self.is_asym_loss: kw = dict(alpha=self.loss_alpha)
        total = self.loss_fn(outputs['mask_logits'], outputs['mask_pred'], batch['mask_true'], **kw)
        # count_head and empty_head stay always-on free probes (count-acc metrics, viz dashboard,
        # OutputSaver all read them regardless of loss_fn/count_loss_weight -- see their __init__
        # comments), so when count_loss_weight==0 or the loss isn't spatial, add their output at
        # weight 0: same total loss value, but keeps them in the backward graph so AdamW always
        # has state for them and a checkpoint can strictly resume.
        if self.count_loss_weight:
            total = total + self.count_loss_weight * count_loss(outputs['count_logits'], batch['info']['n_objects'])
        else:
            total = total + 0 * outputs['count_logits'].sum()
        if not self.is_spatial_loss:
            total = total + 0 * outputs['empty_logit'].sum()
        return total

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, CountAccuracy): metric.update(outputs['count_logits'], batch['info']['n_objects'])
        else: metric.update(outputs['mask_pred'], batch['mask_true'])

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)
