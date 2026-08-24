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
    return (mask_pred - mask_true).square().mean(dim=(-2, -1))

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

def create_metrics(data_info): return {
    "mse": MeanSquaredError(),
    'com-distance': CenterOfMassDistance(),
    'soft-iou': SoftIoU(),
    'soft-precision': SoftPrecision(),
    'soft-recall': SoftRecall(),
    # 'soft-dice': SoftDice(),
    'count-acc': CountAccuracy(num_classes=N_COUNT_CLASSES, average='micro'),
    'count-acc-macro': CountAccuracy(num_classes=N_COUNT_CLASSES, average='macro'),
    }

#***** 2 losses *****

# mse is averaged over (B,H,W) so the error is independent of the out_h out_w we choose
def mse_loss(mask_logits, mask_pred, mask_true): return F.mse_loss(mask_pred, mask_true)
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

LOSSES = {'mse': mse_loss, 'iou': iou_loss, 'dice': dice_loss, 'mse+iou': mse_iou_loss, 'mse+dice': mse_dice_loss,
          'ce-pixel': ce_pixel_loss, 'ce-spatial': spatial_ce_loss, 'ce-spatial-normalized': spatial_ce_normalized_loss,
          'mse-asym': mse_asym_loss, 'ce-pixel-asym': ce_pixel_asym_loss}

#***** 3 decoder *****

class MLPDecoder(nn.Module):
    def __init__(self, d_model, out_h, out_w):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.net = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, out_h * out_w))
    def forward(self, cls): return self.net(cls).view(-1, self.out_h, self.out_w)

class MLPMidDecoder(nn.Module):
    def __init__(self, d_model, out_h, out_w):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.net = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, out_h * out_w),
            )
    def forward(self, cls): return self.net(cls).view(-1, self.out_h, self.out_w)

class AttnDecoder(nn.Module):
    def __init__(self, d_model, out_h, out_w, num_heads:int=2, num_layers:int=2, do_rope:bool=True):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        # v1: one seed shared by every position, so RoPE alone distinguished the queries
        # self.query_seed = nn.Parameter(torch.zeros(1, 1, d_model))
        # v2: one learned query per output position (DETR-style); ties the ckpt to this resolution
        self.query_seed = nn.Parameter(torch.zeros(1, out_h * out_w, d_model))
        nn.init.trunc_normal_(self.query_seed, std=0.02)
        self.register_buffer("freqs_query", precompute_freqs_cis_2d(d_model, out_h, out_w))  # 2D RoPE over the output grid
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model, batch_first=True)
        self.layers = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)
        self.do_rope = do_rope

    def forward(self, memory, memory_key_padding_mask=None):
        # memory: (B,S,D) per-laser token sequence to cross-attend into (S = L+1, includes cls token)
        B = memory.shape[0]
        queries = self.query_seed.expand(B, -1, -1)  # (1,out_h*out_w,D) -> (B,out_h*out_w,D)
        if self.do_rope: queries = apply_rope(queries, self.freqs_query)                    # give each query its 2D grid position
        out = self.layers(queries, memory, memory_key_padding_mask=memory_key_padding_mask)  # (B,out_h*out_w,D)
        return self.head(out).view(B, self.out_h, self.out_w)

def build_decoder(decoder, d_model, out_h, out_w, decoder_num_heads:int=2, decoder_num_layers:int=2):
    if decoder == 'mlp': return MLPDecoder(d_model, out_h, out_w)
    if decoder == 'mlp-mid': return MLPMidDecoder(d_model, out_h, out_w)
    if decoder == 'attn': return AttnDecoder(d_model, out_h, out_w, num_heads=decoder_num_heads, num_layers=decoder_num_layers)
    if decoder == 'attn-no-rope': return AttnDecoder(d_model, out_h, out_w, num_heads=decoder_num_heads, num_layers=decoder_num_layers, do_rope=False)
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
    def __init__(self, patch_size:int, d_model:int, num_heads:int, num_layers:int, signal_length:int, freq_dropout:float, n_channels:int=2):
        super().__init__()
        self.embed = nn.Linear(patch_size * n_channels, d_model)
        self.speakers_embed = nn.Embedding(4, d_model)
        self.layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model, batch_first=True), num_layers=num_layers)
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
        if speaker is not None: x += self.speakers_embed(speaker).unsqueeze(1)  # (B_L,P,D), (B_L,1,D)  -> (B_L,P,D)
        x = torch.cat((self.cls_token.expand(B_L, -1, -1), x), dim=1)           # (B_L,P,D)             -> (B_L,P+1,D)
        output = self.layers(x, src_key_padding_mask=pad_mask(keep, B_L))       # (B_L,P+1,D)           -> (B_L,P+1,D)
        return output[:, 0, :]  # (B_L,P+1,D) -> (B_L,D)

#***** 5 model *****

class VibrationTransformer(ComposerModel):
    def __init__(self, d_model:int=128, pnt_num_heads:int=2, pnt_num_layers:int=2, seq_num_heads:int=2, seq_num_layers:int=2, data_info=None, decoder:str='mlp', decoder_num_heads:int=2, decoder_num_layers:int=2, freq_dropout:float=0.3, laser_dropout:float=0.3, loss_fn:str='mse', loss_alpha:float=0.5, count_loss_weight:float=0.0):
        super().__init__()

        # encoder
        self.freq_encoder = FreqEncoder(data_info['patch_size'], d_model, pnt_num_heads, pnt_num_layers, data_info['n_freqs'], freq_dropout, n_channels=data_info.get('n_channels', 2))
        self.laser_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=seq_num_heads, dim_feedforward=4 * d_model, batch_first=True), num_layers=seq_num_layers)
        self.laser_dropout = laser_dropout

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # Initialize to small random values
        self.register_buffer("freqs_laser", precompute_freqs_cis_2d(d_model, data_info['n_laser_rows'], data_info['n_laser_cols'])) # for laser grid

        # decoder
        self.decoder = build_decoder(decoder, d_model, data_info['out_h'], data_info['out_w'], decoder_num_heads, decoder_num_layers)

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
        if self.count_loss_weight:
            total = total + self.count_loss_weight * count_loss(outputs['count_logits'], batch['info']['n_objects'])
        return total

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, CountAccuracy): metric.update(outputs['count_logits'], batch['info']['n_objects'])
        else: metric.update(outputs['mask_pred'], batch['mask_true'])

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)
