import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanSquaredError, Metric

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

def create_metrics(data_info): return {
    "mse": MeanSquaredError(),
    'com-distance': CenterOfMassDistance(),
    'soft-iou': SoftIoU(),
    # 'soft-dice': SoftDice(),
    }

#***** 2 losses *****

# mse is averaged over (B,H,W) so the error is independent of the out_h out_w we choose
def mse_loss(mask_logits, mask_pred, mask_true): return F.mse_loss(mask_pred, mask_true)
# ce reads logits not preds: sigmoid-then-log saturates to log(0) once the model gets confident
def ce_loss(mask_logits, mask_pred, mask_true): return F.binary_cross_entropy_with_logits(mask_logits, mask_true)
def iou_loss(mask_logits, mask_pred, mask_true): return 1 - soft_iou(mask_pred, mask_true).mean()
def dice_loss(mask_logits, mask_pred, mask_true): return 1 - soft_dice(mask_pred, mask_true).mean()
def mse_iou_loss(mask_logits, mask_pred, mask_true, theta=0.5): return theta * mse_loss(mask_logits, mask_pred, mask_true) + (1 - theta) * iou_loss(mask_logits, mask_pred, mask_true)
def mse_dice_loss(mask_logits, mask_pred, mask_true, theta=0.5): return theta * mse_loss(mask_logits, mask_pred, mask_true) + (1 - theta) * dice_loss(mask_logits, mask_pred, mask_true)

LOSSES = {'mse': mse_loss, 'iou': iou_loss, 'dice': dice_loss, 'mse+iou': mse_iou_loss, 'mse+dice': mse_dice_loss, 'ce': ce_loss}

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
        self.embed = nn.Linear(n_channels * patch_size, d_model)
        self.speakers_embed = nn.Embedding(4, d_model)
        self.layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True), num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("freqs_cis", precompute_freqs_cis(d_model, signal_length // patch_size))
        self.freq_dropout = freq_dropout

    def forward(self, x, speaker=None):
        # x.shape = (B_L,P,C,PS) = (batch_size * n_lasers, n_patches, n_coords, patch_size)
        B_L, P, _, _ = x.shape
        x = self.embed(x.reshape(B_L, P, -1))                                   # (B_L,P,C,PS)          -> (B_L,P,D)
        # drop entire freq patches by setting them to zero, don't actually remove them
        x, keep = dropout(x, (B_L, P), self.freq_dropout, self.training)        # (B_L,P,D)             -> (B_L,P,D), (B_L,P)
        x = apply_rope(x, self.freqs_cis)                                       # (B_L,P,D)             -> (B_L,P,D)
        if speaker is not None: x += self.speakers_embed(speaker).unsqueeze(1)  # (B_L,P,D), (B_L,1,D)  -> (B_L,P,D)
        x = torch.cat((self.cls_token.expand(B_L, -1, -1), x), dim=1)           # (B_L,P,D)             -> (B_L,P+1,D)
        output = self.layers(x, src_key_padding_mask=pad_mask(keep, B_L))       # (B_L,P+1,D)           -> (B_L,P+1,D)
        return output[:, 0, :]  # (B_L,P+1,D) -> (B_L,D)

#***** 5 model *****

class VibrationTransformer(ComposerModel):
    def __init__(self, d_model:int=128, pnt_num_heads:int=2, pnt_num_layers:int=2, seq_num_heads:int=2, seq_num_layers:int=2, data_info=None, decoder:str='mlp', decoder_num_heads:int=2, decoder_num_layers:int=2, freq_dropout:float=0.3, laser_dropout:float=0.3, loss_fn:str='mse'):
        super().__init__()

        # encoder
        self.freq_encoder = FreqEncoder(data_info['patch_size'], d_model, pnt_num_heads, pnt_num_layers, data_info['n_freqs'], freq_dropout, n_channels=data_info.get('n_channels', 2))
        self.laser_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=seq_num_heads, batch_first=True), num_layers=seq_num_layers)
        self.laser_dropout = laser_dropout

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # Initialize to small random values
        self.register_buffer("freqs_laser", precompute_freqs_cis_2d(d_model, data_info['n_laser_rows'], data_info['n_laser_cols'])) # for laser grid

        # decoder
        self.decoder = build_decoder(decoder, d_model, data_info['out_h'], data_info['out_w'], decoder_num_heads, decoder_num_layers)

        # loss and metrics
        self.loss_fn = LOSSES[loss_fn]
        self.train_metrics, self.val_metrics = create_metrics(data_info), create_metrics(data_info)

    def forward(self, batch):
        # B=batch size, L=n_lasers, C=n_coordinates=2, PS=patch_size, D=d_model
        x = batch['fft'] # (B,L,P,2,PS)
        B, L, _, _, _ = x.shape

        # FreqEncoder learns patterns between all frequencies from a single laser
        # flatten so FreqEncoder processes all lasers AND all batches in parallel
        speaker = speaker.repeat_interleave(L) if (speaker := batch.get('speakers_encoded', None)) is not None else None # (BL,D)
        x = self.freq_encoder(x.flatten(0, 1), speaker).reshape(B, L, -1)  # (B,L,P,C,PS) -> (B,L,D)
        # drop entire laser positions by setting them to zero, don't actually remove them
        x, keep = dropout(x, (B, L), self.laser_dropout, self.training)
        key_padding_mask = pad_mask(keep, B)  # (B,L+1), True = ignore; None if not dropping

        # LaserEncoder learns patterns between ALL the lasers shining on the box
        x = apply_rope(x, self.freqs_laser) # (B,L,D) -> (B,L,D)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)  # (B,L,D) (1,1,D) -> (B,L+1,D)
        output = self.laser_encoder(x, src_key_padding_mask=key_padding_mask)  # (B,L+1,D) -> (B,L+1,D)

        # Predict segmentation mask
        decoder_input = output if isinstance(self.decoder, AttnDecoder) else output[:, 0, :]
        mask_logits = self.decoder(decoder_input, key_padding_mask) if isinstance(self.decoder, AttnDecoder) else self.decoder(decoder_input) # (B,L+1,D) or (B,D) -> (B,H,W)
        mask_pred = mask_logits.sigmoid()
        return dict(mask_pred=mask_pred, mask_logits=mask_logits)

    def loss(self, outputs, batch):
        return self.loss_fn(outputs['mask_logits'], outputs['mask_pred'], batch['mask_true'])

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch, outputs, metric):
        metric.update(outputs['mask_pred'], batch['mask_true'])

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)
