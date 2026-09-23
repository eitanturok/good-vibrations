import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

# Isolated behind torch._dynamo.disable so the whole-model torch.compile can't fuse flex_attention
# into the same Triton kernel as neighboring pointwise ops (MaskedAttnDecoder's per-layer
# tanh/mean/mul mask-confidence chain) -- that over-fusion is what crashes Triton codegen at
# --memory-grid 1's Q_LEN=KV_LEN=out_h*out_w=1024 (confirmed: flex_attention compiles fine in
# isolation at this exact shape, only fails when swept into the larger fused kernel). This forces
# a graph break at the attention call instead, with flex_attention compiled as its own separate
# unit -- still gets the O(N) memory benefit over the old dense-mask fallback, just not fused with
# its neighbors.
_compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
@torch._dynamo.disable()
def _flex_attention_isolated(q, k, v, score_mod):
    return _compiled_flex_attention(q, k, v, score_mod=score_mod)
from composer import ComposerModel
from torchmetrics import Metric

from utils.metrics import soft_iou, soft_dice, mass_error, contour_f, localization, LOC_KEYS

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

def precompute_freqs_cis_shared_axis(dim: int, primary_pos: torch.Tensor, secondary_pos: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """Like precompute_freqs_cis_2d, but takes explicit per-token (primary, secondary) coordinates
    instead of assuming a dense h x w grid -- so two DIFFERENT grids (e.g. the wall's (x,z) and the
    floor's (x,y)) can each get their own table while sharing the exact same freqs_x values on the
    primary axis (same dim/theta => same freqs[::2] regardless of what secondary_pos holds). RoPE's
    relative-position equivariance then falls out directly: a query and a key whose primary_pos
    (x) match get a zero relative rotation on those channels, i.e. "same x" attention is a free,
    unlearned inductive bias. primary_pos/secondary_pos are in the SAME physical units as each
    other's counterpart grid (the caller is responsible for that alignment/calibration)."""
    assert primary_pos.shape == secondary_pos.shape
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))            # (dim//2,)
    freqs_x, freqs_s = freqs[::2], freqs[1::2]                                  # same split as precompute_freqs_cis_2d
    angles_x = primary_pos[:, None] * freqs_x[None, :]                         # (N,dim//4)
    angles_s = secondary_pos[:, None] * freqs_s[None, :]                       # (N,dim//4)
    angles = torch.cat([angles_x, angles_s], dim=-1)                           # (N,dim//2)
    return torch.cat([angles.cos(), angles.sin()], dim=-1)                     # (N,dim)

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    shp = [1] * (x.ndim - 2) + [x.shape[1], -1]  # works with 1D + 2D rope
    cos, sin = freqs_cis.reshape(*shp).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

#***** 1 metrics *****

N_COUNT_CLASSES = 4  # n_objects observed in {0,1,2,3}

SEG_KEYS = ('bce', 'iou', *LOC_KEYS, 'contour', 'mass')
# Per-train-batch subset: pure GPU reductions, no connected-component labelling, no
# per-sample Python loop, no device sync. The full suite -- localization* (labels +
# greedy centroid match) and contour (morphology) -- runs on the eval loaders, where
# it is paid once per --eval-interval instead of on every step. `metrics/train/*` for
# the omitted keys then only appear from the boundary `train` evaluator; add an
# Evaluator(label='train') to eval_loaders in run.py to get them at eval cadence.
CHEAP_SEG_KEYS = ('bce', 'iou', 'mass')

# The MaskMetrics sharing a batch see the same (logits, pred, true); compute the suite
# once and cache it. `want` is the key set this caller needs -- localization()/contour_f()
# only run when a wanted key needs them, so the cheap train set skips them entirely.
_seg_cache = {}
def _seg_batch(logits, pred, true, want=SEG_KEYS):
    want = frozenset(want)
    if _seg_cache.get('id') != id(pred) or not want <= _seg_cache.get('want', frozenset()):
        b = len(pred)
        v = {}
        if 'bce' in want:     v['bce'] = (F.binary_cross_entropy_with_logits(logits, true, reduction='sum'), true.numel())
        if 'iou' in want:     v['iou'] = (soft_iou(pred, true).sum(), b)
        if 'mass' in want:    v['mass'] = (mass_error(pred, true).sum(), b)
        if 'contour' in want: v['contour'] = (contour_f(pred, true).sum(), b)
        if want & set(LOC_KEYS):
            nan = lambda t: (t[~t.isnan()].sum(), int((~t.isnan()).sum()))
            v.update({k: nan(t) for k, t in localization(pred, true).items()})
        _seg_cache.clear()
        _seg_cache.update(id=id(pred), want=want, v=v)
    return _seg_cache['v']

class MaskMetric(Metric):
    def __init__(self, key, group=SEG_KEYS):
        super().__init__()
        self.key = key
        self.group = tuple(group)  # the full key set to compute together on the shared batch
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, mask_logits, mask_pred, mask_true, n_objects):
        s, n = _seg_batch(mask_logits, mask_pred.float(), mask_true.float(), self.group)[self.key]
        self.total = self.total + s.to(self.total)
        self.count = self.count + n

    def compute(self): return self.total / self.count.clamp(min=1)

def create_metrics(data_info, keys=SEG_KEYS):
    # the mask metrics read (H,W) as occupancy, which is meaningless on an rgb target
    if data_info.get('out_c', 1) != 1: return {}
    return {k: MaskMetric(k, keys) for k in keys}

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
def ce_pixel_asym_loss(mask_logits, mask_pred, mask_true, alpha=0.5, balanced=False):
    if balanced:
        # pos_weight above only reweights each POSITIVE pixel's own term -- reduction='mean' still
        # divides by ALL pixels, so at sparse occupancy (gastronorm_one_cube: ~0.65% positive) that
        # dilutes even a large pos_weight back down near the unweighted mean (need pos_weight
        # ~(1-p)/p, ~150 here, just to BALANCE the two terms -- alpha=0.85 only gives ~5.7). Instead
        # take the mean over positive pixels and the mean over negative pixels SEPARATELY (each
        # independent of how many pixels of that class exist in the batch), then combine with
        # alpha -- the false-negative/false-positive tradeoff no longer depends on the class prior.
        per_pixel = F.binary_cross_entropy_with_logits(mask_logits, mask_true, reduction='none')
        pos_n, neg_n = mask_true.sum().clamp_min(1), (1 - mask_true).sum().clamp_min(1)
        pos_mean = (per_pixel * mask_true).sum() / pos_n
        neg_mean = (per_pixel * (1 - mask_true)).sum() / neg_n
        return 2 * alpha * pos_mean + 2 * (1 - alpha) * neg_mean
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

class MemoryReshaper(nn.Module):
    """Projects the L+1 laser-token memory onto a NEW learned out_h x out_w grid of tokens before
    the decoder cross-attends into it, so Q and K/V share both a sequence length and a spatial
    index (query i and reshaped-memory i are both "output grid cell i"). Mechanically it's the
    same DETR-style recipe as AttnDecoder's query side: a learned seed per grid cell, 2D RoPE'd
    over that SAME out_h x out_w grid, cross-attending (one MHA layer) into the raw laser memory.
    Without this, memory stays the L+1 laser tokens (10x10 laser grid, not 32x32 -- a different
    space than the queries live in), which is the original AttnDecoder/MaskedAttnDecoder behavior."""
    def __init__(self, d_model, out_h, out_w, num_heads:int=2):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.seed = nn.Parameter(torch.zeros(1, out_h * out_w, d_model))
        nn.init.trunc_normal_(self.seed, std=0.02)
        self.register_buffer("freqs_grid", precompute_freqs_cis_2d(d_model, out_h, out_w))
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory, memory_key_padding_mask=None):
        B = memory.shape[0]
        grid = apply_rope(self.seed.expand(B, -1, -1), self.freqs_grid)  # (B,out_h*out_w,D), 2D box-position embedding
        out, _ = self.attn(grid, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False)
        return self.norm(grid + out)  # (B,out_h*out_w,D), indexed by the SAME out_h x out_w grid as the decoder queries

class LearnedKeyPos(nn.Module):
    """Learned per-laser 2D positional embedding, added to decoder memory (the keys/values the
    queries cross-attend into) right before cross-attention -- so each of the 100 lasers (+ the
    cls token) carries an explicit, LEARNED identity, symmetric with the query side's learned
    query_seed + RoPE, instead of relying on whatever positional structure survived the encoder's
    own (fixed sinusoidal) RoPE. Only meaningful when memory IS the raw laser token sequence;
    once --memory-grid reshapes memory onto the output grid, "key i = laser row r col c" is no
    longer true, so build_decoder asserts the two are mutually exclusive."""
    def __init__(self, d_model, n_laser_rows, n_laser_cols):
        super().__init__()
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.grid_pos = nn.Parameter(torch.zeros(1, n_laser_rows * n_laser_cols, d_model))
        nn.init.trunc_normal_(self.cls_pos, std=0.02)
        nn.init.trunc_normal_(self.grid_pos, std=0.02)

    def forward(self, memory):
        B = memory.shape[0]
        return memory + torch.cat([self.cls_pos, self.grid_pos], dim=1).expand(B, -1, -1)  # (B,1+L,D)

def build_shared_x_freqs(dim: int, out_h: int, out_w: int, n_laser_rows: int, n_laser_cols: int, x_span: float, x_offset: float, theta: float = 10000.0):
    """Query (wall, x-z plane) and key (floor, x-y plane) RoPE tables that share the SAME freqs_x
    channels, so a query and laser with matching x get a zero relative rotation on those channels
    (see precompute_freqs_cis_shared_axis). Wall: rows=z (out_h, vertical), cols=x (out_w,
    horizontal) -- matches run.py's own out_h="height"/out_w="width" docstring. Floor: cols=x
    (n_laser_cols, shared with the wall's out_w), rows=y (n_laser_rows, floor depth) -- assumed
    convention (rows/cols come from camera ROI detection, not labeled by world axis in the data;
    flip x_span's axis here if that assumption turns out backwards).

    The floor's laser grid physically covers only PART of the wall's x-range (the laser doesn't
    span the box's full width) -- x_span/x_offset are a simple, approximate calibration for that
    (fractions of the wall's x-width the laser grid's x-axis covers / starts at), not derived from
    real box geometry yet. Meant as a baseline to later replace with an exact calibration (e.g.
    from box_geometry.json) once that's available."""
    # wall (query): row-major flatten matching precompute_freqs_cis_2d / the decoder's own .view(out_h,out_w,...)
    wall_z, wall_x = torch.meshgrid(torch.arange(out_h).float(), torch.arange(out_w).float(), indexing='ij')
    freqs_query = precompute_freqs_cis_shared_axis(dim, wall_x.reshape(-1), wall_z.reshape(-1), theta)  # (out_h*out_w,dim)

    # floor (key): row-major flatten matching laser_indices' r*n_laser_cols+c ordering
    laser_y, laser_col = torch.meshgrid(torch.arange(n_laser_rows).float(), torch.arange(n_laser_cols).float(), indexing='ij')
    x_scale = x_span * (out_w - 1)
    laser_x = x_offset * (out_w - 1) + laser_col.reshape(-1) / max(n_laser_cols - 1, 1) * x_scale  # laser col -> wall x units
    freqs_laser = precompute_freqs_cis_shared_axis(dim, laser_x, laser_y.reshape(-1), theta)  # (n_lasers,dim)
    cls_freqs = torch.cat([torch.ones(1, dim // 2), torch.zeros(1, dim // 2)], dim=-1)  # angle=0 identity rotation for cls
    freqs_key = torch.cat([cls_freqs, freqs_laser], dim=0)  # (1+n_lasers,dim)
    return freqs_query, freqs_key

class AttnDecoder(nn.Module):
    def __init__(self, d_model, out_h, out_w, num_heads:int=2, num_layers:int=2, do_rope:bool=True, out_c:int=1, ffn_dim:int|None=None, memory_grid:bool=False, key_pos:str='none', n_laser_rows:int|None=None, n_laser_cols:int|None=None, x_span:float=1/3, x_offset:float=0.0):
        super().__init__()
        self.out_h, self.out_w, self.out_c = out_h, out_w, out_c
        # v1: one seed shared by every position, so RoPE alone distinguished the queries
        # self.query_seed = nn.Parameter(torch.zeros(1, 1, d_model))
        # v2: one learned query per output position (DETR-style); ties the ckpt to this resolution
        self.query_seed = nn.Parameter(torch.zeros(1, out_h * out_w, d_model))
        nn.init.trunc_normal_(self.query_seed, std=0.02)
        assert key_pos == 'none' or not memory_grid, "--key-pos is incompatible with --memory-grid 1: reshaped memory is indexed by output grid cell, not laser position"
        self.shared_rope = key_pos == 'shared-rope'
        if self.shared_rope:
            freqs_query, freqs_key = build_shared_x_freqs(d_model, out_h, out_w, n_laser_rows, n_laser_cols, x_span, x_offset)
            self.register_buffer("freqs_query", freqs_query)  # (x,z) RoPE, sharing freqs_x with freqs_key
            self.register_buffer("freqs_key", freqs_key)      # (x,y) RoPE for memory (cls + lasers)
        else:
            self.register_buffer("freqs_query", precompute_freqs_cis_2d(d_model, out_h, out_w))  # 2D RoPE over the output grid
        self.memory_reshaper = MemoryReshaper(d_model, out_h, out_w, num_heads) if memory_grid else None
        self.key_pos = LearnedKeyPos(d_model, n_laser_rows, n_laser_cols) if key_pos == 'learned' else None
        # dropout=0.0: nn.TransformerDecoderLayer's own default is 0.1, silently active regardless
        # of this script's --freq-dropout/--laser-dropout (a different mechanism -- whole-patch/
        # whole-laser masking, not this). Explicit 0 matches --*-dropout 0's stated intent and
        # measured ~25% less activation memory (see perf/TODO.md).
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim or 4 * d_model, dropout=0.0, batch_first=True)
        self.layers = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_c)
        self.do_rope = do_rope

    def forward(self, memory, memory_key_padding_mask=None):
        # memory: (B,S,D) per-laser token sequence to cross-attend into (S = L+1, includes cls token)
        B = memory.shape[0]
        if self.memory_reshaper is not None:
            memory = self.memory_reshaper(memory, memory_key_padding_mask)  # (B,S,D) -> (B,out_h*out_w,D)
            memory_key_padding_mask = None  # reshaped memory is dense (one token per grid cell, none padded)
        elif self.key_pos is not None:
            memory = self.key_pos(memory)  # (B,S,D) -> (B,S,D), each laser + cls gets its learned identity
        elif self.shared_rope:
            memory = apply_rope(memory, self.freqs_key)  # (B,S,D) -> (B,S,D), (x,y) RoPE sharing freqs_x with queries
        queries = self.query_seed.expand(B, -1, -1)  # (1,out_h*out_w,D) -> (B,out_h*out_w,D)
        if self.do_rope or self.shared_rope: queries = apply_rope(queries, self.freqs_query)  # give each query its 2D grid position
        out = self.layers(queries, memory, memory_key_padding_mask=memory_key_padding_mask)  # (B,out_h*out_w,D)
        return self.head(out).view(B, self.out_h, self.out_w, self.out_c).squeeze(-1)

def flex_cross_attn(mha: nn.MultiheadAttention, query, key, value, bias=None, key_padding_mask=None):
    """Cross-attention using an nn.MultiheadAttention's own projection weights, but running the
    actual attention op through FlexAttention instead of mha.forward. Needed for MaskedAttnDecoder:
    its per-query confidence bias is a DENSE additive mask, and flash attention refuses ANY dense
    additive attn_mask outright (verified directly: "Flash Attention does not support non-null
    attn_mask") -- so mha.forward(..., attn_mask=mask) was silently falling back to PyTorch's
    O(N^2)-memory 'math' backend, materializing the full (B*H,num_queries,S) attention matrix.
    FlexAttention expresses the same bias as a fused score_mod instead, at flash-level O(N) memory
    -- numerically verified identical to the old dense-mask path (see perf/TODO.md). Requires
    torch.compile to actually fuse (flex_attention warns and materializes the same O(N^2) matrix
    uncompiled) -- fine here since --compile defaults to on."""
    B, Lq, D = query.shape
    Lk = key.shape[1]
    H = mha.num_heads
    Dh = D // H
    w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0) if mha.in_proj_bias is not None else (None, None, None)
    q = F.linear(query, w_q, b_q).view(B, Lq, H, Dh).transpose(1, 2)  # (B,H,Lq,Dh)
    k = F.linear(key, w_k, b_k).view(B, Lk, H, Dh).transpose(1, 2)    # (B,H,Lk,Dh)
    v = F.linear(value, w_v, b_v).view(B, Lk, H, Dh).transpose(1, 2)  # (B,H,Lk,Dh)

    def score_mod(score, b, h, q_idx, kv_idx):
        if bias is not None: score = score + bias[b, q_idx]
        if key_padding_mask is not None: score = torch.where(key_padding_mask[b, kv_idx], torch.finfo(score.dtype).min, score)
        return score

    out = _flex_attention_isolated(q, k, v, score_mod)  # (B,H,Lq,Dh)
    out = out.transpose(1, 2).reshape(B, Lq, D)
    return mha.out_proj(out)

class MaskedAttnDecoderLayer(nn.Module):
    """One Mask2Former-style decoder layer: cross-attn (gated by the running mask-confidence
    bias) -> self-attn -> FFN, each with a residual + post-norm, matching nn.TransformerDecoderLayer's
    norm_first=False convention so masked-attn is a drop-in swap for AttnDecoder's plain layers."""
    def __init__(self, d_model, num_heads, ffn_dim):
        super().__init__()
        self.num_heads = num_heads
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries, memory, attn_bias=None, memory_key_padding_mask=None):
        # attn_bias: (B,num_queries,S) additive bias on cross-attn logits -- the masked-attn gate,
        # uniform across S (broadcast from a per-query confidence), so only column 0 carries info.
        bias = attn_bias[:, :, 0] if attn_bias is not None else None  # (B,num_queries)
        cross_out = flex_cross_attn(self.cross_attn, queries, memory, memory, bias=bias, key_padding_mask=memory_key_padding_mask)
        x = self.norm1(queries + cross_out)
        x = self.norm2(x + self.self_attn(x, x, x, need_weights=False)[0])
        return self.norm3(x + self.ffn(x))

class MaskedAttnDecoder(nn.Module):
    """Mask2Former-style masked attention, adapted to this task's query/memory domain mismatch
    (queries live on the out_h x out_w output grid, memory is the laser token sequence -- no
    shared spatial index between them, unlike pixel decoder features and instance queries).
    Each layer predicts an intermediate per-query mask logit (sigmoid -> foreground confidence)
    and uses it as a per-query, memory-uniform additive bias on the NEXT layer's cross-attention:
    confident queries attend sharply (large bias magnitude drives near-hard masking at wherever
    they already peaked), unconfident queries stay diffuse. This is a self-mask / confidence-gate
    rather than the literal per-memory-token include/exclude mask2former uses on same-domain
    pixel features, since there is no natural per-laser-token "inside this query's mask" test here."""
    def __init__(self, d_model, out_h, out_w, num_heads:int=2, num_layers:int=2, do_rope:bool=True, out_c:int=1, ffn_dim:int|None=None, mask_temp:float=4.0, memory_grid:bool=False, key_pos:str='none', n_laser_rows:int|None=None, n_laser_cols:int|None=None, x_span:float=1/3, x_offset:float=0.0):
        super().__init__()
        self.out_h, self.out_w, self.out_c = out_h, out_w, out_c
        self.query_seed = nn.Parameter(torch.zeros(1, out_h * out_w, d_model))
        nn.init.trunc_normal_(self.query_seed, std=0.02)
        assert key_pos == 'none' or not memory_grid, "--key-pos is incompatible with --memory-grid 1: reshaped memory is indexed by output grid cell, not laser position"
        self.shared_rope = key_pos == 'shared-rope'
        if self.shared_rope:
            freqs_query, freqs_key = build_shared_x_freqs(d_model, out_h, out_w, n_laser_rows, n_laser_cols, x_span, x_offset)
            self.register_buffer("freqs_query", freqs_query)  # (x,z) RoPE, sharing freqs_x with freqs_key
            self.register_buffer("freqs_key", freqs_key)      # (x,y) RoPE for memory (cls + lasers)
        else:
            self.register_buffer("freqs_query", precompute_freqs_cis_2d(d_model, out_h, out_w))
        self.memory_reshaper = MemoryReshaper(d_model, out_h, out_w, num_heads) if memory_grid else None
        self.key_pos = LearnedKeyPos(d_model, n_laser_rows, n_laser_cols) if key_pos == 'learned' else None
        self.layers = nn.ModuleList([MaskedAttnDecoderLayer(d_model, num_heads, ffn_dim or 4 * d_model) for _ in range(num_layers)])
        self.mask_heads = nn.ModuleList([nn.Linear(d_model, out_c) for _ in range(num_layers)])  # per-layer intermediate mask head, final one IS the output head
        self.do_rope = do_rope
        self.mask_temp = mask_temp  # scales confidence -> attn-logit bias; higher = sharper gating

    def forward(self, memory, memory_key_padding_mask=None):
        # memory: (B,S,D) per-laser token sequence to cross-attend into (S = L+1, includes cls token)
        if self.memory_reshaper is not None:
            memory = self.memory_reshaper(memory, memory_key_padding_mask)  # (B,S,D) -> (B,out_h*out_w,D)
            memory_key_padding_mask = None  # reshaped memory is dense (one token per grid cell, none padded)
        elif self.key_pos is not None:
            memory = self.key_pos(memory)  # (B,S,D) -> (B,S,D), each laser + cls gets its learned identity
        elif self.shared_rope:
            memory = apply_rope(memory, self.freqs_key)  # (B,S,D) -> (B,S,D), (x,y) RoPE sharing freqs_x with queries
        B, S, _ = memory.shape
        queries = self.query_seed.expand(B, -1, -1)
        if self.do_rope or self.shared_rope: queries = apply_rope(queries, self.freqs_query)
        attn_bias = None
        for layer, mask_head in zip(self.layers, self.mask_heads):
            queries = layer(queries, memory, attn_bias=attn_bias, memory_key_padding_mask=memory_key_padding_mask)
            # confidence in [-1,1] via tanh(logit): centered gate, symmetric push for fg/bg queries.
            # Memory-uniform (broadcasts over S) since queries and memory share no spatial index --
            # this sharpens/relaxes each query's attention distribution rather than masking specific tokens.
            conf = torch.tanh(mask_head(queries)).mean(-1, keepdim=True)         # (B,out_h*out_w,1)
            attn_bias = (self.mask_temp * conf).expand(-1, -1, S)                # (B,out_h*out_w,S)
        return self.mask_heads[-1](queries).view(B, self.out_h, self.out_w, self.out_c).squeeze(-1)

def build_decoder(decoder, d_model, out_h, out_w, decoder_num_heads:int=2, decoder_num_layers:int=2, out_c:int=1, ffn_dim:int|None=None, mlp_dec_depth:int|None=None, mlp_dec_hidden:int|None=None, conv_dec_mult:float|None=None, conv_dec_res_blocks:int|None=None, mask_temp:float=4.0, memory_grid:bool=False, key_pos:str='none', n_laser_rows:int|None=None, n_laser_cols:int|None=None, x_span:float=1/3, x_offset:float=0.0):
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
    if decoder == 'attn': return AttnDecoder(d_model, out_h, out_w, num_heads=decoder_num_heads, num_layers=decoder_num_layers, out_c=out_c, ffn_dim=ffn_dim, memory_grid=memory_grid, key_pos=key_pos, n_laser_rows=n_laser_rows, n_laser_cols=n_laser_cols, x_span=x_span, x_offset=x_offset)
    if decoder == 'attn-no-rope': return AttnDecoder(d_model, out_h, out_w, num_heads=decoder_num_heads, num_layers=decoder_num_layers, do_rope=False, out_c=out_c, ffn_dim=ffn_dim, memory_grid=memory_grid, key_pos=key_pos, n_laser_rows=n_laser_rows, n_laser_cols=n_laser_cols, x_span=x_span, x_offset=x_offset)
    if decoder == 'masked-attn': return MaskedAttnDecoder(d_model, out_h, out_w, num_heads=decoder_num_heads, num_layers=decoder_num_layers, out_c=out_c, ffn_dim=ffn_dim, mask_temp=mask_temp, memory_grid=memory_grid, key_pos=key_pos, n_laser_rows=n_laser_rows, n_laser_cols=n_laser_cols, x_span=x_span, x_offset=x_offset)
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
        # dropout=0.0: see AttnDecoder's nn.TransformerDecoderLayer construction for why.
        self.layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim or 4 * d_model, dropout=0.0, batch_first=True), num_layers=num_layers)
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
    def __init__(self, d_model:int=128, pnt_num_heads:int=2, pnt_num_layers:int=2, seq_num_heads:int=2, seq_num_layers:int=2, data_info=None, decoder:str='mlp', decoder_num_heads:int=2, decoder_num_layers:int=2, freq_dropout:float=0.3, laser_dropout:float=0.3, loss_fn:str='mse', loss_alpha:float=0.5, loss_balanced:bool=False, count_loss_weight:float=0.0, ffn_dim:int|None=None, enc_ffn_dim:int|None=None, dec_ffn_dim:int|None=None, mlp_dec_depth:int|None=None, mlp_dec_hidden:int|None=None, conv_dec_mult:float|None=None, conv_dec_res_blocks:int|None=None, mask_temp:float=4.0, memory_grid:bool=False, key_pos:str='none', x_span:float=1/3, x_offset:float=0.0):
        super().__init__()

        # ffn_dim=None keeps the 4*d_model default. It is worth setting explicitly: torch's own
        # default is a FIXED 2048, so runs from before that was pinned to 4*d_model had a 2048-wide
        # FFN at any width -- 16x d_model at d_model=128, not 4x.
        # encoder FFN dimension: enc_ffn_dim overrides ffn_dim; ffn_dim is the fallback
        _enc_ffn_dim = enc_ffn_dim if enc_ffn_dim is not None else ffn_dim
        _dec_ffn_dim = dec_ffn_dim if dec_ffn_dim is not None else ffn_dim
        # encoder
        self.freq_encoder = FreqEncoder(data_info['patch_size'], d_model, pnt_num_heads, pnt_num_layers, data_info['n_freqs'], freq_dropout, n_channels=data_info.get('n_channels', 2), ffn_dim=_enc_ffn_dim)
        # dropout=0.0: see AttnDecoder's nn.TransformerDecoderLayer construction for why.
        self.laser_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=seq_num_heads, dim_feedforward=_enc_ffn_dim or 4 * d_model, dropout=0.0, batch_first=True), num_layers=seq_num_layers)
        self.laser_dropout = laser_dropout

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # Initialize to small random values
        self.register_buffer("freqs_laser", precompute_freqs_cis_2d(d_model, data_info['n_laser_rows'], data_info['n_laser_cols'])) # for laser grid

        # decoder
        out_c = data_info.get('out_c', 1)
        assert out_c == 1 or loss_fn in ('mse', 'ce-pixel'), f"{loss_fn} is mask-only; use mse or ce-pixel with an rgb target"
        self.decoder = build_decoder(decoder, d_model, data_info['out_h'], data_info['out_w'], decoder_num_heads, decoder_num_layers, out_c, ffn_dim=_dec_ffn_dim, mlp_dec_depth=mlp_dec_depth, mlp_dec_hidden=mlp_dec_hidden, conv_dec_mult=conv_dec_mult, conv_dec_res_blocks=conv_dec_res_blocks, mask_temp=mask_temp, memory_grid=memory_grid, key_pos=key_pos, n_laser_rows=data_info['n_laser_rows'], n_laser_cols=data_info['n_laser_cols'], x_span=x_span, x_offset=x_offset)

        # loss and metrics
        self.empty_head = nn.Linear(d_model, 1)  # extra "empty box" class for the spatial losses
        # a single Linear on cls, so count-acc answers exactly "is n_objects linearly decodable from cls"
        self.count_head = nn.Linear(d_model, N_COUNT_CLASSES)
        self.count_loss_weight = count_loss_weight
        self.loss_fn = LOSSES[loss_fn]
        self.is_spatial_loss = loss_fn.startswith('ce-spatial')
        self.is_asym_loss = loss_fn.endswith('-asym')
        self.loss_alpha = loss_alpha
        # 'balanced' is only implemented for ce-pixel-asym (see its comment) -- other -asym losses
        # ignore it, so only pass it through for that one to keep their call signature untouched.
        self.loss_balanced = loss_balanced and loss_fn == 'ce-pixel-asym'
        self.train_metrics = create_metrics(data_info, CHEAP_SEG_KEYS)  # cheap subset per step
        self.val_metrics = create_metrics(data_info)                   # full suite on eval loaders

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
        takes_memory = isinstance(self.decoder, (AttnDecoder, MaskedAttnDecoder))
        decoder_input = output if takes_memory else cls
        mask_logits = self.decoder(decoder_input, key_padding_mask) if takes_memory else self.decoder(decoder_input) # (B,L+1,D) or (B,D) -> (B,H,W)
        mask_pred = mask_logits.sigmoid()
        empty_logit = self.empty_head(cls)  # (B,D) -> (B,1), the "no cube anywhere" class
        count_logits = self.count_head(cls)  # (B,D) -> (B,n_classes), how many objects in the box
        return dict(mask_pred=mask_pred, mask_logits=mask_logits, empty_logit=empty_logit, count_logits=count_logits)

    def loss(self, outputs, batch):
        kw = dict(empty_logit=outputs['empty_logit']) if self.is_spatial_loss else {}
        if self.is_asym_loss: kw = dict(alpha=self.loss_alpha, balanced=self.loss_balanced) if self.loss_balanced else dict(alpha=self.loss_alpha)
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
        metric.update(outputs['mask_logits'], outputs['mask_pred'], batch['mask_true'], batch['info']['n_objects'])

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)
