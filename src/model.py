import argparse, math, os, functools, json
from datetime import datetime
import modal

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from huggingface_hub import snapshot_download
from scipy.signal import butter, sosfiltfilt
from composer import Trainer
from composer.models import ComposerModel
from torchmetrics.regression import MeanSquaredError
from composer.loggers import WandBLogger
from composer.utils.reproducibility import seed_all
from composer.callbacks import CheckpointSaver

from icecream import install
install()

from helpers import BestMetricCheckpointSaver, MaskVisualizationCallback

# **** Modal ****

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"})
    .uv_sync()
    .add_local_dir("src", remote_path="/root")
)

app = modal.App(
    image=image,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    )

# ***** Dataset *****

def clean_shifts(shifts:torch.Tensor, fs:int, lowcut:float=50.0, highcut:float|None=None) -> torch.Tensor:
    # bandpass filter
    if highcut is None: highcut = fs / 2 - 10
    shifts = shifts.numpy()  # (100, N_frames, 2)
    sos = butter(5, [lowcut, highcut], fs=fs, btype='band', output='sos')
    shifts = sosfiltfilt(sos, shifts, axis=1)
    # hann window smoothing
    window = np.hanning(shifts.shape[1])  # (N_frames,)
    shifts = shifts * window[np.newaxis, :, np.newaxis]
    return torch.from_numpy(shifts)

def do_fft(shifts:torch.Tensor, fs:int, min_freq:int=50, max_freq:int=1000):
    fft = torch.fft.rfft(shifts, dim=1)
    freqs = torch.fft.rfftfreq(shifts.shape[1], d=1.0 / fs)
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    fft, freqs = fft[:, mask, :], freqs[mask]
    return fft, freqs

# phase_sync
# for each frequency, rotates all lasers (both x and y) by the same angle so that laser_idx's x-channel has phase=0, i.e. imaginary part is zero and real part is positive
# This removes the absolute phase offset caused by speaker position (propagation delay), making signals comparable across speakers
# This preserves the relative phase differences between all lasers and between x and y channels
def phase_sync(fft, laser_idx=0, xy_idx=0, eps=1e-20):
    ref = fft[laser_idx, :, xy_idx]              # (F,) — x-channel of reference laser
    phase = ref.conj() / (ref.abs() + eps)      # (F,) — unit-phase rotators from x-channel
    fft_synced = fft * phase[None, :, None]     # (L, F, 2) — same rotation applied to x and y
    assert all(abs(v.item()) < 1e-5 for v in fft_synced[laser_idx, :, xy_idx].imag), "phase sync failed: reference laser x-channel is not real after rotation"
    return fft_synced                           # (L, F, 2)

# Global std normalization
# divides by a single scalar (std of all magnitudes), so freq magnitudes are distributed with unit std
# preserves spatial ratios between lasers (loud vs quiet lasers remain proportional)
def global_magnitude(fft, eps=1e-20): return fft / (fft.abs().std() + eps)

# Per-laser std normalization
# divides each laser by its own magnitude std (over frequencies), so freq magnitudes are distributed with unit std
# removes spatial ratios between lasers (a loud laser and a quiet laser look the same after)
def local_magnitude(fft, eps=1e-20): return fft / (fft.abs().std(dim=1, keepdim=True) + eps)

# Both phase sync and global magnitude normalization
def global_magnitude_and_phase_sync(fft, eps=1e-20): return global_magnitude(phase_sync(fft, eps=eps), eps=eps)

# Both phase sync and per-laser magnitude normalization
# removes absolute phase offsets and equalizes per-laser response strength, but destroys spatial amplitude ratios between lasers
def local_magnitude_and_phase_sync(fft, eps=1e-20): return local_magnitude(phase_sync(fft, eps=eps), eps=eps)

# # Strategy 3: Median-laser normalization — computes median per-laser mean magnitude as a robust
# # single scalar, preserving spatial ratios while being robust to outlier lasers.
# # fft: (L,F,2) complex
# def normalize_median(fft, eps=1e-20):
#     per_laser_mean = fft.abs().mean(dim=1)   # (L, 2)
#     s = per_laser_mean.median()              # scalar
#     return fft / (s + eps)

def hermit_poly(t):
    tt = t[None, :] ** torch.arange(4, device=t.device)[:, None]
    A = torch.tensor([[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]], dtype=t.dtype, device=t.device)
    return A @ tt

def interpolate(x, y, xs):
    # x: (n_points,), y: (B, n_points), xs: (n_xs,) -> (B, n_xs)
    m = (y[..., 1:] - y[..., :-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2, m[..., [-1]]], dim=-1)
    idxs = torch.searchsorted(x[1:], xs)
    dx = x[idxs + 1] - x[idxs]
    hh = hermit_poly((xs - x[idxs]) / dx)
    return hh[0] * y[..., idxs] + hh[1] * m[..., idxs] * dx + hh[2] * y[..., idxs + 1] + hh[3] * m[..., idxs + 1] * dx

def frequency_augmentation(shifts:torch.Tensor, fs:int, min_freq:int=100, max_freq:int=2500, generator:torch.Generator|None=None):
    # shifts: (B, n_lasers, n_timesteps, 2) — returns G: (B, n_timesteps)
    B = shifts.shape[0]
    x_points = torch.tensor([500, 1000, 1500, 2000, 2500], device=shifts.device)
    y_points = torch.normal(mean=1.0, std=1, size=(B, len(x_points)), generator=generator)
    domain = torch.linspace(min_freq, max_freq, 10000, device=shifts.device)
    values = interpolate(x_points, y_points, domain)                            # (B, 10000)
    lo, hi = values.min(dim=-1, keepdim=True).values, values.max(dim=-1, keepdim=True).values
    values = (values - lo) / (hi - lo) / 2.5 + 0.8                             # (B, 10000), range [0.8, 1.2]
    freq = torch.fft.fftfreq(shifts.shape[-2], 1/fs, device=shifts.device)
    valid = (freq >= min_freq) & (freq <= max_freq)
    G = torch.zeros(B, len(freq), dtype=torch.float32, device=shifts.device)
    G[:, valid] = values[:, torch.searchsorted(domain, freq[valid])]
    return G                                                                     # (B, n_timesteps)

class VibrationDataset(Dataset):
    """
    Downloads dataset from hf of (shift, mask) pairs
    Download shifts.safetensors and mask.safetensors to CPU via hf_hub_download
    Each __getitem__ reads only the pages for that tensor off disk — the OS never loads the full file into RAM.
    """

    def __init__(self, repo_id:str, split:str="train", disc_mask_h:int=40, disc_mask_w:int=20, patch_size:int=256, floor_cols:int=11, floor_rows:int=12, token:str|None=None, speakers:list|None=None, n_objects:list|None=None):
        self.ds = load_dataset(repo_id, split=split, token=token, columns=["sample_idx", "x_position", "y_position", "object", "fps", "speakers", "n_objects"], verification_mode="no_checks")
        if speakers is not None:
            if not isinstance(speakers[0], list): speakers = [speakers]
            self.ds = self.ds.filter(lambda row: row["speakers"] in speakers)
        if n_objects is not None:
            self.ds = self.ds.filter(lambda row: row["n_objects"] in n_objects)
        print(f"Loaded dataset with {len(self.ds)} samples after filtering for speakers={speakers}, n_objects={n_objects}")

        # load samples into RAM for fast access during training
        sample_patterns = [f"data/sample_{idx}.npz" for idx in self.ds["sample_idx"]]
        self.snapshot_dir = snapshot_download(repo_id, repo_type="dataset", allow_patterns=sample_patterns, token=token)
        def load_npz(sample_idx):
            d = np.load(os.path.join(self.snapshot_dir, f"data/sample_{sample_idx}.npz"))
            return torch.from_numpy(d['shifts'].copy()), torch.from_numpy(d['mask'].copy())
        _shift, _mask = load_npz(self.ds['sample_idx'][0])
        print(f"Sample shape: shifts={_shift.shape} ({_shift.dtype}), mask={_mask.shape} ({_mask.dtype})")
        shifts_bytes = len(self.ds) * _shift.numel() * _shift.element_size()
        masks_bytes = len(self.ds) * _mask.numel() * _mask.element_size()
        print(f"Dataset RAM estimate: shifts={shifts_bytes/1e9:.2f} GB, masks={masks_bytes/1e9:.2f} GB, total={( shifts_bytes+masks_bytes)/1e9:.2f} GB")
        self.shifts, self.masks = torch.empty(len(self.ds), *_shift.shape), torch.empty(len(self.ds), *_mask.shape)
        for i, idx in enumerate(self.ds["sample_idx"]): self.shifts[i], self.masks[i] = load_npz(idx)
        import psutil
        ram = psutil.virtual_memory()
        print(f"RAM after loading dataset: used={ram.used/1e9:.1f} GB, available={ram.available/1e9:.1f} GB, total={ram.total/1e9:.1f} GB")

        self.patch_size, self.disc_mask_h, self.disc_mask_w = patch_size, disc_mask_h, disc_mask_w
        self.floor_cols, self.floor_rows = floor_cols, floor_rows
        self.discretize_fn = F.adaptive_max_pool2d if HARD_MASK else F.adaptive_avg_pool2d

    def __repr__(self): return f"VibrationDataset(split={self.ds.split}, n={len(self.ds)})"
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        row = self.ds[idx]
        shifts, mask = self.shifts[idx], self.masks[idx]                                                                          # (n_lasers, n_timesteps, 2), (H, W)
        if DISCRETIZED_MASK: mask = self.discretize_fn(mask[None, None], (self.disc_mask_h, self.disc_mask_w)).squeeze()    # (H, W) -> (disc_mask_h, disc_mask_w)

        def round_to_floor(x, n): return min(int(x / n), n - 1)
        H, W = mask.shape[-2], mask.shape[-1]
        floor_x, floor_y = round_to_floor(row["x_position"], W / self.floor_cols), round_to_floor(row["y_position"], H / self.floor_rows)

        return {'shifts': shifts, 'mask': mask.float(), 'floor_x': floor_x, 'floor_y': floor_y, 'fps': row["fps"]}

def make_collate(patch_size:int, augment:bool=False, generator:torch.Generator|None=None, normalize:str|None=None):

    normalize_fn = {'global-magnitude': global_magnitude, 'local-magnitude': local_magnitude, 'phase-sync': phase_sync, 'global-magnitude-phase-sync': global_magnitude_and_phase_sync, 'local-magnitude-phase-sync': local_magnitude_and_phase_sync}.get(normalize, None)

    def collate(batch):

        fps = [b["fps"] for b in batch]
        floor_x = torch.tensor([b["floor_x"] for b in batch], dtype=torch.long)
        floor_y = torch.tensor([b["floor_y"] for b in batch], dtype=torch.long)
        shifts, mask = torch.stack([b["shifts"] for b in batch]), torch.stack([b["mask"] for b in batch]).float()

        if augment:
            assert len(set(fps)) == 1, f"all fps in batch must match, got {set(fps)}"
            G = frequency_augmentation(shifts, fps[0], generator=generator)       # (B, T)
            shifts_aug = torch.fft.ifft(torch.fft.fft(shifts, dim=2) * G[:, None, :, None], dim=2).real
            shifts, mask = torch.cat([shifts, shifts_aug]), torch.cat([mask, mask]) # (2B, L, T, 2)
            floor_x, floor_y, fps = torch.cat([floor_x, floor_x]), torch.cat([floor_y, floor_y]), fps + fps

        fft_patches = []
        for i in range(len(shifts)):
            shifts_clean = clean_shifts(shifts[i], fps[i])              # (L,T,2) -> (L,T,2)
            fft, _ = do_fft(shifts_clean, fps[i])                       # (L,T,2) -> (L,F,2)
            if normalize: fft = normalize_fn(fft)                       # (L,F,2) -> (L,F,2)
            fft_patches.append(fft.unfold(1, patch_size, patch_size))   # (L,F,2) -> (L,P,2,PS)

        return torch.stack(fft_patches), mask, floor_x, floor_y
    return collate

def get_dataloaders(repo_id:str, patch_size:int=256, disc_mask_h:int=40, disc_mask_w:int=20, floor_cols:int=11, floor_rows:int=12, batch_size:int=8, eval_batch_size:int=16, shuffle:bool=True, num_workers:int=0, seed:int=42, token:str | None = None, test_split=0.2, speakers:list|None=None, normalize:str|None=None, n_objects:list|None=None):
    dataset = VibrationDataset(repo_id, patch_size=patch_size, disc_mask_h=disc_mask_h, disc_mask_w=disc_mask_w, floor_cols=floor_cols, floor_rows=floor_rows, token=token, speakers=speakers, n_objects=n_objects)
    test_size = int(len(dataset) * test_split)
    print(f'{len(dataset)-test_size} train samples\n{test_size} test samples')
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [len(dataset) - test_size, test_size], generator=generator)
    train_collate_fn, test_collate_fn = make_collate(patch_size, augment=AUGMENT, generator=generator, normalize=normalize), make_collate(patch_size, augment=False, normalize=normalize)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True, collate_fn=train_collate_fn)
    test_loader  = DataLoader(test_set,  batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, generator=generator, pin_memory=True, collate_fn=test_collate_fn)
    fft_patches = test_collate_fn([dataset[0]])[0][0]   # (n_lasers, n_patches, 2, patch_size)
    mask = dataset.masks[0]                             # (orig_h, orig_w)
    data_info = {'floor_cols': floor_cols, 'floor_rows': floor_rows,
                 'mask_h': mask.shape[0], 'mask_w': mask.shape[1], 'disc_mask_h': disc_mask_h, 'disc_mask_w': disc_mask_w,
                 'num_lasers': fft_patches.shape[0], 'n_freqs': fft_patches.shape[1] * patch_size, 'n_patches': fft_patches.shape[1]}
    return train_loader, test_loader, data_info

# ***** metrics *****

def create_metrics(nx: int, ny: int):
    return {"mse": MeanSquaredError()}

# **** loss ****

def soft_dice_fn(logits, targets, axis=None):
    """Computes Soft Dice for the full mask, or just along X or Y directions.
    After collapsing along an axis, normalizes by profile length so the result
    is resolution-independent across different disc_mask_h / disc_mask_w."""
    probs = logits.sigmoid()
    if axis is not None: probs, targets = probs.mean(dim=axis), targets.mean(dim=axis)
    spatial_dims = tuple(range(1, probs.ndim))  # (-2, -1) for dice over entire mask or (-1,) for dice on just x,y profiles
    intersection = (probs * targets).sum(dim=spatial_dims)
    total_sum = probs.sum(dim=spatial_dims) + targets.sum(dim=spatial_dims)
    dice_score = (2 * intersection) / total_sum.clamp(min=1e-6)
    return 1 - dice_score.mean()

def soft_iou_fn(mask_logits, target):
    """Soft (differentiable) IoU — resolution-independent ratio metric."""
    probs = mask_logits.sigmoid()
    intersection = (probs * target).sum(dim=(-2, -1))
    union = (probs + target - probs * target).sum(dim=(-2, -1))
    return 1 - (intersection / union.clamp(min=1e-6)).mean()

def weighted_cross_entropy_fn(mask_logits, target):
    """Weighted BCE where pos_weight is based on pixel fraction, not raw counts,
    so it stays stable across different disc_mask_h / disc_mask_w."""
    n_pixels = target[0].numel()
    pos_frac = target.sum(dim=(-2, -1)).mean() / n_pixels   # mean positive fraction across batch
    neg_frac = 1.0 - pos_frac
    pos_weight = (neg_frac / pos_frac.clamp(min=1e-6)).clamp(max=100.0)
    return F.binary_cross_entropy_with_logits(mask_logits, target, pos_weight=pos_weight)

def focal_loss_fn(mask_logits, target, focal_gamma=2.0):
    """Focal loss: down-weights easy examples via (1-p)^gamma so training focuses on hard pixels.
    gamma=2 is the standard default. gamma=10 crushes gradients to ~0.1% of BCE at init (pt=0.5)."""
    bce = F.binary_cross_entropy_with_logits(mask_logits, target, reduction='none')
    p = mask_logits.sigmoid()
    pt = p * target + (1 - p) * (1 - target)  # prob of correct class per pixel
    return (((1 - pt) ** focal_gamma) * bce).mean()

def asymmetric_focal_loss_fn(mask_logits, target, gamma_neg=4, gamma_pos=0):
    """Asymmetric focal loss: suppresses easy negatives (gamma_neg) without crushing positive gradients.
    gamma_pos=0 means object pixels always get full BCE gradient regardless of confidence.
    gamma_neg=4 aggressively suppresses background pixels the model is already confident about."""
    bce = F.binary_cross_entropy_with_logits(mask_logits, target, reduction='none')
    p = mask_logits.sigmoid()
    weight = target * (1 - p) ** gamma_pos + (1 - target) * p ** gamma_neg
    return (weight * bce).mean()

def boundary_loss_fn(mask_logits, target, kernel_size=3):
    """Boundary loss: weighted BCE that focuses on object edges.
    Extracts boundaries via morphological gradient (dilation - erosion) on the target,
    then doubles the loss weight at those pixels so the model sharpens predicted edges."""
    pad = kernel_size // 2
    dilated  =  F.max_pool2d( target.unsqueeze(1), kernel_size, stride=1, padding=pad).squeeze(1)
    eroded   = -F.max_pool2d(-target.unsqueeze(1), kernel_size, stride=1, padding=pad).squeeze(1)
    boundary = (dilated - eroded).clamp(0, 1)   # (B, H, W) — 1 at edges, 0 elsewhere
    weight   = 1.0 + boundary                    # boundary pixels get 2× weight
    return F.binary_cross_entropy_with_logits(mask_logits, target, weight=weight)

def mse_loss_fn(mask_logits, target):
    """MSE on probabilities (after sigmoid). Averaged over all elements (B, H, W) so it is the percent
    of total pixels that are wrong, making it stable across different disc_mask_h / disc_mask_w."""
    return F.mse_loss(mask_logits.sigmoid(), target)

def tversky_loss_fn(mask_logits, target, alpha=0.3, beta=0.7):
    """Tversky loss: generalization of Dice with asymmetric FP/FN weighting.
    alpha weights FP, beta weights FN. Set beta > alpha to penalize false negatives more.
    Reduces to Dice when alpha=beta=0.5."""
    probs = mask_logits.sigmoid()
    spatial_dims = tuple(range(1, probs.ndim))
    tp = (probs * target).sum(dim=spatial_dims)
    fp = (probs * (1 - target)).sum(dim=spatial_dims)
    fn = ((1 - probs) * target).sum(dim=spatial_dims)
    tversky_score = tp / (tp + alpha * fp + beta * fn).clamp(min=1e-6)
    return 1 - tversky_score.mean()

LOSSES = {'ce': F.cross_entropy, 'wce': weighted_cross_entropy_fn, 'focal': focal_loss_fn, 'asym_focal': asymmetric_focal_loss_fn, 'mse': mse_loss_fn, 'dice': soft_dice_fn, 'tversky': tversky_loss_fn, 'boundary': boundary_loss_fn}

# **** Decoders ****

class MLPDecoder(nn.Module):
    """Baseline: flat MLP from CLS token only."""
    def __init__(self, d_model, out_h, out_w):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, out_h * out_w))
        self.out_h, self.out_w = out_h, out_w
    def forward(self, cls, laser_feats):
        return self.net(cls).view(-1, self.out_h, self.out_w)


class CNNDecoder(nn.Module):
    """Reshape laser tokens to 10x10 grid, upsample to out_h x out_w via transposed convs."""
    def __init__(self, d_model, out_h, out_w):
        super().__init__()
        # 10x10 -> 20x20 -> out_h x out_w  (assumes out_h=40, out_w=20)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(d_model, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(),
        )
        self.cls_proj = nn.Linear(d_model, 256)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(4, 64), nn.GELU(),
        )
        self.head = nn.Conv2d(64, 1, 1)
        self.out_h, self.out_w = out_h, out_w

    def forward(self, cls, laser_feats):
        B = cls.shape[0]
        x = laser_feats.permute(0, 2, 1).reshape(B, -1, 10, 10)  # (B, D, 10, 10)
        x = self.up1(x)                                            # (B, 256, 20, 20)
        x = x + self.cls_proj(cls).reshape(B, 256, 1, 1)          # add CLS context
        x = self.up2(x)                                            # (B, 64, 40, 20)
        return self.head(x).squeeze(1)                             # (B, 40, 20)


class CrossAttnDecoder(nn.Module):
    """Each output pixel queries the laser tokens via stacked cross+self attention."""
    def __init__(self, d_model, out_h, out_w, n_layers=4):
        super().__init__()
        self.mask_queries = nn.Parameter(torch.zeros(1, out_h * out_w, d_model))
        nn.init.trunc_normal_(self.mask_queries, std=0.02)
        self.register_buffer('mask_pos', precompute_freqs_cis_2d(d_model, out_h, out_w))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=0.1),
                'cross_norm': nn.LayerNorm(d_model),
                'self_attn':  nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=0.1),
                'self_norm':  nn.LayerNorm(d_model),
                'ffn':        nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)),
                'ffn_norm':   nn.LayerNorm(d_model),
            })
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1)
        self.out_h, self.out_w = out_h, out_w

    def forward(self, cls, laser_feats):
        B = laser_feats.shape[0]
        q = apply_rope(self.mask_queries.expand(B, -1, -1), self.mask_pos)  # (B, H*W, D)
        for layer in self.layers:
            q = layer['cross_norm'](q + layer['cross_attn'](q, laser_feats, laser_feats)[0])
            q = layer['self_norm'](q + layer['self_attn'](q, q, q)[0])
            q = layer['ffn_norm'](q + layer['ffn'](q))
        return self.head(q).squeeze(-1).view(B, self.out_h, self.out_w)


class PoolDecoder(nn.Module):
    """Multi-scale attention pooling of laser tokens + bilinear upsample + CNN refinement."""
    def __init__(self, d_model, out_h, out_w):
        super().__init__()
        self.pool_25 = nn.Linear(d_model, 25)
        self.pool_4  = nn.Linear(d_model, 4)
        self.to_base = nn.Sequential(nn.Linear(4 * d_model, 5 * 4 * 256), nn.GELU())
        self.refine  = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(16, 256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 1, 1),
        )
        self.out_h, self.out_w = out_h, out_w

    def forward(self, cls, laser_feats):
        B = laser_feats.shape[0]
        global_avg = laser_feats.mean(1)                                                             # (B, D)
        pool25 = (F.softmax(self.pool_25(laser_feats), dim=1).transpose(1, 2) @ laser_feats).mean(1) # (B, D)
        pool4  = (F.softmax(self.pool_4(laser_feats),  dim=1).transpose(1, 2) @ laser_feats).mean(1) # (B, D)
        combined = torch.cat([cls, global_avg, pool25, pool4], dim=-1)                              # (B, 4D)
        x = self.to_base(combined).view(B, 256, 5, 4)                                               # (B, 256, 5, 4)
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode='bilinear', align_corners=False)   # (B, 256, H, W)
        return self.refine(x).squeeze(1)                                                             # (B, H, W)

def build_decoder(decoder, d_model, out_h, out_w, cross_attn_layers=4):
    if decoder == 'mlp':        return MLPDecoder(d_model, out_h, out_w)
    elif decoder == 'cnn':      return CNNDecoder(d_model, out_h, out_w)
    elif decoder == 'cross_attn': return CrossAttnDecoder(d_model, out_h, out_w, cross_attn_layers)
    elif decoder == 'pool':     return PoolDecoder(d_model, out_h, out_w)
    else: raise ValueError(f"Unknown decoder: {decoder}")

# **** model ****

def precompute_freqs_cis(dim:int, end:int, theta:float=10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)] / dim))
    freqs = torch.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

def precompute_freqs_cis_2d(dim:int, h:int, w:int, theta:float = 10000.0) -> torch.Tensor:
    freqs_h, freqs_w = precompute_freqs_cis(dim // 2, h, theta), precompute_freqs_cis(dim // 2, w, theta)
    freqs_h, freqs_w = freqs_h.reshape(h, 1, -1).repeat(1, w, 1), freqs_w.reshape(1, w, -1).repeat(h, 1, 1)
    return torch.cat([freqs_h, freqs_w], dim=-1).reshape(h * w, dim)

def apply_rope(x:torch.Tensor, freqs_cis:torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    shp = [1]*(x.ndim-2) + [x.shape[1], -1] # works with 1D + 2D rope
    cos, sin = freqs_cis.reshape(*shp).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(dim, hidden_dim)  # Learnable embeddings
    def forward(self, x): return x+ self.embed(torch.arange(x.shape[1], device=x.device).unsqueeze(0))

class PointTransformer(nn.Module):
    def __init__(self, patch_size, d_model, num_heads, num_layers, signal_length, signal_is):
        super().__init__()
        self.embed = nn.Linear(2 * patch_size, d_model)
        self.layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True), num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if ROPE:
            self.register_buffer("freqs_cis", precompute_freqs_cis(d_model, signal_length//patch_size))
        else:
            self.freq_pos_embd = LearnablePositionalEncoding(signal_length // patch_size, d_model)
        self.raw_to_tokens = {'magnitude': lambda t: t.abs(), 'complex': lambda t: torch.cat([t.real, t.imag], dim=-1), 'mag_phase': lambda t: torch.cat([t.abs(), t.angle()], dim=-1)}[signal_is]

    def forward(self, x):
        # x.shape = (B_L,P,C,_PS) = (batch_size * n_lasers, n_patches, n_coords, patch_size)
        B_L, P, _, _ = x.shape
        x = self.raw_to_tokens(x).float()                                                   # (B_L,P,C,_PS) -> (B_L,P,C,PS) where PS=_PS or 2*_PS
        x = self.embed(x.reshape(B_L, P, -1))                                               # (B_L,P,C,PS) -> (B_L,P,D)
        x = apply_rope(x, self.freqs_cis.to(x.device)) if ROPE else self.freq_pos_embd(x)   # (B_L,P,D)
        x = torch.cat((self.cls_token.expand(B_L, -1, -1).to(x.device), x), dim=1)          # (B_L,P+1,D)
        output = self.layers(x)                                                             # (B_L,P+1,D)
        return output[:, 0, :]                                                              # (B_L,D)

class SignalTransformer(ComposerModel):
    def __init__(self, d_model, pnt_num_heads, pnt_num_layers, seq_num_heads, seq_num_layers, patch_size, signal_is, data_info, alpha, beta, loss, gamma=0.5, delta=0.5, decoder='mlp', cross_attn_layers=4):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.floor_cols, self.floor_rows = data_info['floor_cols'], data_info['floor_rows']
        laser_rows, laser_cols = int(math.sqrt(data_info['num_lasers'])), int(math.sqrt(data_info['num_lasers']))
        self.out_h, self.out_w = (data_info['disc_mask_h'], data_info['disc_mask_w']) if DISCRETIZED_MASK else (data_info['mask_h'], data_info['mask_w'])
        self.pnt_trans = PointTransformer(patch_size, d_model, pnt_num_heads, pnt_num_layers, data_info['n_freqs'], signal_is)
        self.seq_trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=seq_num_heads, batch_first=True), num_layers=seq_num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=.02)  # Initialize to small random values

        # positional embeddings for laser grid
        if ROPE:
            self.register_buffer("freqs_laser", precompute_freqs_cis_2d(d_model, laser_rows, laser_cols))
        else:
            self.laser_pos_embd = LearnablePositionalEncoding(laser_rows * laser_cols, d_model)

        loss_fn = LOSSES[loss]
        if loss == 'tversky':
            loss_fn = functools.partial(loss_fn, alpha=alpha, beta=beta)
        elif loss == 'focal':
            loss_fn = functools.partial(loss_fn, focal_gamma=gamma)
        elif loss == 'asym_focal':
            loss_fn = functools.partial(loss_fn, gamma_neg=gamma, gamma_pos=delta)
        self.loss_fn = loss_fn

        # Prediction heads
        self.mlp_head_floor_x = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, self.floor_cols))
        self.mlp_head_floor_y = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, self.floor_rows))
        self.decoder = build_decoder(decoder, d_model, self.out_h, self.out_w, cross_attn_layers)

        # SORD loss cost matrix
        self.register_buffer('cost_matrix_x', self._init_cost_matrix(self.floor_cols))
        self.register_buffer('cost_matrix_y', self._init_cost_matrix(self.floor_rows))

        # Metrics
        self.train_metrics, self.val_metrics = create_metrics(self.floor_cols, self.floor_rows), create_metrics(self.floor_cols, self.floor_rows)

    def _init_cost_matrix(self, num_classes, multiplier=0.5):
        indices = torch.arange(num_classes)
        return multiplier * (indices.unsqueeze(1) - indices.unsqueeze(0)).abs() ** 2

    def forward(self, batch):
        # B=batch size, L=n_lasers, C=n_coordinates=2, PS=patch size, D=d_model
        x, *_ = batch
        B, L,_, _, _ = x.shape

        # PointTransformer learns patterns between all frequencies from a single laser
        # flatten so PointTransformer processes all lasers AND all batches in parallel
        x = self.pnt_trans(x.flatten(0, 1)).reshape(B, L, -1)                                   # (B,L,P,C,PS) -> (B,L,D)

        # SequenceTransformer learns patterns between all lasers in the the laser grid
        x = apply_rope(x, self.freqs_laser.to(x.device)) if ROPE else self.laser_pos_embd(x)    # (B,L,D) -> (B,L,D)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)                             # (B,L,D) (1,1,D) -> (B,L+1,D)
        output = self.seq_trans(x)                                                              # (B,L+1,D) -> (B,L+1,D)
        cls_embedding = output[:, 0, :]                                                         # (B,D)

        # Predict x position, y position, and segmentation mask
        laser_feats = output[:, 1:, :]                              # (B, L, D)
        x_logits = self.mlp_head_floor_x(cls_embedding)
        y_logits = self.mlp_head_floor_y(cls_embedding)
        mask_logits = self.decoder(cls_embedding, laser_feats)
        return x_logits, y_logits, cls_embedding, mask_logits

    def loss(self, outputs, batch):
        _, mask_true, floor_x_true, floor_y_true = batch
        x_logits, y_logits, _, mask_logits = outputs
        return self.loss_fn(mask_logits, mask_true)

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch, outputs, metric):
        _, mask_true, floor_x_true, floor_y_true = batch
        x_logits, y_logits, _, mask_logits = outputs
        metric_name, pred_type, pos = getattr(metric, 'metric_name', None), getattr(metric, 'pred_type', None), getattr(metric, 'pos', None)


    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

# ***** main *****

def count_parameters(model: nn.Module) -> int: return sum([p_.numel() for p_ in model.parameters()])

def get_parser():
    parser = argparse.ArgumentParser()

    # abalations
    parser.add_argument('--sord', type=int, default=0)
    parser.add_argument('--mask-loss', type=int, default=1)
    parser.add_argument('--position-loss', type=int, default=1)
    parser.add_argument('--discretized-mask', type=int, default=1)
    parser.add_argument('--rope', type=int, default=1)
    parser.add_argument('--focal', type=int, default=0)
    parser.add_argument('--augment', type=int, default=1)
    parser.add_argument('--hard-mask', type=int, default=0)
    parser.add_argument('--normalize', type=str, default=None, choices=['global-magnitude', 'local-magnitude', 'phase-sync', 'global-magnitude-phase-sync', 'local-magnitude-phase-sync'])
    parser.add_argument('--loss', type=str, default='dice', choices=list(LOSSES.keys()), help='Loss functions')

    # data
    parser.add_argument('--data-dir', type=str, default='eturok-weizmann/vibrations')
    parser.add_argument('--signal-is', type=str, default='magnitude')
    parser.add_argument('--speakers', type=str, default=None, help='JSON list of speakers to include, e.g. \'[[0,1,0,0],[1,0,0,0]]\'')
    parser.add_argument('--n-objects', type=int, nargs='+', default=[1], help='List of n_objects values to include, e.g. --n-objects 1 2')

    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--disc-mask-h', type=int, default=40)
    parser.add_argument('--disc-mask-w', type=int, default=20)

    # model arch
    parser.add_argument('--decoder', type=str, default='mlp', choices=['mlp', 'cnn', 'cross_attn', 'pool'])
    parser.add_argument('--cross-attn-layers', type=int, default=4)

    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--pnt-num-heads', type=int, default=2)
    parser.add_argument('--seq-num-heads', type=int, default=2)
    parser.add_argument('--pnt-num-layers', type=int, default=2)
    parser.add_argument('--seq-num-layers', type=int, default=2)

    # training
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)

    # evaluation
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--max-duration', type=str, default='10_000ep')
    parser.add_argument('--eval-interval', type=str, default='10ep')

    # logging
    parser.add_argument('--mask-viz-train-interval', type=int, default=10)
    parser.add_argument('--mask-viz-thresholds', type=str, default='0.3,0.5,0.7,0.9')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--best-metric', type=str, default='mse', help='Eval metric key to gate best-checkpoint saving on')
    parser.add_argument('--best-metric-higher-is-better', action='store_true', default=False, help='Set if larger metric values are better (e.g. accuracy)')
    parser.add_argument('--checkpoint-interval', type=str, default='100ep', help='Interval for saving checkpoints to HF; see https://modal.com/docs/guide/checkpoints#checkpoint-intervals for supported formats')

    # loss
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--delta', type=float, default=0.4)
    return parser

@app.function(
    gpu="A100",
    timeout=86_400, # maximum timeout is 24 hours or 86_400 seconds; see https://modal.com/docs/guide/timeouts#timeouts
    retries=3,
    )
def train(**kwargs):
    args = get_parser().parse_args([])  # get defaults
    args.__dict__.update(kwargs)        # apply overrides
    global SORD, MASK, POSITION, ROPE, DISCRETIZED_MASK, FOCAL, AUGMENT, HARD_MASK # environment variables
    SORD, MASK, POSITION, ROPE, DISCRETIZED_MASK, FOCAL, AUGMENT, HARD_MASK = args.sord, args.mask_loss, args.position_loss, args.rope, args.discretized_mask, args.focal, args.augment, args.hard_mask

    run_id = f"{args.run_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    seed_all(args.seed) # must seed before initializing model + dataloader
    speakers = json.loads(args.speakers) if args.speakers else None
    n_objects = args.n_objects
    train_loader, test_loader, data_info = get_dataloaders(args.data_dir, args.patch_size, args.disc_mask_h, args.disc_mask_w, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, seed=args.seed, speakers=speakers, normalize=args.normalize, n_objects=n_objects)
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    model = SignalTransformer(args.d_model, args.pnt_num_heads, args.pnt_num_layers, args.seq_num_heads, args.seq_num_layers, args.patch_size, args.signal_is, data_info, args.alpha, args.beta, args.loss, gamma=args.gamma, delta=args.delta, decoder=args.decoder, cross_attn_layers=args.cross_attn_layers)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    config = {'n_params': count_parameters(model), **data_info, 'delta': args.delta, 'SORD': SORD, 'MASK': MASK, 'POSITION': POSITION, 'data_dir': args.data_dir, 'seed': args.seed, 'signal_is': args.signal_is, 'd_model': args.d_model, 'pnt_num_heads': args.pnt_num_heads, 'seq_num_heads': args.seq_num_heads, 'pnt_num_layers': args.pnt_num_layers, 'seq_num_layers': args.seq_num_layers, 'patch_size': args.patch_size, 'batch_size': args.batch_size, 'eval_batch_size': args.eval_batch_size, 'lr': args.lr, 'alpha': args.alpha, 'beta': args.beta, 'gamma': args.gamma, 'max_duration': args.max_duration, 'eval_interval': args.eval_interval, 'decoder': args.decoder, 'cross_attn_layers': args.cross_attn_layers}
    logger = WandBLogger('good-vibrations', group='loss', name=run_id, init_kwargs={'config': config, 'save_code': True})
    resume_saver = CheckpointSaver(folder=f'hf://{args.data_dir}/checkpoints/{run_id}', save_interval=args.checkpoint_interval, num_checkpoints_to_keep=1, overwrite=True)
    best_saver = BestMetricCheckpointSaver(metric_name=args.best_metric, higher_is_better=args.best_metric_higher_is_better, folder=f'hf://{args.data_dir}/checkpoints/{run_id}/best', save_interval=args.eval_interval, num_checkpoints_to_keep=1, overwrite=True)
    mask_viz = MaskVisualizationCallback(n_samples=args.eval_batch_size, save_dir="visualizations", train_viz_interval=args.mask_viz_train_interval)
    ic(config)

    trainer = Trainer(
        run_name=run_id, model=model, train_dataloader=train_loader, eval_dataloader=test_loader,
        max_duration=args.max_duration, eval_interval=args.eval_interval,
        optimizers=optimizer, device=device, seed=args.seed,
        loggers=logger, log_to_console=True, auto_log_hparams=True, save_metrics=True,
        callbacks=[mask_viz, resume_saver, best_saver],
    )

    trainer.fit()
    ic(trainer.state.train_metrics, trainer.state.eval_metrics)
    trainer.close()

@app.local_entrypoint()
def main(*args):
    train.remote(**vars(get_parser().parse_args(args)))  # runs on Modal GPU

if __name__ == '__main__':
    train.local(**vars(get_parser().parse_args()))
