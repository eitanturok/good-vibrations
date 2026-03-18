import argparse, math
import modal

import torch
import wandb
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from scipy.signal import butter, sosfiltfilt
from composer import Trainer
from composer.models import ComposerModel
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import MulticlassAccuracy, BinaryJaccardIndex
from torchmetrics.regression import MeanSquaredError
from composer.loggers import WandBLogger
from composer.utils.reproducibility import seed_all

from icecream import install
install()

from helpers import getenv, HFChkptUploader, MaskVisualizationCallback

# **** Modal ****

image = modal.Image.debian_slim().apt_install("git").uv_sync().add_local_dir("src", remote_path="/root")
app = modal.App(image=image)

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

def hermit_poly(t):
    tt = t[None, :] ** torch.arange(4, device=t.device)[:, None]
    A = torch.tensor([[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]], dtype=t.dtype, device=t.device)
    return A @ tt

def interpolate(x, y, xs):
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[1:], xs)
    dx = (x[idxs + 1] - x[idxs])
    hh = hermit_poly((xs - x[idxs]) / dx)
    return hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx

def frequency_augmentation(shifts:torch.Tensor, fs:int, min_freq:int=100, max_freq:int=2500):
    # Initial points and domain
    x_points = torch.tensor([500, 1000, 1500, 2000, 2500], device=shifts.device)
    y_points = torch.normal(mean=1.0, std=1, size=(len(x_points),), device=shifts.device)
    domain = torch.linspace(min_freq, max_freq, 10000, device=shifts.device)  # TODO: arguments of F^{sample} the FIXED frequency domain

    # Interpolate values over the domain
    values = interpolate(x_points, y_points, domain)

    # Normalize values
    values = (values - torch.min(values)) / (torch.max(values) - torch.min(values))

    # Spline between 0.8 to 1.2
    normalized_values = values / 2.5 + 0.8

    # Frequency range for the FFT
    f = torch.fft.fftfreq(10200, 1 / 5100, device=shifts.device)  # TODO: arguments of F^{sample} the FIXED frequency domain

    # Filter frequencies in the desired range and assign values
    valid_freq_mask = (f >= min_freq) & (f <= max_freq)
    G = torch.zeros_like(f, dtype=torch.float32, device=shifts.device)
    G[valid_freq_mask] = normalized_values[torch.searchsorted(domain, f[valid_freq_mask])]
    return G

class VibrationDataset(torch.utils.data.Dataset):
    """
    Downloads shifts.safetensors once via hf_hub_download (cached to disk after first run),
    then memory-maps it with safe_open. Each __getitem__ reads only the pages for that
    tensor off disk — the OS never loads the full file into RAM.
    """

    def __init__(self, repo_id:str, split:str="train", mask_h:int=40, mask_w:int=20, patch_size:int=256, token:str|None=None):
        self.ds = load_dataset(repo_id, split=split, token=token, columns=["shifts_idx", "mask_idx", "x_position", "y_position", "object", "fps"])
        self.st_shifts = safe_open(hf_hub_download(repo_id, "shifts.safetensors", repo_type="dataset", token=token), framework="pt", device="cpu")
        self.st_masks = safe_open(hf_hub_download(repo_id, "masks.safetensors",  repo_type="dataset", token=token),  framework="pt", device="cpu")
        self.patch_size, self.mask_h, self.mask_w = patch_size, mask_h, mask_w
        # remap raw position values to 0-indexed class labels
        self.x_labels = torch.tensor(self.ds["x_position"]).unique(return_inverse=True)[1]
        self.y_labels = torch.tensor(self.ds["y_position"]).unique(return_inverse=True)[1]
    def __repr__(self): return f"VibrationDataset(split={self.ds.split}, n={len(self.ds)})"
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        row = self.ds[idx]

        # discretize the mask
        mask = self.st_masks.get_tensor(f"mask_{row['mask_idx']}")                                      # (H, W) bool
        if DISCRETIZED_MASK: mask = F.adaptive_avg_pool2d(mask.float()[None, None], (self.mask_h, self.mask_w)).squeeze()    # (mask_h, mask_w)

        # clean + fft the laser shifts
        shifts = self.st_shifts.get_tensor(f"shifts_{row['shifts_idx']}")       # (n_lasers, n_timesteps, 2)
        shifts = clean_shifts(shifts, row["fps"])                               # (n_lasers, n_timesteps, 2)
        fft, _ = do_fft(shifts, row["fps"])                                     # (n_lasers, n_freqs, 2)
        fft_patches = fft.unfold(1, self.patch_size, self.patch_size).float()   # (n_lasers, n_freqs, 2) -> (n_lasers, n_patches, 2, patch_size)

        return fft_patches, (mask, self.x_labels[idx], self.y_labels[idx])

def get_dataloaders(repo_id:str, patch_size:int=256, mask_h:int=40, mask_w:int=20, batch_size:int=8, eval_batch_size:int=16, shuffle:bool=True, num_workers:int=0, seed:int=42, token:str | None = None):
    train_set = VibrationDataset(repo_id, split="train", patch_size=patch_size, mask_h=mask_h, mask_w=mask_w, token=token)
    test_set = VibrationDataset(repo_id, split="test", patch_size=patch_size, mask_h=mask_h, mask_w=mask_w, token=token)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size,     shuffle=shuffle,  num_workers=num_workers, generator=generator, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=eval_batch_size, shuffle=False,   num_workers=num_workers, generator=generator, pin_memory=True)
    num_x_positions = int(train_set.x_labels.max()) + 1
    num_y_positions = int(train_set.y_labels.max()) + 1
    return train_loader, test_loader, num_x_positions, num_y_positions

# ***** metrics *****

def _tag(m, metric_name:str, pred_type:str, pos:str|None=None):
    m.metric_name = metric_name
    m.pred_type = pred_type     # 'x', 'y', or 'mask'
    m.pos = pos                 # 'x', 'y', or None (for profile axis)
    return m

def create_metrics(nx: int, ny: int):
    return {
        "pos/x_Acc":        _tag(MulticlassAccuracy(num_classes=nx), 'Accuracy', 'x'),
        "pos/y_Acc":        _tag(MulticlassAccuracy(num_classes=ny), 'Accuracy', 'y'),
        "pos/x_rMSE":       _tag(MeanSquaredError(squared=False),   'rMSE', 'x'),
        "pos/y_rMSE":       _tag(MeanSquaredError(squared=False),   'rMSE', 'y'),
        "pos/x_CE":         _tag(MeanMetric(), 'CE', 'x'),
        "pos/y_CE":         _tag(MeanMetric(), 'CE', 'y'),
        "mask/IoU":         _tag(BinaryJaccardIndex(), 'IoU', 'mask'),
        "pos/WeightedCE":   _tag(MeanMetric(), 'WeightedCE', 'mask'),
        "mask/SoftDice":    _tag(MeanMetric(), 'SoftDice',   'mask'),
        "mask/x_SoftDice":  _tag(MeanMetric(), 'SoftDice', 'mask', 'x'),
        "mask/y_SoftDice":  _tag(MeanMetric(), 'SoftDice', 'mask', 'y'),
    }

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

def sord_loss(predictions, targets, cost_matrix):
    """SORD loss with soft labels based on ordinal distance."""
    soft_labels = torch.exp(-cost_matrix[targets])
    soft_labels = F.normalize(soft_labels, p=1, dim=1)
    log_predictions = F.log_softmax(predictions, dim=-1)
    return -(soft_labels * log_predictions).sum(dim=1).mean()

def soft_dice_fn(logits, targets, axis=None):
    """Computes Soft Dice for the full mask, or just along X or Y directions"""
    probs = logits.sigmoid()
    if axis is not None: probs, targets = probs.mean(dim=axis), targets.mean(dim=axis)
    spatial_dims = tuple(range(1, probs.ndim))  # (-2, -1) for dice over entire mask or (-1,) for dice on just x,y profiles
    intersection = (probs * targets).sum(dim=spatial_dims)
    total_sum = probs.sum(dim=spatial_dims) + targets.sum(dim=spatial_dims)
    dice_score = (2 * intersection + 1) / (total_sum + 1)
    return 1 - dice_score.mean()

def weighted_cross_entropy_fn(mask_logits, target):
    pos_weight = (1 - target).sum() / target.sum().clamp(min=1e-6)
    return F.binary_cross_entropy_with_logits(mask_logits, target, pos_weight=pos_weight)

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
        x = self.raw_to_tokens(x)                                                           # (B_L,P,C,_PS) -> (B_L,P,C,PS) where PS=_PS or 2*_PS
        x = self.embed(x.reshape(B_L, P, -1))                                               # (B_L,P,C,PS) -> (B_L,P,D)
        x = apply_rope(x, self.freqs_cis.to(x.device)) if ROPE else self.freq_pos_embd(x)   # (B_L,P,D)
        x = torch.cat((self.cls_token.expand(B_L, -1, -1).to(x.device), x), dim=1)          # (B_L,P+1,D)
        output = self.layers(x)                                                             # (B_L,P+1,D)
        return output[:, 0, :]                                                              # (B_L,D)


class SignalTransformer(ComposerModel):
    def __init__(self, d_model, pnt_num_heads, pnt_num_layers, seq_num_heads, seq_num_layers, patch_size, signal_length, signal_is, num_x_positions, num_y_positions, num_lasers, alpha, beta, gamma=0.5, delta=0.5, mask_h=100, mask_w=100):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.delta, self.mask_h, self.mask_w, self.num_x_pos, self.num_y_pos = alpha, beta, gamma, delta, mask_h, mask_w, num_x_positions, num_y_positions
        self.pnt_trans = PointTransformer(patch_size, d_model, pnt_num_heads, pnt_num_layers, signal_length, signal_is)
        self.seq_trans = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=seq_num_heads, batch_first=True), num_layers=seq_num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=.02)  # Initialize to small random values

        # positional embeddings for segmentation mask and laser grid
        if ROPE:
            self.register_buffer("freqs_mask", precompute_freqs_cis_2d(d_model, mask_h, mask_w))
        else:
            self.laser_pos_embd = LearnablePositionalEncoding(num_lasers, d_model)
        self.register_buffer("freqs_laser", precompute_freqs_cis_2d(d_model, int(math.sqrt(num_lasers)), int(math.sqrt(num_lasers))))

        # Prediction heads
        self.mlp_head_x_position = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, self.num_x_pos))
        self.mlp_head_y_position = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, self.num_y_pos))
        self.mlp_head_mask = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, mask_h * mask_w))

        # SORD loss cost matrix
        self.register_buffer('cost_matrix_x', self._init_cost_matrix(self.num_x_pos))
        self.register_buffer('cost_matrix_y', self._init_cost_matrix(self.num_y_pos))

        # Metrics
        self.train_metrics, self.val_metrics = create_metrics(self.num_x_pos, self.num_y_pos), create_metrics(self.num_x_pos, self.num_y_pos)

    def _init_cost_matrix(self, num_classes, multiplier=0.5):
        indices = torch.arange(num_classes)
        return multiplier * (indices.unsqueeze(1) - indices.unsqueeze(0)).abs() ** 2

    def forward(self, batch):
        # B=batch size, L=n_lasters, C=n_coordinates=2, PS=patch size, D=d_model
        x, _ = batch
        B, L,_, _, _ = x.shape

        # PointTransformer learns patterns between all frequencies from a single laser
        # flatten so PointTransformer processes all lasers across all batches in parallel
        x = self.pnt_trans(x.flatten(0, 1)).reshape(B, L, -1)                                   # (B,L,P,C,PS) -> (B,L,D)

        # SequenceTransformer learns patterns between all lasers in the the laser grid
        x = apply_rope(x, self.freqs_laser.to(x.device)) if ROPE else self.laser_pos_embd(x)    # (B,L,D) -> (B,L,D)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)                             # (B,L,D) (1,1,D) -> (B,L+1,D)
        output = self.seq_trans(x)                                                              # (B,L+1,D) -> (B,L+1,D)
        cls_embedding = output[:, 0, :]                                                         # (B,D)

        # Predict x position, y position, and segmentation mask
        x_logits = self.mlp_head_x_position(cls_embedding)
        y_logits = self.mlp_head_y_position(cls_embedding)
        mask_logits = self.mlp_head_mask(cls_embedding).view(B, self.mask_h, self.mask_w)
        return x_logits, y_logits, cls_embedding, mask_logits

    def loss(self, outputs, batch):
        _, (mask_true, x_true, y_true) = batch
        x_logits, y_logits, _, mask_logits = outputs
        position_loss = mask_loss = 0.0
        loss_log = {}

        if POSITION:
            ce_loss_x = F.cross_entropy(x_logits, x_true)
            ce_loss_y = F.cross_entropy(y_logits, y_true)
            if not SORD:
                position_loss = self.beta * ce_loss_x + (1 - self.beta) * ce_loss_y
                loss_log.update({'loss/train/ce_x': ce_loss_x, 'loss/train/ce_y': ce_loss_y})
            else:
                sord_loss_x = sord_loss(x_logits, x_true, self.cost_matrix_x)
                sord_loss_y = sord_loss(y_logits, y_true, self.cost_matrix_y)
                ce_sord_loss_x = self.alpha * sord_loss_x + (1 - self.alpha) * ce_loss_x
                ce_sord_loss_y = self.alpha * sord_loss_y + (1 - self.alpha) * ce_loss_y
                position_loss = self.beta * ce_sord_loss_x + (1 - self.beta) * ce_sord_loss_y
                loss_log.update({'loss/train/ce_x': ce_loss_x, 'loss/train/ce_y': ce_loss_y,
                                 'loss/train/sord_x': sord_loss_x, 'loss/train/sord_y': sord_loss_y,
                                 'loss/train/ce_sord_x': ce_sord_loss_x, 'loss/train/ce_sord_y': ce_sord_loss_y})
            loss_log['loss/train/position'] = position_loss

        if MASK:
            mask_dice_loss = soft_dice_fn(mask_logits, mask_true)
            mask_ce_loss = weighted_cross_entropy_fn(mask_logits, mask_true)
            mask_loss = self.delta * mask_dice_loss + (1 - self.delta) * mask_ce_loss
            loss_log.update({'loss/train/mask_dice': mask_dice_loss, 'loss/train/mask_cross_entropy': mask_ce_loss, 'loss/train/mask_total': mask_loss})

        if POSITION and MASK:
            total_loss = position_loss * (1 - self.gamma) + self.gamma * mask_loss
        else:
            total_loss = position_loss if POSITION else mask_loss
        loss_log['loss/train/total'] = total_loss

        self.logger.log_metrics({k: v.item() for k, v in loss_log.items()})
        return total_loss

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch, outputs, metric):
        _, (mask_true, x_true, y_true) = batch
        x_logits, y_logits, _, mask_logits = outputs
        metric_name, pred_type, pos = getattr(metric, 'metric_name', None), getattr(metric, 'pred_type', None), getattr(metric, 'pos', None)

        if pred_type == 'x':
            if metric_name == 'rMSE': metric.update(x_logits.argmax(-1), x_true)
            elif metric_name == 'CE': metric.update(F.cross_entropy(x_logits, x_true))
            elif metric_name == 'Accuracy': metric.update(x_logits, x_true) # multiclass accuracy
            else: raise ValueError(f"did not recognize {metric_name=}")
        elif pred_type == 'y':
            if metric_name == 'rMSE': metric.update(y_logits.argmax(-1), y_true)
            elif metric_name == 'CE': metric.update(F.cross_entropy(y_logits, y_true))
            elif metric_name == 'Accuracy': metric.update(y_logits, y_true)
            else: raise ValueError(f"did not recognize {metric_name=}")
        elif pred_type == 'mask':
            if metric_name == 'SoftDice':
                axis = 1 if pos == 'x' else (0 if pos == 'y' else None)
                metric.update(soft_dice_fn(mask_logits, mask_true, axis=axis))
            elif metric_name == 'IoU': metric.update((mask_logits.sigmoid() > 0.5).int(), mask_true.int())
            elif metric_name == 'WeightedCE': metric.update(weighted_cross_entropy_fn(mask_logits, mask_true))
            else: raise ValueError(f"did not recognize {metric_name=}")
        else:
            raise ValueError(f"did not recognize {pred_type=}")

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

# ***** main *****

def count_parameters(model: nn.Module) -> int: return sum([p_.numel() for p_ in model.parameters()])

def get_parser():
    parser = argparse.ArgumentParser()

    # flags
    parser.add_argument('--sord', type=int, default=1)
    parser.add_argument('--mask', type=int, default=1)
    parser.add_argument('--position', type=int, default=1)
    parser.add_argument('--discretized-mask', type=int, default=1)
    parser.add_argument('--rope', type=int, default=1)

    # data
    parser.add_argument('--data-dir', type=str, default='eturok-weizmann/vibration-data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--signal-is', type=str, default='magnitude')

    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--mask-h', type=int, default=40)
    parser.add_argument('--mask-w', type=int, default=20)

    # model arch
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--pnt-num-heads', type=int, default=2)
    parser.add_argument('--seq-num-heads', type=int, default=2)
    parser.add_argument('--pnt-num-layers', type=int, default=2)
    parser.add_argument('--seq-num-layers', type=int, default=2)

    # learning
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-duration', type=str, default='1_000ep')
    parser.add_argument('--eval-interval', type=str, default='10ep')
    parser.add_argument('--run-name', type=str, default=None)

    # loss
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=0.5)
    return parser

@app.function(
    gpu="A100",
    timeout=86_400, # maximum timeout is 24 hours; see https://modal.com/docs/guide/timeouts#timeouts
    retries=3,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    )
def train(**kwargs):
    args = get_parser().parse_args([])  # get defaults
    args.__dict__.update(kwargs)        # apply overrides
    global SORD, MASK, POSITION, ROPE, DISCRETIZED_MASK # environment variables
    SORD, MASK, POSITION, ROPE, DISCRETIZED_MASK = args.sord, args.mask, args.position, args.rope, args.discretized_mask

    seed_all(args.seed) # must seed before initializing model + dataloader (which has random shuffle)
    train_loader, test_loader, num_x_positions, num_y_positions = get_dataloaders(args.data_dir, args.patch_size, args.mask_h, args.mask_w, args.batch_size, args.eval_batch_size, seed=args.seed)
    num_lasers, n_freqs_used, n_patches = 100, 3328, 13
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    model = SignalTransformer(args.d_model, args.pnt_num_heads, args.pnt_num_layers, args.seq_num_heads, args.seq_num_layers, args.patch_size, n_freqs_used, args.signal_is, num_x_positions, num_y_positions, num_lasers, args.alpha, args.beta, gamma=args.gamma, delta=args.delta, mask_h=args.mask_h, mask_w=args.mask_w)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    config = {'n_params': count_parameters(model), 'num_x_positions': num_x_positions, 'num_y_positions': num_y_positions, 'n_patches':n_patches, 'num_lasers':num_lasers, 'delta': args.delta, 'SORD': SORD, 'MASK': MASK, 'POSITION': POSITION, 'data_dir': args.data_dir, 'seed': args.seed, 'signal_is': args.signal_is, 'd_model': args.d_model, 'pnt_num_heads': args.pnt_num_heads, 'seq_num_heads': args.seq_num_heads, 'pnt_num_layers': args.pnt_num_layers, 'seq_num_layers': args.seq_num_layers, 'patch_size': args.patch_size, 'batch_size': args.batch_size, 'eval_batch_size': args.eval_batch_size, 'lr': args.lr, 'alpha': args.alpha, 'beta': args.beta, 'gamma': args.gamma, 'max_duration': args.max_duration, 'eval_interval': args.eval_interval}
    logger = WandBLogger('good-vibrations', 'seg-mask', init_kwargs={'config': config, 'save_code': True})
    hf_ckpt_upload = HFChkptUploader("eturok-weizmann/good-vibrations", interval=args.eval_interval, monitor="x/rMSE", save_local=True)
    mask_viz = MaskVisualizationCallback(n_samples=args.eval_batch_size, save_dir="visualizations")
    ic(config)

    trainer = Trainer(
        model=model, train_dataloader=train_loader, eval_dataloader=test_loader,
        max_duration=args.max_duration, eval_interval=args.eval_interval,
        optimizers=optimizer, device=device, seed=args.seed,
        loggers=logger, log_to_console=True, auto_log_hparams=True, save_metrics=True,
        # callbacks=[hf_ckpt_upload, mask_viz])
        callbacks=[mask_viz])
    # override wandb run name
    wandb.run.name = args.run_name if args.run_name else '-'.join(wandb.run.name.split('-')[1:]) + f'_lr{float(args.lr)}_{args.gamma=}'

    trainer.fit()
    ic(trainer.state.train_metrics, type(trainer.state.train_metrics))
    ic(trainer.state.eval_metrics)

    trainer.close()

@app.local_entrypoint()
def main(*args):
    train.remote(**vars(get_parser().parse_args(args)))  # runs on Modal GPU

if __name__ == '__main__':
    train.local(**vars(get_parser().parse_args()))
