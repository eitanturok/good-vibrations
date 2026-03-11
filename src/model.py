import argparse
from pathlib import Path

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
from composer.metrics import CrossEntropy
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.regression import MeanSquaredError
from composer.loggers import WandBLogger
from composer.utils.reproducibility import seed_all

from icecream import install
install()

from helpers import getenv, HFChkptUploader

SORD = getenv("SORD", 1)
DEBUG = getenv("DEBUG", 0)

# ***** Dataset *****

def clean_shifts(shifts: torch.Tensor, fs: int, lowcut: float = 50.0, highcut: float | None = None) -> torch.Tensor:
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

class VibrationDataset(torch.utils.data.Dataset):
    """
    Downloads shifts.safetensors once via hf_hub_download (cached to disk after first run),
    then memory-maps it with safe_open. Each __getitem__ reads only the pages for that
    tensor off disk — the OS never loads the full file into RAM.
    """

    def __init__(self, repo_id:str, split:str="train", patch_size:int=256, token:str|None=None):
        self.ds = load_dataset(repo_id, split=split, token=token, columns=["shifts_idx", "mask_idx", "x_position", "y_position", "object", "fps"])
        self.st_shifts = safe_open(hf_hub_download(repo_id, "shifts.safetensors", repo_type="dataset", token=token), framework="pt", device="cpu")
        self.st_masks = safe_open(hf_hub_download(repo_id, "masks.safetensors",  repo_type="dataset", token=token),  framework="pt", device="cpu")
        self.patch_size = patch_size
        # remap raw position values to 0-indexed class labels (same as vibration_transformer.py)
        self.x_labels = torch.tensor(self.ds["x_position"]).unique(return_inverse=True)[1]
        self.y_labels = torch.tensor(self.ds["y_position"]).unique(return_inverse=True)[1]
    def __repr__(self): return f"VibrationDataset(split={self.ds.split}, n={len(self.ds)})"
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        row = self.ds[idx]
        mask = self.st_masks.get_tensor(f"mask_{row['mask_idx']}")          # (H, W) bool
        shifts = self.st_shifts.get_tensor(f"shifts_{row['shifts_idx']}")   # (n_lasers, n_timesteps, 2)
        # clean + fft the shift data
        shifts = clean_shifts(shifts, row["fps"])                           # (n_lasers, n_timesteps, 2)
        fft, freqs = do_fft(shifts, row["fps"])                             # (n_lasers, n_freqs, 2)
        # patchify: unfold freq dim into non-overlapping patches -> (n_lasers, n_patches, n_coords, patch_size)
        fft_patches = fft.unfold(1, self.patch_size, self.patch_size)
        return fft_patches.float(), (mask, self.x_labels[idx], self.y_labels[idx])

def get_dataloaders(repo_id:str, patch_size:int=256, batch_size:int=8, eval_batch_size:int=16, shuffle:bool=True, num_workers:int=0, seed:int=42, token:str | None = None):
    train_set = VibrationDataset(repo_id, split="train", patch_size=patch_size, token=token)
    test_set = VibrationDataset(repo_id, split="test", patch_size=patch_size, token=token)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size,     shuffle=shuffle,  num_workers=num_workers, generator=generator, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=eval_batch_size, shuffle=False,   num_workers=num_workers, generator=generator, pin_memory=True)
    num_x_positions = int(train_set.x_labels.max()) + 1
    num_y_positions = int(train_set.y_labels.max()) + 1
    return train_loader, test_loader, num_x_positions, num_y_positions

# ***** model *****

def sord_loss(predictions, targets, cost_matrix):
    """SORD loss with soft labels based on ordinal distance."""
    soft_labels = torch.exp(-cost_matrix[targets])
    soft_labels = F.normalize(soft_labels, p=1, dim=1)
    log_predictions = F.log_softmax(predictions, dim=-1)
    return -(soft_labels * log_predictions).sum(dim=1).mean()

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(dim, hidden_dim)  # Learnable embeddings

    def forward(self, x):
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        return x + self.embed(positions)

class PointTransformer(nn.Module):
    def __init__(self, patch_size, d_model, num_heads, num_layers, signal_length, signal_is, dropout_prob=0.5):
        super().__init__()
        self.signal_is = signal_is

        # Token embedding projection
        self.embedding = nn.Linear(2 * patch_size, d_model)

        # Positional encoding
        self.learnable_positional_encoding = LearnablePositionalEncoding(signal_length // patch_size, d_model)

        # Transformer encoder setup
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Token dropout probability
        self.token_dropout_prob = dropout_prob

    def _raw_signal_to_tokenizable_one(self, tokens):
        # tokens are: (B, num_tokens, 2, patch_size)  of type COMPLEX!
        if self.signal_is == 'magnitude':
            tokens = tokens.abs()
        elif self.signal_is == 'complex':
            tokens = torch.cat([tokens.real, tokens.imag], dim=-1)
        elif self.signal_is == 'mag_phase':
            tokens = torch.cat([tokens.abs(), tokens.angle()], dim=-1)
        else:
            raise RuntimeError(f'unknown signal type {self.signal_is}')
        return tokens


    def forward(self, tokens):
        tokens = self._raw_signal_to_tokenizable_one(tokens)
        B, P, _, _ = tokens.shape # (B, num_patches, 2, patch_size)
        tokens_emb = self.embedding(tokens.reshape(B, P, -1))  # (B, num_patches, 2, patch_size) -> (B, num_patches, D)

        # Add positional encoding to token embeddings
        tokens_emb = self.learnable_positional_encoding(tokens_emb)  # (B, num_tokens, d_model)

        # Add PNT token
        cls_token_expanded = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens_emb = torch.cat((cls_token_expanded, tokens_emb), dim=1)  # (B, num_tokens + 1, d_model)

        # Transformer encoder
        output = self.transformer_encoder(tokens_emb)  # (B, num_tokens + 1, d_model)

        # Return PNT token representation
        PNT = output[:, 0, :]  # (B, d_model)
        return PNT


class SignalTransformer(ComposerModel):
    def __init__(self, d_model, pnt_num_heads, pnt_num_layers, seq_num_heads, seq_num_layers, patch_size, signal_length, signal_is, num_x_positions, num_y_positions, num_lasers, alpha, beta):
        super().__init__()
        self.beta = beta

        # Initialize one PointTransformer for every point
        self.point_transformer = PointTransformer(patch_size, d_model, pnt_num_heads, pnt_num_layers, signal_length, signal_is)

        # Positional encoding for point embeddings
        self.learnable_positional_encoding = LearnablePositionalEncoding(num_lasers, d_model)

        # Sequence-level transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=seq_num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=seq_num_layers)

        # Learnable [CLS] token for the sequence-level transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=.02)  # Initialize to small random values

        # Prediction heads
        self.mlp_head_x_position = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, num_x_positions))
        self.mlp_head_y_position = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, num_y_positions))

        # SORD loss cost matrix
        self.alpha = alpha
        self.register_buffer('cost_matrix_x', self._init_cost_matrix(num_x_positions))
        self.register_buffer('cost_matrix_y', self._init_cost_matrix(num_y_positions))

        # metrics - tag each with position attribute
        def _tag(m, pos):
            m.position = pos
            return m
        self.train_accuracy_x = _tag(MulticlassAccuracy(num_classes=num_x_positions, average='micro'), 'x')
        self.val_accuracy_x = _tag(MulticlassAccuracy(num_classes=num_x_positions, average='micro'), 'x')
        self.train_cross_entropy_x = _tag(CrossEntropy(), 'x')
        self.val_cross_entropy_x = _tag(CrossEntropy(), 'x')
        self.train_rmse_x = _tag(MeanSquaredError(squared=False), 'x')
        self.val_rmse_x = _tag(MeanSquaredError(squared=False), 'x')

        self.train_accuracy_y = _tag(MulticlassAccuracy(num_classes=num_y_positions, average='micro'), 'y')
        self.val_accuracy_y = _tag(MulticlassAccuracy(num_classes=num_y_positions, average='micro'), 'y')
        self.train_cross_entropy_y = _tag(CrossEntropy(), 'y')
        self.val_cross_entropy_y = _tag(CrossEntropy(), 'y')
        self.train_rmse_y = _tag(MeanSquaredError(squared=False), 'y')
        self.val_rmse_y = _tag(MeanSquaredError(squared=False), 'y')

    def _init_cost_matrix(self, num_classes, multiplier=0.5):
        indices = torch.arange(num_classes)
        return multiplier * (indices.unsqueeze(1) - indices.unsqueeze(0)).abs() ** 2

    def forward(self, batch):
        inputs, _ = batch
        B, N_LASERS, N_PATCHES, PATCH_SIZE, N_COORDS = inputs.shape

        pnt_tokens = []
        # TODO: batch process all together!!
        # Pass each point's data through PointTransformer
        points_to_process = range(inputs.size(1))
        for i in points_to_process:
            point_data = inputs[:, i]  # Shape: (B, num_tokens, patch_size)
            pnt_token = self.point_transformer(point_data)  # Output shape: (B, d_model)
            pnt_tokens.append(pnt_token)

        # Stack CLS tokens from PointTransformers: (B, num_points, d_model)
        pnt_tokens = torch.stack(pnt_tokens, dim=1)

        # Add positional encoding
        pnt_tokens = self.learnable_positional_encoding(pnt_tokens)

        # Add learnable sequence CLS token: (B, 1, d_model)
        cls_token = self.cls_token.expand(B, -1, -1)

        transformer_input = torch.cat((cls_token, pnt_tokens), dim=1)  # (B, num_points + 1, d_model)

        # Process through the sequence-level transformer
        output = self.transformer_encoder(transformer_input)  # (B, num_points + 1, d_model)

        # Extract the sequence-level CLS embedding
        cls_embedding = output[:, 0, :]  # Shape: (B, d_model)

        # Predict x and y position
        logits_x_position = self.mlp_head_x_position(cls_embedding)
        logits_y_position = self.mlp_head_y_position(cls_embedding)
        return logits_x_position, logits_y_position, cls_embedding

    def loss(self, outputs, batch):
        _, (_, targets_x, targets_y) = batch
        logits_x, logits_y, _ = outputs

        ce_loss_x = F.cross_entropy(logits_x, targets_x)
        ce_loss_y = F.cross_entropy(logits_y, targets_y)
        if not SORD: return self.beta * ce_loss_x + (1 - self.beta) * ce_loss_y

        sord_loss_x = sord_loss(logits_x, targets_x, self.cost_matrix_x)
        sord_loss_y = sord_loss(logits_y, targets_y, self.cost_matrix_y)
        ce_sord_loss_x = self.alpha * sord_loss_x + (1 - self.alpha) * ce_loss_x
        ce_sord_loss_y = self.alpha * sord_loss_y + (1 - self.alpha) * ce_loss_y
        return self.beta * ce_sord_loss_x + (1 - self.beta) * ce_sord_loss_y

    def update_metric(self, batch, outputs, metric):
        _, (_, targets_x, targets_y) = batch
        logits_x_position, logits_y_position, _ = outputs

        is_x_metric = metric.position == 'x'
        logits = logits_x_position if is_x_metric else logits_y_position
        targets = targets_x if is_x_metric else targets_y

        if isinstance(metric, MeanSquaredError):
            metric.update(logits.argmax(dim=-1), targets)
        else:
            metric.update(logits, targets)

    def get_metrics(self, is_train=False):
        if is_train: return {'x/MulticlassAccuracy': self.train_accuracy_x, 'x/CrossEntropy': self.train_cross_entropy_x, 'x/rMSE': self.train_rmse_x,
                             'y/MulticlassAccuracy': self.train_accuracy_y, 'y/CrossEntropy': self.train_cross_entropy_y, 'y/rMSE': self.train_rmse_y}
        return {'x/MulticlassAccuracy': self.val_accuracy_x, 'x/CrossEntropy': self.val_cross_entropy_x, 'x/rMSE': self.val_rmse_x,
                'y/MulticlassAccuracy': self.val_accuracy_y, 'y/CrossEntropy': self.val_cross_entropy_y, 'y/rMSE': self.val_rmse_y}

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

# ***** main *****

def count_parameters(model: nn.Module) -> int: return sum([p_.numel() for p_ in model.parameters()])

def get_parser():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data_dir', type=str, default='eturok-weizmann/vibration-data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--signal_is', type=str, default='magnitude')

    # model arch
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--pnt_num_heads', type=int, default=2)
    parser.add_argument('--seq_num_heads', type=int, default=2)
    parser.add_argument('--pnt_num_layers', type=int, default=2)
    parser.add_argument('--seq_num_layers', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=256)

    # learning
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--max_duration', type=str, default='1_000ep')
    parser.add_argument('--eval_interval', type=str, default='5ep')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    seed_all(args.seed) # must seed before initializing the model

    train_loader, test_loader, num_x_positions, num_y_positions = get_dataloaders(args.data_dir, args.patch_size, args.batch_size, args.eval_batch_size, seed=args.seed)
    num_lasers, n_freqs_used, n_patches = 100, 3328, 13
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    model = SignalTransformer(args.d_model, args.pnt_num_heads, args.pnt_num_layers, args.seq_num_heads, args.seq_num_layers, args.patch_size, n_freqs_used, args.signal_is, num_x_positions, num_y_positions, num_lasers, args.alpha, args.beta)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    config = {'n_params': count_parameters(model), 'num_x_positions': num_x_positions, 'num_y_positions': num_y_positions, 'n_patches':n_patches, 'num_lasers':num_lasers, 'SORD': SORD} | vars(args)
    # logger = WandBLogger('good-vibe-rations', 'classify-position', init_kwargs={'config': config, 'save_code': True})
    logger = WandBLogger('good-vibe-rations', 'classify-position', init_kwargs={'save_code': True})
    hf_ckpt_upload = HFChkptUploader("eturok-weizmann/good-vibrations", interval=args.eval_interval, monitor="x/rMSE", save_local=True)
    ic(config)

    trainer = Trainer(
        model=model, train_dataloader=train_loader, eval_dataloader=test_loader,
        max_duration=args.max_duration, eval_interval=args.eval_interval,
        optimizers=optimizer, device=device, seed=args.seed,
        loggers=logger, log_to_console=True, auto_log_hparams=True, save_metrics=True,
        callbacks=[hf_ckpt_upload,])
    # override wandb run name
    wandb.run.name = '-'.join(wandb.run.name.split('-')[1:]) + f'_lr{float(args.lr)}_SORD{SORD}_alpha{args.alpha}_beta{args.beta}'

    trainer.fit()
    ic(trainer.state.train_metrics, type(trainer.state.train_metrics))
    ic(trainer.state.eval_metrics)

    trainer.close()

if __name__ == '__main__':
    main()
