import argparse
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
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

# ***** Dataset *****

class VibrationDataset(Dataset):
    """Dataset that returns (fft_mag, (x_label, y_label))."""
    def __init__(self, data_path:Path, x_label_path:Path, y_label_path:Path, patch_size:int):
        self.fft_vals = torch.load(data_path)
        # Load and normalize labels to start from 0 (required for cross_entropy)
        x_labels_raw = torch.load(x_label_path)
        y_labels_raw = torch.load(y_label_path)
        _, self.x_labels = x_labels_raw.unique(return_inverse=True)
        _, self.y_labels = y_labels_raw.unique(return_inverse=True)
        self.patch_size = patch_size 
    def __len__(self): return len(self.x_labels)
    def __getitem__(self, idx):
        n_lasers, n_freqs, n_coords = self.fft_vals[idx].shape  # index into n_samples of (n_samples, n_lasers, n_freqs, 2)
        n_patches = n_freqs // self.patch_size
        n_freqs_used = n_patches * self.patch_size
        # if n_freqs_used < n_freqs: print(f'WARNING: dropping {n_freqs - n_freqs_used} highest frequencies')
        patches = self.fft_vals[idx][:, :n_freqs_used, :].reshape(n_lasers, n_patches, self.patch_size, n_coords).transpose(-2, -1) # (n_lasers, n_freqs, 2) -> (n_lasers, n_patches, 2, patch_size)
        return patches, (self.x_labels[idx], self.y_labels[idx])

def get_dataloaders(data_dir:Path, patch_size:int, batch_size:int=8, eval_batch_size=16, test_split:float=0.2, shuffle:bool=True, num_workers:int=0, seed:int=42):
    dataset = VibrationDataset(data_dir / "fft_vals.pt", data_dir / "x_labels.pt", data_dir / "y_labels.pt", patch_size)
    test_size = int(len(dataset) * test_split)
    print(f'{len(dataset)-test_size} train samples\n{test_size} test samples')
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [len(dataset) - test_size, test_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True)
    num_x_positions = len(dataset.x_labels.unique())
    num_y_positions = len(dataset.y_labels.unique())
    return train_loader, test_loader, dataset, num_x_positions, num_y_positions

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
        _, (targets_x, targets_y) = batch
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
        _, (targets_x, targets_y) = batch
        logits_x_position, logits_y_position, _ = outputs

        is_x_metric = metric.position == 'x'
        logits = logits_x_position if is_x_metric else logits_y_position
        targets = targets_x if is_x_metric else targets_y

        if isinstance(metric, MeanSquaredError):
            metric.update(logits.argmax(dim=-1), targets)
        else:
            metric.update(logits, targets)

    def get_metrics(self, is_train=False):
        if is_train:
            return {
                'x/MulticlassAccuracy': self.train_accuracy_x, 'x/CrossEntropy': self.train_cross_entropy_x, 'x/rMSE': self.train_rmse_x,
                'y/MulticlassAccuracy': self.train_accuracy_y, 'y/CrossEntropy': self.train_cross_entropy_y, 'y/rMSE': self.train_rmse_y,
            }
        return {
            'x/MulticlassAccuracy': self.val_accuracy_x, 'x/CrossEntropy': self.val_cross_entropy_x, 'x/rMSE': self.val_rmse_x,
            'y/MulticlassAccuracy': self.val_accuracy_y, 'y/CrossEntropy': self.val_cross_entropy_y, 'y/rMSE': self.val_rmse_y,
        }

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

def count_parameters(model: nn.Module) -> int:
    return sum([p_.numel() for p_ in model.parameters()])


def get_parser():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data_dir', type=str, default='data/processed')
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
    parser.add_argument('--batch_size', type=int, default=64)
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

    train_loader, test_loader, dataset, num_x_positions, num_y_positions = get_dataloaders(
        Path(args.data_dir), args.patch_size, args.batch_size, args.eval_batch_size, seed=args.seed)
    n_patches = dataset.fft_vals.shape[2] // args.patch_size
    n_freqs_used = n_patches * args.patch_size
    num_lasers = dataset.fft_vals.shape[1] # number of laser positions

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SignalTransformer(args.d_model, args.pnt_num_heads, args.pnt_num_layers, args.seq_num_heads,
                              args.seq_num_layers, args.patch_size, n_freqs_used, args.signal_is, num_x_positions, num_y_positions, num_lasers, args.alpha, args.beta)
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
