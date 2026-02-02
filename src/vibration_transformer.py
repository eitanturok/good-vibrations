import argparse
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from composer import Trainer
from composer.models import ComposerModel
from composer.metrics import CrossEntropy
from torchmetrics.classification import MulticlassAccuracy
from composer.loggers import WandBLogger
from composer.utils.reproducibility import seed_all

from icecream import install

install()

# ***** Dataset *****

class VibrationDataset(Dataset):
    """Dataset that returns (fft_mag, label)."""
    def __init__(self, data_path:Path, label_path:Path, patch_size:int):
        self.fft_vals, self.labels, self.patch_size = torch.load(data_path), torch.load(label_path), patch_size
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        n_lasers, n_freqs, n_coords = self.fft_vals[idx].shape  # index into n_samples of (n_samples, n_lasers, n_freqs, 2)
        n_patches = n_freqs // self.patch_size
        n_freqs_used = n_patches * self.patch_size
        # if n_freqs_used < n_freqs: print(f'WARNING: dropping {n_freqs - n_freqs_used} highest frequencies')
        patches = self.fft_vals[idx][:, :n_freqs_used, :].reshape(n_lasers, n_patches, self.patch_size, n_coords).transpose(-2, -1) # (n_lasers, n_freqs, 2) -> (n_lasers, n_patches, 2, patch_size)
        return patches, self.labels[idx]

def get_dataloaders(data_dir:Path, target:str, patch_size:int, batch_size:int=8, test_split:float=0.2, shuffle:bool=True, num_workers:int=0, seed:int=42):
    dataset = VibrationDataset(data_dir / "fft_vals.pt", data_dir / f"{target}_labels.pt", patch_size)
    test_size = int(len(dataset) * test_split)
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [len(dataset) - test_size, test_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True)
    return train_loader, test_loader, dataset

# ***** model *****

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
        # if self.training:  # Apply dropout only during training
        #     B, num_tokens, c = tokens_emb.size()

        #     # Apply random dropout for the remaining tokens
        #     new_n = max(1, int((1.0 - self.token_dropout_prob) * num_tokens))
        #     with torch.no_grad():
        #         r = torch.rand_like(tokens_emb[..., 0])  # B-N
        #         ridx = torch.argsort(r, dim=-1)[:, :new_n]
        #     tokens_emb = torch.gather(tokens_emb, dim=1, index=ridx[..., None].expand(-1, -1, c))

        # Add PNT token
        cls_token_expanded = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens_emb = torch.cat((cls_token_expanded, tokens_emb), dim=1)  # (B, num_tokens + 1, d_model)

        # Transformer encoder
        output = self.transformer_encoder(tokens_emb)  # (B, num_tokens + 1, d_model)

        # Return PNT token representation
        PNT = output[:, 0, :]  # (B, d_model)
        return PNT


class SignalTransformer(ComposerModel):
    def __init__(self, d_model, pnt_num_heads, pnt_num_layers, cls_num_heads, cls_num_layers, patch_size, signal_length, signal_is, num_classes, num_positions):
        super().__init__()

        # Initialize one PointTransformer for every point
        self.point_transformer = PointTransformer(patch_size, d_model, pnt_num_heads, pnt_num_layers, signal_length, signal_is)

        # Positional encoding for point embeddings
        self.learnable_positional_encoding = LearnablePositionalEncoding(num_positions, d_model)

        # Sequence-level transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=pnt_num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=pnt_num_layers)

        # Learnable [CLS] token for the sequence-level transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=.02)  # Initialize to small random values

        # Prediction heads
        self.mlp_head_capacity = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, num_classes))
        self.mlp_head_classification = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, num_classes))

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

        transformer_input = torch.cat((cls_token, pnt_tokens), dim=1)  # Shape: (B, num_points + 1, d_model)

        # Process through the sequence-level transformer
        output = self.transformer_encoder(transformer_input)  # Shape: (B, num_points + 1, d_model)

        # Extract the sequence-level CLS embedding
        cls_embedding = output[:, 0, :]  # Shape: (B, d_model)

        # Predict capacity and classification outputs
        logits_capacity_pred = self.mlp_head_capacity(cls_embedding)  # Shape: (B, VECTOR_SIZE)
        logits_class_pred = self.mlp_head_classification(cls_embedding)  # Shape: (B, NUM_CLASSES)

        return logits_capacity_pred, logits_class_pred, cls_embedding

    def loss(self, outputs, batch):
        _, targets = batch
        logits_capacity_pred, logits_class_pred, cls_embedding = outputs
        return F.cross_entropy(logits_class_pred, targets)


def get_parser():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--signal_is', type=str, default='magnitude')

    # model arch
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--pnt_num_heads', type=int, default=4)
    parser.add_argument('--cls_num_heads', type=int, default=4)
    parser.add_argument('--pnt_num_layers', type=int, default=8)
    parser.add_argument('--cls_num_layers', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=128)

    # learning
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    seed_all(args.seed) # must seed before initializing the model

    train_loader, test_loader, dataset = get_dataloaders(Path(args.data_dir), 'x', args.patch_size, args.batch_size, seed=args.seed)
    n_classes = len(dataset.labels.unique())
    n_patches = dataset.fft_vals.shape[2] // args.patch_size
    n_freqs_used = n_patches * args.patch_size
    num_positions = dataset.fft_vals.shape[1] # number of laser positions

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SignalTransformer(args.d_model, args.pnt_num_heads, args.pnt_num_layers, args.cls_num_heads,
                              args.cls_num_layers, args.patch_size, n_freqs_used, args.signal_is, n_classes, num_positions)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # metrics = [CrossEntropy(), MulticlassAccuracy(num_classes=n_classes, average='micro')]

    trainer = Trainer(
        model=model, train_dataloader=train_loader, eval_dataloader=test_loader,
        max_duration="1ep", optimizers=optimizer, device=device, seed=args.seed,
        loggers=WandBLogger('good-vibe-rations', 'classify-position', init_kwargs={'config': args}))
    trainer.fit()
    print(trainer.state.train_metrics)
    print(trainer.state.eval_metrics)

if __name__ == '__main__':
    main()
