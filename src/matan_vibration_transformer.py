import argparse
import os
import multiprocessing as mp

import torch
from torch import nn
import torch.nn.functional as nnf
from torch.utils.data import Dataset, DataLoader
from torch.fft import fft, ifft, fftfreq
import numpy as np
import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import data_frames

fill_levels = [0., 0.2, 0.4, 0.6, 0.8, 1.0]

containers_classes = {
    'Shampoo_Plastic': 0,
    'Coconut_Water_Carton': 1,
    'Almond_Milk_Carton': 2,
    'Tomato_Juice_Carton': 3,
    'Ananas_Juice_Carton': 4,
    'Rice_Milk_Carton': 5,
    'Energy_Drink': 6,
    'Short_Beer_Can': 7,
    'Coke_Can': 8,
    'Tall_Beer_Can': 9,
    'Pineapple_Nectar_Tin': 10,
    'Green_Tea_Plastic': 11,
    'Oil_Tin_Can': 12,
    "Shai's": 13,  # Silver Vacuum Flask
    'Black_Vacuum_flask': 14,
    'Champagne_Glass': 15,
    'Orange_Vacuum_flask': 16,
    'Pitcher_Vacuum_flask': 17,
    'Conditioner': 18,
    'Oatly': 19,
    'Contigo': 20,
    'Delta': 21,  # Matte Vacuum Flask
}


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_length', type=int, default=10200)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--token_size', type=int, default=100)
    parser.add_argument('--signal_is', type=str, default='magnitude', choices=('magnitude', 'complex', 'mag_phase'))

    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--pnt_transformer_num_heads', type=int, default=4)
    parser.add_argument('--pnt_transformer_num_layers', type=int, default=8)

    parser.add_argument('--cls_transformer_num_heads', type=int, default=4)
    parser.add_argument('--cls_transformer_num_layers', type=int, default=8)
    # ablations
    parser.add_argument('--ablation_point', type=int, default=-1,
                        help='for ablating working on one point')
    parser.add_argument('--ablation_set_prediction', type=bool, default=False,
                        help='do not add LPE for the ShaptTransformer, discard information on position of points')

    parser.add_argument('--token_dropout_prob', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--data_version', type=str, default='v5')

    # save the experiment
    parser.add_argument('--exp_base', type=str, default='/home/kichlerm/speckle-vibration-analysis/experiments_iccv')
    parser.add_argument('--exp_name', type=str, default='data_v5_baseline')
    parser.add_argument('--tag', type=str, default='v0')
    parser.add_argument('--validate_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=500)
    return parser


def args_to_hyperparams(args):
    hyperparameters = {k.upper(): v for k, v in args.__dict__.items()}
    return hyperparameters


def hermit_poly(t):
    tt = t[None, :] ** torch.arange(4, device=t.device)[:, None]
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=t.device)
    return A @ tt


def interp(x, y, xs):
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[1:], xs)
    dx = (x[idxs + 1] - x[idxs])
    hh = hermit_poly((xs - x[idxs]) / dx)
    return hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx


def frequency_augmentation(local_device):
    # Initial points and domain
    x_points = torch.tensor([500, 1000, 1500, 2000, 2500], dtype=dtype, device=local_device)
    y_points = torch.normal(mean=1.0, std=1, size=(len(x_points),), device=local_device)
    domain = torch.linspace(100, 2500, 10000, device=local_device)  # TODO: arguments of F^{sample} the FIXED frequency domain

    # Interpolate values over the domain
    values = interp(x_points, y_points, domain)

    # Normalize values
    values = (values - torch.min(values)) / (torch.max(values) - torch.min(values))

    # # Spline between 0.5 to 2
    # normalized_values = values * 1.5 + 0.5

    # Spline between 0.8 to 1.2
    normalized_values = values / 2.5 + 0.8

    # Frequency range for the FFT
    f = fftfreq(10200, 1 / 5100, device=local_device)  # TODO: arguments of F^{sample} the FIXED frequency domain

    # Filter frequencies in the desired range and assign values
    valid_freq_mask = (f >= 100) & (f <= 2500)
    G = torch.zeros_like(f, dtype=torch.float32, device=local_device)
    G[valid_freq_mask] = normalized_values[torch.searchsorted(domain, f[valid_freq_mask])]

    return G


class SignalDataset(Dataset):
    def __init__(self, data_frame, name, hyperparameters, augmentation=True):
        super(SignalDataset, self).__init__()
        self.name = name  # name the dataset
        self.token_size = hyperparameters["TOKEN_SIZE"]
        #  self.num_tokens = hyperparameters["SIGNAL_LENGTH"] // hyperparameters["TOKEN_SIZE"]
        self.apply_augmentation = augmentation
        self.data_frame = data_frame

    def __len__(self):
        return self.data_frame.shape[0]

    def __getitem__(self, idx):
        item = self.data_frame.iloc[idx]

        raw_signal = torch.from_numpy(item['signal']).to(dtype)  # 3-2-T
        # local_device = 'cpu'
        # augmentation
        if self.apply_augmentation:
            G = frequency_augmentation(raw_signal.device)[None, None, :]
        else:
            G = 1.0
        fourier_signal = fft(raw_signal, dim=-1)
        cplx = (G * (fourier_signal))[..., 200:5000]  # 100Hz - 2500Hz. TODO sample here!
        # normalizing the magnitude of the fourier transform
        cplx /= cplx.abs().std(dim=-1, keepdim=True)
        # print(f'mag={cplx.shape} {cplx.dtype}')  # shape 3-2-4800
        sample = self.split_signal(cplx, self.token_size)  # , self.num_tokens)
        # print(f'sample={sample.shape} {sample.dtype}')   # shape 3-n_tokens(48)-2-100
        capacity_label = float(item['level'])
        class_label = containers_classes['_'.join(item['container'].split(' '))]

        return sample, capacity_label, class_label

    @staticmethod
    def split_signal(sample, token_size):  # , num_tokens):
        # force number of tokens
        sig_len = sample.shape[-1]
        num_tokens = sig_len // token_size
        if num_tokens * token_size < sig_len:
            print(f'WARNING: truncating signal of len {sig_len} to fit {num_tokens} tokens of size {token_size}')
            sample = sample[..., :(num_tokens * token_size)]
        reshaped_sample = torch.zeros((3, num_tokens, 2, token_size), dtype=sample.dtype)  # .to(device)

        for i in range(3):
            point_data = sample[i]

            # Reshape to (num_tokens, 2, token_size)
            tokens = point_data.reshape(2, num_tokens, token_size).permute(1, 0, 2)
            reshaped_sample[i] = tokens

        return reshaped_sample


class TokensLinearProjection(nn.Module):
    def __init__(self, token_size, d_model):
        super(TokensLinearProjection, self).__init__()
        self.projection = nn.Linear(2 * token_size, d_model)

    def forward(self, tokens):
        """
        Args:
            tokens: Tensor of shape (batch_size, num_tokens, 2, token_size)
                   where the last two dimensions represent two components (e.g., channels) and the token size.
        Returns:
            Tensor of shape (batch_size, num_tokens, d_model)
        """
        # Flatten the last two dimensions (2, token_size) -> (2 * token_size)
        flattened_tokens = tokens.view(tokens.size(0), tokens.size(1), -1)  # (batch_size, num_tokens, 2 * token_size)

        # Project to d_model dimensions
        projected_tokens = self.projection(flattened_tokens)  # (batch_size, num_tokens, d_model)
        return projected_tokens


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        d_model = hyperparameters["D_MODEL"]
        num_tokens = hyperparameters["SIGNAL_LENGTH"] // hyperparameters["TOKEN_SIZE"]
        self.embed = nn.Embedding(num_tokens, d_model)  # Learnable embeddings

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.embed(positions)


class PointTransformer(nn.Module):
    def __init__(self, hyperparameters):
        super(PointTransformer, self).__init__()

        token_size = hyperparameters["TOKEN_SIZE"]
        d_model = hyperparameters["D_MODEL"]
        num_heads = hyperparameters["PNT_TRANSFORMER_NUM_HEADS"]
        num_layers = hyperparameters["PNT_TRANSFORMER_NUM_LAYERS"]
        self.signal_is = hyperparameters.get("SIGNAL_IS", "magnitude")  # "magnitude", "comlex" or "mag_phase"
        if self.signal_is == 'magnitude':
            pass
        elif self.signal_is == 'complex':
            token_size *= 2
        elif self.signal_is == 'mag_phase':
            token_size *= 2

        # Token embedding projection
        self.embedding = TokensLinearProjection(token_size, d_model)

        # Positional encoding
        self.learnable_positional_encoding = LearnablePositionalEncoding(hyperparameters)

        # Transformer encoder setup
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Token dropout probability
        self.token_dropout_prob = hyperparameters["TOKEN_DROPOUT_PROB"]

    def _raw_signal_to_tokenizable_one(self, tokens):
        # tokens are: (batch_size, num_tokens, 2, token_size)  of type COMPLEX!
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
        # Input tokens shape: (batch_size, num_tokens, 2, token_size)
        tokens_emb = self.embedding(tokens)  # Project tokens: (batch_size, num_tokens, d_model)

        # Add positional encoding to token embeddings
        tokens_emb = self.learnable_positional_encoding(tokens_emb)  # (batch_size, num_tokens, d_model)
        if self.training:  # Apply dropout only during training
            batch_size, num_tokens, c = tokens_emb.size()

            # Apply random dropout for the remaining tokens
            new_n = max(1, int((1.0 - self.token_dropout_prob) * num_tokens))
            with torch.no_grad():
                r = torch.rand_like(tokens_emb[..., 0])  # B-N
                ridx = torch.argsort(r, dim=-1)[:, :new_n]
            tokens_emb = torch.gather(tokens_emb, dim=1, index=ridx[..., None].expand(-1, -1, c))

        # Add PNT token
        batch_size = tokens_emb.size(0)
        cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        tokens_emb = torch.cat((cls_token_expanded, tokens_emb), dim=1)  # (batch_size, num_tokens + 1, d_model)

        # Transformer encoder
        output = self.transformer_encoder(tokens_emb)  # (batch_size, num_tokens + 1, d_model)

        # Return PNT token representation
        PNT = output[:, 0, :]  # (batch_size, d_model)
        return PNT


def sord_loss(predictions, hard_targets, cost_matrix):
    num_classes = predictions.size(1)

    batch_size = hard_targets.size(0)
    soft_labels = torch.zeros(batch_size, num_classes, device=predictions.device)
    for idx in range(batch_size):
        true_label = round(hard_targets[idx].item() * (num_classes - 1))
        soft_labels[idx] = torch.exp(-cost_matrix[true_label])

    soft_labels = nnf.normalize(soft_labels, p=1, dim=1)  # Normalize to create a probability distribution

    # Log-softmax for predictions
    log_predictions = nnf.log_softmax(predictions, dim=-1)
    # Cross-entropy loss between soft labels and predictions
    loss = -(soft_labels * log_predictions).sum(dim=1).mean()
    return loss


class SignalTransformer(nn.Module):
    def __init__(self, hyperparameters):
        super(SignalTransformer, self).__init__()

        # Extract hyperparameters
        num_points = 3  # Number of PointTransformers
        d_model = hyperparameters["D_MODEL"]
        num_heads = hyperparameters["CLS_TRANSFORMER_NUM_HEADS"]
        num_layers = hyperparameters["CLS_TRANSFORMER_NUM_LAYERS"]
        alpha = hyperparameters["ALPHA"]
        self.alpha = alpha

        # Initialize one PointTransformer for every point
        self.point_transformer = PointTransformer(hyperparameters)

        # Positional encoding for point embeddings
        if hyperparameters.get("ABLATION_SET_PREDICTION", False):
            print('\n', '*'*40, f'ABLATION STUDY -- NO ShapeTrans as SET PREDICTION!!!\n', '*'*40, '\n')
            self.learnable_positional_encoding = nn.Identity()
        else:
            self.learnable_positional_encoding = LearnablePositionalEncoding(hyperparameters)

        self.pick_point = hyperparameters.get("ABLATION_POINT", -1)
        if self.pick_point >= 0:
            print('\n', '*'*40, f'ABLATION STUDY -- NO ShapeTransformer!!!\n selecting point {self.pick_point}', '\n')
        # Sequence-level transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable [CLS] token for the sequence-level transformer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=.02)  # Initialize to small random values

        # Prediction heads
        self.mlp_head_capacity = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, len(fill_levels))
        )
        self.mlp_head_classification = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, len(containers_classes.keys()))
        )

        # Precompute ordinality matrix for SORD loss

    #         self.ordinality_matrix = self.create_ordinality_matrix(VECTOR_SIZE)

    def forward(self, samples):
        batch_size = samples.size(0)
        pnt_tokens = []

        # TODO: batch process all together!!
        # Pass each point's data through PointTransformer
        points_to_process = range(samples.size(1)) if self.pick_point == -1 else [self.pick_point,]
        for i in points_to_process:
            point_data = samples[:, i]  # Shape: (batch_size, num_tokens, token_size)
            pnt_token = self.point_transformer(point_data)  # Output shape: (batch_size, d_model)
            pnt_tokens.append(pnt_token)

        # if self.pick_point == -1:
        # regular mode
        # Stack CLS tokens from PointTransformers: (batch_size, num_points, d_model)
        pnt_tokens = torch.stack(pnt_tokens, dim=1)

        # Add positional encoding
        pnt_tokens = self.learnable_positional_encoding(pnt_tokens)

        # Add learnable sequence CLS token: (batch_size, 1, d_model)
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        transformer_input = torch.cat((cls_token, pnt_tokens), dim=1)  # Shape: (batch_size, num_points + 1, d_model)

        # Process through the sequence-level transformer
        output = self.transformer_encoder(transformer_input)  # Shape: (batch_size, num_points + 1, d_model)

        # Extract the sequence-level CLS embedding
        cls_embedding = output[:, 0, :]  # Shape: (batch_size, d_model)
        # else:
        #     # ablation mode
        #     assert len(pnt_tokens) == 1
        #     cls_embedding = pnt_tokens[0]

        # Predict capacity and classification outputs
        logits_capacity_pred = self.mlp_head_capacity(cls_embedding)  # Shape: (batch_size, VECTOR_SIZE)
        logits_class_pred = self.mlp_head_classification(cls_embedding)  # Shape: (batch_size, NUM_CLASSES)

        return logits_capacity_pred, logits_class_pred, cls_embedding

    def compute_loss(self, logits_capacity_pred, logits_class_pred, capacity_target, class_target):
        """
        Compute combined loss using SORD loss for capacity and cross-entropy for classification.
        """
        # Create the cost matrix for SORD loss
        num_classes = logits_capacity_pred.size(1)
        multiplier = 2

        # Step 1: Create cost matrix
        cost_matrix = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                cost_matrix[i, j] = multiplier * abs(i - j) ** 2

        # Capacity loss (SORD Loss)
        capacity_loss = sord_loss(logits_capacity_pred, capacity_target, cost_matrix)

        # Classification loss (Cross-Entropy)
        classification_loss = nnf.cross_entropy(logits_class_pred, class_target)

        # Combined loss with weighting factor alpha
        total_loss = self.alpha * capacity_loss + (1 - self.alpha) * classification_loss

        return total_loss, capacity_loss, classification_loss


def count_parameters(model: nn.Module) -> int:
    return sum([p_.numel() for p_ in model.parameters()])


@torch.no_grad()
def validate(epoch, model, val, writer=None, detailed=False, latex=False, l_acc_as_nan=False):
    model.eval()
    p_ = next(model.parameters())
    device, dtype = p_.device, p_.dtype

    acc_c = []
    acc_l = []
    err_l = []
    container = []

    level = torch.tensor(fill_levels, device=device, dtype=dtype, requires_grad=False)[None, :]
    for step, (x, l, c) in enumerate(val):
        x = x.to(device)  # B x 3 x SeqLen x 2 x WinSize
        c = c.to(device)
        l = l.to(device, dtype)
        l_int = (l * 5.).to(torch.int64)  # integer level labels

        l_pred, c_pred, _ = model(x)

        acc_c.append((c_pred.argmax(dim=-1) == c).to(dtype))

        el = ((torch.softmax(l_pred, dim=-1) * level).sum(dim=-1) - l).abs()
        err_l.append(el)
        # err2_l += (el ** 2).sum().item()

        acc_l.append((torch.argmax(l_pred, dim=-1) == l_int).to(dtype))
        container.append(c)

    acc_c = torch.concatenate(acc_c, dim=0).to(dtype)
    acc_l = torch.concatenate(acc_l, dim=0).to(dtype)
    err_l = torch.concatenate(err_l, dim=0).to(dtype)
    container = torch.concatenate(container, dim=0)

    if detailed:
        # per container
        nc = len(containers_classes.keys())  # number of containers
        per_cont = torch.index_add(input=torch.zeros((nc, 5), device=device), dim=0, index=container,
                                   source=torch.stack([acc_c, acc_l, err_l, err_l**2, torch.ones_like(acc_c)],
                                                      dim=1))
        print(f'\n**{val.dataset.name}:**')
        if not latex:
            print(f'\t#l {"container":>20} | {"MAE":>15} | {"Acc L":>6} | {"Acc C":>6} | count')
        else:
            print('          &          & \multicolumn{2}{c|}{Level Pred.}           & Container \t \\\\')
            print('Container & \\#Samp. & Acc.\\(\\uparrow\\) & MAE\\(\\downarrow\\) & Acc.\\(\\uparrow\\) \t \\\\\n\hline')
        for cname, clabel in containers_classes.items():
            fancy_cname = cname.replace('_', ' ')
            n = per_cont[clabel, -1].item()  # number of instances of this container in the set
            if n > 0:
                # we have this container in this set
                cca = (per_cont[clabel, 0] / n).item()
                cla = 'N/A' if l_acc_as_nan else f'{(per_cont[clabel, 1] / n).item():.4f}'
                cle = (per_cont[clabel, 2] / n).item()
                cle2 = (per_cont[clabel, 3] / n).item()
                cle_std = np.sqrt(cle2 - cle ** 2)
                if not latex:
                    print(f'\t{clabel:02d} {cname:>20} | {cle:.4f} ({cle_std:.4f}) | {cla} | {cca:.4f} | {int(n)}')
                else:
                    print(f'{fancy_cname} & {int(n)} & {cla} & {cle:.4f} \\err{{{cle_std:.4f}}} & {cca:.4f} \\\\')
        if latex:
            print('\\hline')
    # global statistics
    n = acc_c.shape[0]
    acc_c = acc_c.mean().item()
    acc_l = 'N/A' if l_acc_as_nan else f'{acc_l.mean().item():.4f}'
    l_err = err_l.mean().item()
    l_err_std = err_l.std().item()

    if detailed:
        if latex:
            print(f'Total: & {n} & {acc_l} & {l_err:.4f} \\err{{{l_err_std:.4f}}} & {acc_c:.4f} \\\\\n\\hline')
    else:
        if latex:
            print(f'{val.dataset.name} & {acc_l} & {l_err:.4f} \\err{{{l_err_std:.4f}}} & {acc_c:.4f} \\\\')
        else:
            print(f'Done validation {val.dataset.name} epoch {epoch}. Classification accuracy = {acc_c:.4f} '
                f'\t level L1 err = {l_err:.4f} ({l_err_std:.4f}), acc = {acc_l}\n')
    if writer is not None:
        writer.add_scalar(f'{val.dataset.name}/MAE', l_err, epoch)
        writer.add_scalar(f'{val.dataset.name}/AccL', acc_l, epoch)
        writer.add_scalar(f'{val.dataset.name}/Acc', acc_c, epoch)
    return l_err


def train(epoch, model, opt, train_loader, writer):
    model.train()
    p_ = next(model.parameters())
    device, dtype = p_.device, p_.dtype

    acc_c = 0
    acc_l = 0
    pbar = tqdm.tqdm(iterable=train_loader, total=len(train_loader))
    for step, (x, l, c) in enumerate(pbar):
        opt.zero_grad()
        x = x.to(device)
        c = c.to(device)
        l = l.to(device, dtype)

        l_pred, c_pred, _ = model(x)

        loss, c_loss, l_loss = model.compute_loss(l_pred, c_pred, l, c)

        acc_c += c_loss.item()
        acc_l += l_loss.item()
        pbar.set_description(f'Train [{epoch:02d}] c={acc_c / (step + 1):.4f} l={acc_l / (step + 1):.4f}')

        loss.backward()
        opt.step()
        with torch.no_grad():
            writer.add_scalar('Train/c_loss', c_loss.item(), epoch * len(train_loader) + step)
            writer.add_scalar('Train/l_loss', l_loss.item(), epoch * len(train_loader) + step)
            writer.add_scalar('Train/loss', loss.item(), epoch * len(train_loader) + step)
    print(f'Done training epoch {epoch}. Classification loss = {acc_c / len(train_loader.dataset):.4f} '
          f'\t level loss = {acc_l / len(train_loader.dataset):.4f}')


def get_dataframes(data_version):
    # DATA_BASE = '/mnt/mark_lab/DATA/dataset_v1'
    DATA_BASE = '/home/kichlerm/speckle-vibration-analysis'
    train_df = pd.read_pickle(os.path.join(DATA_BASE, f'train_{data_version}.pkl'))
    dup_df = pd.read_pickle(os.path.join(DATA_BASE, f'duplicates_testset_{data_version}.pkl'))
    test_df = [(pd.read_pickle(os.path.join(DATA_BASE, f'test0_{data_version}.pkl')), 'test0'),
               (dup_df, 'test_dup')]
    # add each container in dup test
    for container in dup_df['container'].unique():
        test_df.append((dup_df[dup_df['container'] == container], container))
    return train_df, test_df


def main():
    parser = get_parser()
    args = parser.parse_args()
    hyperparameters = args_to_hyperparams(args)

    dst = os.path.join(args.exp_base, args.exp_name + '_' + args.tag)
    os.makedirs(dst, exist_ok=True)

    NUM_WORKERS = 0 if 'dbg' in args.tag.lower() else 2
    train_set, test_sets = data_frames.make_data_frames()  #  get_dataframes(hyperparameters["DATA_VERSION"])
    train_loader = DataLoader(SignalDataset(train_set, name='train', hyperparameters=hyperparameters),
                              batch_size=hyperparameters['BATCH_SIZE'],
                              shuffle=True,
                              drop_last=False,
                              num_workers=NUM_WORKERS,
                              persistent_workers=NUM_WORKERS > 0)
    test_loaders = [DataLoader(SignalDataset(test_set_, name=t_name_,
                                             hyperparameters=hyperparameters,
                                             augmentation=False),
                               batch_size=hyperparameters['BATCH_SIZE'],
                               shuffle=False,
                               drop_last=False,
                               num_workers=NUM_WORKERS,
                               persistent_workers=NUM_WORKERS > 0) for (test_set_, t_name_) in test_sets]

    model = SignalTransformer(hyperparameters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['LR'])

    # sanity check on datasets
    print(f'***\nRunning on {device} with {dtype}\n***\n')
    print(f'\n\nTraining set sanity: |train|={len(train_loader.dataset)} |loader|={len(train_loader)}')
    print('train containers = ', set([train_loader.dataset[i_][2] for i_ in range(len(train_loader.dataset))]))
    print('train levels = ', set([train_loader.dataset[i_][1] for i_ in range(len(train_loader.dataset))]))
    for v_ in test_loaders:
        print(f'{v_.dataset.name} sanity: |{v_.dataset.name}|={len(v_.dataset)} |loader|={len(v_)}')
        print(v_.dataset.name, ' containers = ', set([v_.dataset[i_][2] for i_ in range(len(v_.dataset))]))
        print(v_.dataset.name, ' levels = ', set([v_.dataset[i_][1] for i_ in range(len(v_.dataset))]))
    print('\n\n')
    print(f'\nTraining with\n{hyperparameters}\n')

    writer = SummaryWriter(log_dir=dst)
    [validate(-1, model, v_, writer) for v_ in test_loaders]
    for epoch in range(hyperparameters['EPOCHS']):
        train(epoch, model, optimizer, train_loader, writer)
        # validate
        if ((epoch + 1) % args.validate_every) == 0:
            [validate(epoch, model, v_, writer) for v_ in test_loaders]
        if ((epoch + 1) % args.save_every) == 0:
            torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(),
                        'epoch': epoch, 'hyp': hyperparameters, 'args': args}, f'{dst}/checkpoint-{epoch:04d}.pth')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
