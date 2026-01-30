import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from icecream import install

install()


class VibrationDataset(Dataset):
    """Dataset that returns (fft_vals, label)."""

    def __init__(self, data_path: Path, label_path: Path):
        self.fft_vals = torch.load(data_path)
        self.labels = torch.load(label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # compute the fft magnitude with abs()
        return self.fft_vals[idx].abs(), self.labels[idx]


def get_dataloaders(
    data_dir: Path,
    target: str = 'x',
    batch_size: int = 8,
    test_split: float = 0.2,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: int = 42,
):
    """Create train/test dataloaders for position prediction.

    Args:
        data_dir: Path to directory containing fft_vals.pt and label files.
        target: 'x' or 'y' to specify which position labels to use.
        batch_size: Batch size for dataloaders.
        test_split: Fraction of data to use for testing.
        shuffle: Whether to shuffle training data.
        num_workers: Number of workers for data loading.
        seed: Random seed for reproducibility.

    Returns:
        tuple: (train_loader, test_loader)
    """
    dataset = VibrationDataset(data_dir / "fft_vals.pt", data_dir / f"{target}_labels.pt")

    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=generator)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=generator)

    return train_loader, test_loader


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    train_loader, test_loader = get_dataloaders(Path(args.data_dir), target='x', batch_size=args.batch_size, seed=args.seed)

    ic(len(train_loader.dataset), len(test_loader.dataset))
    for fft_vals, pos in train_loader:
        ic(fft_vals.shape, pos.shape)
        break


if __name__ == '__main__':
    main()
