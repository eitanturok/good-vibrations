import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch.utils.data import DataLoader

REPO_ID = "eturok-weizmann/vibration-data"


class VibrationDataset(torch.utils.data.Dataset):
    """
    Downloads shifts.safetensors once via hf_hub_download (cached to disk after first run),
    then memory-maps it with safe_open. Each __getitem__ reads only the pages for that
    tensor off disk — the OS never loads the full file into RAM.
    """

    def __init__(self, repo_id: str = REPO_ID, split: str = "train", token: str|None = None):
        self.ds = load_dataset(repo_id, split=split, token=token,
                               columns=["shifts_idx", "mask_idx", "x_position", "y_position", "object"])
        shifts_path = hf_hub_download(repo_id, "shifts.safetensors", repo_type="dataset", token=token)
        masks_path  = hf_hub_download(repo_id, "masks.safetensors",  repo_type="dataset", token=token)
        self.st_shifts = safe_open(shifts_path, framework="pt", device="cpu")
        self.st_masks  = safe_open(masks_path,  framework="pt", device="cpu")

    def __repr__(self): return f"VibrationDataset(split={self.ds.split}, n={len(self.ds)})"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        X = self.st_shifts.get_tensor(f"shifts_{row['shifts_idx']}")  # (100, N_frames, 2)
        y = self.st_masks.get_tensor(f"mask_{row['mask_idx']}")       # (H, W) bool
        return X, y


def get_dataloader(repo_id: str = REPO_ID, split: str = "train", token: str|None=None, **kwargs) -> DataLoader:
    dataset = VibrationDataset(repo_id, split, token)
    return DataLoader(dataset, **kwargs)


if __name__ == "__main__":
    loader = iter(get_dataloader(split="train", batch_size=2))

    for i in range(3):
        X, y = next(loader)
        print(f"batch {i}  X={X.shape}  y={y.shape}")
