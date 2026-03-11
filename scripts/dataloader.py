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
    tensor off disk — the OS never loads the full ~900 MB into RAM.
    """

    def __init__(self, repo_id: str = REPO_ID, split: str = "train", token: str = None):
        self.ds = load_dataset(repo_id, split=split, token=token)
        # Downloads on first run, returns cached path on subsequent runs
        path = hf_hub_download(repo_id, "shifts.safetensors", repo_type="dataset", token=token)
        # Memory-mapped: file stays on disk, OS pages in only what's accessed
        self.st = safe_open(path, framework="pt", device="cpu")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = dict(self.ds[idx])
        row["shifts"] = self.st.get_tensor(f"shifts_{row['shifts_idx']}")  # reads ~7 MB off disk
        row["overhead_image"] = TF.to_tensor(row["overhead_image"])
        return row


def get_dataloader(repo_id: str = REPO_ID, split: str = "train", token: str = None, **kwargs) -> DataLoader:
    return DataLoader(VibrationDataset(repo_id, split, token), **kwargs)


if __name__ == "__main__":
    loader = iter(get_dataloader(split="train", batch_size=2))

    for i in range(10):
        batch = next(loader)
        print(f'batch {i}')
        print("shifts:", batch["shifts"].shape)  # (2, 100, 9000, 2)
        print("x:", batch["x_position"])
        print("fps:", batch["fps"])
