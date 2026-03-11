import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import DataLoader, random_split

REPO_ID = "eturok-weizmann/vibration-data"

def clean_shifts(shifts: torch.Tensor, fs: int, lowcut: float = 50.0, highcut: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Bandpass filter, Hann window"""

    # bandpass filter
    if highcut is None:
        highcut = fs / 2 - 10
    shifts = shifts.numpy()  # (100, N_frames, 2)
    sos = butter(5, [lowcut, highcut], fs=fs, btype='band', output='sos')
    shifts = sosfiltfilt(sos, shifts, axis=1)

    # hann window smoothing
    window = np.hanning(shifts.shape[1])  # (N_frames,)
    shifts = shifts * window[np.newaxis, :, np.newaxis]
    return shifts

def fft(shifts:np.ndarray, fs:int, min_freq:int=50, max_freq:int=1000):
    fft_val = np.fft.rfft(shifts, axis=1)
    freqs = np.fft.rfftfreq(shifts.shape[1], d=1.0 / fs)
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    fft_val, freqs = fft_val[:, mask, :], freqs[mask]
    return torch.from_numpy(fft_val), torch.from_numpy(freqs)

class VibrationDataset(torch.utils.data.Dataset):
    """
    Downloads shifts.safetensors once via hf_hub_download (cached to disk after first run),
    then memory-maps it with safe_open. Each __getitem__ reads only the pages for that
    tensor off disk — the OS never loads the full file into RAM.
    """

    def __init__(self, repo_id: str = REPO_ID, split: str = "train", patch_size: int = 16, token: str|None = None):
        self.ds = load_dataset(repo_id, split=split, token=token, columns=["shifts_idx", "mask_idx", "x_position", "y_position", "object", "fps"])
        self.st_shifts = safe_open(hf_hub_download(repo_id, "shifts.safetensors", repo_type="dataset", token=token), framework="pt", device="cpu")
        self.st_masks = safe_open(hf_hub_download(repo_id, "masks.safetensors",  repo_type="dataset", token=token),  framework="pt", device="cpu")
        self.patch_size = patch_size
    def __repr__(self): return f"VibrationDataset(split={self.ds.split}, n={len(self.ds)})"
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        row = self.ds[idx]
        mask = self.st_masks.get_tensor(f"mask_{row['mask_idx']}")          # (H, W) bool
        shifts = self.st_shifts.get_tensor(f"shifts_{row['shifts_idx']}")   # (n_lasers, n_timesteps, 2)
        # clean + fft the shift data
        shifts = clean_shifts(shifts, row["fps"])                           # (n_lasers, n_timesteps, 2)
        fft_val, freqs = fft(shifts, row["fps"])                            # (n_lasers, n_freqs, 2)
        # patchify the fft
        n_lasers, n_freqs, n_coords = fft_val.shape
        n_patches = n_freqs // self.patch_size
        n_freqs_used = n_patches * self.patch_size
        if n_freqs_used < n_freqs: print(f'WARNING: dropping {n_freqs - n_freqs_used} highest frequencies')
        patches = fft_val[:, :n_freqs_used, :].reshape(n_lasers, n_patches, self.patch_size, n_coords).transpose(-2, -1) # (n_lasers, n_freqs, 2) -> (n_lasers, n_patches, 2, patch_size)
        return patches, (mask, row["x_position"], row["y_position"])

def get_dataloaders(repo_id: str = REPO_ID, patch_size: int = 16, batch_size: int = 8, eval_batch_size: int = 16, test_split: float = 0.2, shuffle: bool = True, num_workers: int = 0, seed: int = 42, token: str | None = None) -> tuple[DataLoader, DataLoader]:
    dataset = VibrationDataset(repo_id, split="train", patch_size=patch_size, token=token)
    test_size = int(len(dataset) * test_split)
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [len(dataset) - test_size, test_size], generator=generator)
    train_loader = DataLoader(train_set, batch_size=batch_size,     shuffle=shuffle,  num_workers=num_workers, generator=generator, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=eval_batch_size, shuffle=False,   num_workers=num_workers, generator=generator, pin_memory=True)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(batch_size=2, eval_batch_size=2)
    for i, (X, (y, x_pos, y_pos)) in enumerate(train_loader): # (B, n_lasers, n_patches, 2, patch_size), ((B,H,W), (B,), (B,))
        print(f"batch {i}  X={X.shape}  y={y.shape}, {x_pos=}, {y_pos=}")
        if i >= 2: break
