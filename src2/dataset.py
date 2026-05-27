import os, json, math

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from torch.nn import functional as F
from huggingface_hub import snapshot_download
from composer.core import Evaluator, DataSpec
from streaming import StreamingDataset

from src2.helpers import SPEAKER_EMBD

DATA_INFO = {'out_h': 40, 'out_w': 20, 'n_freqs': 3328, 'n_laser_rows': 10, 'n_laser_cols': 10, 'patch_size': 256,
             'x_pos': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None], 'y_pos': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, None]}
SPEAKER_TYPE = {'1000': 'white', '0100': 'white', '0010': 'black', '0001': 'black'}
SPEAKER_TYPE_REVERSE = {'white': ['1000', '0100'], 'black': ['0010', '0001']}

# BAD_SAMPLES = {
#     "outlier-fft-magnitude": {
#         "0001": [525, 529, 337],
#         "0010": [526, 393, 530, 90, 94, 50, 54, 58, 62, 70, 74, 78, 82, 30, 38, 42],
#         "0100": [527, 531, 394],
#         "1000": [480, 528, 395, 532, 396, 52, 56, 60, 64, 68, 72, 76, 80, 84, 28, 32, 36, 40, 44],
#     },
# }
# BAD_SAMPLE_IDS = {id for _bad_samples in BAD_SAMPLES.values() for ids in _bad_samples.values() for id in ids}
BAD_SAMPLE_IDS = {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 89, 90, 91, 94, 337, 393, 394, 395, 480, 525, 526, 527, 528, 529, 530, 531, 532}

def process_masks(masks:torch.Tensor, out_h:torch.Tensor, out_w:torch.Tensor, verbose:bool=True):
    # discretize masks and cast to float
    if verbose: print('Processing masks...')
    masks = F.adaptive_avg_pool2d(masks, (out_h, out_w)).squeeze()
    if verbose: print(f"{masks.shape=}\t{masks.dtype=}\n")
    return masks

def raw_to_tokens(x:torch.Tensor, signal_mode:str) -> torch.Tensor:
    if signal_mode == "magnitude": return x.abs()
    if signal_mode == "complex": return torch.cat([x.real, x.imag], dim=-1)
    if signal_mode == "mag_phase": return torch.cat([x.abs(), x.angle()], dim=-1)
    raise ValueError(f"Unknown signal mode: {signal_mode}")

def normalize_fft(x:torch.Tensor, normalize_mode:str, speakers, verbose:bool=True) -> torch.Tensor:
    if normalize_mode is None: return x
    if normalize_mode == 'std-sample':
        std = x.std(dim=(1,2,3), keepdim=True).clamp(min=1e-8)
        if verbose: print(f'Normalize {normalize_mode}\n{std.shape=}\n{std.squeeze()=}')
        return x / std
    if normalize_mode == 'z-sample':
        mean, std = x.mean(dim=(1,2,3), keepdim=True), x.std(dim=(1,2,3), keepdim=True).clamp(min=1e-8)
        if verbose: print(f'Normalize {normalize_mode}\n{mean.shape=}\t{std.shape=}\n{mean.squeeze()=}\n{std.squeeze()=}')
        return (x-mean) / std
    if normalize_mode == 'z-global-sample':
        mean, std = x.mean(dim=0), x.std(dim=0).clamp(min=1e-8)
        if verbose: print(f'Normalize {normalize_mode}\n{mean.shape=}\t{std.shape=}\n{mean=}\n{std=}')
        return (x - mean) / std
    if normalize_mode == 'z-global':
        mean, std = x.mean(), x.std().clamp(min=1e-8)
        if verbose: print(f'Normalize {normalize_mode}\nmean={mean:.4f}  std={std:.4f}')
        return (x - mean) / std
    if normalize_mode == 'z-speaker':
        if verbose: print(f'Normalize {normalize_mode}')
        for spk in sorted(set(speakers)):
            idx = [i for i, s in enumerate(speakers) if s == spk]
            mean, std = x[idx].mean(), x[idx].std().clamp(min=1e-8)
            x[idx] = (x[idx] - mean) / std
            if verbose: print(f'\t{spk=}  n={len(idx)}  mean={mean.mean():.4f}  std={std.mean():.4f}')
        return x
    if normalize_mode == 'z-speaker-type':
        for speaker_type, speakers_in_type in SPEAKER_TYPE_REVERSE.items():
            idx = [i for i, s in enumerate(speakers) if s in speakers_in_type]
            mean, std = x[idx].mean(), x[idx].std().clamp(min=1e-8)
            x[idx] = (x[idx] - mean) / std
            if verbose: print(f'\t{speaker_type=}\tspeakers={speakers_in_type}\tn={len(idx)}\tmean={mean.mean():.4f}\tstd={std.mean():.4f}')
        return x
    raise ValueError(f"Unknown normalize mode: {normalize_mode}")

def process_fft(fft:torch.Tensor, patch_size, signal_mode:str, normalize_mode:str, speakers:list, patchify:bool=True, verbose:bool=True):
    # signal-ify, normalize, and patchify FFTs
    if verbose: print('Processing FFTs...')
    # Note: F_ is the actual num freqs and F=F_ or 2*F_ depending on signal_mode
    fft = raw_to_tokens(fft, signal_mode).float()    # (B,L,F_,2) -> (B,L,F,2)
    fft = normalize_fft(fft, normalize_mode, speakers, verbose)         # (B,L,F,2) -> (B,L,F,2)

    # Note: unfold drops entries that do not fully fit into patch_size
    if patchify: fft = fft.unfold(2, patch_size, patch_size)     # (B,L,F,2) -> (B,L,P,2,PS)
    if verbose: print(f'{fft.shape=}\t{fft.dtype=}\n')
    return fft

class VibrationDataset(StreamingDataset):
    def __init__(self, repo_id:str, patch_size:int, out_h:int, out_w:int, normalize_mode:str='z', signal_mode:str='magnitude',
                 speakers:list[int,str]|list[int]|list[str]|str|None=None, n_objects:list[int]|int|None=None, n_samples:int=None,
                 num_proc:int=8, dry_run:bool=False):
        print(f'Downloading dataset {repo_id}...')
        self.ds = load_dataset(repo_id, split="train", num_proc=num_proc) # this is `data/metadata.jsonl`
        self.ds = self.ds.remove_columns(['overhead_file_name', 'speckle_vibrations_file_name', 'speckle_shifts_ifft_audio_file_name', 'audio_file_name', 'mask_file_name'])
        print(f"Loaded dataset with {len(self.ds)} samples\n")

        # record all the x, y positions in the box
        self.x_pos_encoder, self.y_pos_encoder = LabelEncoder().fit(self.ds['x_position']), LabelEncoder().fit(self.ds['y_position'])
        self.x_positions, self.y_positions = self.x_pos_encoder.classes_, self.y_pos_encoder.classes_
        print(f"x positions: {self.x_positions}\ny positions: {self.y_positions}\n")

        # map all speakers to integers
        self.speakers_encoded = LabelEncoder().fit_transform(self.ds['speakers'])

        # Filter the dataset
        print('Filtering dataset...')
        print(f'Remove bad samples\n{BAD_SAMPLE_IDS=}')
        self.ds = self.ds.filter(lambda row: row["sample_id"] not in BAD_SAMPLE_IDS)
        if speakers is not None:
            self.ds = self.ds.filter(lambda row: row["speakers"] in (str(speakers) if isinstance(speakers, list) else [speakers]), num_proc=num_proc)
            print(f"Filtered dataset to {len(self.ds)} samples with speakers={speakers}")
        if n_objects is not None:
            self.ds = self.ds.filter(lambda row: row["n_objects"] in (n_objects if isinstance(n_objects, list) else [n_objects]), num_proc=num_proc)
            print(f"Filtered dataset to {len(self.ds)} samples with n_objects={n_objects}")
        if n_samples is not None:
            n_samples_ = min(n_samples, len(self.ds))
            self.ds = self.ds.select(range(n_samples_))
            print(f"Selected first {n_samples_} samples from the dataset")
        print(f"Final dataset contains {len(self.ds)} samples\n")

        if dry_run:
            print("Dry run mode - skipping download and preprocessing of masks and FFTs")
            return

        # Download the segmentation masks and speckle shift FFTs for the filtered dataset
        print('Downloading masks and FFTs...')
        def get_path(paths): return [json.loads(manifest)['artifacts'][paths] for manifest in list(self.ds['manifest'])]
        mask_paths, fft_paths = get_path('mask_npz'), get_path('speckle_shifts_fft')
        print(f"Mask paths: {mask_paths[:5]}...\nFFT paths: {fft_paths[:5]}...")
        snapshot_dir = snapshot_download(repo_id, repo_type="dataset", allow_patterns=set(mask_paths+fft_paths)) # might be duplicate paths for masks
        print(f"Downloaded snapshot to {snapshot_dir}\n")

        # Load the masks and FFTs into RAM
        print('Loading masks and FFTs...')
        def load_sample(paths, key): return torch.stack([torch.from_numpy(np.load(os.path.join(snapshot_dir, path))[key]) for path in paths])
        self.masks, self.fft = load_sample(mask_paths, 'mask'), load_sample(fft_paths, 'fft') # (B,H,W) (B,L,F_,2)
        print(f"masks.shape={self.masks.shape}\tmasks.dtype={self.masks.dtype}\nfft.shape={self.fft.shape}\tfft.dtype={self.fft.dtype}\n")

        # process them
        self.masks = process_masks(self.masks[:, None].float(), out_h, out_w)
        self.fft = process_fft(self.fft, patch_size, signal_mode, normalize_mode, self.ds['speakers'])

    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        def pos(idx, axis): return -1 if self.ds[idx][f'{axis}_position'] is None else self.ds[idx][f'{axis}_position']
        info = dict(sample_id=self.ds[idx]['sample_id'], x_position=pos(idx, 'x'), y_position=pos(idx, 'y'), n_objects=self.ds[idx]['n_objects'], speakers=self.ds[idx]['speakers'])
        return dict(mask_true=self.masks[idx], fft=self.fft[idx], info=info) | (dict(speakers_encoded=self.speakers_encoded[idx]) if SPEAKER_EMBD else {})

def build_dataset(repo_id:str='eturok-weizmann/laser-vibrations', patch_size:int=256, out_h:int=40, out_w:int=20, batch_size:int=64, eval_batch_size:int=64,
                  seed:int=42, generator=None, test_size:float=0.2, num_workers:int=8, speakers:list[int,str]|list[int]|list[str]|str|None=None,
                  n_objects:int|None=None, n_samples:int|None=None, num_proc:int=8, dry_run:bool=False,
                  normalize_mode:str='z-global', signal_mode:str='magnitude'):
    if generator is None: generator = torch.Generator().manual_seed(seed)
    dataset = VibrationDataset(repo_id, patch_size, out_h, out_w, normalize_mode, signal_mode, speakers, n_objects, n_samples, num_proc, dry_run)

    # define indices for the train and eval sets (base, unseen positions, multi-object)
    held_out_positions = set([(3, 4), (7, 8), (1, 9), (2, 2), (6, 6)])
    unseen_pos_indices = set([i for i, d in enumerate(dataset.ds) if (d['x_position'], d['y_position']) in held_out_positions])
    multi_object_indices = set([i for i, d in enumerate(dataset.ds) if d['n_objects'] > 1])
    available_indices = set(range(len(dataset))) - unseen_pos_indices - multi_object_indices
    train_indices, eval_indices = train_test_split(list(available_indices), test_size=test_size, random_state=seed, shuffle=True)
    print(f'{held_out_positions=}')
    print(f'{len(dataset)} total samples\t{len(train_indices)} train samples\t{len(eval_indices)} eval samples\t{len(unseen_pos_indices)} unseen position eval samples\t{len(multi_object_indices)} multi-object eval samples\n')
    print(f'{train_indices=}\n{eval_indices=}\n{unseen_pos_indices=}\n{multi_object_indices=}\n')

    # save data info (needed for model architecture)
    data_info = dict(out_h=dataset.masks.shape[1], out_w=dataset.masks.shape[2], n_freqs=dataset.fft.shape[2] * dataset.fft.shape[4],
                    n_laser_rows=int(math.sqrt(dataset.fft.shape[1])), n_laser_cols=int(math.sqrt(dataset.fft.shape[1])), patch_size=patch_size,
                    x_pos=list(set(dataset.x_positions)), y_pos=list(set(dataset.y_positions)))
    print(f'{data_info=}')

    def get_num_samples_in_batch(batch: dict) -> int: return batch['mask_true'].shape[0]
    def make_loader(indices, bs, shuffle):
        dl = DataLoader(Subset(dataset, list(indices)), batch_size=bs, shuffle=shuffle, num_workers=num_workers,
                        generator=generator, pin_memory=True, persistent_workers=num_workers>0,
                        prefetch_factor=4 if num_workers>0 else None)
        return DataSpec(dataloader=dl, get_num_samples_in_batch=get_num_samples_in_batch)

    train_loader = make_loader(train_indices, batch_size, shuffle=True)
    print(f"Train: batch_size={batch_size}, batches={len(train_loader.dataloader)}, n_samples={len(train_indices)}")

    eval_splits = [('eval/base', eval_indices), ('eval/unseen_pos', unseen_pos_indices), ('eval/multi_object', multi_object_indices)]
    eval_loaders = []
    for label, indices in eval_splits:
        spec = make_loader(indices, eval_batch_size, shuffle=False)
        eval_loaders.append(Evaluator(label=label, dataloader=spec))
        print(f"{label}: batch_size={eval_batch_size}, batches={len(spec.dataloader)}, n_samples={len(indices)}")

    return train_loader, eval_loaders, data_info
