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

class VibrationDataset(Dataset):
    def __init__(self, repo_id:str, patch_size:int, out_h:int, out_w:int, speakers:list[int,str]|list[int]|list[str]|str|None=None, n_objects:list[int]|int|None=None, n_samples:int=None, num_proc:int=8, dry_run:bool=False):
        print(f'Downloading dataset {repo_id}...')
        self.ds = load_dataset(repo_id, split="train", num_proc=num_proc) # this is `data/metadata.jsonl`
        self.ds = self.ds.remove_columns(['segmented_overhead_file_name', 'speckle_vibrations_file_name', 'speckle_shifts_ifft_audio_file_name', 'audio_file_name', 'mask_file_name'])
        print(f"Loaded dataset with {len(self.ds)} samples\n")

        # record all the x, y positions in the box
        self.x_pos_encoder, self.y_pos_encoder = LabelEncoder().fit(self.ds['x_position']), LabelEncoder().fit(self.ds['y_position'])
        self.x_positions, self.y_positions = self.x_pos_encoder.classes_, self.y_pos_encoder.classes_
        print(f"x positions: {self.x_positions}\ny positions: {self.y_positions}\n")

        # Filter the dataset
        print('Filtering dataset...')
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

        # Load the masks and FFTs
        print('Loading masks and FFTs...')
        def load_sample(paths, key): return torch.stack([torch.from_numpy(np.load(os.path.join(snapshot_dir, path))[key]) for path in paths])
        self.masks, self.fft = load_sample(mask_paths, 'mask'), load_sample(fft_paths, 'fft')
        print(f"masks.shape={self.masks.shape}\tmasks.dtype={self.masks.dtype}\nfft.shape={self.fft.shape}\tfft.dtype={self.fft.dtype}\n")

        # discretize masks and cast to float
        print('Discretizing masks...')
        self.masks = F.adaptive_avg_pool2d(self.masks[:, None].float(), (out_h, out_w)).squeeze()
        print(f"masks.shape={self.masks.shape}\tmasks.dtype={self.masks.dtype}\n")

        # normalize and patchify FFTs
        print('Normalizing and patchifying FFTs...')
        # self.fft = self.fft.abs() # take magnitude of FFTs
        # drops entries that do not fully fit into patch_size
        self.fft = self.fft.unfold(2, patch_size, patch_size) # (B,L,F,2) -> (B,L,P,2,PS)
        print(f'fft.shape={self.fft.shape}\t{self.fft.dtype=}\n')

    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        info = dict(sample_id=self.ds[idx]['sample_id'], x_position=self.ds[idx]['x_position'], y_position=self.ds[idx]['y_position'], n_objects=self.ds[idx]['n_objects'], speakers=self.ds[idx]['speakers'])
        return dict(mask_true=self.masks[idx], fft=self.fft[idx], info=info)

def build_dataset(repo_id, patch_size, out_h, out_w, batch_size, eval_batch_size, seed, generator, test_size, num_workers, speakers, n_objects, n_samples, num_proc, save_folder, dry_run:bool=False):
    dataset = VibrationDataset(repo_id, patch_size, out_h, out_w, speakers, n_objects, n_samples, num_proc, dry_run)

    # define indices for the train and eval sets (base, unseen positions, multi-object)
    held_out_positions = set([(3, 4), (7, 8), (1, 9), (2, 2), (6, 6)])
    unseen_pos_indices = set([i for i, d in enumerate(dataset.ds) if (d['x_position'], d['y_position']) in held_out_positions])
    multi_object_indices = set([i for i, d in enumerate(dataset.ds) if d['n_objects'] > 1])
    available_indices = set(range(len(dataset))) - unseen_pos_indices - multi_object_indices
    train_indices, eval_indices = train_test_split(list(available_indices), test_size=test_size, random_state=seed, shuffle=True)
    print(f'{len(dataset)} total samples\t{len(train_indices)} train samples\t{len(eval_indices)} eval samples\t{len(unseen_pos_indices)} unseen position eval samples\t{len(multi_object_indices)} multi-object eval samples\n')
    print(f'{held_out_positions=}')

    # save data info (needed for model architecture)
    data_info = dict(out_h=dataset.masks.shape[1], out_w=dataset.masks.shape[2], n_freqs=dataset.fft.shape[2] * dataset.fft.shape[4],
                    n_laser_rows=int(math.sqrt(dataset.fft.shape[1])), n_laser_cols=int(math.sqrt(dataset.fft.shape[1])), patch_size=patch_size,
                    x_pos=list(set(dataset.x_positions)), y_pos=list(set(dataset.y_positions)))
    print(f'{data_info=}')

    # since dataset.__getitem__ returns info, which is a dict not a tensor,
    # we need to tell the trainer how many samples are in each batch
    def get_num_samples_in_batch(batch: dict) -> int: return batch['mask_true'].shape[0]

    # train
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator, pin_memory=False, persistent_workers=num_workers>0, prefetch_factor=4 if num_workers>0 else None)
    train_loader = DataSpec(dataloader=train_loader, get_num_samples_in_batch=get_num_samples_in_batch)
    print(f"Train dataloader: batch_size={batch_size}, batches={len(train_loader.dataloader)}, n_samples={len(train_indices)}")

    # eval (general)
    eval_loader_base = DataLoader(Subset(dataset, eval_indices), batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, generator=generator, pin_memory=False, persistent_workers=num_workers>0, prefetch_factor=4 if num_workers>0 else None)
    eval_loader_base = DataSpec(dataloader=eval_loader_base, get_num_samples_in_batch=get_num_samples_in_batch)
    eval_loader_base = Evaluator(label='eval/base', dataloader=eval_loader_base)
    print(f"Eval dataloader: batch_size={eval_batch_size}, batches={len(eval_loader_base.dataloader.dataloader)}, n_samples={len(eval_indices)}")

    # eval (unseen position)
    eval_loader_unseen_pos = DataLoader(Subset(dataset, list(unseen_pos_indices)), batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, generator=generator, pin_memory=False, persistent_workers=num_workers>0, prefetch_factor=4 if num_workers>0 else None)
    eval_loader_unseen_pos = DataSpec(dataloader=eval_loader_unseen_pos, get_num_samples_in_batch=get_num_samples_in_batch)
    eval_loader_unseen_pos = Evaluator(label='eval/unseen_pos', dataloader=eval_loader_unseen_pos)
    print(f"Eval dataloader (unseen positions): batch_size={eval_batch_size}, batches={len(eval_loader_unseen_pos.dataloader.dataloader)}, n_samples={len(unseen_pos_indices)}")

    # eval (multi-object)
    eval_loader_multi_object = DataLoader(Subset(dataset, list(multi_object_indices)), batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, generator=generator, pin_memory=False, persistent_workers=num_workers>0, prefetch_factor=4 if num_workers>0 else None)
    eval_loader_multi_object = DataSpec(dataloader=eval_loader_multi_object, get_num_samples_in_batch=get_num_samples_in_batch)
    eval_loader_multi_object = Evaluator(label='eval/multi_object', dataloader=eval_loader_multi_object)
    print(f"Eval dataloader (multi-object): batch_size={eval_batch_size}, batches={len(eval_loader_multi_object.dataloader.dataloader)}, n_samples={len(multi_object_indices)}")

    eval_loader = [eval_loader_base, eval_loader_unseen_pos, eval_loader_multi_object]
    return train_loader, eval_loader, data_info
