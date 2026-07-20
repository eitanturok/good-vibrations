#****** 1 process image *****

def downsample(mask: Image.Image, out_h: int, out_w: int) -> Image.Image:
    # BOX resampling area-averages over the full H x W mask (unlike a floor-division block
    # reshape, which silently truncates to block_h*out_h x block_w*out_w and drops the
    # bottom/right edge whenever out_h/out_w don't evenly divide H/W).
    return mask.resize((out_w, out_h), resample=Image.BOX)

def noisy_blur(mask: np.ndarray, rng: np.random.Generator, sigma:float=0.8, noise_std:float=0.05) -> np.ndarray:
    # Gaussian blur, then add noise back onto originally-nonzero pixels only. mask: (B,H,W).
    blurred = gaussian_filter(mask, sigma=sigma, radius=1, mode='nearest', axes=(1, 2))
    noise = rng.normal(0.0, noise_std, size=mask.shape)
    out = blurred.copy()
    out[mask != 0] += noise[mask != 0]
    return np.clip(out, 0, 1)

def process_image(mask: np.ndarray, out_h:int, out_w:int, rng: np.random.Generator|None=None, augment_fn=noisy_blur) -> np.ndarray:
    """mask: (B,H,W) float32 in [0,1]. Returns (B,H,W) float32.
    rng=None skips mask augmentation (offline baseline / eval)."""
    if rng is not None: mask = augment_fn(mask, rng)
    return downsample(mask, out_h, out_w)

#***** 2 post process vibration *****

def extract_signal(x: np.ndarray, signal_mode: str) -> np.ndarray:
    # Cast to complex128 before abs/angle: np.abs on complex64 loses precision for large values
    # because sqrt(re²+im²) is computed in float32; PyTorch promotes internally so we match it.
    if signal_mode == "magnitude": return np.abs(x.astype(np.complex128))
    if signal_mode == "complex": return np.concatenate([x.real, x.imag], axis=-1)
    if signal_mode == "mag_phase": return np.concatenate([np.abs(x.astype(np.complex128)), np.angle(x.astype(np.complex128))], axis=-1)
    raise ValueError(f"Unknown signal mode: {signal_mode}")

def normalize_fft(x: np.ndarray, normalize_mode: str) -> np.ndarray:
    if normalize_mode is None: return x
    # Compute stats in float64 with ddof=1 to match PyTorch's std behavior
    x64 = x.astype(np.float64)
    if normalize_mode == 'std':
        std = np.maximum(x64.std(axis=(1, 2, 3), ddof=1, keepdims=True), 1e-8).astype(np.float32)
        return x / std
    if normalize_mode == 'z':
        mean = x64.mean(axis=(1, 2, 3), keepdims=True).astype(np.float32)
        std = np.maximum(x64.std(axis=(1, 2, 3), ddof=1, keepdims=True), 1e-8).astype(np.float32)
        return (x - mean) / std
    raise ValueError(f"Unknown normalize mode: {normalize_mode}")

def tokenize(x: np.ndarray, patch_size:int):
    # Note: unfold drops entries that do not fully fit into patch_size
    if patch_size <= 0: return x
    B, L, F, C = x.shape
    P = F // patch_size
    return x[:, :, :P * patch_size, :].reshape(B, L, P, patch_size, C)  # (B,L,P,PS,C)

def random_frequency_gain(freqs: np.ndarray, rng: np.random.Generator, n_control:int=5, gain_range:tuple=(0.8, 1.2)) -> np.ndarray:
    # Smooth random multiplicative gain curve: random values at a few control frequencies,
    # connected with a monotonic cubic spline, rescaled into gain_range.
    control_freqs = np.linspace(freqs.min(), freqs.max(), n_control)
    control_gains = rng.normal(loc=1.0, scale=1.0, size=n_control)
    spline = PchipInterpolator(control_freqs, control_gains)
    gain = spline(freqs)
    gain = (gain - gain.min()) / (gain.max() - gain.min())  # -> [0, 1]
    lo, hi = gain_range
    return (gain * (hi - lo) + lo).astype(np.float32)

def augment_vibration(fft: np.ndarray, freqs: np.ndarray, rng: np.random.Generator, n_control:int=5, gain_range:tuple=(0.8, 1.2)) -> np.ndarray:
    # fft: (B,L,F,C) complex. One shared gain curve per call (batch); draw a fresh rng per
    # row upstream (e.g. make_rng(epoch, sample_id)) if per-sample gain curves are wanted.
    gain = random_frequency_gain(freqs, rng, n_control, gain_range)
    return fft * gain[None, None, :, None]

def process_vibration(fft: np.ndarray, freqs: np.ndarray, signal_mode:str, normalize_mode:str, patch_size:int, rng: np.random.Generator|None=None, gain_kwargs: dict|None=None) -> np.ndarray:
    """fft: (B,L,F,C) complex. Returns tokenized (B,L,P,patch_size,C) float32.
    rng=None skips frequency augmentation (offline baseline / eval)."""
    if rng is not None: fft = augment_vibration(fft, freqs, rng, **(gain_kwargs or {}))
    x = extract_signal(fft, signal_mode).astype(np.float32)  # (B,L,F_,C) -> (B,L,F,C)
    x = normalize_fft(x, normalize_mode)
    return tokenize(x, patch_size)

#***** 3 dataset *****

class VibrationDataset(StreamingDataset):
    def __init__(self, local: str | Path, shuffle: bool = False, augment_site: AugmentSite = "none", out_h:int=40, out_w:int=20,
                 signal_mode: str = "magnitude", normalize_mode: str = "std-sample", patch_size: int = 256, **kwargs):
        super().__init__(local=str(local), shuffle=shuffle, batch_size=kwargs.pop("batch_size", None), **kwargs)
        self.augment_site = augment_site
        self.signal_mode, self.normalize_mode, self.patch_size = signal_mode, normalize_mode, patch_size
        self.freqs = np.load(Path(local) / "freqs.npy")
        # self.rng =  # todo: define rng here, don't need epoch.

    def __getitem__(self, idx):
        s = super().__getitem__(idx)
        X, y = s.pop("X"), s.pop("y")  # X: (L,F,C,2) real/imag fft, y: (H,W) raw mask
        sample_id = s["sample_id"]
        info = dict(sample_id=sample_id, output_id=s["output_id"], n_objects=s["n_objects"], speaker=s["speaker"], box=s["box"], is_empty_box=s["is_empty_box"], x_com=s["downsampled_com_x"], y_com=s["downsampled_com_y"])

        if self.augment_site == "collate": return dict(fft=X[..., 0] + 1j * X[..., 1], mask_true=y, info=info)

        fft_complex = (X[..., 0] + 1j * X[..., 1])[None]  # (1,L,F,C)
        fft_processed = process_vibration(fft_complex, self.freqs, self.signal_mode, self.normalize_mode, self.patch_size, rng=rng)[0]
        mask_processed = process_image(y[None], rng=rng)[0]
        return dict(fft=torch.from_numpy(fft_processed.copy()), mask_true=torch.from_numpy(mask_processed.copy()), info=info)

#***** 4 split *****

def augment(batch: list[dict], epoch: int, signal_mode: str, normalize_mode: str, patch_size: int, freqs: np.ndarray, augment: bool):
    """Batched counterpart to VibrationDataset(augment_site='getitem'): stacks raw complex fft +
    raw mask across the batch, then runs process_vibration/process_image once on the whole
    batch instead of once per sample. Only meaningful when the dataset was built with
    augment_site='collate' (otherwise __getitem__ already returned processed tensors)."""
    fft = np.stack([b["fft"] for b in batch])          # (B,L,F,C) complex
    mask = np.stack([b["mask_true"] for b in batch])   # (B,H,W)
    infos = [b["info"] for b in batch]
    rng = make_rng(epoch, infos[0]["sample_id"]) if augment else None  # one shared draw per batch
    fft_processed = process_vibration(fft, freqs, signal_mode, normalize_mode, patch_size, rng=rng)
    mask_processed = process_image(mask, rng=rng)
    return dict(fft=torch.from_numpy(fft_processed.copy()), mask_true=torch.from_numpy(mask_processed.copy()), info=infos)

def _matches(row: dict, speakers, n_objects, box) -> bool:
    if speakers is not None and row["speaker"] not in (speakers if isinstance(speakers, list) else [speakers]): return False
    if n_objects is not None and row["n_objects"] not in (n_objects if isinstance(n_objects, list) else [n_objects]): return False
    if box is not None and row["box"] not in (box if isinstance(box, list) else [box]): return False
    return True

EXP25_GROUPS = {
    "purple_cube":         {"train_range": (3, 29),   "eval_ranges": [(30, 59)]},
    "green_cube":          {"train_range": (110, 124), "eval_ranges": [(125, 127)]},
    "purple_green_cubes":  {"train_range": (60, 88),  "eval_ranges": [(89, 109)]},
}
EXP25_ALWAYS_TRAIN_RANGE = (0, 2) # empty-box

def exp25_split(mds_path: str | Path, batch_size: int = 64, eval_batch_size: int = 64, test_size: float = 0.20,
                  unseen_pos_frac: float = 0.15, unseen_pos_speaker_frac: float = 0.05, seed: int = 42, num_workers: int = 8, augment_site: str = "none",
                  speakers=None, n_objects=None, box=None, n_samples: int | None = None,
                  verbose: int = 1):
    """Return (train_loader, eval_loaders) using an already-written MDS, split by object-type.

    Each object-type (purple_cube, green_cube, purple_green_cubes) has an output_id (== image_id)
    range that is always train (its "grid-1" layout, or the train-side layout for the mixed type)
    and one or more output_id ranges we may only draw eval samples from (its "grid-2" layout(s)).
    empty-box is always train, no eval carve-out.

    For each type:
    - combined pool = train-range samples + eval-range samples.
    - target_eval = test_size * len(combined pool).
    - If the eval-range pool is smaller than target_eval, the *whole* eval-range pool becomes eval
      candidates (we never dip into the train range to make up the difference) -> this type ends up
      with < test_size eval, and all of its train-range samples stay in train.
    - Otherwise, target_eval samples (whole output_ids) are drawn from the eval-range pool as eval
      candidates, and the leftover eval-range output_ids fall back into train.
    - The eval candidates are then split unseen_pos_frac / unseen_pos_speaker_frac (of the combined
      pool, not of the eval candidates) into:
        - eval/{type}: whole held-out output_ids -> positions never seen in train, from any speaker.
        - eval/{type}_speaker: individual samples from output_ids that DO appear in train -> this
          exact (position, speaker) pair is new, but the position was seen from other speakers.
      If the eval-range pool was capped smaller than target_eval, these two eval sets are scaled down
      proportionally to fit inside what's available.
    """
    generator = torch.Generator().manual_seed(seed)
    dataset = VibrationDataset(local=mds_path, augment_site=augment_site)
    eval_dataset = VibrationDataset(local=mds_path, augment_site="none")  # eval never augments, even when train uses augment_site="getitem"

    # load metadata.jsonl (no dataset-level header line in this format: every line is a sample)
    lines = (Path(mds_path) / "metadata.jsonl").read_text().strip().splitlines()
    index = [json.loads(line) for line in lines if line]

    # filter the dataset by speakers, n_objects, box, n_samples
    keep = [i for i, row in enumerate(index) if _matches(row, speakers, n_objects, box)]
    if n_samples is not None: keep = keep[:n_samples]

    def in_range(i, lo, hi): return lo <= int(index[i]["output_id"]) <= hi

    train_idx = [i for i in keep if in_range(i, *EXP25_ALWAYS_TRAIN_RANGE)]
    evals = {}
    for name, group in EXP25_GROUPS.items():
        train_range_idx = [i for i in keep if in_range(i, *group["train_range"])]
        eval_pool_idx = [i for i in keep if any(in_range(i, lo, hi) for lo, hi in group["eval_ranges"])]
        combined_n = len(train_range_idx) + len(eval_pool_idx)

        target_eval = test_size * combined_n
        eval_pool_output_ids = sorted({index[i]["output_id"] for i in eval_pool_idx})

        if len(eval_pool_idx) <= target_eval:
            # worst case: the whole eval-range pool becomes eval candidates, nothing more
            eval_candidate_idx = eval_pool_idx
            train_idx += train_range_idx
            scale = len(eval_pool_idx) / target_eval if target_eval > 0 else 0.0
        else:
            n_output_ids = round(target_eval / len(eval_pool_idx) * len(eval_pool_output_ids))
            n_output_ids = min(max(n_output_ids, 0), len(eval_pool_output_ids))
            eval_candidate_output_ids, _ = train_test_split(
                eval_pool_output_ids, train_size=n_output_ids, random_state=seed, shuffle=True)
            eval_candidate_output_ids = set(eval_candidate_output_ids)
            eval_candidate_idx = [i for i in eval_pool_idx if index[i]["output_id"] in eval_candidate_output_ids]
            train_idx += train_range_idx + [i for i in eval_pool_idx if index[i]["output_id"] not in eval_candidate_output_ids]
            scale = 1.0

        # split eval candidates' output_ids: unseen_pos_frac -> whole new positions, rest -> pool for _speaker
        pos_frac = min(unseen_pos_frac * scale, unseen_pos_frac + unseen_pos_speaker_frac)
        eval_candidate_output_ids = sorted({index[i]["output_id"] for i in eval_candidate_idx})
        n_unseen_pos = round(pos_frac / test_size * len(eval_candidate_output_ids)) if test_size > 0 else 0
        n_unseen_pos = min(max(n_unseen_pos, 0), len(eval_candidate_output_ids))
        unseen_pos_output_ids, speaker_pool_output_ids = train_test_split(
            eval_candidate_output_ids, train_size=n_unseen_pos, random_state=seed, shuffle=True) if 0 < n_unseen_pos < len(eval_candidate_output_ids) \
            else (eval_candidate_output_ids, []) if n_unseen_pos == len(eval_candidate_output_ids) \
            else ([], eval_candidate_output_ids)
        unseen_pos_output_ids = set(unseen_pos_output_ids)
        evals[f"eval/{name}"] = [i for i in eval_candidate_idx if index[i]["output_id"] in unseen_pos_output_ids]

        # unseen_pos_speaker_frac of the combined pool, taken as individual samples from the
        # speaker-pool output_ids (these output_ids are NOT held out -> they stay available to train too)
        speaker_pool_idx = [i for i in eval_candidate_idx if index[i]["output_id"] in set(speaker_pool_output_ids)]
        n_speaker = round(unseen_pos_speaker_frac * scale / test_size * combined_n) if test_size > 0 else 0
        n_speaker = min(max(n_speaker, 0), len(speaker_pool_idx))
        if n_speaker < len(speaker_pool_idx):
            remainder_idx, speaker_idx = train_test_split(speaker_pool_idx, test_size=n_speaker, random_state=seed, shuffle=True)
        else:
            remainder_idx, speaker_idx = [], speaker_pool_idx
        evals[f"eval/{name}_speaker"] = speaker_idx
        train_idx += remainder_idx

    if verbose:
        counts = ", ".join(f"{label}={len(idxs)}" for label, idxs in evals.items())
        print(f"{len(index)} total samples, {len(keep)} after filtering -> train={len(train_idx)}, {counts}")

    def num_samples(batch): return batch["mask_true"].shape[0]
    def loader(idxs, bs, shuffle=False, drop_last=False, augment=False):
        # collate site: __getitem__ always hands back raw data regardless of augment, so both
        # train and eval can share `dataset` -- augmenting_collate's own `augment` flag gates it.
        # getitem site (or "none"): augmentation is baked into __getitem__ itself, so eval must
        # use a separate augment_site="none" instance to guarantee it never augments.
        ds = dataset if (augment or augment_site == "collate") else eval_dataset
        collate_fn = (lambda batch: augmenting_collate(batch, dataset._epoch, dataset.signal_mode, dataset.normalize_mode, dataset.patch_size, dataset.freqs, augment)) if augment_site == "collate" else None
        dl = DataLoader(Subset(ds, idxs), batch_size=bs, shuffle=shuffle, num_workers=num_workers, generator=generator, pin_memory=True, persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None, drop_last=drop_last, collate_fn=collate_fn)
        return DataSpec(dataloader=dl, get_num_samples_in_batch=num_samples)

    train_loader = loader(train_idx, batch_size, shuffle=True, drop_last=True, augment=True)
    eval_loaders = [Evaluator(label=label, dataloader=loader(idxs, eval_batch_size)) for label, idxs in evals.items()]
    return train_loader, eval_loaders

#***** 5 build *****

SPLIT_METHODS = {
    "exp22": exp22_split,
    "exp23": exp23_split,
    "exp24": exp24_split,
    "exp25": exp25_split,
}

def build_dataset(mds_path: str | Path, split: str = "exp22", **kwargs):
    if split not in SPLIT_METHODS: raise ValueError(f"Unknown split {split!r}, expected one of {sorted(SPLIT_METHODS)}")
    return SPLIT_METHODS[split](mds_path, **kwargs)
