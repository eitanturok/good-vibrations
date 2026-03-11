import os, json, re, random
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import torch
from safetensors.torch import save_file
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage
from huggingface_hub import HfApi

REPO_ID = "eturok-weizmann/vibration-data"
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
SAFETENSORS_PATH = "/tmp/shifts.safetensors"
SEED = 42


def load_sample(idx, cube_path):
    recovery = np.load(os.path.join(cube_path, "RECOVERY.npz"), allow_pickle=True)
    shifts = recovery["all_shifts"]  # (100, 9000, 2)
    config = json.load(open(os.path.join(cube_path, "experiment_config.json")))
    fps = int(config["FPS"])
    m = re.search(r"cube-(\d+)x-(\d+)y", os.path.basename(cube_path))
    tensors = {f"shifts_{idx}": torch.from_numpy(shifts)}
    meta = {
        "shifts_idx": idx,
        "overhead_image": Image.open(os.path.join(cube_path, "box_overhead_image.png")),
        "x_position": int(m.group(1)),
        "y_position": int(m.group(2)),
        "experiment_config": json.dumps(config),
        "fps": fps,
        "object": "cube",
    }
    return tensors, meta


# Collect all cube dirs with their (x, y) group label
cube_paths, groups = [], []
for exp in ["experiment-04", "experiment-05"]:
    exp_dir = os.path.join(DATA_ROOT, exp)
    for d in sorted(os.listdir(exp_dir)):
        path = os.path.join(exp_dir, d)
        if not os.path.isdir(path): continue
        m = re.search(r"cube-(\d+)x-(\d+)y", d)
        cube_paths.append(path)
        groups.append(f"{m.group(1)}x-{m.group(2)}y")

# Split on groups so the same (x,y) position is never in both train and test
train_idx, test_idx = next(GroupShuffleSplit(test_size=0.2, random_state=SEED).split(cube_paths, groups=groups))
train_paths = [cube_paths[i] for i in train_idx]
test_paths  = [cube_paths[i] for i in test_idx]
cube_paths  = train_paths + test_paths

# Load all samples
all_tensors, all_meta = {}, []
for i, path in enumerate(cube_paths):
    print(f"Loading {i+1}/{len(cube_paths)}: {os.path.basename(path)}")
    tensors, meta = load_sample(i, path)
    all_tensors.update(tensors)
    all_meta.append(meta)

# Save safetensors file
print(f"Saving safetensors to {SAFETENSORS_PATH}...")
save_file(all_tensors, SAFETENSORS_PATH)

# Build HF DatasetDict
features = Features({
    "shifts_idx": Value("int32"),
    "overhead_image": HFImage(),
    "x_position": Value("int32"),
    "y_position": Value("int32"),
    "experiment_config": Value("string"),
    "fps": Value("int32"),
    "object": Value("string"),
})
n_train = len(train_paths)
ds = DatasetDict({
    "train": Dataset.from_list(all_meta[:n_train], features=features),
    "test":  Dataset.from_list(all_meta[n_train:], features=features),
})

# Upload both to HF
api = HfApi()
api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
print("Pushing dataset to HF...")
ds.push_to_hub(REPO_ID)
print("Uploading shifts.safetensors...")
api.upload_file(path_or_fileobj=SAFETENSORS_PATH, path_in_repo="shifts.safetensors",
                repo_id=REPO_ID, repo_type="dataset")

print(f"Done: {n_train} train, {len(cube_paths)-n_train} test samples")
