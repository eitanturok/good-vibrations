import argparse
import io
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
from datasets import Dataset, Image as HFImage
from huggingface_hub import HfApi


sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from segment import segment_sample
from watch import watch
from status import claim, finish
from helpers import sample_npz_path

REMOTE_HOST = 'mcluster11'


def slurm_job_completed(sample_name):
    result = subprocess.run(
        ['ssh', REMOTE_HOST, f'sacct --name {sample_name} -n -o State -X'],
        capture_output=True, text=True, check=True
    )
    return 'COMPLETED' in result.stdout


IMAGE_COLS = ["raw_image", "cropped_image", "overlay_image"]


def sample_idx_from_path(path):
    match = re.fullmatch(r"data/sample_(\d+)\.npz", path)
    return int(match.group(1)) if match else None


def to_webp(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="WEBP", quality=85)
    return {"bytes": buf.getvalue(), "path": None}


def upload_sample(repo_id, shifts, mask, data):
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    files = list(api.list_repo_files(repo_id, repo_type="dataset"))
    existing_indices = {
        sample_idx for f in files if (sample_idx := sample_idx_from_path(f)) is not None
    }
    idx = max(existing_indices, default=-1) + 1

    buf = io.BytesIO()
    np.savez_compressed(buf, shifts=shifts, mask=mask)
    buf.seek(0)
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=sample_npz_path(idx),
        repo_id=repo_id,
        repo_type="dataset",
    )

    row = {
        "object": data["object"],
        "n_objects": data["n_objects"],
        "speakers": data["speakers"],
        "box_material": data["box_material"],
        "x_position": data["x_position"],
        "y_position": data["y_position"],
        "raw_image": to_webp(data["raw_image"]),
        "cropped_image": to_webp(data["cropped_image"]),
        "overlay_image": to_webp(data["overlay_image"]),
        "fps": data["fps"],
        "sample_idx": idx,
        "experiment_config": data["experiment_config"],
    }
    ds = Dataset.from_list([row])
    for col in IMAGE_COLS:
        ds = ds.cast_column(col, HFImage())
    # upload as a new shard — HF concatenates all data/train-*.parquet shards on load,
    # so we never need to download existing data just to append a row
    buf = io.BytesIO()
    ds.to_parquet(buf)
    buf.seek(0)
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=f"data/train-{idx:05d}.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )

def cast(value, _type): return _type(value) if value is not None else None

def build_process(shared_dir, hf_dataset, left, right, up, down):
    @watch(shared_dir)
    def process(sample_path):
        result = claim(sample_path, "upload", prerequisite="move_data")
        if result == "finished": return True
        if result == "waiting":  return False
        if not slurm_job_completed(sample_path.name):
            return False

        config = json.load(open(os.path.join(sample_path, "experiment_config.json")))

        print(f"Segmenting {sample_path.name}...")
        t0 = time.time()
        mask, vision = segment_sample(sample_path=sample_path, left=left, right=right, up=up, down=down, object=config.get("object"), box_material=config.get("box_material", "cardboard"))
        print(f"Segmented. ({time.time() - t0:.1f}s)")

        recovery = np.load(os.path.join(sample_path, "RECOVERY.npz"), allow_pickle=True)
        shifts = recovery["all_shifts"]
        fps = config.get("FPS")
        n_objects = config.get("n_objects")
        data = {**vision, "object": config.get("object", ""), "speakers": config.get("speakers", ""), "fps": cast(fps, int),
                "n_objects": cast(n_objects, int), "box_material": config.get("box_material", ""), "experiment_config": json.dumps(config),
                "mask_area": float(mask.sum() / mask.size * 100)}

        print('Uploading...')
        t0 = time.time()
        upload_sample(hf_dataset, shifts, mask, data)
        print(f"Uploaded. ({time.time() - t0:.1f}s)")
        print(f"Done. See dataset at https://huggingface.co/datasets/{hf_dataset}")

        finish(sample_path, "upload")
        return True
    return process


def main():
    parser = argparse.ArgumentParser(description="Watch a directory and upload each new sample to HuggingFace.")
    parser.add_argument("--shared-dir",  required=True, help="Mounted shared dir to watch for new sample subdirectories")
    parser.add_argument("--hf-dataset",  default='eturok-weizmann/vibrations')
    parser.add_argument("--left",        type=float, default=0.15)
    parser.add_argument("--right",       type=float, default=0.67)
    parser.add_argument("--up",          type=float, default=0.08)
    parser.add_argument("--down",        type=float, default=0.7)
    args = parser.parse_args()

    process = build_process(args.shared_dir, args.hf_dataset, args.left, args.right, args.up, args.down)
    process()


if __name__ == "__main__":
    main()
