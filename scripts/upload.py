import argparse
import io
import json
import os
import sys
import time

import numpy as np
from datasets import Dataset, Image as HFImage, load_dataset
from huggingface_hub import HfApi


sys.path.insert(0, os.path.dirname(__file__))
from segment import segment_sample


IMAGE_COLS = ["raw_image", "cropped_image", "overlay_image"]


def extract_sample(sample_path, left=0.15, right=0.67, up=0.08, down=0.7):
    """Run segmentation and combine with sample metadata into a full record."""
    recovery = np.load(os.path.join(sample_path, "RECOVERY.npz"), allow_pickle=True)
    shifts = recovery["all_shifts"]  # (100, N_frames, 2)
    config = json.load(open(os.path.join(sample_path, "experiment_config.json")))

    mask, vision = segment_sample(sample_path, left=left, right=right, up=up, down=down, object=config["object"])

    data = {
        **vision,
        "object":            config["object"],
        "speakers":          config["speakers"],
        "fps":               int(config["FPS"]),
        "experiment_config": json.dumps(config),
    }
    return shifts, mask, data


def to_webp(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="WEBP", quality=85)
    return {"bytes": buf.getvalue(), "path": None}


def upload_sample(repo_id, shifts, mask, data):
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    files = list(api.list_repo_files(repo_id, repo_type="dataset"))
    idx = sum(1 for f in files if f.startswith("data/sample_") and f.endswith(".npz"))

    buf = io.BytesIO()
    np.savez_compressed(buf, shifts=shifts, mask=mask)
    buf.seek(0)
    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=f"data/sample_{idx}.npz",
        repo_id=repo_id,
        repo_type="dataset",
    )

    row = {**data, "sample_idx": idx, **{col: to_webp(data[col]) for col in IMAGE_COLS}}
    try:
        ds = load_dataset(repo_id, split="train")
        ds = ds.add_item(row)
    except Exception:
        ds = Dataset.from_list([row])
    for col in IMAGE_COLS:
        ds = ds.cast_column(col, HFImage())
    ds.push_to_hub(repo_id)


def main():
    parser = argparse.ArgumentParser(description="Extract a sample and upload to HuggingFace.")
    parser.add_argument("--hf-dataset", default='eturok-weizmann/vibrations', help="HuggingFace dataset repo id, e.g. eturok-weizmann/vibrations")
    parser.add_argument("--sample-path", required=True, help="Path to sample directory")
    parser.add_argument("--left",        type=float, default=0.15)
    parser.add_argument("--right",       type=float, default=0.67)
    parser.add_argument("--up",          type=float, default=0.08)
    parser.add_argument("--down",        type=float, default=0.7)
    args = parser.parse_args()

    print("Extracting and segmenting sample...")
    t0 = time.time()
    shifts, mask, data = extract_sample(
        args.sample_path,
        left=args.left, right=args.right, up=args.up, down=args.down,
    )
    print(f"Done. ({time.time() - t0:.1f}s)")

    print("Uploading sample...")
    t0 = time.time()
    upload_sample(args.hf_dataset, shifts, mask, data)
    print(f"Done. ({time.time() - t0:.1f}s)")

    print(f"Done. https://huggingface.co/datasets/{args.hf_dataset}")


if __name__ == "__main__":
    main()
