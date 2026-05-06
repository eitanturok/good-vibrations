"""
Fix empty box segmentation artifacts.

For each empty box sample:
  - Replace mask with all-zeros
  - Regenerate segmented_overhead.png = cropped_overhead + speaker overlay (no mask)
  - Set x_com, y_com = null; n_objects = 0; object = "empty"

Updates both the experiment-16 directory on mcluster11 and the HF repo.
"""
import io
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
SPEAKER_DIR = ASSETS_DIR / "speakers"
SPEAKER_FILES = ("1000", "0100", "0010", "0001")
PADDED_BG = (232, 232, 232)

REMOTE_HOST = "ethantu@mcluster11.wisdom.weizmann.ac.il"
REMOTE_EXPERIMENT_DIR = "/net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA/experiment-16"
HF_REPO = "eturok-weizmann/laser-vibrations"

SAMPLES = [
    {"sample_id": 613, "speakers": "0001", "image_dir": "empty_0001-POSx-POSy-1obj-cardboard-2026-03-31-18-14-17"},
    {"sample_id": 614, "speakers": "0010", "image_dir": "empty_0010-POSx-POSy-1obj-cardboard-2026-03-31-18-14-15"},
    {"sample_id": 615, "speakers": "0100", "image_dir": "empty_0100-POSx-POSy-1obj-cardboard-2026-03-31-18-14-13"},
    {"sample_id": 616, "speakers": "1000", "image_dir": "empty_1000-POSx-POSy-1obj-cardboard-2026-03-31-18-14-12"},
]


def remote_read(remote_path: str) -> bytes:
    result = subprocess.run(
        ["ssh", REMOTE_HOST, f"cat {shlex.quote(remote_path)}"],
        check=True, capture_output=True,
    )
    return result.stdout


def remote_write(local_path: Path, remote_path: str) -> None:
    with open(local_path, "rb") as f:
        subprocess.run(
            ["ssh", REMOTE_HOST, f"cat > {shlex.quote(remote_path)}"],
            check=True, stdin=f,
        )


def apply_speaker_overlay(img: Image.Image, speakers: str) -> Image.Image:
    inner = img.convert("RGB")
    inner_width, inner_height = inner.size

    target_icon_height = int(inner_height * 0.40)
    speaker_icon = Image.open(SPEAKER_DIR / "1000.png").convert("RGBA")
    orig_w, orig_h = speaker_icon.size
    target_icon_width = int(orig_w * target_icon_height / orig_h)

    padded_width = inner_width + target_icon_width
    padded_height = inner_height + target_icon_height
    canvas = Image.new("RGB", (padded_width, padded_height), PADDED_BG)
    canvas.paste(inner, (target_icon_width // 2, target_icon_height // 2))

    if "1" in speakers:
        composite = canvas.convert("RGBA")
        for bit, key in zip(speakers, SPEAKER_FILES):
            if bit == "1":
                icon = Image.open(SPEAKER_DIR / f"{key}.png").convert("RGBA")
                icon = icon.resize((target_icon_width, target_icon_height), Image.LANCZOS)
                if key == "1000":
                    px, py = 0, padded_height // 2 - icon.height // 2
                elif key == "0100":
                    px, py = padded_width // 3 - icon.width // 2, padded_height - icon.height
                elif key == "0010":
                    px, py = 2 * padded_width // 3 - icon.width // 2, padded_height - icon.height
                elif key == "0001":
                    px, py = padded_width - icon.width, padded_height // 2 - icon.height // 2
                else:
                    px, py = 0, 0
                composite.alpha_composite(icon, (px, py))
        canvas = composite.convert("RGB")

    return canvas


def fix_sample(sample: dict, tmp: Path, api: HfApi) -> None:
    sample_id = sample["sample_id"]
    speakers = sample["speakers"]
    image_dir = sample["image_dir"]
    remote_image_root = f"{REMOTE_EXPERIMENT_DIR}/image/{image_dir}"

    print(f"\n[{sample_id}] Processing {image_dir}")

    # Download cropped overhead from cluster
    cropped_bytes = remote_read(f"{remote_image_root}/cropped_overhead.png")
    cropped = Image.open(io.BytesIO(cropped_bytes)).convert("RGB")
    w, h = np.array(cropped).shape[1], np.array(cropped).shape[0]

    # Zero mask
    mask = np.zeros((h, w), dtype=np.float32)

    # Save mask.png (all black)
    mask_png_path = tmp / f"{sample_id}_mask.png"
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(mask_png_path)
    print(f"[{sample_id}] Created zero mask.png ({w}x{h})")

    # Save mask.npz
    mask_npz_path = tmp / f"{sample_id}_mask.npz"
    np.savez_compressed(mask_npz_path, mask=mask, left=0.15, right=0.67, up=0.08, down=0.7, prompt=None)
    print(f"[{sample_id}] Created zero mask.npz")

    # Generate segmented_overhead with speaker overlay (no mask tint)
    segmented = apply_speaker_overlay(cropped, speakers)
    segmented_path = tmp / f"{sample_id}_segmented_overhead.png"
    segmented.save(segmented_path)
    print(f"[{sample_id}] Created segmented_overhead.png with speaker overlay for '{speakers}'")

    # Upload to cluster
    remote_write(mask_png_path, f"{remote_image_root}/mask.png")
    print(f"[{sample_id}] Pushed mask.png to cluster")
    remote_write(mask_npz_path, f"{remote_image_root}/mask.npz")
    print(f"[{sample_id}] Pushed mask.npz to cluster")
    remote_write(segmented_path, f"{remote_image_root}/segmented_overhead.png")
    print(f"[{sample_id}] Pushed segmented_overhead.png to cluster")

    # Upload image files to HF
    for local_path, hf_rel in [
        (mask_png_path, f"image/{image_dir}/mask.png"),
        (mask_npz_path, f"image/{image_dir}/mask.npz"),
        (segmented_path, f"image/{image_dir}/segmented_overhead.png"),
    ]:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hf_rel,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"Fix empty box {image_dir}: zero mask + speaker overlay",
        )
        print(f"[{sample_id}] Uploaded {hf_rel} to HF")


def update_metadata(tmp: Path, api: HfApi) -> None:
    print("\n[metadata] Updating metadata.jsonl")
    remote_metadata_path = f"{REMOTE_EXPERIMENT_DIR}/data/metadata.jsonl"
    raw = remote_read(remote_metadata_path).decode("utf-8")

    fix_ids = {s["sample_id"] for s in SAMPLES}
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = int(row.get("sample_id", -1))
        if sid in fix_ids:
            row["x_com"] = None
            row["y_com"] = None
            row["n_objects"] = 0
            row["object"] = "empty"
            print(f"[metadata] Fixed sample_id={sid}: x_com=null, y_com=null, n_objects=0, object=empty")
        rows.append(row)

    rows.sort(key=lambda r: int(r.get("sample_id", 0)))
    updated = "\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n"

    metadata_path = tmp / "metadata.jsonl"
    metadata_path.write_text(updated, encoding="utf-8")

    # Write back to cluster
    remote_write(metadata_path, remote_metadata_path)
    print("[metadata] Pushed metadata.jsonl to cluster")

    # Upload to HF
    api.upload_file(
        path_or_fileobj=str(metadata_path),
        path_in_repo="data/metadata.jsonl",
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message="Fix empty box metadata: zero mask, n_objects=0, object=empty",
    )
    print("[metadata] Uploaded metadata.jsonl to HF")


def main() -> None:
    token_path = Path("~/.cache/huggingface/token").expanduser()
    api = HfApi(token=token_path.read_text().strip() if token_path.exists() else None)

    with tempfile.TemporaryDirectory(prefix="fix-empty-seg-") as tmp_dir:
        tmp = Path(tmp_dir)
        for sample in SAMPLES:
            fix_sample(sample, tmp, api)
        update_metadata(tmp, api)

    print("\n[done] All empty box samples fixed.")


if __name__ == "__main__":
    main()
