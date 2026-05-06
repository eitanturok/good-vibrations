import argparse
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw


DEFAULT_REPO_ID = "eturok-weizmann/laser-vibrations"
REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
SPEAKER_DIR = ASSETS_DIR / "speakers"
SPEAKER_FILES = ("1000", "0100", "0010", "0001")
PADDED_BG = (232, 232, 232)


def sample_dir_rel(sample_id: int) -> str:
    return f"data/{int(sample_id):07d}"


def _artifact_repo_path(value: str | None) -> str | None:
    marker = "/resolve/main/"
    if not isinstance(value, str) or not value:
        return None
    if marker in value:
        return value.split(marker, 1)[1]
    return value.lstrip("/")


def _download_image(repo_id: str, repo_path: str) -> Image.Image:
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=repo_path)
    with Image.open(local_path) as img:
        return img.convert("RGB")


def _load_manifest(sample_id: int, repo_id: str) -> tuple[dict, str]:
    manifest_repo_path = f"{sample_dir_rel(sample_id)}/manifest.json"
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=manifest_repo_path)
    return json.loads(Path(local_path).read_text(encoding="utf-8")), manifest_repo_path


def _load_mask(repo_id: str, manifest: dict) -> tuple[np.ndarray, str]:
    artifacts = manifest.get("artifacts", {})
    mask_npz_repo_path = _artifact_repo_path(artifacts.get("mask_npz"))
    if mask_npz_repo_path is not None:
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=mask_npz_repo_path)
        with np.load(local_path, allow_pickle=True) as payload:
            if "mask" in payload:
                return np.asarray(payload["mask"], dtype=np.float32), mask_npz_repo_path

    mask_png_repo_path = _artifact_repo_path(artifacts.get("mask_png"))
    if mask_png_repo_path is None:
        raise FileNotFoundError("manifest does not contain mask_npz or mask_png")

    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=mask_png_repo_path)
    with Image.open(local_path) as img:
        mask = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    return mask, mask_png_repo_path


def build_segmented_overhead(cropped_image: Image.Image, mask: np.ndarray, x_com: float | None, y_com: float | None) -> Image.Image:
    cropped = np.asarray(cropped_image.convert("RGB"), dtype=np.uint8)
    mask_arr = np.asarray(mask, dtype=np.float32)
    alpha = np.clip(mask_arr, 0.0, 1.0)[..., None] * 0.5
    tint = np.zeros_like(cropped, dtype=np.float32)
    tint[..., 1] = 204.0
    blended = (cropped.astype(np.float32) * (1.0 - alpha) + tint * alpha).astype(np.uint8)
    image = Image.fromarray(blended)
    if x_com is not None and y_com is not None:
        draw = ImageDraw.Draw(image)
        radius = 20
        draw.line([(x_com - radius, y_com), (x_com + radius, y_com)], fill=(255, 0, 0), width=4)
        draw.line([(x_com, y_com - radius), (x_com, y_com + radius)], fill=(255, 0, 0), width=4)
    return image


def apply_speaker_overlay(img: Image.Image, speakers: str) -> Image.Image:
    inner = img.convert("RGB")
    inner_width, inner_height = inner.size

    reference_icon_path = SPEAKER_DIR / "1000.png"
    if not reference_icon_path.exists():
        reference_icon_path = SPEAKER_DIR / "speaker.png"
    reference_icon = Image.open(reference_icon_path).convert("RGBA")
    target_icon_height = int(inner_height * 0.40)
    orig_w, orig_h = reference_icon.size
    target_icon_width = int(orig_w * target_icon_height / orig_h)

    padded_width = inner_width + target_icon_width
    padded_height = inner_height + target_icon_height
    canvas = Image.new("RGB", (padded_width, padded_height), PADDED_BG)
    canvas.paste(inner, (target_icon_width // 2, target_icon_height // 2))

    if "1" in speakers:
        composite = canvas.convert("RGBA")
        for bit, key in zip(speakers, SPEAKER_FILES):
            if bit != "1":
                continue
            icon_path = SPEAKER_DIR / f"{key}.png"
            if not icon_path.exists():
                icon_path = SPEAKER_DIR / "speaker.png"
            icon = Image.open(icon_path).convert("RGBA")
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


def regenerate_segmented_overhead_for_sample(
    sample_id: int,
    repo_id: str = DEFAULT_REPO_ID,
    output_path: str | Path | None = None,
) -> dict:
    manifest, manifest_repo_path = _load_manifest(sample_id, repo_id)
    artifacts = manifest.get("artifacts", {})
    cropped_repo_path = _artifact_repo_path(artifacts.get("cropped_overhead"))
    if cropped_repo_path is None:
        raise FileNotFoundError(f"manifest {manifest_repo_path} does not contain artifacts.cropped_overhead")

    mask, mask_repo_path = _load_mask(repo_id, manifest)
    cropped = _download_image(repo_id, cropped_repo_path)
    segmentation = manifest.get("segmentation", {})
    speakers = str(manifest.get("sample", {}).get("speakers") or "")
    rebuilt = apply_speaker_overlay(
        build_segmented_overhead(cropped, mask, segmentation.get("x_com"), segmentation.get("y_com")),
        speakers,
    )

    if output_path is None:
        output_path = REPO_ROOT / "tmp" / "segmented_overhead_repairs" / f"{int(sample_id):07d}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.save(output_path)

    return {
        "sample_id": int(sample_id),
        "repo_id": repo_id,
        "manifest_repo_path": manifest_repo_path,
        "cropped_overhead_repo_path": cropped_repo_path,
        "mask_repo_path": mask_repo_path,
        "speakers": speakers,
        "output_path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate segmented_overhead.png for one dataset sample.")
    parser.add_argument("sample_id", type=int)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = regenerate_segmented_overhead_for_sample(
        sample_id=args.sample_id,
        repo_id=args.repo_id,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
