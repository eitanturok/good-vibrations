import argparse
import json
import os
from pathlib import Path

import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from PIL import Image, ImageDraw


DEFAULT_REPO_ID = "eturok-weizmann/laser-vibrations"
SPEAKER_FILES = ("1000", "0100", "0010", "0001")
PADDED_BG = (232, 232, 232)
SPEAKER_SCALE_MULTIPLIER = 1
REFERENCE_SPEAKER_ICON = "speaker.png"


def resolve_repo_root() -> Path:
    candidates = []

    env_root = os.getenv("GOOD_VIBRATIONS_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    candidates.append(Path("~/good-vibrations").expanduser())
    candidates.append(Path(__file__).resolve().parent.parent)

    for candidate in candidates:
        if (candidate / "assets" / "speakers").exists() and (candidate / "src").exists():
            return candidate

    return Path(__file__).resolve().parent.parent


REPO_ROOT = resolve_repo_root()
SPEAKER_DIR = REPO_ROOT / "assets" / "speakers"


def sample_dir_rel(sample_id: int) -> str:
    return f"data/{int(sample_id):07d}"


def hf_file_url(repo_id: str, repo_path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{repo_path}"


def artifact_repo_path(value: str | None) -> str | None:
    marker = "/resolve/main/"
    if not isinstance(value, str) or not value:
        return None
    if marker in value:
        return value.split(marker, 1)[1]
    return value.lstrip("/")


def read_json(repo_id: str, repo_path: str) -> dict:
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=repo_path)
    return json.loads(Path(local_path).read_text(encoding="utf-8"))


def read_jsonl_rows(repo_id: str, repo_path: str) -> list[dict]:
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=repo_path)
    return [json.loads(line) for line in Path(local_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def download_image(repo_id: str, repo_path: str) -> Image.Image:
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=repo_path)
    with Image.open(local_path) as image:
        return image.convert("RGB")


def load_mask(repo_id: str, manifest: dict) -> tuple[np.ndarray, str]:
    artifacts = manifest.get("artifacts", {})
    mask_npz_repo_path = artifact_repo_path(artifacts.get("mask_npz"))
    if mask_npz_repo_path is not None:
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=mask_npz_repo_path)
        with np.load(local_path, allow_pickle=True) as payload:
            if "mask" in payload:
                return np.asarray(payload["mask"], dtype=np.float32), mask_npz_repo_path

    mask_png_repo_path = artifact_repo_path(artifacts.get("mask_png"))
    if mask_png_repo_path is None:
        raise FileNotFoundError("manifest does not contain artifacts.mask_npz or artifacts.mask_png")
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=mask_png_repo_path)
    with Image.open(local_path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0, mask_png_repo_path


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


def build_sample_overhead(segmented_overhead: Image.Image, speakers: str) -> Image.Image:
    inner = segmented_overhead.convert("RGB")
    inner_width, inner_height = inner.size

    reference_icon_path = SPEAKER_DIR / REFERENCE_SPEAKER_ICON
    with Image.open(reference_icon_path) as reference_icon_rgba:
        reference_icon = reference_icon_rgba.convert("RGBA")
        base_icon_height = int(inner_height * 0.40)
        orig_w, orig_h = reference_icon.size
        base_icon_width = int(orig_w * base_icon_height / orig_h)
        target_icon_height = int(base_icon_height * SPEAKER_SCALE_MULTIPLIER)
        target_icon_width = int(base_icon_width * SPEAKER_SCALE_MULTIPLIER)

    padded_width = inner_width + base_icon_width
    padded_height = inner_height + base_icon_height
    canvas = Image.new("RGB", (padded_width, padded_height), PADDED_BG)
    canvas.paste(inner, (base_icon_width // 2, base_icon_height // 2))

    if "1" not in speakers:
        return canvas

    composite = canvas.convert("RGBA")
    for bit, key in zip(speakers, SPEAKER_FILES):
        if bit != "1":
            continue
        icon_path = SPEAKER_DIR / REFERENCE_SPEAKER_ICON
        with Image.open(icon_path) as icon_rgba:
            icon = icon_rgba.convert("RGBA")
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
    return composite.convert("RGB")


def write_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


def rebuild_overhead_artifacts_for_sample(
    sample_id: int,
    repo_id: str = DEFAULT_REPO_ID,
    output_root: str | Path | None = None,
) -> dict:
    sample_dir = sample_dir_rel(sample_id)
    manifest_repo_path = f"{sample_dir}/manifest.json"
    manifest = read_json(repo_id, manifest_repo_path)
    metadata_rows = read_jsonl_rows(repo_id, "data/metadata.jsonl")

    artifacts = manifest.get("artifacts", {})
    cropped_repo_path = artifact_repo_path(artifacts.get("cropped_overhead"))
    if cropped_repo_path is None:
        raise FileNotFoundError(f"manifest {manifest_repo_path} does not contain artifacts.cropped_overhead")

    cropped_image = download_image(repo_id, cropped_repo_path)
    mask, mask_repo_path = load_mask(repo_id, manifest)
    segmentation = manifest.get("segmentation", {})
    x_com = segmentation.get("x_com")
    y_com = segmentation.get("y_com")
    speakers = str(manifest.get("sample", {}).get("speakers") or "")
    image_dir = str(manifest.get("sample", {}).get("image_dir") or "")
    if not image_dir:
        raise ValueError(f"manifest {manifest_repo_path} does not contain sample.image_dir")

    segmented_overhead = build_segmented_overhead(cropped_image, mask, x_com, y_com)
    sample_overhead = build_sample_overhead(segmented_overhead, speakers)

    segmented_repo_path = f"image/{image_dir}/segmented_overhead.png"
    overhead_repo_path = f"{sample_dir}/overhead.png"

    updated_manifest = json.loads(json.dumps(manifest))
    updated_manifest.setdefault("artifacts", {})["segmented_overhead"] = segmented_repo_path
    updated_manifest["artifacts"]["overhead"] = overhead_repo_path

    matching_rows = [row for row in metadata_rows if int(row.get("sample_id", -1)) == int(sample_id)]
    if len(matching_rows) != 1:
        raise ValueError(f"expected exactly 1 metadata row for sample_id={sample_id}, found {len(matching_rows)}")
    updated_metadata_rows = []
    for row in metadata_rows:
        if int(row.get("sample_id", -1)) != int(sample_id):
            updated_metadata_rows.append(row)
            continue
        updated_row = dict(row)
        updated_row["segmented_overhead_file_name"] = hf_file_url(repo_id, overhead_repo_path)
        updated_row["manifest"] = json.dumps(updated_manifest, ensure_ascii=True)
        updated_metadata_rows.append(updated_row)

    if output_root is None:
        output_root = REPO_ROOT / "tmp" / "overhead_artifact_repairs" / f"{int(sample_id):07d}"
    output_root = Path(output_root)

    segmented_local_path = output_root / segmented_repo_path
    overhead_local_path = output_root / overhead_repo_path
    manifest_local_path = output_root / manifest_repo_path
    metadata_local_path = output_root / "data" / "metadata.jsonl"

    write_image(segmented_local_path, segmented_overhead)
    write_image(overhead_local_path, sample_overhead)
    write_json(manifest_local_path, updated_manifest)
    write_jsonl(metadata_local_path, updated_metadata_rows)

    return {
        "sample_id": int(sample_id),
        "repo_id": repo_id,
        "speakers": speakers,
        "cropped_overhead_repo_path": cropped_repo_path,
        "mask_repo_path": mask_repo_path,
        "segmented_overhead_repo_path": segmented_repo_path,
        "overhead_repo_path": overhead_repo_path,
        "manifest_repo_path": manifest_repo_path,
        "metadata_repo_path": "data/metadata.jsonl",
        "output_root": str(output_root),
        "files_written": [
            str(segmented_local_path),
            str(overhead_local_path),
            str(manifest_local_path),
            str(metadata_local_path),
        ],
    }


def upload_rebuilt_artifacts(result: dict, commit_message: str | None = None) -> dict:
    repo_id = result["repo_id"]
    output_root = Path(result["output_root"])
    operations = []
    for repo_path in [
        result["segmented_overhead_repo_path"],
        result["overhead_repo_path"],
        result["manifest_repo_path"],
        result["metadata_repo_path"],
    ]:
        operations.append(
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=str(output_root / repo_path),
            )
        )
    api = HfApi()
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=commit_message or f"Repair overhead artifacts for sample {result['sample_id']}",
    )
    return {
        "commit_oid": getattr(commit, "oid", None),
        "commit_url": getattr(commit, "commit_url", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild shared segmented_overhead and per-sample overhead for one sample.")
    parser.add_argument("sample_id", type=int)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--upload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--commit-message", default=None)
    args = parser.parse_args()

    result = rebuild_overhead_artifacts_for_sample(
        sample_id=args.sample_id,
        repo_id=args.repo_id,
        output_root=args.output_root,
    )
    if args.upload:
        result["upload"] = upload_rebuilt_artifacts(result, commit_message=args.commit_message)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
