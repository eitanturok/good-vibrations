import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
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
DATASET_CARD_PATH = REPO_ROOT / "hf_data" / "README.md"


def log(message: str) -> None:
    print(f"[info] {message}", flush=True)


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


def candidate_repo_paths(repo_path: str | None) -> list[str]:
    if repo_path is None:
        return []
    candidates = [repo_path]
    if repo_path.startswith("data/image/"):
        candidates.append(repo_path.removeprefix("data/"))
    elif repo_path.startswith("image/"):
        candidates.append(f"data/{repo_path}")
    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def download_repo_file(repo_id: str, repo_path: str, label: str, sample_id: int | None = None) -> tuple[Path, str]:
    candidates = candidate_repo_paths(repo_path)
    last_error = None
    for candidate in candidates:
        prefix = f"sample_id={sample_id} " if sample_id is not None else ""
        log(f"{prefix}trying {label}: {candidate}")
        try:
            local_path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=candidate))
            return local_path, candidate
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"no repo path candidates for {label}")


def read_json(repo_id: str, repo_path: str) -> dict:
    local_path, _ = download_repo_file(repo_id, repo_path, label="json")
    return json.loads(Path(local_path).read_text(encoding="utf-8"))


def read_jsonl_rows(repo_id: str, repo_path: str) -> list[dict]:
    local_path, _ = download_repo_file(repo_id, repo_path, label="jsonl")
    return [json.loads(line) for line in Path(local_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def download_image(repo_id: str, repo_path: str, sample_id: int | None = None) -> tuple[Image.Image, str]:
    local_path, resolved_repo_path = download_repo_file(repo_id, repo_path, label="image", sample_id=sample_id)
    with Image.open(local_path) as image:
        return image.convert("RGB"), resolved_repo_path


def load_mask(repo_id: str, manifest: dict, sample_id: int | None = None) -> tuple[np.ndarray, str]:
    artifacts = manifest.get("artifacts", {})
    mask_npz_repo_path = artifact_repo_path(artifacts.get("mask_npz"))
    if mask_npz_repo_path is not None:
        local_path, resolved_repo_path = download_repo_file(repo_id, mask_npz_repo_path, label="mask npz", sample_id=sample_id)
        with np.load(local_path, allow_pickle=True) as payload:
            if "mask" in payload:
                return np.asarray(payload["mask"], dtype=np.float32), resolved_repo_path

    mask_png_repo_path = artifact_repo_path(artifacts.get("mask_png"))
    if mask_png_repo_path is None:
        raise FileNotFoundError("manifest does not contain artifacts.mask_npz or artifacts.mask_png")
    local_path, resolved_repo_path = download_repo_file(repo_id, mask_png_repo_path, label="mask png", sample_id=sample_id)
    with Image.open(local_path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0, resolved_repo_path


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


def copy_dataset_card(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "README.md"
    shutil.copy2(DATASET_CARD_PATH, target)
    return target


def validate_staged_sample(output_root: Path, sample_id: int, repo_id: str) -> bool:
    sample_dir = sample_dir_rel(sample_id)
    overhead_path = output_root / sample_dir / "overhead.png"
    manifest_path = output_root / sample_dir / "manifest.json"
    if not overhead_path.exists() or not manifest_path.exists():
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    artifacts = manifest.get("artifacts", {})
    expected_overhead_repo_path = f"{sample_dir}/overhead.png"
    if artifacts.get("overhead") != expected_overhead_repo_path:
        return False

    segmented_repo_path = artifacts.get("segmented_overhead")
    if not isinstance(segmented_repo_path, str):
        return False
    segmented_path = output_root / segmented_repo_path
    if not segmented_path.exists():
        return False

    expected_metadata_url = hf_file_url(repo_id, expected_overhead_repo_path)
    metadata_path = output_root / "data" / "metadata.jsonl"
    if not metadata_path.exists():
        return True
    try:
        rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return False
    matches = [row for row in rows if int(row.get("sample_id", -1)) == int(sample_id)]
    if not matches:
        return True
    if matches[0].get("overhead_file_name") != expected_metadata_url:
        return False
    return True


def build_metadata_row_with_overhead(row: dict, repo_id: str, overhead_repo_path: str, manifest: dict) -> dict:
    updated = {"sample_id": int(row["sample_id"]), "overhead_file_name": hf_file_url(repo_id, overhead_repo_path)}
    for key, value in row.items():
        if key in {"sample_id", "segmented_overhead_file_name", "overhead_file_name", "manifest"}:
            continue
        updated[key] = value
    updated["manifest"] = json.dumps(manifest, ensure_ascii=True)
    return updated


def rebuild_sample_artifacts(sample_id: int, repo_id: str, metadata_row: dict) -> dict:
    sample_dir = sample_dir_rel(sample_id)
    manifest_repo_path = f"{sample_dir}/manifest.json"
    log(f"sample_id={sample_id} loading manifest {manifest_repo_path}")
    manifest = read_json(repo_id, manifest_repo_path)

    artifacts = manifest.get("artifacts", {})
    cropped_repo_path = artifact_repo_path(artifacts.get("cropped_overhead"))
    if cropped_repo_path is None:
        raise FileNotFoundError(f"manifest {manifest_repo_path} does not contain artifacts.cropped_overhead")

    log(f"sample_id={sample_id} manifest cropped_overhead={cropped_repo_path}")
    cropped_image, cropped_repo_path = download_image(repo_id, cropped_repo_path, sample_id=sample_id)
    mask, mask_repo_path = load_mask(repo_id, manifest, sample_id=sample_id)
    log(f"sample_id={sample_id} resolved cropped_overhead={cropped_repo_path}")
    log(f"sample_id={sample_id} resolved mask={mask_repo_path}")
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

    return {
        "sample_id": int(sample_id),
        "repo_id": repo_id,
        "speakers": speakers,
        "cropped_overhead_repo_path": cropped_repo_path,
        "mask_repo_path": mask_repo_path,
        "segmented_overhead_repo_path": segmented_repo_path,
        "overhead_repo_path": overhead_repo_path,
        "manifest_repo_path": manifest_repo_path,
        "segmented_overhead": segmented_overhead,
        "sample_overhead": sample_overhead,
        "manifest": updated_manifest,
        "metadata_row": build_metadata_row_with_overhead(metadata_row, repo_id, overhead_repo_path, updated_manifest),
    }


def target_sample_ids(metadata_rows: list[dict], sample_id: int | None, all_samples: bool) -> list[int]:
    if all_samples:
        return sorted({int(row["sample_id"]) for row in metadata_rows})
    if sample_id is None:
        raise ValueError("provide a sample_id or pass --all-samples")
    return [int(sample_id)]


def default_output_root(sample_ids: list[int], all_samples: bool) -> Path:
    if all_samples:
        return REPO_ROOT / "tmp" / "overhead_artifact_repairs" / "all-samples"
    return REPO_ROOT / "tmp" / "overhead_artifact_repairs" / f"{int(sample_ids[0]):07d}"


def stage_rebuilt_overhead_artifacts(
    sample_id: int | None = None,
    *,
    all_samples: bool = False,
    repo_id: str = DEFAULT_REPO_ID,
    output_root: str | Path | None = None,
    resume: bool = True,
) -> dict:
    metadata_rows = read_jsonl_rows(repo_id, "data/metadata.jsonl")
    sample_ids = target_sample_ids(metadata_rows, sample_id=sample_id, all_samples=all_samples)

    rows_by_id = {}
    for row in metadata_rows:
        sid = int(row.get("sample_id", -1))
        if sid in rows_by_id:
            raise ValueError(f"duplicate metadata row for sample_id={sid}")
        rows_by_id[sid] = row

    if output_root is None:
        output_root = default_output_root(sample_ids, all_samples=all_samples)
    output_root = Path(output_root)

    updated_rows_by_id = {}
    files_written = []
    sample_summaries = []
    skipped_sample_ids = []
    for sid in sample_ids:
        row = rows_by_id.get(sid)
        if row is None:
            raise ValueError(f"metadata.jsonl is missing sample_id={sid}")
        if resume and validate_staged_sample(output_root, sid, repo_id):
            log(f"sample_id={sid} already staged; skipping rebuild")
            skipped_sample_ids.append(sid)
            manifest_path = output_root / sample_dir_rel(sid) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            updated_rows_by_id[sid] = build_metadata_row_with_overhead(row, repo_id, f"{sample_dir_rel(sid)}/overhead.png", manifest)
            sample_summaries.append({
                "sample_id": sid,
                "skipped": True,
                "overhead_repo_path": f"{sample_dir_rel(sid)}/overhead.png",
                "manifest_repo_path": f"{sample_dir_rel(sid)}/manifest.json",
            })
            continue
        sample_result = rebuild_sample_artifacts(sid, repo_id, row)

        segmented_local_path = output_root / sample_result["segmented_overhead_repo_path"]
        overhead_local_path = output_root / sample_result["overhead_repo_path"]
        manifest_local_path = output_root / sample_result["manifest_repo_path"]

        write_image(segmented_local_path, sample_result["segmented_overhead"])
        write_image(overhead_local_path, sample_result["sample_overhead"])
        write_json(manifest_local_path, sample_result["manifest"])

        updated_rows_by_id[sid] = sample_result["metadata_row"]
        files_written.extend([str(segmented_local_path), str(overhead_local_path), str(manifest_local_path)])
        sample_summaries.append({
            "sample_id": sid,
            "skipped": False,
            "speakers": sample_result["speakers"],
            "segmented_overhead_repo_path": sample_result["segmented_overhead_repo_path"],
            "overhead_repo_path": sample_result["overhead_repo_path"],
            "manifest_repo_path": sample_result["manifest_repo_path"],
        })

    final_metadata_rows = [updated_rows_by_id.get(int(row.get("sample_id", -1)), row) for row in metadata_rows]
    metadata_local_path = output_root / "data" / "metadata.jsonl"
    write_jsonl(metadata_local_path, final_metadata_rows)
    files_written.append(str(metadata_local_path))

    dataset_card_local_path = copy_dataset_card(output_root)
    files_written.append(str(dataset_card_local_path))

    return {
        "repo_id": repo_id,
        "all_samples": all_samples,
        "sample_ids": sample_ids,
        "n_samples": len(sample_ids),
        "n_skipped": len(skipped_sample_ids),
        "skipped_sample_ids": skipped_sample_ids,
        "metadata_repo_path": "data/metadata.jsonl",
        "dataset_card_repo_path": "README.md",
        "output_root": str(output_root),
        "suggested_upload_command": f"hf upload-large-folder {repo_id} {output_root} --repo-type dataset",
        "sample_summaries": sample_summaries,
        "files_written": files_written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild shared segmented_overhead and per-sample overhead for one sample or all samples.")
    parser.add_argument("sample_id", type=int, nargs="?")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--all-samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    result = stage_rebuilt_overhead_artifacts(
        sample_id=args.sample_id,
        all_samples=args.all_samples,
        repo_id=args.repo_id,
        output_root=args.output_root,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
