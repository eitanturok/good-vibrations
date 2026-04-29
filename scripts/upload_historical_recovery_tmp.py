import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STAGING_ROOT = REPO_ROOT / "tmp"
DEFAULT_REPO_ID = "eturok-weizmann/laser-vibrations"


def log(message: str) -> None:
    print(f"[info] {message}", flush=True)


def load_jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def historical_sample_ids(metadata_rows: list[dict]) -> list[int]:
    ids = []
    for row in metadata_rows:
        manifest = row.get("manifest")
        if isinstance(manifest, str) and "historical_from_vibrations_only" in manifest:
            ids.append(int(row["sample_id"]))
    return sorted(ids)


def validate_staging_root(staging_root: Path, existing_repo_files: set[str] | None = None) -> tuple[list[dict], list[int]]:
    metadata_path = staging_root / "data" / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing staged metadata file: {metadata_path}")

    metadata_rows = load_jsonl_rows(metadata_path)
    staged_ids = historical_sample_ids(metadata_rows)
    if not staged_ids:
        raise ValueError("no historical recovery rows found in staged data/metadata.jsonl")

    missing_sample_dirs = []
    missing_sample_files = []
    missing_image_files = []
    duplicate_sample_ids = []
    seen = set()

    for sample_id in staged_ids:
        if sample_id in seen:
            duplicate_sample_ids.append(sample_id)
        seen.add(sample_id)

        sample_dir = staging_root / "data" / f"{sample_id:07d}"
        if not sample_dir.exists():
            missing_sample_dirs.append(sample_id)
            continue

        required_sample_files = [
            "speckle_shifts.npz",
            "speckle_shifts_clean.npz",
            "speckle_shifts_fft.npz",
            "speckle_shifts_ifft_audio.wav",
            "manifest.json",
            "recovery_summary.json",
        ]
        missing = [name for name in required_sample_files if not (sample_dir / name).exists()]
        if missing:
            missing_sample_files.append((sample_id, missing))

    for row in metadata_rows:
        manifest = row.get("manifest")
        if not (isinstance(manifest, str) and "historical_from_vibrations_only" in manifest):
            continue
        for key in ["segmented_overhead_file_name", "mask_file_name"]:
            url = row.get(key)
            if not isinstance(url, str) or "/resolve/main/" not in url:
                continue
            repo_path = url.split("/resolve/main/", 1)[1]
            local_path = staging_root / repo_path
            if not local_path.exists() and (existing_repo_files is None or repo_path not in existing_repo_files):
                missing_image_files.append((int(row["sample_id"]), key, repo_path))

    if duplicate_sample_ids:
        raise ValueError(f"duplicate historical sample_ids in staged metadata: {sorted(set(duplicate_sample_ids))}")
    if missing_sample_dirs:
        raise FileNotFoundError(f"missing staged sample directories for sample_ids={missing_sample_dirs}")
    if missing_sample_files:
        raise FileNotFoundError(f"missing staged sample files: {missing_sample_files[:10]}")
    if missing_image_files:
        raise FileNotFoundError(f"missing staged image files: {missing_image_files[:10]}")

    return metadata_rows, staged_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload staged historical recovery files from tmp/ to laser-vibrations using upload_large_folder.")
    parser.add_argument("--staging-root", default=str(DEFAULT_STAGING_ROOT))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    staging_root = Path(args.staging_root)
    api = HfApi()
    log(f"listing existing repo files from {args.repo_id}")
    existing_repo_files = set(api.list_repo_files(args.repo_id, repo_type="dataset"))
    log(f"validating staged recovery root {staging_root}")
    metadata_rows, staged_ids = validate_staging_root(staging_root, existing_repo_files=existing_repo_files)
    log(f"validated {len(staged_ids)} staged historical samples")
    log(f"sample_id range: {min(staged_ids)}..{max(staged_ids)}")
    log(f"staged metadata rows total: {len(metadata_rows)}")
    log("upload will include only data/** and image/** from the staging root")

    if args.dry_run:
        print(json.dumps({
            "staging_root": str(staging_root),
            "repo_id": args.repo_id,
            "num_workers": args.num_workers,
            "n_historical_samples": len(staged_ids),
            "sample_ids_first20": staged_ids[:20],
            "allow_patterns": ["data/**", "image/**"],
        }, indent=2))
        return

    log(f"creating repo if needed: {args.repo_id}")
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    log("starting upload_large_folder")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(staging_root),
        allow_patterns=["data/**", "image/**"],
        num_workers=args.num_workers,
        print_report=True,
        print_report_every=60,
    )
    log(f"upload complete: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
