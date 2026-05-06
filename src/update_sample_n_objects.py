import argparse
import json
import re
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi, hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError


DEFAULT_REPO_ID = "eturok-weizmann/laser-vibrations"
EXPECTED_SHARED_IMAGE_FILES = (
    "raw_overhead.png",
    "cropped_overhead.png",
    "segmented_overhead.png",
    "mask.png",
    "mask.npz",
)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tmp" / "n_objects_updates"


def log(message: str) -> None:
    print(f"[info] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[warn] {message}", flush=True)


def sample_dir_rel(sample_id: int) -> str:
    return f"data/{int(sample_id):07d}"


def hf_file_url(repo_id: str, repo_path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{repo_path}"


def parse_updates(values: list[str]) -> list[tuple[int, int]]:
    if len(values) % 2 != 0:
        raise ValueError("updates must be provided as pairs: <sample_id> <new_n_objects>")
    updates = []
    for idx in range(0, len(values), 2):
        updates.append((int(values[idx]), int(values[idx + 1])))
    return updates


def load_metadata_rows(repo_id: str) -> tuple[list[dict], Path]:
    metadata_path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="data/metadata.jsonl"))
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, metadata_path


def load_manifest(repo_id: str, sample_id: int) -> dict:
    manifest_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f"{sample_dir_rel(sample_id)}/manifest.json")
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def replace_n_objects_token(image_dir: str, new_n_objects: int) -> str:
    updated, count = re.subn(r"-(?P<n>\d+)obj-", f"-{int(new_n_objects)}obj-", image_dir, count=1)
    if count != 1:
        raise ValueError(f"could not uniquely replace n_objects token in image_dir={image_dir!r}")
    return updated


def current_manifest_from_row(row: dict) -> dict:
    manifest = row.get("manifest")
    if not isinstance(manifest, str):
        raise ValueError(f"sample_id={row.get('sample_id')} row is missing manifest string")
    return json.loads(manifest)


def build_rows_by_id(rows: list[dict]) -> dict[int, dict]:
    rows_by_id = {}
    for row in rows:
        sid = int(row["sample_id"])
        if sid in rows_by_id:
            raise ValueError(f"duplicate metadata row for sample_id={sid}")
        rows_by_id[sid] = row
    return rows_by_id


def shared_image_paths(image_dir: str) -> dict[str, str]:
    root = f"image/{image_dir}"
    return {
        "raw_overhead": f"{root}/raw_overhead.png",
        "cropped_overhead": f"{root}/cropped_overhead.png",
        "segmented_overhead": f"{root}/segmented_overhead.png",
        "mask_png": f"{root}/mask.png",
        "mask_npz": f"{root}/mask.npz",
    }


def update_manifest_for_image_dir(manifest: dict, new_n_objects: int, new_image_dir: str) -> dict:
    payload = json.loads(json.dumps(manifest))
    payload.setdefault("sample", {})["n_objects"] = int(new_n_objects)
    payload["sample"]["image_dir"] = new_image_dir
    payload.setdefault("artifacts", {}).update(shared_image_paths(new_image_dir))
    return payload


def update_metadata_row(row: dict, repo_id: str, new_n_objects: int, manifest: dict) -> dict:
    payload = dict(row)
    payload["n_objects"] = int(new_n_objects)
    payload["mask_file_name"] = hf_file_url(repo_id, manifest["artifacts"]["mask_png"])
    payload["manifest"] = json.dumps(manifest, ensure_ascii=True)
    return payload


def list_dir_files(api: HfApi, repo_id: str, path_in_repo: str) -> dict[str, object]:
    try:
        entries = list(api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo=path_in_repo, recursive=False))
    except RemoteEntryNotFoundError:
        return {}
    return {Path(entry.path).name: entry for entry in entries}


def lfs_sha(entry: object) -> str | None:
    lfs = getattr(entry, "lfs", None)
    if lfs is None:
        return None
    return getattr(lfs, "sha256", None)


def validate_existing_target_dir(api: HfApi, repo_id: str, old_image_dir: str, new_image_dir: str) -> None:
    old_entries = list_dir_files(api, repo_id, f"image/{old_image_dir}")
    new_entries = list_dir_files(api, repo_id, f"image/{new_image_dir}")
    missing_old = [name for name in EXPECTED_SHARED_IMAGE_FILES if name not in old_entries]
    missing_new = [name for name in EXPECTED_SHARED_IMAGE_FILES if name not in new_entries]
    if missing_old:
        raise FileNotFoundError(f"old image_dir image/{old_image_dir} is missing files: {missing_old}")
    if missing_new:
        raise FileNotFoundError(f"existing target image_dir image/{new_image_dir} is missing files: {missing_new}")
    mismatched = []
    for name in EXPECTED_SHARED_IMAGE_FILES:
        if lfs_sha(old_entries[name]) != lfs_sha(new_entries[name]):
            mismatched.append(name)
    if mismatched:
        raise ValueError(
            f"existing target image_dir image/{new_image_dir} does not match image/{old_image_dir}; mismatched files: {mismatched}"
        )


def validate_old_dir_exists(api: HfApi, repo_id: str, old_image_dir: str) -> None:
    old_entries = list_dir_files(api, repo_id, f"image/{old_image_dir}")
    missing_old = [name for name in EXPECTED_SHARED_IMAGE_FILES if name not in old_entries]
    if missing_old:
        raise FileNotFoundError(f"old image_dir image/{old_image_dir} is missing files: {missing_old}")


def image_dir_exists(api: HfApi, repo_id: str, image_dir: str) -> bool:
    return bool(list_dir_files(api, repo_id, f"image/{image_dir}"))


def scan_rows_referencing_image_dir(rows: list[dict], image_dir: str) -> list[int]:
    target_fragment = f"image/{image_dir}/"
    sample_ids = []
    for row in rows:
        manifest = current_manifest_from_row(row)
        if manifest.get("sample", {}).get("image_dir") == image_dir:
            sample_ids.append(int(row["sample_id"]))
            continue
        if target_fragment in json.dumps(manifest, ensure_ascii=True):
            sample_ids.append(int(row["sample_id"]))
            continue
        if target_fragment in json.dumps(row, ensure_ascii=True):
            sample_ids.append(int(row["sample_id"]))
    return sorted(set(sample_ids))


def ensure_no_references_to_deleted_dirs(rows: list[dict], deleted_image_dirs: set[str]) -> None:
    leftovers = {}
    for image_dir in sorted(deleted_image_dirs):
        refs = scan_rows_referencing_image_dir(rows, image_dir)
        if refs:
            leftovers[image_dir] = refs
    if leftovers:
        raise ValueError(f"stale references remain to deleted image_dirs: {leftovers}")


def write_local_outputs(output_root: Path, rows: list[dict], updated_manifest_by_id: dict[int, dict], report: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "data" / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")
    for sample_id, manifest in updated_manifest_by_id.items():
        manifest_path = output_root / sample_dir_rel(sample_id) / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")


def build_commit_operations(
    repo_id: str,
    rows: list[dict],
    updated_manifest_by_id: dict[int, dict],
    image_dir_moves: dict[str, str],
    reused_image_dirs: set[str],
) -> list[object]:
    operations: list[object] = []
    moved_old_dirs = set(image_dir_moves)
    for old_image_dir, new_image_dir in image_dir_moves.items():
        if new_image_dir in reused_image_dirs:
            continue
        for filename in EXPECTED_SHARED_IMAGE_FILES:
            operations.append(
                CommitOperationCopy(
                    src_path_in_repo=f"image/{old_image_dir}/{filename}",
                    path_in_repo=f"image/{new_image_dir}/{filename}",
                )
            )
    metadata_bytes = ("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n").encode("utf-8")
    operations.append(CommitOperationAdd(path_in_repo="data/metadata.jsonl", path_or_fileobj=metadata_bytes))
    for sample_id, manifest in sorted(updated_manifest_by_id.items()):
        body = json.dumps(manifest, indent=2, ensure_ascii=True).encode("utf-8")
        operations.append(CommitOperationAdd(path_in_repo=f"{sample_dir_rel(sample_id)}/manifest.json", path_or_fileobj=body))
    for old_image_dir in sorted(moved_old_dirs):
        for filename in EXPECTED_SHARED_IMAGE_FILES:
            operations.append(CommitOperationDelete(path_in_repo=f"image/{old_image_dir}/{filename}"))
    return operations


def apply_updates(
    rows: list[dict],
    updates: list[tuple[int, int]],
    repo_id: str,
    api: HfApi,
) -> tuple[list[dict], dict[int, dict], dict[str, str], set[str], list[dict]]:
    rows_by_id = build_rows_by_id(rows)
    updated_manifest_by_id: dict[int, dict] = {}
    image_dir_moves: dict[str, str] = {}
    reused_image_dirs: set[str] = set()
    update_reports: list[dict] = []

    for sample_id, new_n_objects in updates:
        if sample_id not in rows_by_id:
            raise ValueError(f"sample_id={sample_id} not found in metadata.jsonl")
        target_row = rows_by_id[sample_id]
        target_manifest = updated_manifest_by_id.get(sample_id) or load_manifest(repo_id, sample_id)
        old_image_dir = str(target_manifest.get("sample", {}).get("image_dir") or "")
        if not old_image_dir:
            raise ValueError(f"sample_id={sample_id} manifest missing sample.image_dir")
        old_n_objects = int(target_manifest.get("sample", {}).get("n_objects"))
        new_image_dir = replace_n_objects_token(old_image_dir, new_n_objects)

        if old_n_objects == int(new_n_objects) and old_image_dir == new_image_dir:
            warn(
                f"sample_id={sample_id} already has n_objects={new_n_objects} "
                f"and image_dir={old_image_dir}; skipping no-op update"
            )
            update_reports.append(
                {
                    "requested_sample_id": sample_id,
                    "new_n_objects": int(new_n_objects),
                    "old_image_dir": old_image_dir,
                    "new_image_dir": new_image_dir,
                    "affected_sample_ids": [sample_id],
                    "reused_existing_target": False,
                    "skipped_noop": True,
                }
            )
            continue

        affected_sample_ids = scan_rows_referencing_image_dir(rows, old_image_dir)
        if not affected_sample_ids:
            raise ValueError(f"no samples reference image_dir={old_image_dir}")

        log(
            f"sample_id={sample_id} old_n_objects={old_n_objects} new_n_objects={new_n_objects} "
            f"old_image_dir={old_image_dir} new_image_dir={new_image_dir} affected_samples={affected_sample_ids}"
        )

        if old_image_dir != new_image_dir and old_image_dir not in image_dir_moves:
            validate_old_dir_exists(api, repo_id, old_image_dir)
            if image_dir_exists(api, repo_id, new_image_dir):
                validate_existing_target_dir(api, repo_id, old_image_dir, new_image_dir)
                warn(f"target image_dir image/{new_image_dir} already exists and matches; reusing it")
                reused_image_dirs.add(new_image_dir)
            else:
                log(f"target image_dir image/{new_image_dir} does not exist; will rename shared files")
                image_dir_moves[old_image_dir] = new_image_dir
            image_dir_moves[old_image_dir] = new_image_dir
        elif old_image_dir in image_dir_moves and image_dir_moves[old_image_dir] != new_image_dir:
            raise ValueError(
                f"conflicting requested rename for image/{old_image_dir}: "
                f"{image_dir_moves[old_image_dir]} vs {new_image_dir}"
            )

        for affected_id in affected_sample_ids:
            row = rows_by_id[affected_id]
            manifest = updated_manifest_by_id.get(affected_id) or load_manifest(repo_id, affected_id)
            current_image_dir = str(manifest.get("sample", {}).get("image_dir") or "")
            if current_image_dir != old_image_dir:
                raise ValueError(
                    f"sample_id={affected_id} no longer points to old image_dir={old_image_dir}; current={current_image_dir}"
                )
            new_manifest = update_manifest_for_image_dir(manifest, new_n_objects, new_image_dir)
            new_row = update_metadata_row(row, repo_id, new_n_objects, new_manifest)
            rows_by_id[affected_id] = new_row
            updated_manifest_by_id[affected_id] = new_manifest

        rows = [rows_by_id[int(row["sample_id"])] for row in rows]
        update_reports.append(
            {
                "requested_sample_id": sample_id,
                "new_n_objects": int(new_n_objects),
                "old_image_dir": old_image_dir,
                "new_image_dir": new_image_dir,
                "affected_sample_ids": affected_sample_ids,
                "reused_existing_target": new_image_dir in reused_image_dirs,
                "skipped_noop": False,
            }
        )

    ensure_no_references_to_deleted_dirs(rows, set(image_dir_moves))
    return rows, updated_manifest_by_id, image_dir_moves, reused_image_dirs, update_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Update n_objects for one or more samples and rename shared image_dirs on HF.")
    parser.add_argument("updates", nargs="+", help="Pairs of <sample_id> <new_n_objects>")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--apply", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--commit-message", default=None)
    args = parser.parse_args()

    updates = parse_updates(args.updates)
    repo_id = str(args.repo_id)
    output_root = Path(args.output_root)

    metadata_rows, _ = load_metadata_rows(repo_id)
    api = HfApi()
    updated_rows, updated_manifest_by_id, image_dir_moves, reused_image_dirs, update_reports = apply_updates(
        metadata_rows,
        updates,
        repo_id,
        api,
    )

    operations = build_commit_operations(
        repo_id=repo_id,
        rows=updated_rows,
        updated_manifest_by_id=updated_manifest_by_id,
        image_dir_moves=image_dir_moves,
        reused_image_dirs=reused_image_dirs,
    )

    report = {
        "repo_id": repo_id,
        "updates": update_reports,
        "n_modified_metadata_rows": len(updated_manifest_by_id),
        "modified_sample_ids": sorted(updated_manifest_by_id),
        "image_dir_moves": image_dir_moves,
        "reused_image_dirs": sorted(reused_image_dirs),
        "n_operations": len(operations),
        "operations_summary": {
            "copy": sum(isinstance(op, CommitOperationCopy) for op in operations),
            "add": sum(isinstance(op, CommitOperationAdd) for op in operations),
            "delete": sum(isinstance(op, CommitOperationDelete) for op in operations),
        },
        "output_root": str(output_root),
        "apply": bool(args.apply),
    }

    write_local_outputs(output_root, updated_rows, updated_manifest_by_id, report)

    if args.apply:
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=args.commit_message or f"Update n_objects for samples {[sid for sid, _ in updates]}",
        )
        report["commit"] = {
            "oid": getattr(commit, "oid", None),
            "url": getattr(commit, "commit_url", None),
        }
        (output_root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
