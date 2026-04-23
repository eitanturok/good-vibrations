import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


DEFAULT_REPO_ID = "eturok-weizmann/laser-vibrations"
DEFAULT_REMOTE_HOST = "ethantu@mcluster11.wisdom.weizmann.ac.il"
DEFAULT_REMOTE_EXPERIMENT_DIR = "/net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA/experiment-16"
REQUIRED_METADATA_KEYS = [
    "segmented_overhead_file_name",
    "mask_file_name",
    "speckle_vibrations_file_name",
    "speckle_shifts_ifft_audio_file_name",
    "audio_file_name",
]


def stage(label, fn):
    result = fn()
    print(f"[done] {label}", flush=True)
    return result


def repo_path_from_hf_url(url: str) -> str | None:
    marker = "/resolve/main/"
    if not isinstance(url, str) or marker not in url:
        return None
    return url.split(marker, 1)[1]


def load_metadata_rows(repo_id: str) -> tuple[list[dict], Path]:
    metadata_path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="data/metadata.jsonl"))
    rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, metadata_path


def missing_repo_paths(rows: list[dict], existing: set[str]) -> list[str]:
    missing = []
    for row in rows:
        for key in REQUIRED_METADATA_KEYS:
            repo_path = repo_path_from_hf_url(row.get(key, ""))
            if repo_path and repo_path not in existing:
                missing.append(repo_path)
    return sorted(set(missing))


def fetch_remote_file(remote_host: str, remote_path: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as f:
        subprocess.run(
            ["ssh", remote_host, f"cat {shlex.quote(remote_path)}"],
            check=True,
            stdout=f,
        )
    return local_path


def fetch_remote_batch(remote_host: str, remote_experiment_dir: str, repo_paths: list[str], tmp_root: Path) -> list[str]:
    if not repo_paths:
        return []
    quoted_paths = " ".join(shlex.quote(path) for path in repo_paths)
    remote_cmd = (
        "bash -lc "
        + shlex.quote(
            f"set -euo pipefail; "
            f"cd {shlex.quote(remote_experiment_dir)}; "
            f"tar -cf - {quoted_paths}"
        )
    )
    ssh_proc = subprocess.Popen(["ssh", remote_host, remote_cmd], stdout=subprocess.PIPE)
    try:
        subprocess.run(["tar", "-xf", "-", "-C", str(tmp_root)], stdin=ssh_proc.stdout, check=True)
    finally:
        if ssh_proc.stdout is not None:
            ssh_proc.stdout.close()
    rc = ssh_proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, ["ssh", remote_host, remote_cmd])
    return repo_paths


def create_and_merge_pr(api: HfApi, repo_id: str, operations: list[CommitOperationAdd], commit_message: str) -> None:
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=commit_message,
        create_pr=True,
    )
    discussion_num = getattr(commit, "pr_num", None)
    if discussion_num is None:
        pr_url = getattr(commit, "pr_url", None) or ""
        if "/discussions/" not in pr_url:
            raise RuntimeError(f"Could not determine PR number for commit: {commit}")
        discussion_num = int(pr_url.rstrip("/").rsplit("/", 1)[1])
    api.merge_pull_request(repo_id=repo_id, repo_type="dataset", discussion_num=int(discussion_num))
    print(f"[hf] merged PR #{discussion_num}: {commit_message}", flush=True)


def batched(items: list[str], batch_size: int):
    for idx in range(0, len(items), batch_size):
        yield items[idx:idx + batch_size]


def upload_missing_files(repo_id: str, remote_host: str, remote_experiment_dir: str, batch_size: int, max_files: int | None) -> list[str]:
    api = HfApi()
    rows, _ = load_metadata_rows(repo_id)
    existing = set(api.list_repo_files(repo_id, repo_type="dataset"))
    missing = missing_repo_paths(rows, existing)
    if max_files is not None:
        missing = missing[:max_files]
    print(f"[info] missing referenced files on HF: {len(missing)}", flush=True)
    if not missing:
        return []

    uploaded = []
    with tempfile.TemporaryDirectory(prefix="hf-recover-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for batch_idx, batch in enumerate(batched(missing, batch_size), start=1):
            stage(
                f"fetch batch {batch_idx} ({len(batch)} files)",
                lambda b=batch: fetch_remote_batch(remote_host, remote_experiment_dir, b, tmp_root),
            )
            operations = []
            for repo_path in batch:
                local_path = tmp_root / repo_path
                operations.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path)))
            create_and_merge_pr(
                api,
                repo_id,
                operations,
                commit_message=f"Recover missing laser-vibrations assets batch {batch_idx}",
            )
            uploaded.extend(batch)
    return uploaded


def build_filtered_metadata_bytes(rows: list[dict], existing: set[str]) -> bytes:
    kept = []
    for row in rows:
        required_paths = []
        for key in REQUIRED_METADATA_KEYS:
            repo_path = repo_path_from_hf_url(row.get(key, ""))
            if repo_path:
                required_paths.append(repo_path)
        if required_paths and all(path in existing for path in required_paths):
            kept.append(json.dumps(row, ensure_ascii=True))
    return (("\n".join(kept) + "\n") if kept else "").encode("utf-8")


def upload_filtered_metadata(repo_id: str) -> None:
    api = HfApi()
    rows, _ = load_metadata_rows(repo_id)
    existing = set(api.list_repo_files(repo_id, repo_type="dataset"))
    body = build_filtered_metadata_bytes(rows, existing)
    create_and_merge_pr(
        api,
        repo_id,
        [CommitOperationAdd(path_in_repo="data/metadata.jsonl", path_or_fileobj=body)],
        commit_message="Filter metadata.jsonl to committed viewer assets",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover missing HF viewer assets for laser-vibrations.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-experiment-dir", default=DEFAULT_REMOTE_EXPERIMENT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    uploaded = stage(
        "upload missing referenced files",
        lambda: upload_missing_files(
            repo_id=args.repo_id,
            remote_host=args.remote_host,
            remote_experiment_dir=args.remote_experiment_dir,
            batch_size=args.batch_size,
            max_files=args.max_files,
        ),
    )
    print(f"[info] uploaded files: {len(uploaded)}", flush=True)
    stage("upload filtered metadata.jsonl", lambda: upload_filtered_metadata(args.repo_id))


if __name__ == "__main__":
    main()
