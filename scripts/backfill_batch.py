"""
Batch backfill with a single persistent Modal segmentation container.

Opens app.run() once so the GPU container stays warm across all samples,
eliminating the ~35s cold-start penalty that the per-sample script pays.

Pipeline per sample:
  1. pre_segment  — runs on mcluster11: package raw files, FFT, crop overhead
  2. segment      — runs locally:       call segmenter.segment.remote() (warm container)
  3. push mask    — push mask.npz + mask.png back to mraid20 over SSH
  4. post_segment — runs on mcluster11: overlay, speaker overlay, manifest, metadata

Pipelining: pre_segment for sample N+1 runs in a background thread while
modal inference runs for sample N, hiding most of the ~30s I/O cost.

Usage (one sample):
    python scripts/backfill_batch.py --samples cube-00x01y_0001--31-03-18-21-24:1

Usage (multiple samples):
    python scripts/backfill_batch.py --samples cube-00x01y_0001--31-03-18-21-24:1 cube-01x02y_0001--01-04-10-00-00:2

Usage (all unprocessed samples, with failure tracking):
    python scripts/backfill_batch.py --all
    python scripts/backfill_batch.py --all --upload-to-hf
"""
import datetime
import io
import json
import re
import shlex
import subprocess
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from utils.segment import app, Segmenter
from migrate_experiment15_to_16_one import build_image_dir_name

REMOTE_HOST      = "ethantu@mcluster11.wisdom.weizmann.ac.il"
REMOTE_VENV      = "$HOME/venvs/experiment16-migrate"
REMOTE_SCRIPT    = "/home/ethantu/tmp/migrate_experiment15_to_16_one.py"
REMOTE_AUDIO     = "/home/ethantu/tmp/chirp_50_1000_3.0sec.wav"
REMOTE_README    = "/home/ethantu/tmp/hf_data/README.md"
REMOTE_EXP_CFG   = "/home/ethantu/tmp/hf_data/base_experiment_config.json"
REMOTE_PROC_CFG  = "/home/ethantu/tmp/hf_data/base_processing_config.json"
OLD_DIR       = "/net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA/experiment-15"
NEW_DIR       = "/net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA/experiment-16"
HF_REPO       = "eturok-weizmann/laser-vibrations"
DEFAULT_PROMPT = "A black metal cube sitting on the floor of an open cardboard box from a bird's eye view."

ASSIGNMENTS_FILE    = REPO_ROOT / "backfill_assignments.jsonl"
FAILURES_FILE       = REPO_ROOT / "backfill_failures.jsonl"
HF_UPLOAD_JOBS_FILE = REPO_ROOT / "backfill_hf_upload_jobs.jsonl"


# ── SSH helpers ───────────────────────────────────────────────────────────────

def _ssh_run(cmd: str) -> None:
    """Run a remote command, streaming output to stdout."""
    subprocess.run(
        ["ssh", REMOTE_HOST, f"bash -lc {shlex.quote(cmd)}"],
        check=True,
    )


def _ssh_fetch(cmd: str, stdin: bytes | None = None) -> bytes:
    """Run a remote command and capture its stdout."""
    result = subprocess.run(
        ["ssh", REMOTE_HOST, f"bash -lc {shlex.quote(cmd)}"],
        check=True, capture_output=True, input=stdin,
    )
    return result.stdout


def _sync_static_files() -> None:
    """Sync the migration script and static assets to mcluster11 (checksum-skipped if unchanged)."""
    import hashlib

    def _local_md5(path: Path) -> str:
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _remote_md5(remote_path: str) -> str | None:
        cmd = f"md5sum {shlex.quote(remote_path)} 2>/dev/null || true"
        line = _ssh_fetch(cmd).decode().strip()
        return line.split()[0] if line else None

    def sync(local: Path, remote: str, label: str) -> None:
        t = time.perf_counter()
        if _local_md5(local) == _remote_md5(remote):
            print(f"[sync] {label}: skipped (unchanged)", flush=True)
            return
        _ssh_fetch(f"mkdir -p $(dirname {shlex.quote(remote)}) && cat > {shlex.quote(remote)}", stdin=local.read_bytes())
        print(f"[sync] {label}: {time.perf_counter() - t:.2f}s", flush=True)

    migrate_script = REPO_ROOT / "scripts" / "migrate_experiment15_to_16_one.py"
    sync(migrate_script, REMOTE_SCRIPT, "migrate script")
    for key in ("1000", "0100", "0010", "0001", "speaker"):
        sync(REPO_ROOT / "assets" / "speakers" / f"{key}.png", f"/home/ethantu/assets/speakers/{key}.png", f"speaker {key}.png")
    audio_local = REPO_ROOT / "data" / "audio_samples" / "chirp_50_1000_3.0sec.wav"
    sync(audio_local, REMOTE_AUDIO, "audio chirp")
    sync(REPO_ROOT / "hf_data" / "README.md",                   REMOTE_README,   "hf_data README")
    sync(REPO_ROOT / "hf_data" / "base_experiment_config.json", REMOTE_EXP_CFG,  "hf_data base_experiment_config")
    sync(REPO_ROOT / "hf_data" / "base_processing_config.json", REMOTE_PROC_CFG, "hf_data base_processing_config")


def _run_remote_stage(stage: str, source_dir_name: str, sample_id: int) -> None:
    # Note: REMOTE_VENV contains $HOME which must NOT be single-quoted so bash expands it.
    quoted_args = " ".join(shlex.quote(a) for a in [
        REMOTE_SCRIPT, "--remote-worker",
        "--old-dir", OLD_DIR, "--new-dir", NEW_DIR, "--hf-repo", HF_REPO,
        "--source-dir-name", source_dir_name,
        "--sample-id", str(sample_id),
        "--stage", stage, "--overwrite",
        "--remote-audio-path", REMOTE_AUDIO,
    ])
    _ssh_run(f"{REMOTE_VENV}/bin/python {quoted_args}")


# ── Path helpers ──────────────────────────────────────────────────────────────

def _image_dir(
    source_dir_name: str,
    object_name: str | None = None,
    n_objects: int = 1,
    box_material: str = "cardboard",
    tags: list[str] | None = None,
) -> str:
    """Canonical image directory name matching migrate_experiment15_to_16_one.build_image_dir_name."""
    m = re.search(r"(?P<x>\d{2})x(?P<y>\d{2})y", source_dir_name)
    x = int(m.group("x")) if m else None
    y = int(m.group("y")) if m else None
    obj = object_name or source_dir_name.split("-")[0].strip().lower()
    return build_image_dir_name(
        source_dir_name,
        object_name=obj,
        x_position=x,
        y_position=y,
        n_objects=n_objects,
        box_material=box_material,
        tags=tags,
    )


# ── --all mode helpers ────────────────────────────────────────────────────────

def _list_source_dirs() -> list[str]:
    """Return sorted list of all source dir names in OLD_DIR."""
    raw = _ssh_fetch(f"ls {shlex.quote(OLD_DIR)}").decode()
    return sorted(line.strip() for line in raw.splitlines() if line.strip())


def _processed_experiment_ids() -> set[str]:
    """Read experiment_ids already present in experiment-16 metadata.jsonl."""
    try:
        raw = _ssh_fetch(
            f"cat {shlex.quote(NEW_DIR + '/data/metadata.jsonl')} 2>/dev/null || true"
        ).decode()
    except Exception:
        return set()
    ids = set()
    for line in raw.splitlines():
        line = line.strip()
        if line:
            try:
                ids.add(json.loads(line)["experiment_id"])
            except Exception:
                pass
    return ids


def _load_or_create_assignments(source_dirs: list[str], start_id: int) -> list[tuple[str, int]]:
    """
    Load existing assignments from ASSIGNMENTS_FILE, appending new entries for any
    source_dir not yet assigned. Returns the ordered list of (source_dir_name, sample_id)
    for the given source_dirs.
    """
    existing: dict[str, int] = {}
    if ASSIGNMENTS_FILE.exists():
        for line in ASSIGNMENTS_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[rec["source_dir_name"]] = int(rec["sample_id"])

    next_id = max([start_id - 1] + list(existing.values())) + 1
    new_lines = []
    for d in source_dirs:
        if d not in existing:
            existing[d] = next_id
            new_lines.append(json.dumps({"source_dir_name": d, "sample_id": next_id}))
            next_id += 1

    if new_lines:
        with ASSIGNMENTS_FILE.open("a", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"[assignments] wrote {len(new_lines)} new entries to {ASSIGNMENTS_FILE}", flush=True)

    return [(d, existing[d]) for d in source_dirs]


def _missing_experiment_configs(source_dir_names: list[str]) -> set[str]:
    """Return the subset of source_dir_names whose experiment_config.json is absent."""
    py_script = (
        f"import os\n"
        f"base = {OLD_DIR!r}\n"
        f"names = {source_dir_names!r}\n"
        f"missing = [n for n in names if not os.path.exists(os.path.join(base, n, 'experiment_config.json'))]\n"
        f"print('\\n'.join(missing))\n"
    )
    # Write script to cluster tmp file then run it (avoids SSH command-length limits)
    _ssh_fetch(f"cat > /tmp/_check_cfg.py", stdin=py_script.encode())
    raw = _ssh_fetch("python3 /tmp/_check_cfg.py").decode()
    return set(line.strip() for line in raw.splitlines() if line.strip())


def _ensure_hf_shared_files() -> None:
    """Upload README.md and shared audio to HF if not already present.

    These files must exist on HF before any sample rows are visible in the
    dataset viewer — README.md defines the features schema, and audio/
    holds the shared excitation WAV referenced by every metadata row.
    """
    script = (
        f"import os\n"
        f"os.environ['HF_HUB_DISABLE_XET'] = '1'\n"
        f"from huggingface_hub import HfApi\n"
        f"api = HfApi()\n"
        f"existing = set(api.list_repo_files({HF_REPO!r}, repo_type='dataset'))\n"
        f"# README always re-uploaded (content may have changed)\n"
        f"api.upload_file(path_or_fileobj={REMOTE_README!r},\n"
        f"                path_in_repo='README.md', repo_id={HF_REPO!r},\n"
        f"                repo_type='dataset', commit_message='Sync README')\n"
        f"print('[hf-shared] uploaded README.md', flush=True)\n"
        f"# Config files always re-uploaded (content may have changed)\n"
        f"api.upload_file(path_or_fileobj={REMOTE_EXP_CFG!r},\n"
        f"                path_in_repo='base_experiment_config.json', repo_id={HF_REPO!r},\n"
        f"                repo_type='dataset', commit_message='Sync base_experiment_config')\n"
        f"print('[hf-shared] uploaded base_experiment_config.json', flush=True)\n"
        f"api.upload_file(path_or_fileobj={REMOTE_PROC_CFG!r},\n"
        f"                path_in_repo='base_processing_config.json', repo_id={HF_REPO!r},\n"
        f"                repo_type='dataset', commit_message='Sync base_processing_config')\n"
        f"print('[hf-shared] uploaded base_processing_config.json', flush=True)\n"
        f"# Delete old data/ locations if they still exist\n"
        f"for old in ('data/base_experiment_config.json', 'data/base_processing_config.json'):\n"
        f"    if old in existing:\n"
        f"        api.delete_file(path_in_repo=old, repo_id={HF_REPO!r}, repo_type='dataset',\n"
        f"                        commit_message=f'Remove {{old}} (moved to top-level)')\n"
        f"        print(f'[hf-shared] deleted {{old}}', flush=True)\n"
        f"# Audio only uploaded if missing (content never changes)\n"
        f"audio_repo = 'audio/chirp_50_1000_3.0sec.wav'\n"
        f"audio_local = {NEW_DIR!r} + '/audio/chirp_50_1000_3.0sec.wav'\n"
        f"if audio_repo not in existing and os.path.exists(audio_local):\n"
        f"    api.upload_file(path_or_fileobj=audio_local, path_in_repo=audio_repo,\n"
        f"                    repo_id={HF_REPO!r}, repo_type='dataset',\n"
        f"                    commit_message='Add shared audio')\n"
        f"    print('[hf-shared] uploaded audio', flush=True)\n"
        f"else:\n"
        f"    print('[hf-shared] audio already present', flush=True)\n"
    )
    _ssh_fetch(f"cat > /tmp/_ensure_hf_shared.py", stdin=script.encode())
    _ssh_run(f"source /home/ethantu/venvs/experiment16-migrate/bin/activate && python3 /tmp/_ensure_hf_shared.py")


def _processed_sample_ids() -> list[int]:
    """Return sample_ids already written to experiment-16 metadata.jsonl."""
    try:
        raw = _ssh_fetch(
            f"cat {shlex.quote(NEW_DIR + '/data/metadata.jsonl')} 2>/dev/null || true"
        ).decode()
    except Exception:
        return []
    ids = []
    for line in raw.splitlines():
        if line.strip():
            try:
                ids.append(int(json.loads(line)["sample_id"]))
            except Exception:
                pass
    return sorted(ids)


def _submitted_sample_ids() -> set[int]:
    """Return all sample_ids already recorded in an HF upload job."""
    if not HF_UPLOAD_JOBS_FILE.exists():
        return set()
    ids: set[int] = set()
    for line in HF_UPLOAD_JOBS_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                ids.update(json.loads(line).get("new_sample_ids", []))
            except Exception:
                pass
    return ids


def _hf_upload_job_running() -> bool:
    """Return True if an hf-upload sbatch job is currently queued or running."""
    out = _ssh_fetch("squeue -u ethantu --name=hf-upload -h").decode().strip()
    return bool(out)


def _submit_hf_upload_sbatch() -> str:
    """Submit an HF upload sbatch job and return the job_id string."""
    # HF_XET_CACHE must point to local disk (not NFS) on the compute node to avoid
    # slow chunk writes over the network. /tmp is local on all cluster nodes.
    wrap_cmd = (
        f"HF_XET_HIGH_PERFORMANCE=1 HF_XET_CACHE=/tmp/hf_xet_cache"
        f" /home/ethantu/venvs/experiment16-migrate/bin/python {REMOTE_SCRIPT}"
        f" --new-dir {shlex.quote(NEW_DIR)}"
        f" --hf-repo {shlex.quote(HF_REPO)}"
        f" --upload-to-hf --remote-worker"
    )
    sbatch_cmd = (
        f"sbatch --job-name=hf-upload --mem=32G --time=6:00:00 --partition=normal.q"
        f" --output=/home/ethantu/tmp/hf_upload_sbatch_%j.log"
        f" --wrap={shlex.quote(wrap_cmd)}"
    )
    out = _ssh_fetch(sbatch_cmd).decode().strip()
    return out.split()[-1]  # "Submitted batch job 167295" → "167295"


def _maybe_submit_hf_upload(processed_ids: list[int], threshold: int = 5) -> None:
    """
    Submit an HF upload sbatch job if enough new sample_ids have accumulated.

    - Skips if an hf-upload job is already queued or running (prevents concurrent uploads).
    - Records submitted jobs in HF_UPLOAD_JOBS_FILE so reruns don't double-count.
    - upload_large_folder handles file-level dedup (skips already-committed files).
    """
    submitted = _submitted_sample_ids()
    new_ids = sorted(sid for sid in processed_ids if sid not in submitted)
    if len(new_ids) < threshold:
        return
    if _hf_upload_job_running():
        print(f"[hf-upload] skipping — job already running ({len(new_ids)} samples pending)", flush=True)
        return
    job_id = _submit_hf_upload_sbatch()
    rec = {
        "job_id": job_id,
        "submitted_at": datetime.datetime.utcnow().isoformat(),
        "new_sample_ids": new_ids,
    }
    with HF_UPLOAD_JOBS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[hf-upload] submitted sbatch job {job_id} ({len(new_ids)} new samples)", flush=True)


def _log_failure(source_dir_name: str, sample_id: int, exc: Exception) -> None:
    rec = {
        "source_dir_name": source_dir_name,
        "sample_id": sample_id,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    with FAILURES_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[FAILED] {source_dir_name} (sample_id={sample_id}): {exc}", flush=True)


# ── Per-sample pipeline ───────────────────────────────────────────────────────

def process_sample(source_dir_name: str, sample_id: int, segmenter: Segmenter,
                   pre_segment_future: Future | None = None) -> None:
    t0 = time.perf_counter()
    image_dir = _image_dir(source_dir_name)
    remote_image_root = f"{NEW_DIR}/image/{image_dir}"
    object_name = source_dir_name.split("-")[0].strip().lower()

    # 1. Wait for pre_segment to finish (may already be done if pipelined)
    if pre_segment_future is not None:
        t = time.perf_counter()
        pre_segment_future.result()  # raises if it failed
        print(f"[timing] pre_segment:   {time.perf_counter() - t:.2f}s (waited)", flush=True)
    else:
        t = time.perf_counter()
        _run_remote_stage("pre_segment", source_dir_name, sample_id)
        print(f"[timing] pre_segment:   {time.perf_counter() - t:.2f}s", flush=True)

    # 2. Fetch cropped overhead image for local Modal call
    t = time.perf_counter()
    cropped_bytes = _ssh_fetch(f"cat {shlex.quote(remote_image_root + '/cropped_overhead.png')}")
    cropped_arr = np.array(Image.open(io.BytesIO(cropped_bytes)).convert("RGB"), dtype=np.uint8)
    print(f"[timing] fetch cropped: {time.perf_counter() - t:.2f}s", flush=True)

    # 3. Segment via Modal (container already warm — no cold start)
    t = time.perf_counter()
    mask, _ = segmenter.segment.remote(cropped_arr, object_name, "cardboard", DEFAULT_PROMPT)
    print(f"[timing] modal segment: {time.perf_counter() - t:.2f}s", flush=True)

    # 4. Push mask back to mraid20
    t = time.perf_counter()
    mask_npz_buf = io.BytesIO()
    np.savez_compressed(mask_npz_buf, mask=mask)
    _ssh_fetch(f"cat > {shlex.quote(remote_image_root + '/mask.npz')}", stdin=mask_npz_buf.getvalue())
    mask_png_buf = io.BytesIO()
    Image.fromarray((np.clip(mask.astype(np.float32), 0, 1) * 255).astype(np.uint8)).save(mask_png_buf, format="PNG")
    _ssh_fetch(f"cat > {shlex.quote(remote_image_root + '/mask.png')}", stdin=mask_png_buf.getvalue())
    print(f"[timing] push mask:     {time.perf_counter() - t:.2f}s", flush=True)

    # 5. Post-segmentation: overlay, speaker overlay, manifest, metadata
    t = time.perf_counter()
    _run_remote_stage("post_segment", source_dir_name, sample_id)
    print(f"[timing] post_segment:  {time.perf_counter() - t:.2f}s", flush=True)

    print(f"[timing] sample total:  {time.perf_counter() - t0:.2f}s", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--samples", nargs="+", metavar="DIR_NAME:SAMPLE_ID",
        help="e.g. cube-00x01y_0001--31-03-18-21-24:1",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Discover and process all unprocessed experiment-15 dirs automatically",
    )
    p.add_argument(
        "--upload-to-hf", action="store_true",
        help="Run HF upload after all samples finish (only applies with --all)",
    )
    p.add_argument(
        "--upload-pending", action="store_true",
        help="Submit HF upload sbatch job for any processed samples not yet submitted, then exit",
    )
    return p.parse_args()


def _run_batch(samples: list[tuple[str, int]], on_failure=None, on_success=None) -> None:
    """Core batch loop: process samples with pipelined pre_segment and optional per-sample callbacks."""
    segmenter = Segmenter()

    with app.run(), ThreadPoolExecutor(max_workers=1) as executor:
        pre_future = executor.submit(_run_remote_stage, "pre_segment", samples[0][0], samples[0][1])

        for i, (source_dir_name, sample_id) in enumerate(samples):
            print(f"\n[batch] {source_dir_name} (sample_id={sample_id}) [{i+1}/{len(samples)}]", flush=True)

            next_future = None
            if i + 1 < len(samples):
                next_name, next_id = samples[i + 1]
                next_future = executor.submit(_run_remote_stage, "pre_segment", next_name, next_id)

            try:
                process_sample(source_dir_name, sample_id, segmenter, pre_segment_future=pre_future)
                if on_success is not None:
                    on_success(source_dir_name, sample_id)
            except Exception as exc:
                if on_failure is not None:
                    on_failure(source_dir_name, sample_id, exc)
                else:
                    raise

            pre_future = next_future


def main():
    args = parse_args()

    if args.upload_pending:
        _sync_static_files()
        _ensure_hf_shared_files()
        processed = _processed_sample_ids()
        print(f"[hf-upload] {len(processed)} samples processed on cluster", flush=True)
        _maybe_submit_hf_upload(processed, threshold=1)
        return

    if args.all:
        _sync_static_files()
        _ensure_hf_shared_files()

        processed = _processed_experiment_ids()
        all_dirs = _list_source_dirs()
        remaining = [d for d in all_dirs if d not in processed]
        print(f"[batch] {len(all_dirs)} total dirs, {len(processed)} already done, {len(remaining)} to process", flush=True)

        if not remaining:
            print("[batch] nothing to do", flush=True)
            return

        assignments = _load_or_create_assignments(remaining, start_id=len(processed) + 1)
        print(f"[batch] sample_ids {assignments[0][1]}–{assignments[-1][1]}", flush=True)

        # Pre-flight: find dirs missing experiment_config.json and skip them upfront
        print("[batch] checking for missing experiment_config.json ...", flush=True)
        missing_cfg = _missing_experiment_configs([d for d, _ in assignments])
        if missing_cfg:
            print(f"[batch] {len(missing_cfg)} dirs missing experiment_config.json — logging and skipping", flush=True)

        # Clear failures file at start of this run so counts reflect only the current attempt
        FAILURES_FILE.write_text("", encoding="utf-8")

        for source_dir_name, sample_id in assignments:
            if source_dir_name in missing_cfg:
                _log_failure(source_dir_name, sample_id,
                             FileNotFoundError(f"{OLD_DIR}/{source_dir_name}/experiment_config.json not found"))

        runnable = [(d, sid) for d, sid in assignments if d not in missing_cfg]
        print(f"[batch] {len(runnable)} samples to run", flush=True)

        # Upload any already-processed samples not yet submitted before starting
        existing_ids = _processed_sample_ids()
        if existing_ids:
            _maybe_submit_hf_upload(existing_ids, threshold=1)

        completed_ids: list[int] = []

        def _on_success(_dir: str, sid: int) -> None:
            completed_ids.append(sid)
            if len(completed_ids) % 5 == 0:
                _maybe_submit_hf_upload(completed_ids)

        _run_batch(runnable, on_failure=_log_failure, on_success=_on_success)

        # Upload any remainder not yet submitted (fewer than 5 at end of run)
        if completed_ids:
            _maybe_submit_hf_upload(completed_ids, threshold=1)

        failed = []
        if FAILURES_FILE.exists():
            for line in FAILURES_FILE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    failed.append(json.loads(line)["source_dir_name"])
        print(f"\n[batch] done. {len(remaining) - len(failed)}/{len(remaining)} succeeded, {len(failed)} failed.", flush=True)
        if failed:
            print(f"[batch] failures written to {FAILURES_FILE}", flush=True)

        if args.upload_to_hf:
            # Periodic sbatch uploads (every 5 samples) already cover this.
            # Run one final upload for any samples not yet submitted.
            all_processed = _processed_sample_ids()
            _maybe_submit_hf_upload(all_processed, threshold=1)

    elif args.samples:
        samples = []
        for s in args.samples:
            name, sid = s.rsplit(":", 1)
            samples.append((name, int(sid)))

        _sync_static_files()
        _run_batch(samples)

    else:
        import argparse
        raise argparse.ArgumentError(None, "one of --samples or --all is required")


if __name__ == "__main__":
    main()
