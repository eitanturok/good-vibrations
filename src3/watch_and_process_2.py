#!/usr/bin/env python3
"""Two-pool vibration watcher.

Decouples the bandwidth-bound UPLOAD from the GPU COMPUTE (Modal) and DOWNLOAD so
no thread ever blocks on remote processing:

  upload pool    : raw file ready -> upload -> spawn modal job (non-blocking)
  download poller: cheaply polls each running job; the instant one finishes,
                   hands the (small) download to the download pool

One `jobs` dict is the single source of truth (mirrored to jobs.jsonl for crash
recovery + idempotency). Separate --upload-workers / --download-workers knobs.
"""
import sys
import time
import argparse
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent))

import modal
from io_utils import Timing, modal_upload, modal_download, fix_symlinks, append, load
from vibrations_pipeline import app, volume, process_vibrations_modal, VIBRATION_FILES

MIN_READY_BYTES = 1 * 2**20  # reject files caught mid-write at a few KB/MB

def is_file_ready(path: Path, check_interval: float = 1.0) -> bool:
    """True once the raw file's size is stable (done writing) and above a sanity floor."""
    try:
        s1 = path.stat().st_size
        time.sleep(check_interval)
        return s1 == path.stat().st_size and s1 >= MIN_READY_BYTES
    except (OSError, PermissionError):
        return False  # locked by the writer; retry next tick

def poll_status(call_id: str) -> tuple[str, tuple[str, str] | None]:
    """get(timeout=0) IS the done-check: returns if finished, raises TimeoutError if
    still running, raises the remote error if the job crashed. On failure the returned
    error is (message, traceback) so the poller can persist why the job died."""
    try:
        modal.FunctionCall.from_id(call_id).get(timeout=0)
        return "done", None
    except modal.exception.TimeoutError:
        return "running", None
    except Exception as e:
        return "failed", (f"{type(e).__name__}: {e}", traceback.format_exc())

def main():
    p = argparse.ArgumentParser(description="Two-pool (upload / download) vibration watcher.")
    p.add_argument("--dir", required=True, help="Directory to watch for raw vibration files.")
    p.add_argument("--upload-workers", type=int, default=2, help="Parallel uploaders (bandwidth-bound; keep low).")
    p.add_argument("--download-workers", type=int, default=4, help="Parallel downloaders (results are small).")
    p.add_argument("--verbose", type=int, default=1, help="Verbosity level.")
    p.add_argument("--poll-rate", type=float, default=2.0, help="Seconds between scan / job-poll ticks.")
    args = p.parse_args()

    watch_path = Path(args.dir).resolve()
    ledger_path = watch_path / "jobs.jsonl"
    failed_path = watch_path / "failed_samples.jsonl"
    verbose = args.verbose

    jobs = {}                    # sample_id -> {"status", "call_id", "sample_dir"}  (single source of truth)
    lock = threading.Lock()      # guards `jobs` and the ledger / failed-log appends

    def set_status(sid, status, sample_dir, call_id=None):
        row = {"sample_id": sid, "status": status, "call_id": call_id, "sample_dir": str(sample_dir),
               "time": datetime.now(timezone.utc).isoformat()}
        with lock:
            jobs[sid] = {"status": status, "call_id": call_id, "sample_dir": str(sample_dir)}
            append(row, ledger_path)  # append-only; last row per sample wins on reload

    def record_failure(sid, sample_dir, phase, error, tb, call_id=None):
        """Append a failed modal instance to failed_samples.jsonl (sample id, error, traceback, timestamp)."""
        row = {"sample_id": sid, "phase": phase, "error": error, "traceback": tb, "call_id": call_id,
               "sample_dir": str(sample_dir), "time": datetime.now(timezone.utc).isoformat()}
        with lock:
            append(row, failed_path)

    print("=" * 80)
    print(f"👁️  TWO-POOL WATCHER | {watch_path} | up={args.upload_workers} down={args.download_workers}")
    print("=" * 80)

    # recover from ledger: keep done (skip) + running (reconnect); retry everything else from scratch
    if ledger_path.exists():
        latest = {r["sample_id"]: r for r in load(ledger_path)}
        jobs.update({sid: {k: r[k] for k in ("status", "call_id", "sample_dir")}
                     for sid, r in latest.items() if r["status"] in ("done", "running")})
        running = sum(j["status"] == "running" for j in jobs.values())
        if running:
            print(f"🔄 reconnected to {running} in-flight modal job(s) from ledger.")

    with app.run():
        upload_pool = ThreadPoolExecutor(max_workers=args.upload_workers, thread_name_prefix="up")
        download_pool = ThreadPoolExecutor(max_workers=args.download_workers, thread_name_prefix="down")

        def upload_worker(sample_dir):
            sid = sample_dir.name
            try:
                with Timing(f"⬆️  [sample {sid}] upload: ", enabled=verbose >= 1):
                    modal_upload(volume, sample_dir, verbose=verbose)
                fc = process_vibrations_modal.spawn(sid, pclk_batch_size=1024, pclk_mode="sequential", verbose=verbose)
                set_status(sid, "running", sample_dir, fc.object_id)
                print(f"🚀 [sample {sid}] spawned modal job {fc.object_id}")
            except Exception as e:
                print(f"❌ [sample {sid}] upload/spawn failed: {e}", file=sys.stderr)
                record_failure(sid, sample_dir, "upload", f"{type(e).__name__}: {e}", traceback.format_exc())
                set_status(sid, "failed", sample_dir)  # scan will retry it

        def download_worker(sample_dir, call_id):
            sid = sample_dir.name
            try:
                with Timing(f"⬇️  [sample {sid}] download: ", enabled=verbose >= 1):
                    for f in VIBRATION_FILES:
                        modal_download(volume, f"{sid}/inputs/{f}", sample_dir / f"inputs/{f}")
                    fix_symlinks(sample_dir)
                set_status(sid, "done", sample_dir, call_id)
                print(f"✅ [sample {sid}] complete.")
            except Exception as e:
                print(f"❌ [sample {sid}] download failed: {e}", file=sys.stderr)
                record_failure(sid, sample_dir, "download", f"{type(e).__name__}: {e}", traceback.format_exc(), call_id)
                set_status(sid, "failed", sample_dir, call_id)  # scan will retry it

        def poll_loop():
            while True:
                with lock:
                    running = [(sid, j["call_id"], Path(j["sample_dir"])) for sid, j in jobs.items() if j["status"] == "running"]
                for sid, call_id, sample_dir in running:
                    status, err = poll_status(call_id)
                    if status == "done":
                        set_status(sid, "downloading", sample_dir, call_id)  # leaves "running" so we stop polling it
                        download_pool.submit(download_worker, sample_dir, call_id)
                    elif status == "failed":
                        msg, tb = err
                        print(f"❌ [sample {sid}] modal job {call_id} FAILED: {msg}", file=sys.stderr)
                        record_failure(sid, sample_dir, "modal_job", msg, tb, call_id)
                        set_status(sid, "failed", sample_dir, call_id)
                time.sleep(args.poll_rate)

        threading.Thread(target=poll_loop, daemon=True).start()

        try:
            print("⚡ Modal app running. Watching for raw vibrations...")
            while True:
                for npy_path in watch_path.rglob("**/inputs/00_raw_vibrations.npy"):
                    sample_dir = npy_path.parents[1]
                    sid = sample_dir.name
                    with lock:
                        if sid in jobs and jobs[sid]["status"] != "failed":
                            continue  # in-flight or done
                    if (sample_dir / "inputs/05_processed_fft.npy").exists():
                        set_status(sid, "done", sample_dir)
                        continue
                    if not is_file_ready(npy_path):
                        continue
                    set_status(sid, "uploading", sample_dir)
                    print(f"📥 [sample {sid}] raw ready ({npy_path.stat().st_size / 1e9:.2f} GB). Queuing upload...")
                    upload_pool.submit(upload_worker, sample_dir)
                time.sleep(args.poll_rate)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down pools...")
            upload_pool.shutdown(wait=True)
            download_pool.shutdown(wait=True)
            print("👋 Stopped. In-flight modal jobs are tracked in the ledger.")

if __name__ == "__main__":
    main()
