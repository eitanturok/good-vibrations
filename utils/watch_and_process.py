#!/usr/bin/env python3
"""Watch a directory for raw vibration captures and process them with pclk, either
locally or on Modal.

Local: pclk saturates the GPU on its own (batched_optimized batches all ROIs into
one call), so a single background worker processes samples one at a time — extra
threads would just contend for VRAM, not add throughput.

Modal: upload (bandwidth-bound) and remote compute are decoupled into two pools so
neither blocks the other — upload workers spawn jobs and move on; a poller hands
each finished job to a download worker as soon as it's ready.
"""
import sys
import time
import queue
import argparse
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))           # utils.*
sys.path.insert(0, str(REPO / "src"))   # data.*

import matplotlib
matplotlib.use("Agg")  # plots render in worker threads; Tk backends require the main thread

import modal

from utils.io_utils import modal_upload, modal_download, fix_symlinks, append, load
from utils.helpers import Timing
from data.vibrate import app, volume, process_vibrations, _process_vibrations_modal, PROCESSED_FILES

MIN_READY_BYTES = 1 * 2**20  # reject files caught mid-write at a few KB/MB

def is_file_ready(path: Path, check_interval: float = 1.0) -> bool:
    """True once the raw file's size is stable (done writing) and above a sanity floor."""
    try:
        s1 = path.stat().st_size
        time.sleep(check_interval)
        return s1 == path.stat().st_size and s1 >= MIN_READY_BYTES
    except (OSError, PermissionError):
        return False  # locked by the writer; retry next tick

class LocalEngine:
    """Runs pclk on this machine's GPU. One worker: batched_optimized already
    saturates the GPU per-sample, so concurrent samples would only add VRAM
    contention, not speed."""
    def __init__(self, pclk_mode: str, pclk_batch_size: int, verbose: int, cleanup_raw_vibrations: str, use_pc: bool = True):
        self.pclk_mode, self.pclk_batch_size, self.verbose, self.cleanup_raw_vibrations, self.use_pc = pclk_mode, pclk_batch_size, verbose, cleanup_raw_vibrations, use_pc
        self.q: queue.Queue[Path] = queue.Queue()
        self.done = set()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, sample_dir: Path):
        if sample_dir.name in self.done: return
        self.done.add(sample_dir.name)
        self.q.put(sample_dir)

    def _worker(self):
        while True:
            sample_dir = self.q.get()
            sid = sample_dir.name
            try:
                process_vibrations(sample_dir, use_modal=False, pclk_mode=self.pclk_mode,
                                    pclk_batch_size=self.pclk_batch_size, do_save=True, verbose=self.verbose,
                                    cleanup_raw_vibrations=self.cleanup_raw_vibrations, use_PC=self.use_pc)
                print(f"✅ [sample {sid}] complete.")
            except Exception as e:
                print(f"❌ [sample {sid}] local processing failed: {e}", file=sys.stderr)
                traceback.print_exc()

class ModalEngine:
    """Two-pool upload/spawn/poll/download so the bandwidth-bound upload never
    blocks on remote GPU compute, and results download the moment they're ready."""
    def __init__(self, pclk_mode: str, pclk_batch_size: int, verbose: int, cleanup_raw_vibrations: str, watch_path: Path,
                 upload_workers: int, download_workers: int, poll_rate: float, use_pc: bool = True):
        self.pclk_mode, self.pclk_batch_size, self.verbose, self.cleanup_raw_vibrations, self.use_pc = pclk_mode, pclk_batch_size, verbose, cleanup_raw_vibrations, use_pc
        self.poll_rate = poll_rate
        self.ledger_path = watch_path / "jobs.jsonl"
        self.failed_path = watch_path / "failed_samples.jsonl"
        self.jobs = {}       # sample_id -> {"status", "call_id", "sample_dir"}
        self.lock = threading.Lock()
        if self.ledger_path.exists():
            latest = {r["sample_id"]: r for r in load(self.ledger_path)}
            self.jobs.update({sid: {k: r[k] for k in ("status", "call_id", "sample_dir")}
                               for sid, r in latest.items() if r["status"] in ("done", "running")})
            running = sum(j["status"] == "running" for j in self.jobs.values())
            if running: print(f"🔄 reconnected to {running} in-flight modal job(s) from ledger.")

        self.upload_pool = ThreadPoolExecutor(max_workers=upload_workers, thread_name_prefix="up")
        self.download_pool = ThreadPoolExecutor(max_workers=download_workers, thread_name_prefix="down")
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def _set_status(self, sid, status, sample_dir, call_id=None):
        row = {"sample_id": sid, "status": status, "call_id": call_id, "sample_dir": str(sample_dir),
               "time": datetime.now(timezone.utc).isoformat()}
        with self.lock:
            self.jobs[sid] = {"status": status, "call_id": call_id, "sample_dir": str(sample_dir)}
            append(row, self.ledger_path)

    def _record_failure(self, sid, sample_dir, phase, error, tb, call_id=None):
        row = {"sample_id": sid, "phase": phase, "error": error, "traceback": tb, "call_id": call_id,
               "sample_dir": str(sample_dir), "time": datetime.now(timezone.utc).isoformat()}
        with self.lock:
            append(row, self.failed_path)

    def submit(self, sample_dir: Path):
        sid = sample_dir.name
        with self.lock:
            if sid in self.jobs and self.jobs[sid]["status"] != "failed": return
        self._set_status(sid, "uploading", sample_dir)
        self.upload_pool.submit(self._upload_worker, sample_dir)

    def _upload_worker(self, sample_dir: Path):
        sid = sample_dir.name
        try:
            with Timing(f"⬆️  [sample {sid}] upload: ", enabled=self.verbose >= 1):
                modal_upload(volume, sample_dir, verbose=self.verbose)
            fc = _process_vibrations_modal.spawn(sid, pclk_batch_size=self.pclk_batch_size,
                                                  pclk_mode=self.pclk_mode, verbose=self.verbose, cleanup_raw_vibrations=self.cleanup_raw_vibrations,
                                                  use_PC=self.use_pc)
            self._set_status(sid, "running", sample_dir, fc.object_id)
            print(f"🚀 [sample {sid}] spawned modal job {fc.object_id}")
        except Exception as e:
            print(f"❌ [sample {sid}] upload/spawn failed: {e}", file=sys.stderr)
            self._record_failure(sid, sample_dir, "upload", f"{type(e).__name__}: {e}", traceback.format_exc())
            self._set_status(sid, "failed", sample_dir)  # scan will retry it

    def _download_worker(self, sample_dir: Path, call_id: str):
        sid = sample_dir.name
        try:
            files = PROCESSED_FILES + (["00_raw_vibrations.npy.bz2"] if self.cleanup_raw_vibrations == 'compress' else [])
            with Timing(f"⬇️  [sample {sid}] download: ", enabled=self.verbose >= 1):
                for f in files:
                    modal_download(volume, f"{sid}/vibration/{f}", sample_dir / f"vibration/{f}")
                fix_symlinks(sample_dir)
            self._set_status(sid, "done", sample_dir, call_id)
            print(f"✅ [sample {sid}] complete.")
        except Exception as e:
            print(f"❌ [sample {sid}] download failed: {e}", file=sys.stderr)
            self._record_failure(sid, sample_dir, "download", f"{type(e).__name__}: {e}", traceback.format_exc(), call_id)
            self._set_status(sid, "failed", sample_dir, call_id)  # scan will retry it

    def _poll_status(self, call_id: str) -> tuple[str, tuple[str, str] | None]:
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
            return "done", None
        except modal.exception.TimeoutError:
            return "running", None
        except Exception as e:
            return "failed", (f"{type(e).__name__}: {e}", traceback.format_exc())

    def _poll_loop(self):
        while True:
            with self.lock:
                running = [(sid, j["call_id"], Path(j["sample_dir"])) for sid, j in self.jobs.items() if j["status"] == "running"]
            for sid, call_id, sample_dir in running:
                status, err = self._poll_status(call_id)
                if status == "done":
                    self._set_status(sid, "downloading", sample_dir, call_id)  # leaves "running" so we stop polling it
                    self.download_pool.submit(self._download_worker, sample_dir, call_id)
                elif status == "failed":
                    msg, tb = err
                    print(f"❌ [sample {sid}] modal job {call_id} FAILED: {msg}", file=sys.stderr)
                    self._record_failure(sid, sample_dir, "modal_job", msg, tb, call_id)
                    self._set_status(sid, "failed", sample_dir, call_id)
            time.sleep(self.poll_rate)

def main():
    p = argparse.ArgumentParser(description="Watch a directory and process raw vibrations with pclk, locally or on Modal.")
    p.add_argument("--dir", required=True, help="Directory to watch for raw vibration files.")
    p.add_argument("--modal", action="store_true", help="Process on Modal instead of locally (default: local).")
    p.add_argument("--pclk-mode", default="batched_optimized", choices=["sequential", "batched", "batched_optimized"])
    p.add_argument("--pclk-batch-size", type=int, default=256)
    p.add_argument("--upload-workers", type=int, default=2, help="[modal only] parallel uploaders (bandwidth-bound; keep low).")
    p.add_argument("--download-workers", type=int, default=4, help="[modal only] parallel downloaders (results are small).")
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--poll-rate", type=float, default=2.0, help="Seconds between scan / job-poll ticks.")
    p.add_argument("--cleanup-raw-vibrations", default="delete", choices=["compress", "delete"],
                    help="What to do with the raw vibrations file once pclk is done with it.")
    p.add_argument("--use-pc", type=int, default=1, choices=[0, 1],
                    help="1 (default) runs the phase-correlation pre-alignment step before LK; 0 skips it and runs LK directly on the raw frames.")
    args = p.parse_args()
    args.use_pc = bool(args.use_pc)

    watch_path = Path(args.dir).resolve()
    print("=" * 80)
    print(f"👁️  WATCHER | {watch_path} | engine={'modal' if args.modal else 'local'} | pclk={args.pclk_mode} (batch={args.pclk_batch_size}) | use_pc={args.use_pc} | cleanup={args.cleanup_raw_vibrations}")
    print("=" * 80)

    def watch_loop(engine):
        seen_done = set()
        while True:
            for npy_path in watch_path.rglob("**/vibration/00_raw_vibrations.npy"):
                sample_dir = npy_path.parents[1]
                sid = sample_dir.name
                if sid in seen_done: continue
                if not is_file_ready(npy_path): continue
                pclk_done = (sample_dir / "vibration/01_raw_shifts.npy").exists()
                status = f"pclk already done, {args.cleanup_raw_vibrations} only" if pclk_done else f"pclk + {args.cleanup_raw_vibrations}"
                print(f"📥 [sample {sid}] raw ready ({npy_path.stat().st_size / 1e9:.2f} GB) -- {status}. Queuing...")
                engine.submit(sample_dir)
                # mark as seen once queued, not on some "done" file -- the raw .npy only
                # disappears once _process_vibrations compresses/deletes it, and relying on
                # e.g. fft.npz existing missed samples processed before this step existed
                seen_done.add(sid)
            time.sleep(args.poll_rate)

    try:
        if args.modal:
            with app.run():
                engine = ModalEngine(args.pclk_mode, args.pclk_batch_size, args.verbose, args.cleanup_raw_vibrations,
                                      watch_path, args.upload_workers, args.download_workers, args.poll_rate, use_pc=args.use_pc)
                watch_loop(engine)
        else:
            engine = LocalEngine(args.pclk_mode, args.pclk_batch_size, args.verbose, args.cleanup_raw_vibrations, use_pc=args.use_pc)
            watch_loop(engine)
    except KeyboardInterrupt:
        print("\n🛑 Stopped. In-flight modal jobs (if any) are tracked in the ledger.")

if __name__ == "__main__":
    main()
