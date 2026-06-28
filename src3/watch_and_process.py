#!/usr/bin/env python3
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys

# Ensure local workspace modules can resolve imports seamlessly
sys.path.insert(0, str(Path(__file__).parent))

from io_utils import Timing
from vibrations_pipeline import process_vibrations, app

def is_file_ready(path: Path, sample_id: str, check_interval: float = 1.0) -> bool:
    """Ensures the 3GB file has finished writing completely to disk before processing."""
    if not path.exists():
        return False
    try:
        size_1 = path.stat().st_size
        time.sleep(check_interval)
        size_2 = path.stat().st_size
        # If size hasn't changed and is larger than 0, writing is finished
        return size_1 == size_2 and size_1 > 0
    except (OSError, PermissionError):
        # File might be locked by the main process writing it; retry on next tick
        return False

def processing_worker(sample_dir: Path, use_modal: bool, verbose: int):
    """Worker task executed inside the ThreadPoolExecutor queue."""
    sample_id = sample_dir.name
    print(f"\n🚀 [watcher] [sample {sample_id}] Thread assigned. Starting pipeline...")
    
    try:
        # Match your exact timing log standard
        with Timing(f"[watcher] [sample {sample_id}] full remote pipeline total: ", enabled=verbose >= 1):
            process_vibrations(sample_dir, use_modal=use_modal, do_save=True, verbose=verbose)
        print(f"✅ [watcher] [sample {sample_id}] Completed successfully.")
    except Exception as e:
        print(f"❌ [watcher] [sample {sample_id}] CRITICAL ERROR during processing: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Continuous background watcher for experimental vibration processing.")
    parser.add_argument("--dir", type=str, required=True, help="Base experiment or sample storage directory to watch.")
    parser.add_argument("--workers", type=int, default=2, help="Maximum number of parallel processing workers (default: 2).")
    parser.add_argument("--no-modal", action="store_true", help="Run process_vibrations locally instead of deploying on Modal.")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level for processing output logs.")
    parser.add_argument("--poll-rate", type=float, default=2.0, help="Seconds to wait between directory scan ticks.")
    args = parser.parse_args()

    watch_path = Path(args.dir).resolve()
    use_modal = not args.no_modal

    print("=" * 80)
    print(f"👁️  VIBRATION PIPELINE DAEMON ACTIVATED")
    print(f"📂 Watching Directory: {watch_path}")
    print(f"⚙️  Max Workers:       {args.workers}")
    print(f"☁️  Modal Engine:      {'ENABLED' if use_modal else 'DISABLED'}")
    print("=" * 80)

    if not watch_path.exists():
        print(f"⚠️  [watcher] Warning: Target path '{watch_path}' does not exist yet. Waiting for creation...")

    # Set up the persistent thread pool queue
    executor = ThreadPoolExecutor(max_workers=args.workers)
    
    # Active tracking set to prevent double-submitting tasks during a runtime session
    active_jobs = set()

    try:
        # Wrap the daemon tracking execution loop inside the authenticated app run state
        print("⚡ [watcher] Authenticating and initializing Modal context app session...")
        with app.run():
            print("🚀 [watcher] Modal application running. Beginning directory watch polling loop.")
            while True:
                if watch_path.exists():
                    # Recursively search for any raw vibration array matches in the tree
                    for npy_path in watch_path.rglob("**/inputs/00_raw_vibrations.npy"):
                        sample_dir = npy_path.parents[1] # Path to sample directory
                        sample_id = sample_dir.name
                        
                        # 1. Skip if already processed or currently active in the thread pool
                        final_output_file = sample_dir / "inputs/05_processed_fft.npy"
                        if final_output_file.exists() or sample_id in active_jobs:
                            continue
                        
                        # 2. Safety check: Ensure the file isn't currently being written by the experiment
                        if not is_file_ready(npy_path, sample_id):
                            if args.verbose >= 2:
                                print(f"⏳ [watcher] [sample {sample_id}] File detected but size is changing. Waiting...")
                            continue

                        # 3. Submit to queue
                        rel_path = npy_path.relative_to(watch_path)
                        print(f"📥 [watcher] [sample {sample_id}] New raw data ready: {rel_path} ({npy_path.stat().st_size / 1e9:.2f} GB). Adding to queue...")
                        active_jobs.add(sample_id)
                        executor.submit(processing_worker, sample_dir, use_modal, args.verbose)

                time.sleep(args.poll_rate)

    except KeyboardInterrupt:
        print("\n🛑 [watcher] Shutdown signal received. Cleaning up thread pool queue...")
        executor.shutdown(wait=True)
        print("👋 [watcher] Daemon stopped cleanly.")

if __name__ == "__main__":
    main()