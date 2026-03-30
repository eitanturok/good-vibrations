#!/usr/bin/env python3
"""
Continuously monitors a source directory and moves its contents to a destination,
then deletes the source directory. Prints timing for each item moved.

Usage:
    python move_data.py <source> <destination>
"""

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path


POLL_INTERVAL_SECONDS = 5
IDLE_TIMEOUT_SECONDS = 3600  # stop if no new files appear for this long (1 hour)


def file_checksum(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def verify_move(src: Path, dst: Path) -> bool:
    """Confirm that dst exists, has the same size, and the same MD5 checksum as src."""
    if not dst.exists():
        print(f"  [VERIFY FAILED] Destination does not exist: {dst}")
        return False
    if src.stat().st_size != dst.stat().st_size:
        print(f"  [VERIFY FAILED] Size mismatch for {dst.name}: "
              f"src={src.stat().st_size} dst={dst.stat().st_size}")
        return False
    if file_checksum(src) != file_checksum(dst):
        print(f"  [VERIFY FAILED] Checksum mismatch for {dst.name}")
        return False
    return True


def move_item(src: Path, dst_root: Path) -> bool:
    """
    Copy src (file or directory tree) into dst_root, verify, then delete src.
    Returns True if the item was successfully moved and verified.
    """
    dst = dst_root / src.name

    t_start = time.perf_counter()

    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        elapsed = time.perf_counter() - t_start
        print(f"  Copied dir  '{src.name}' in {elapsed:.3f}s — verifying files...")

        # Verify every file inside
        files = list(src.rglob("*"))
        failed = False
        for f in files:
            if f.is_file():
                rel = f.relative_to(src)
                dst_file = dst / rel
                if not verify_move(f, dst_file):
                    failed = True
        if failed:
            return False

        print(f"  [OK] Dir  '{src.name}' verified ({len([f for f in files if f.is_file()])} files)")
        shutil.rmtree(src)

    else:
        shutil.copy2(src, dst)
        elapsed = time.perf_counter() - t_start
        print(f"  Copied file '{src.name}' in {elapsed:.3f}s — verifying...")

        if not verify_move(src, dst):
            return False

        print(f"  [OK] File '{src.name}' verified")
        src.unlink()

    total_elapsed = time.perf_counter() - t_start
    print(f"  Total time for '{src.name}': {total_elapsed:.3f}s")
    return True


def process_source(source: Path, destination: Path) -> None:
    """Move all top-level items from source into destination, then remove source dir."""
    items = sorted(source.iterdir())
    if not items:
        print(f"[INFO] Source directory is empty: {source}")
        return

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found {len(items)} item(s) in '{source}'")

    all_ok = True
    for item in items:
        print(f"\n  -> Moving: {item.name}")
        ok = move_item(item, destination)
        if not ok:
            print(f"  [ERROR] Failed to move '{item.name}'. Skipping deletion of this item.")
            all_ok = False

    if all_ok:
        # Source dir should now be empty; remove it
        try:
            source.rmdir()
            print(f"\n[DONE] Source directory removed: {source}")
        except OSError as e:
            print(f"\n[WARN] Could not remove source directory '{source}': {e}")
    else:
        print(f"\n[WARN] Some items failed to move. Source directory NOT deleted: {source}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously move data from source to destination, then delete source."
    )
    parser.add_argument("source", type=Path, help="Source directory path")
    parser.add_argument("destination", type=Path, help="Destination directory path")
    parser.add_argument(
        "--idle-timeout", type=float, default=IDLE_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=f"Stop if no new files appear for this many seconds (default: {IDLE_TIMEOUT_SECONDS})",
    )
    args = parser.parse_args()

    source: Path = args.source.resolve()
    destination: Path = args.destination.resolve()

    idle_timeout: float = args.idle_timeout

    if not destination.exists():
        destination.mkdir(parents=True)
        print(f"[INFO] Created destination directory: {destination}")

    print(f"[INFO] Watching source:      {source}")
    print(f"[INFO] Moving to dest:       {destination}")
    print(f"[INFO] Poll interval:        {POLL_INTERVAL_SECONDS}s")
    print(f"[INFO] Idle timeout:         {idle_timeout}s")
    print("[INFO] Press Ctrl+C to stop.\n")

    last_activity = time.monotonic()

    try:
        while True:
            has_items = source.exists() and source.is_dir() and any(source.iterdir())
            if has_items:
                last_activity = time.monotonic()
                process_source(source, destination)
            else:
                idle_secs = time.monotonic() - last_activity
                if idle_secs >= idle_timeout:
                    print(f"[INFO] No new files for {idle_secs:.0f}s (timeout={idle_timeout}s). Exiting.")
                    break
                print(f"[{time.strftime('%H:%M:%S')}] Waiting... idle for {idle_secs:.0f}s / {idle_timeout}s")

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
