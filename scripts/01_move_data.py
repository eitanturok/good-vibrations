#!/usr/bin/env python3
"""
Continuously monitors local-dir and moves its contents to shared-dir.
Prints timing for each item moved. Stops after --idle-timeout seconds with no new files.

Usage:
    python 01_move_data.py <local-dir> <shared-dir>
"""

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
from watch import watch


POLL_INTERVAL_SECONDS = 5
IDLE_TIMEOUT_SECONDS = 3600


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

        files = list(src.rglob("*"))
        failed = any(
            not verify_move(f, dst / f.relative_to(src))
            for f in files if f.is_file()
        )
        if failed:
            return False

        print(f"  [OK] Dir  '{src.name}' verified ({sum(1 for f in files if f.is_file())} files)")
        shutil.rmtree(src)

    else:
        shutil.copy2(src, dst)
        elapsed = time.perf_counter() - t_start
        print(f"  Copied file '{src.name}' in {elapsed:.3f}s — verifying...")

        if not verify_move(src, dst):
            return False

        print(f"  [OK] File '{src.name}' verified")
        src.unlink()

    print(f"  Total time for '{src.name}': {time.perf_counter() - t_start:.3f}s")
    return True


def build_process(source, destination, idle_timeout=IDLE_TIMEOUT_SECONDS):
    @watch(source, idle=idle_timeout, poll=POLL_INTERVAL_SECONDS)
    def process(item):
        print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Moving: {item.name}")
        move_item(item, destination)
    return process


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously move data from source to destination."
    )
    parser.add_argument("local_dir", type=Path, help="Local dir with raw experiment results")
    parser.add_argument("shared_dir",     type=Path, help="Mounted shared dir accessible by both local machine and cluster")
    parser.add_argument("--idle-timeout", type=float, default=IDLE_TIMEOUT_SECONDS, metavar="SECONDS")
    args = parser.parse_args()

    local_dir: Path = args.local_dir.resolve()
    shared_dir:     Path = args.shared_dir.resolve()
    shared_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Watching: {local_dir}  ->  {shared_dir}")

    process = build_process(local_dir, shared_dir, args.idle_timeout)
    process()


if __name__ == "__main__":
    main()
