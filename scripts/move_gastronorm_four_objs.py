"""Merge the two halves of 2026_09_27_gastronorm_four_objs onto E: (D ran out of room mid-experiment).

    D:/eturok/2026_09_27_gastronorm_four_objs  ->  E:/eturok/2026_09_27_gastronorm_four_objs_part_1  (copy, verify, delete D)
    E:/eturok/2026_09_27_gastronorm_four_objs  ->  E:/eturok/2026_09_27_gastronorm_four_objs_part_2  (rename, same drive)

Sample/position ids overlap between the parts -- untouched here, fixed later.
Pause post-processing first (see the notebook cell in the PR/chat); the script refuses to run
while any file in either dir was modified in the last --quiet-minutes.

Every step is skipped if already done, so a crashed run can simply be rerun. D is copied into a
`.partial` dir first, so a half-finished copy is never mistaken for part_1.

Dry run by default:
    python scripts/move_gastronorm_four_objs.py
    python scripts/move_gastronorm_four_objs.py --execute
"""
import os
import time
import shutil
import argparse
from pathlib import Path

NAME = "2026_09_27_gastronorm_four_objs"
SRC_D, SRC_E = Path("D:/eturok") / NAME, Path("E:/eturok") / NAME
PART_1, PART_2 = Path("E:/eturok") / f"{NAME}_part_1", Path("E:/eturok") / f"{NAME}_part_2"
PARTIAL_1 = PART_1.with_name(PART_1.name + ".partial")


def files(root: Path) -> dict[Path, os.stat_result]:
    """relative path -> stat, for every file under root"""
    return {p.relative_to(root): p.stat() for p in root.rglob("*") if p.is_file()}


def summary(stats: dict) -> str:
    raws = sum(p.name == "01_raw_vibrations.npy" for p in stats)
    return f"{len(stats)} files, {sum(s.st_size for s in stats.values()) / 1e9:.1f} GB, {raws} unprocessed raws"


def check_quiet(root: Path, stats: dict, quiet_minutes: float, dry: bool):
    if not stats:
        return
    newest = max(stats, key=lambda p: stats[p].st_mtime)
    age = (time.time() - stats[newest].st_mtime) / 60
    if age < quiet_minutes:
        msg = (f"{root / newest} was modified {age:.1f} min ago -- pause post-processing / recording "
               f"and wait until nothing has changed for {quiet_minutes:g} min")
        if not dry:
            raise SystemExit(msg)
        print(f"    WARNING (--execute would refuse): {msg}")


def verify(src: dict, dst_root: Path):
    """every source file exists in dst with the same size, and dst has nothing extra"""
    dst = files(dst_root)
    missing = [p for p in src if p not in dst]
    wrong_size = [p for p in src if p in dst and dst[p].st_size != src[p].st_size]
    extra = [p for p in dst if p not in src]
    for label, ps in (("missing", missing), ("wrong size", wrong_size), ("extra", extra)):
        if ps:
            raise SystemExit(f"verify FAILED: {len(ps)} {label} in {dst_root}, e.g. {ps[:5]} -- D left untouched")
    print(f"  verified {dst_root}: {summary(dst)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="actually move (default: dry run)")
    ap.add_argument("--quiet-minutes", type=float, default=2.0, help="refuse if anything changed more recently than this")
    args = ap.parse_args()
    dry = not args.execute
    print("DRY RUN -- nothing will be changed\n" if dry else "EXECUTING\n")

    # 1. rename E -> part_2 (same drive: instant; fails loudly if a file inside is still open)
    if PART_2.exists():
        print(f"[1] {PART_2} already exists -- rename already done")
        if SRC_E.exists():
            raise SystemExit(f"both {SRC_E} and {PART_2} exist -- something recreated the old dir, sort it out by hand")
    else:
        e = files(SRC_E)
        print(f"[1] rename {SRC_E}  ->  {PART_2.name}\n    {summary(e)}")
        check_quiet(SRC_E, e, args.quiet_minutes, dry)
        if not dry:
            try:
                SRC_E.rename(PART_2)
            except PermissionError as err:
                raise SystemExit(f"rename failed, a file in {SRC_E} is still open (post-processing not paused?): {err}")

    # 2. copy D -> part_1.partial, verify, then promote to part_1
    if PART_1.exists():
        print(f"[2] {PART_1} already exists -- copy already done")
    else:
        d = files(SRC_D)
        free = shutil.disk_usage(PART_1.anchor).free
        need = sum(s.st_size for s in d.values())
        print(f"[2] copy {SRC_D}  ->  {PARTIAL_1}, verify, rename to {PART_1.name}\n"
              f"    {summary(d)}; E: has {free / 1e9:.0f} GB free")
        check_quiet(SRC_D, d, args.quiet_minutes, dry)
        if need > free:
            raise SystemExit(f"not enough room on E: ({need / 1e9:.1f} GB needed)")
        if not dry:
            t0 = time.time()
            shutil.copytree(SRC_D, PARTIAL_1, dirs_exist_ok=True)  # rerun after a crash: overwrites the partial copy
            print(f"  copied in {(time.time() - t0) / 60:.1f} min")
            verify(d, PARTIAL_1)
            PARTIAL_1.rename(PART_1)

    # 3. delete D, only after re-verifying part_1 against it (D must not have changed since the copy)
    if not SRC_D.exists():
        print(f"[3] {SRC_D} already gone -- delete already done")
    else:
        print(f"[3] verify {PART_1.name} against {SRC_D} again, then delete {SRC_D}")
        if not dry:
            verify(files(SRC_D), PART_1)
            shutil.rmtree(SRC_D)
            print(f"  deleted {SRC_D}")

    print("\ndry run done -- rerun with --execute" if dry else "\ndone")


if __name__ == "__main__":
    main()
