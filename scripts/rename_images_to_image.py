"""Rename each sample's `images/` directory to `image/` so the gastronorm
experiment matches the breadbox (experiment-25) layout.

Usage:
    python scripts/rename_images_to_image.py --dry-run
    python scripts/rename_images_to_image.py
"""

import argparse
from pathlib import Path

GASTRONORM = Path(
    "/home/ethantu/workspace/good-vibrations/experiments/31_07_2026_gastronorm_exp1/samples"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=GASTRONORM)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    renamed = skipped = conflicts = 0

    for sample in sorted(p for p in args.samples.iterdir() if p.is_dir()):
        src, dst = sample / "images", sample / "image"

        if not src.is_dir():
            skipped += 1
            continue
        if dst.exists():
            print(f"CONFLICT: {dst} already exists, leaving {src} alone")
            conflicts += 1
            continue

        print(f"{src} -> {dst}")
        if not args.dry_run:
            src.rename(dst)
        renamed += 1

    verb = "would rename" if args.dry_run else "renamed"
    print(f"\n{verb} {renamed} | skipped (no images/) {skipped} | conflicts {conflicts}")


if __name__ == "__main__":
    main()
