"""python -m viz -- segmentation mask comparison across training runs."""

import argparse
from pathlib import Path

import uvicorn

from viz import app as app_module
from viz import config


def main():
    ap = argparse.ArgumentParser(prog="viz", description=__doc__)
    ap.add_argument("--experiment", type=Path, default=config.EXPERIMENT_DIR,
                    help="a single experiment dir (has samples/ directly), or a parent "
                         "dir holding several -- every child dir with samples/ loads and "
                         "is shown at once, filterable by its 'box' metadata. Defaults to "
                         "experiments/, the parent of every experiment on disk.")
    ap.add_argument("--runs", type=Path, default=config.RUNS_DIR)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mask", metavar="HxW",
                    help="target mask grid, e.g. 20x40 or 30x30, applied to every loaded "
                         "experiment. Defaults to each experiment's own best size (most "
                         "usable, most-trained-at). An experiment that doesn't ship this "
                         "size is skipped -- or, with a single experiment dir, a hard "
                         "error, exactly as before.")
    args = ap.parse_args()

    mask_override = None
    if args.mask:
        try:
            h, w = (int(v) for v in args.mask.lower().split("x", 1))
        except ValueError:
            ap.error(f"--mask must look like HxW, got {args.mask!r}")
        mask_override = (h, w)

    app_module.init(args.experiment, args.runs, mask_override)
    print(f"[viz] http://{args.host}:{args.port}")
    uvicorn.run(app_module.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
