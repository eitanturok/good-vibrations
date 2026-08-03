"""python -m viz2 -- segmentation mask comparison across training runs."""

import argparse
from pathlib import Path

import uvicorn

from viz2 import app as app_module
from viz2 import config


def main():
    ap = argparse.ArgumentParser(prog="viz2", description=__doc__)
    ap.add_argument("--experiment", type=Path, default=config.EXPERIMENT_DIR)
    ap.add_argument("--runs", type=Path, default=config.RUNS_DIR)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--mask", metavar="HxW",
                    help="target mask grid, e.g. 20x40 or 30x30. Defaults to the only "
                         "size the dataset ships, or 20x40 when it ships several. Runs "
                         "trained on a different size are listed as incompatible.")
    args = ap.parse_args()

    # Must happen before anything builds a Layout or reads MASK_H/MASK_W.
    available = config.mask_shapes(args.experiment / "samples")
    if args.mask:
        try:
            h, w = (int(v) for v in args.mask.lower().split("x", 1))
        except ValueError:
            ap.error(f"--mask must look like HxW, got {args.mask!r}")
        if available and (h, w) not in available:
            sizes = ", ".join(f"{a}x{b}" for a, b in available)
            ap.error(f"{args.mask} not in {args.experiment}; available: {sizes}")
        config.set_mask_shape(h, w)
    elif len(available) == 1:
        # Only one size on disk: use it, so a dataset that ships just 30x30 needs no flag.
        config.set_mask_shape(*available[0])
    elif len(available) > 1:
        sizes = ", ".join(f"{a}x{b}" for a, b in available)
        print(f"[viz2] {len(available)} mask sizes available ({sizes}); using "
              f"{config.MASK_H}x{config.MASK_W}. Pass --mask to pick another.")

    app_module.init(args.experiment, args.runs)
    print(f"[viz2] http://{args.host}:{args.port}")
    uvicorn.run(app_module.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
