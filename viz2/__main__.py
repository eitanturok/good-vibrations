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
    else:
        # Pick the default off the data instead of a hardcoded size. `usable` drops sizes
        # whose masks carry no real mass -- the gastronorm 20x40 files are non-zero but
        # ~300x too sparse, and defaulting to those scores every prediction against a
        # near-empty target while looking like a working viz2.
        usable = config.usable_mask_shapes(args.experiment / "samples")
        # Among the usable sizes, prefer the one MOST RUNS were trained at. Resolution is
        # the wrong tiebreak: experiment-25 ships both 20x40 and 30x30 targets but nearly
        # all its runs are 20x40, so defaulting to the finer grid left 1 of 68 runs
        # comparable. Other sizes still appear in the table, this only sets the default.
        if usable:
            from viz2 import data
            best = data.most_trained_shape(args.runs, usable) or usable[0]
            config.set_mask_shape(*best)
        if len(available) > 1:
            sizes = ", ".join(f"{a}x{b}" for a, b in available)
            unusable = [f"{a}x{b}" for a, b in available if (a, b) not in usable]
            note = f"; no usable masks at {', '.join(unusable)}" if unusable else ""
            print(f"[viz2] mask sizes available: {sizes}{note}. Using "
                  f"{config.MASK_H}x{config.MASK_W} as the default grid; runs trained at "
                  f"another size are shown alongside it.")

    app_module.init(args.experiment, args.runs)
    print(f"[viz2] http://{args.host}:{args.port}")
    uvicorn.run(app_module.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
