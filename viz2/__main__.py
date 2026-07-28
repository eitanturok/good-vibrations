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
    args = ap.parse_args()

    app_module.init(args.experiment, args.runs)
    print(f"[viz2] http://{args.host}:{args.port}")
    uvicorn.run(app_module.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
