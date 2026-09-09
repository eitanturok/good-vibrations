"""python -m viz2 <experiment-dir | dir-of-experiments>

Point it at one experiment (a dir with samples/) or a parent dir holding several; in the
second case every experiment shows up in the step-1 box picker and switches in place.
"""
import argparse, socket
from pathlib import Path
import uvicorn


def free_port(host, port):
    """The dashboard should start, not refuse over a busy port."""
    for p in range(port, port + 20):
        with socket.socket() as s:
            if s.connect_ex((host, p)) != 0:
                return p
    return port


def main():
    ap = argparse.ArgumentParser(prog="viz2", description=__doc__)
    ap.add_argument("experiment", type=Path,
                    help="one experiment dir, or a dir containing several")
    ap.add_argument("--port", type=int, default=8505)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    from viz2 import app as m
    n = m.init(a.experiment)
    from viz2 import data
    port = free_port(a.host, a.port)
    print(f"[viz2] {n} experiment(s) from {a.experiment}; loaded {data.CURRENT} "
          f"({len(data.DIRS)} samples)")
    print(f"[viz2] http://{a.host}:{port}")
    uvicorn.run(m.app, host=a.host, port=port, log_level="warning")


main()
