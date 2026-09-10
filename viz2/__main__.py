"""python -m viz2 <dataset-dir | dir-of-datasets>

Point it at one dataset (a dir with samples/) or a parent dir holding several; in the
second case every dataset shows up in the step-1 box picker and switches in place.
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
    ap.add_argument("dataset", type=Path,
                    help="one dataset dir, or a dir containing several")
    ap.add_argument("--port", type=int, default=8505)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    from viz2 import app as m
    n = m.init(a.dataset)
    from viz2 import data
    port = free_port(a.host, a.port)
    print(f"[viz2] {n} dataset(s) from {a.dataset}; loaded {data.CURRENT} "
          f"({len(data.DIRS)} samples)")
    print(f"[viz2] http://{a.host}:{port}")
    uvicorn.run(m.app, host=a.host, port=port, log_level="warning")


main()
