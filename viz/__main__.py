"""Vibrations dashboard server.

    python -m viz              # http://localhost:8501
    python viz                 # same thing (a directory with __main__.py is runnable)
    python viz/app.py          # same thing, kept working for muscle memory

Run from the repo root, or with PYTHONPATH=. so viz/data.py can import utils.metrics.

Everything is read live from data/ and runs/ — new samples or runs appear
without a restart (the frontend polls /api/version).
"""
import argparse
import errno
import socket
import sys
from pathlib import Path


def main(argv=None):
    import uvicorn

    ap = argparse.ArgumentParser(prog="python -m viz", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args(argv)

    url = f"http://{args.host}:{args.port}"

    # Check the port up front: with reload=True uvicorn binds inside its reloader supervisor,
    # which reports a bare "[Errno 98] Address already in use" naming neither the address nor
    # the holder, and then hangs instead of exiting.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((args.host, args.port))
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        sys.exit(f"[viz] cannot start: {url} is already in use.\n"
                 f"      find the holder:  ss -lptn 'sport = :{args.port}'\n"
                 f"      then stop it:     kill <pid>\n"
                 f"      or pick another:  python -m viz --port {args.port + 1}")
    finally:
        probe.close()

    print(f"[viz] serving on {url}")
    # app_dir puts viz/ on sys.path in the reloader subprocess, so `import data` resolves
    # there the same way it does here.
    uvicorn.run("app:app", host=args.host, port=args.port,
                app_dir=str(Path(__file__).resolve().parent), reload=True)


if __name__ == "__main__":
    main()
