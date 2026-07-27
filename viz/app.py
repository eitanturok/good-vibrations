"""Vibrations dashboard server.

    python viz/app.py            # http://localhost:8501

Everything is read live from data/ and runs/ — new samples or runs appear
without a restart (the frontend polls /api/version).
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import data

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="good-vibrations dashboard")


@app.middleware("http")
async def no_cache(request, call_next):
    resp = await call_next(request)
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/manifest")
def manifest(dataset: str | None = None, run: str | None = None):
    # a run implies its dataset, so the sample list always matches the selected run
    if dataset is None:
        dataset = data.dataset_for_run(run) if run else data.DEFAULT_DATASET
    man = data.build_manifest(dataset)
    man["version"] = data.data_version()
    return JSONResponse(man)


@app.get("/api/version")
def version():
    return {"version": data.data_version()}


@app.get("/api/fft")
def fft(ids: str, lasers: str = "all", dirs: str = "xy", norm: bool = False,
        dataset: str | None = None, run: str | None = None):
    ds = dataset or (data.dataset_for_run(run) if run else data.DEFAULT_DATASET)
    laser_idx = None if lasers == "all" else [int(i) for i in lasers.split(",") if i != ""]
    if laser_idx == []:
        laser_idx = None
    curves, freqs = {}, None
    for sid in [int(i) for i in ids.split(",") if i != ""]:
        try:
            curve, freqs = data.fft_curve(sid, laser_idx, dirs, norm, ds)
        except (FileNotFoundError, KeyError):
            continue
        curves[str(sid)] = [round(float(v), 4) for v in curve]
    if freqs is None:
        raise HTTPException(404, "no fft found for requested ids")
    return {"freqs": [round(float(f), 2) for f in freqs], "curves": curves}


@app.get("/api/run/{run}")
def run_info(run: str, epoch: int | None = None):
    if run not in data.list_runs():
        raise HTTPException(404, f"unknown run: {run}")
    return data.run_payload(run, epoch)


@app.get("/api/run/{run}/masks")
def masks(run: str, ids: str, epoch: int | None = None):
    if run not in data.list_runs():
        raise HTTPException(404, f"unknown run: {run}")
    return data.run_masks(run, [int(i) for i in ids.split(",") if i != ""], epoch)


@app.get("/api/gt_masks")
def gt_masks(ids: str, dataset: str | None = None, run: str | None = None):
    ds = dataset or (data.dataset_for_run(run) if run else data.DEFAULT_DATASET)
    out = {}
    for sid in [int(i) for i in ids.split(",") if i != ""]:
        try:
            out[str(sid)] = data.gt_mask(sid, ds)
        except (FileNotFoundError, KeyError):
            pass
    return out


MEDIA = {
    "overhead": "image/04_overhead_scored.png",  # masks + boxes + confidence drawn on
    "thumb": "image/01_cropped.png",             # plain cropped overhead
    "smask": "image/02_smask.png",
    "audio": "audio.wav",
    "recovered": "recovered_audio.wav",
}


@app.get("/media/{sample_id}/{kind}")
def media(sample_id: int, kind: str):
    """Per-sample images/audio live in the raw capture dirs, not in the MDS dataset (which
    only carries X/y). Served only when a raw sample dir is configured and still on disk."""
    if kind not in MEDIA:
        raise HTTPException(404, f"unknown media kind: {kind}")
    base = data.raw_sample_dir(sample_id)
    if base is None:
        raise HTTPException(404, f"no raw sample dir for {sample_id}")
    path = base / MEDIA[kind]
    if not path.exists():
        raise HTTPException(404, str(path))
    return FileResponse(path)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import argparse
    import errno
    import socket
    import sys

    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8501)
    args = ap.parse_args()

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
                 f"      or pick another:  python viz/app.py --port {args.port + 1}")
    finally:
        probe.close()

    print(f"[viz] serving on {url}")
    uvicorn.run("app:app", host=args.host, port=args.port,
                app_dir=str(Path(__file__).parent), reload=True)
