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
def manifest():
    man = data.build_manifest()
    man["version"] = data.data_version()
    return JSONResponse(man)


@app.get("/api/version")
def version():
    return {"version": data.data_version()}


@app.get("/api/fft")
def fft(ids: str, lasers: str = "all", dirs: str = "xy", norm: bool = False):
    laser_idx = None if lasers == "all" else [int(i) for i in lasers.split(",") if i != ""]
    if laser_idx == []:
        laser_idx = None
    curves, freqs = {}, None
    for sid in [int(i) for i in ids.split(",") if i != ""]:
        try:
            curve, freqs = data.fft_curve(sid, laser_idx, dirs, norm)
        except FileNotFoundError:
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
def gt_masks(ids: str):
    out = {}
    for sid in [int(i) for i in ids.split(",") if i != ""]:
        try:
            out[str(sid)] = data.gt_mask(sid)
        except FileNotFoundError:
            pass
    return out


MEDIA = {
    "overhead": "overhead.png",
    "thumb": data.THUMB_NAME,
    "audio": "audio.wav",
    "recovered": "recovered_audio.wav",
}


@app.get("/media/{sample_id}/{kind}")
def media(sample_id: int, kind: str):
    if kind not in MEDIA:
        raise HTTPException(404, f"unknown media kind: {kind}")
    path = data.SAMPLES_DIR / f"{sample_id:06d}" / MEDIA[kind]
    if not path.exists():
        raise HTTPException(404, str(path))
    return FileResponse(path)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8501,
                app_dir=str(Path(__file__).parent), reload=True)
