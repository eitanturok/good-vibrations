"""FastAPI routes for viz2.

The server does only what the browser cannot: decode .pt files, compute metrics once
per run, and render PNGs. Sorting and filtering are entirely client-side -- the whole
dataset is a few hundred KB and already in memory there, so a round-trip per slider
frame would only add latency.
"""

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from viz2 import config, render
from viz2.data import Registry

app = FastAPI(title="viz2")
registry: Registry = None  # set by init()

STATIC = Path(__file__).parent / "static"
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}


def init(experiment_dir: Path = None, runs_dir: Path = None) -> Registry:
    global registry
    registry = Registry(experiment_dir or config.EXPERIMENT_DIR, runs_dir or config.RUNS_DIR)
    for name in registry.defaults():  # warm so the first paint has data
        registry.run(name)
    return registry


@app.on_event("startup")
def _startup():
    if registry is None:
        init()


def _sid(sid) -> int:
    try:
        return registry.sample_index(sid)
    except (KeyError, ValueError, TypeError):
        raise HTTPException(404, "unknown sample")


def _run(name: str):
    try:
        return registry.run(name)
    except KeyError:
        raise HTTPException(404, "unknown or incompatible run")


def _clean(v):
    """NaN/Inf are not valid JSON; emit null so the client can treat them as missing."""
    f = float(v)
    return None if not np.isfinite(f) else round(f, 6)


# ***** metadata *****


@app.get("/api/runs")
def api_runs():
    runs = [{"name": e.name, "compatible": e.compatible, "reason": e.reason,
             "mtime": e.mtime, "epoch": e.epoch, "eval_splits": e.eval_splits,
             "family": e.family} for e in registry.entries]
    return {"runs": runs, "default_selected": registry.defaults(),
            "n_samples": len(registry.gt), "render_version": config.RENDER_VERSION}


@app.get("/api/samples")
def api_samples():
    gt = registry.gt
    out = []
    for i, sid in enumerate(gt.sample_ids):
        m = gt.meta[i]
        com = gt.avg_com[i]
        out.append({
            "i": i,
            "sample_id": sid,
            "output_id": m.get("output_id"),
            "layout": m.get("layout"),
            "n_objects": m.get("n_objects"),
            "speaker": m.get("speaker"),
            "is_empty_box": bool(m.get("is_empty_box")),
            # full-resolution image coords, for the position scatter; [-1,-1] sentinel
            # on empty-box samples means "no position"
            "avg_com": [_clean(com[0]), _clean(com[1])],
            "com_gt": [_clean(gt.com_gt[i][0]), _clean(gt.com_gt[i][1])],
        })
    return {"samples": out}


@app.get("/api/run/{name}")
def api_run(name: str):
    rd = _run(name)
    samples = {}
    for i, sid in enumerate(rd.sample_ids):
        samples[int(sid)] = {
            "split": rd.splits[i],
            "mse": _clean(rd.mse[i]),
            "iou": _clean(rd.iou[i]),
            "comdist": _clean(rd.comdist[i]),
            "com": [_clean(rd.com_pred[i][0]), _clean(rd.com_pred[i][1])],
        }
    return {"name": rd.name, "epoch": rd.epoch, "family": rd.family,
            "skipped_files": rd.skipped_files, "n": len(rd.sample_ids),
            "samples": samples}


# ***** mask images *****


@app.get("/api/mask.png")
def api_mask(run: str, sid: int, mode: str = "pred", bg: int = 1):
    i = _sid(sid)
    rd = _run(run)
    if i not in rd.row_of:
        raise HTTPException(404, "no prediction for this sample")
    mode = "diff" if mode == "diff" else "pred"
    img = render.cached_mask("run", run, i, mode, bool(bg))
    return Response(img, media_type=render.media_type(img), headers=IMMUTABLE)


@app.get("/api/gt_mask.png")
def api_gt_mask(sid: int, bg: int = 1):
    img = render.cached_mask("gt", "", _sid(sid), "pred", bool(bg))
    return Response(img, media_type=render.media_type(img), headers=IMMUTABLE)


@app.get("/api/colorbar/{mode}.png")
def api_colorbar(mode: str):
    mode = "diff" if mode == "diff" else "pred"
    return Response(render.cached_colorbar(mode), media_type="image/png", headers=IMMUTABLE)


@app.get("/api/values")
def api_values(sid: int, run: str = "", mode: str = "pred"):
    """Grid values behind one cell, for the hover tooltip. Fetched on first hover and
    memoized client-side -- never prefetched."""
    i = _sid(sid)
    if not run:
        values = registry.gt.masks[i]
    else:
        rd = _run(run)
        if i not in rd.row_of:
            raise HTTPException(404, "no prediction for this sample")
        values = rd.masks[rd.row_of[i]]
        if mode == "diff":
            values = values - registry.gt.masks[i]
    return JSONResponse({"v": np.round(values.astype(np.float64), 4).tolist()},
                        headers=IMMUTABLE)


# ***** per-sample media + detail *****


@app.get("/api/overhead/{sid}.png")
def api_overhead(sid: int):
    p = registry.sample_dir(_sid(sid)) / config.OVERHEAD_REL
    if not p.exists():
        raise HTTPException(404, "no overhead image")
    return FileResponse(p, media_type="image/png", headers=IMMUTABLE)


@app.get("/api/vibration/{sid}/{which}.png")
def api_vibration(sid: int, which: str):
    pat = config.VIBRATION_GLOB.get(which)
    if pat is None:
        raise HTTPException(404, "unknown image")
    hits = sorted(registry.sample_dir(_sid(sid)).glob(pat))
    if not hits:
        raise HTTPException(404, "not generated for this sample")
    return FileResponse(hits[0], media_type="image/png", headers=IMMUTABLE)


@app.get("/api/audio/{sid}/{which}")
def api_audio(sid: int, which: str):
    rel = config.AUDIO_REL.get(which)
    if rel is None:
        raise HTTPException(404, "unknown audio")
    p = registry.sample_dir(_sid(sid)) / rel
    if not p.exists():
        raise HTTPException(404, "not generated for this sample")
    # FileResponse handles Range requests, which <audio> needs in order to seek.
    return FileResponse(p, media_type="audio/wav")


@app.get("/api/detail/{sid}")
def api_detail(sid: int):
    i = _sid(sid)
    m = registry.gt.meta[i]
    d = registry.sample_dir(i)
    keys = ["sample_id", "output_id", "description", "layout", "n_objects", "objects",
            "coms", "avg_com", "box", "is_empty_box", "speaker", "min_freq", "max_freq",
            "n_lasers", "fps", "n_capture_seconds"]
    out = {k: m.get(k) for k in keys}
    out["com_gt_grid"] = [_clean(registry.gt.com_gt[i][0]), _clean(registry.gt.com_gt[i][1])]
    out["has"] = {
        "original": (d / config.AUDIO_REL["original"]).exists(),
        "recovered": (d / config.AUDIO_REL["recovered"]).exists(),
        "spectrogram": bool(list(d.glob(config.VIBRATION_GLOB["spectrogram"]))),
        "fft": bool(list(d.glob(config.VIBRATION_GLOB["fft"]))),
    }
    return out


@app.get("/")
def index():
    """Serve index.html with the asset URLs stamped by file mtime, so an edited app.js
    or style.css can never be served from a stale browser cache."""
    html = (STATIC / "index.html").read_text()
    for asset in ("app.js", "style.css"):
        v = int((STATIC / asset).stat().st_mtime)
        html = html.replace(f"/{asset}", f"/{asset}?v={v}")
    return Response(html, media_type="text/html",
                    headers={"Cache-Control": "no-cache, must-revalidate"})


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
