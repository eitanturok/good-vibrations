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

from viz2 import config, data, render
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
    # Picks up runs that appeared since startup, so a model that finishes training while
    # viz2 is open shows up in the picker on its own.
    registry.maybe_rescan()
    runs = [{"name": e.name, "compatible": e.compatible, "reason": e.reason,
             "mtime": e.mtime, "epoch": e.epoch, "eval_splits": e.eval_splits,
             "family": e.family, "status": e.status} for e in registry.entries]
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
            # position_id is the gastronorm spelling of output_id: the scene identity
            # shared by the 8 samples that differ only in which speaker played.
            "output_id": m.get("output_id") or m.get("position_id"),
            "layout": m.get("layout"),
            "n_objects": m.get("n_objects"),
            # Object types present, independent of layout: lets the UI ask "contains a
            # cylinder?" once the dataset holds more than cubes.
            "objects": sorted(m.get("objects") or {}),
            # The enclosure the scene sits in. Experiment-25 is entirely "metal", so the
            # filter shows a single chip today; it populates itself if other boxes appear.
            "box": m.get("box"),
            "speaker": m.get("speaker"),
            "is_empty_box": bool(m.get("is_empty_box")),
            # full-resolution image coords, for the position scatter; [-1,-1] sentinel
            # on empty-box samples means "no position"
            "avg_com": [_clean(com[0]), _clean(com[1])],
            "com_gt": [_clean(gt.com_gt[i][0]), _clean(gt.com_gt[i][1])],
        })
    return {"samples": out}


@app.get("/api/run/{name}")
def api_run(name: str, reload: int = 0, epoch: int | None = None):
    """Per-sample metrics for one run. `epoch` scores that saved epoch instead of the
    latest, so the numbers and sorting match the masks on screen while scrubbing."""
    # reload=1 re-reads the run's prediction files, picking up epochs written since it
    # was first loaded (a run still training keeps producing them).
    if reload:
        registry.rescan()
    try:
        rd = registry.run(name, reload=bool(reload), epoch=epoch)
    except KeyError:
        raise HTTPException(404, "unknown or incompatible run")
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
            "epochs": registry.epochs(name),   # drives the epoch slider
            "samples": samples}


# ***** mask images *****


@app.get("/api/mask.png")
def api_mask(run: str, sid: int, mode: str = "pred", bg: int = 1):
    i = _sid(sid)
    rd = _run(run)
    if i not in rd.row_of:
        raise HTTPException(404, "no prediction for this sample")
    mode = mode if mode in ("diff", "overlay", "stacked") else "pred"
    img = render.cached_mask("run", run, i, mode, bool(bg))
    return Response(img, media_type=render.media_type(img), headers=IMMUTABLE)


@app.get("/api/gt_mask.png")
def api_gt_mask(sid: int, bg: int = 1):
    img = render.cached_mask("gt", "", _sid(sid), "pred", bool(bg))
    return Response(img, media_type=render.media_type(img), headers=IMMUTABLE)


@app.get("/api/backdrop/{sid}.jpg")
def api_backdrop(sid: int):
    """The overhead frame at cell size, shown behind canvas-drawn masks. Fetched once per
    sample and reused across every epoch, so scrubbing never re-downloads it."""
    img = render.cached_backdrop(_sid(sid))
    if img is None:
        raise HTTPException(404, "no backdrop")
    return Response(img, media_type="image/jpeg", headers=IMMUTABLE)


@app.get("/api/colorbar/{mode}.png")
def api_colorbar(mode: str):
    mode = mode if mode in ("diff", "truth") else "pred"
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
        elif mode in ("overlay", "stacked"):
            # Two masks are on screen, so report both rather than leaving the reader to
            # guess which one a single number belongs to.
            return JSONResponse(
                {"v": np.round(values.astype(np.float64), 4).tolist(),
                 "t": np.round(registry.gt.masks[i].astype(np.float64), 4).tolist()},
                headers=IMMUTABLE)
    return JSONResponse({"v": np.round(values.astype(np.float64), 4).tolist()},
                        headers=IMMUTABLE)


# ***** per-sample media + detail *****


@app.get("/api/overhead/{sid}.png")
def api_overhead(sid: int):
    p = registry.sample_dir(_sid(sid)) / registry.gt.layout.overhead
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
    rel = registry.gt.layout.audio.get(which)
    if rel is None:
        # Either an unknown name or a track this layout never produces (the gastronorm
        # captures ship no source audio, only the recovered waveform).
        raise HTTPException(404, "unknown audio")
    p = registry.sample_dir(_sid(sid)) / rel
    if not p.exists():
        raise HTTPException(404, "not generated for this sample")
    # FileResponse handles Range requests, which <audio> needs in order to seek.
    return FileResponse(p, media_type="audio/wav")


@app.get("/api/lut")
def api_lut():
    """The colormaps, so the client draws cells with exactly the colours the server uses
    for the modal and ground-truth images. One source of truth for colour."""
    return {"pred": render.SEQ_LUT.tolist(), "truth": render.TRUE_SEQ_LUT.tolist(),
            "diff": render.DIV_LUT.tolist(), "gamma": render.GAMMA,
            "gain": render.OVERLAY_GAIN,   # alpha scale when drawn over the backdrop
            "h": config.MASK_H, "w": config.MASK_W}


@app.get("/api/frames")
def api_frames(run: str, sids: str, epochs: str = ""):
    """Raw mask values for a set of samples across a set of epochs, as one fp16 blob.

    A 20x40 mask is 1600 bytes -- usually smaller than a PNG of it -- so shipping values
    and drawing them on the client is both lighter than per-frame images and fast enough
    to scrub and animate without touching the network again.

    Layout: float16[n_epochs][n_sids][H*W], C-order. Samples the run never predicted are
    filled with NaN so the client can show its "no prediction" state.
    """
    ids = [registry.sample_index(s) for s in sids.split(",") if s.strip()]
    if not ids:
        raise HTTPException(400, "no sids")
    eps = [int(e) for e in epochs.split(",") if e.strip()] or registry.epochs(run)
    if not eps:
        raise HTTPException(404, "run has no saved epochs")

    if run not in registry.by_name or not registry.by_name[run].compatible:
        raise HTTPException(404, "unknown or incompatible run")

    want = set(ids)
    out = np.full((len(eps), len(ids), config.MASK_H * config.MASK_W), np.nan, dtype=np.float16)
    for ei, ep in enumerate(eps):
        masks = data.load_epoch_masks(run, registry.runs_dir, ep, want)
        for si, i in enumerate(ids):
            m = masks.get(i)
            if m is not None:
                out[ei, si] = m.reshape(-1)
    return Response(out.tobytes(), media_type="application/octet-stream", headers=IMMUTABLE)


@app.get("/api/neighbors")
def api_neighbors(run: str, sid: int, k: int = 5):
    """Ground-truth samples whose center of mass is closest to / furthest from what this
    run predicted for `sid`.

    Answers "the model put the mass here -- which real scenes actually look like that?",
    so a prediction that resembles a different sample than its own target is visible.
    """
    i = _sid(sid)
    rd = _run(run)
    if i not in rd.row_of:
        raise HTTPException(404, "no prediction for this sample")
    pred = np.asarray(rd.com_pred[rd.row_of[i]], dtype=np.float64)

    gt = registry.gt
    # Grid cells are not square (20 rows x 40 cols over the same scene), so normalise to
    # [0,1] on each axis before measuring -- otherwise a column offset counts double.
    scale = np.array([config.MASK_H - 1, config.MASK_W - 1], dtype=np.float64)
    target = pred / scale
    coms = gt.com_gt / scale

    # Empty boxes carry a (-1,-1) sentinel rather than a position; ranking them would
    # fill the "least similar" list with samples that have no center of mass at all.
    valid = np.array([not m.get("is_empty_box") and gt.com_gt[j][0] >= 0
                      for j, m in enumerate(gt.meta)])
    d = np.linalg.norm(coms - target, axis=-1)
    d[~valid] = np.nan

    order = np.argsort(np.where(np.isnan(d), np.inf, d))
    order = [j for j in order if not np.isnan(d[j])]
    k = max(1, min(int(k), 25))

    def position_key(m: dict):
        """What identifies a physical scene, independent of which speaker played.

        experiment-25 calls it output_id, the gastronorm captures call it position_id.
        Falling back to the sample id when neither is present keeps every candidate
        distinct -- a key of None for every sample would otherwise collapse the whole
        list to a single entry rather than merely failing to dedupe.
        """
        for key in ("output_id", "position_id"):
            if m.get(key) is not None:
                return (key, m[key])
        return ("sample_id", m.get("sample_id"))

    def distinct(seq):
        """One entry per physical position: 8 samples share each position (one per
        speaker) with identical ground-truth COM, so without this every slot in the list
        would be the same scene repeated."""
        seen, out = set(), []
        for j in seq:
            key = position_key(gt.meta[j])
            if key in seen:
                continue
            seen.add(key)
            out.append(j)
            if len(out) == k:
                break
        return out

    def pack(idx):
        m = gt.meta[idx]
        return {"i": int(idx), "sample_id": gt.sample_ids[idx],
                "output_id": m.get("output_id") or m.get("position_id"),
                "speaker": m.get("speaker"),
                "layout": m.get("layout"), "n_objects": m.get("n_objects"),
                "com": [_clean(gt.com_gt[idx][0]), _clean(gt.com_gt[idx][1])],
                "distance": _clean(d[idx])}

    return {"run": run, "sample_id": gt.sample_ids[i],
            "pred_com": [_clean(pred[0]), _clean(pred[1])],
            "gt_com": [_clean(gt.com_gt[i][0]), _clean(gt.com_gt[i][1])],
            "n_candidates": len(order),
            "most_similar": [pack(j) for j in distinct(order)],
            "least_similar": [pack(j) for j in distinct(order[::-1])]}


@app.get("/api/detail/{sid}")
def api_detail(sid: int):
    i = _sid(sid)
    m = registry.gt.meta[i]
    d = registry.sample_dir(i)
    # Union of both eras' metadata: experiment-25 writes output_id/n_lasers, the
    # gastronorm captures write position_id/laser_idx/timestamp/n_rows/n_cols. Missing
    # keys come back null and the modal omits them, so one list serves both.
    keys = ["sample_id", "output_id", "position_id", "description", "layout",
            "n_objects", "objects", "coms", "avg_com", "box", "is_empty_box", "speaker",
            "min_freq", "max_freq", "n_lasers", "n_rows", "n_cols", "laser_idx",
            "fps", "n_capture_seconds", "timestamp"]
    out = {k: m.get(k) for k in keys}
    # coms/avg_com are numpy reprs on the gastronorm layout; normalise the one the UI
    # actually plots so the modal never prints a raw "[603.1 901.2]" string.
    out["avg_com"] = data.parse_com(m.get("avg_com"))
    out["com_gt_grid"] = [_clean(registry.gt.com_gt[i][0]), _clean(registry.gt.com_gt[i][1])]
    audio = registry.gt.layout.audio
    out["has"] = {
        # A track the layout does not define is simply absent, so the modal hides it
        # rather than offering a control that 404s.
        "original": bool(audio.get("original")) and (d / audio["original"]).exists(),
        "recovered": bool(audio.get("recovered")) and (d / audio["recovered"]).exists(),
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
