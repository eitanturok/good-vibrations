"""Routes. The server does numpy; the browser draws."""

import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from viz2 import data, render

app = FastAPI()
STATIC = Path(__file__).parent / "static"
CACHE = {"Cache-Control": "public, max-age=31536000, immutable"}
# The cache-buster every rendered-media URL carries (see api/samples' "rv"). Every file
# whose code actually decides what a byte on the wire looks like, not just render.py -- a
# thumb size/quality change made here in app.py (e.g. the box= passed to render.thumb)
# wouldn't bump this otherwise, and every browser would keep serving its year-cached,
# immutable copy from before the change forever, since the URL never changed either.
RENDER_V = int(max(Path(f).stat().st_mtime for f in (
    render.__file__, __file__, data.__file__,                    # data.recspec draws a PNG too,
    Path(__file__).parent.parent / "utils" / "viz.py")))         # with utils.viz's renderer


@app.get("/")
def index():
    """index.html with the asset URLs stamped by mtime.

    Without this the browser keeps a cached app.js/style.css across restarts, so code
    changes appear not to take effect -- which is a genuinely confusing failure, because
    the server is serving the new file and the page is running the old one.
    """
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for name in ("app.js", "style.css"):
        v = int((STATIC / name).stat().st_mtime)
        html = html.replace(f'"/{name}"', f'"/{name}?v={v}"')
    return Response(html, media_type="text/html",
                    headers={"Cache-Control": "no-cache"})


def init(exp):
    n = data.init(exp)
    threading.Thread(target=_prewarm, daemon=True, name="thumb-prewarm").start()
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return n


def _d(sid):
    try:
        return data.d(sid)
    except KeyError:
        raise HTTPException(404, "unknown sample")


def _wav(pcm, sr):
    import wave, io
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return b.getvalue()


def _payload():
    # rv is the cache-buster on every sid-keyed media URL (thumb/scene/mask/heat/...), and
    # those endpoints are "immutable" cached forever -- sample ids are only unique WITHIN a
    # dataset (both loaded datasets can have a "000009"), so rv must fold in the dataset
    # name too, or switching datasets serves the old dataset's cached image back. It also
    # folds in data.EPOCH, which data.py bumps whenever a sample is actually added/removed/
    # reprocessed -- otherwise a sample deleted and recaptured under the same id keeps
    # serving its old (wrong) cached photo forever, since nothing else about the URL changed.
    return {"samples": list(data.META.values()), "info": data.INFO,
            "rv": f"{RENDER_V}-{data.EPOCH}-{data.CURRENT}",
            "datasets": sorted(data.DATASETS), "dataset": data.CURRENT,
            "dataset_counts": data.COUNTS}


@app.get("/api/samples")
def samples(rv: str = ""):
    """Also picks up samples (and new experiment dirs) the watcher finished writing since
    the last call -- cheap (a scandir + one stat per sample), so the client just polls this
    to go live. A poll passes the rv it already has; if nothing moved, only rv comes back
    instead of re-encoding and re-sending the whole ~1.6MB sample list every 0.5s."""
    data.rescan_datasets()
    data.rescan()
    p = _payload()
    return {"rv": p["rv"]} if rv and rv == p["rv"] else p


@app.get("/api/switch/{name}")
def switch(name: str):
    """Load a different dataset (box) and hand back the same shape as /api/samples."""
    try:
        data.switch(name)
    except KeyError:
        raise HTTPException(404, "unknown dataset")
    return _payload()


@app.get("/api/box/{name}.jpg")
def box_thumb(name: str):
    """Row icon for the dataset picker: the bare box (an empty-box sample if there is one)."""
    if name not in data.DATASETS:
        raise HTTPException(404, "unknown dataset")
    p = data.box_photo(name)
    if not p:
        raise HTTPException(404, "no photo")
    return Response(render.thumb(p), media_type="image/jpeg", headers=CACHE)


@app.get("/api/thumb/{sid}.jpg")
def sample_thumb(sid: str, seg: int = 0, sel: int = -1):
    """THE sample photo -- the gallery cards, the sample viewer and the sidebar's current
    sample all show this same URL (app.js photoFig), so the viewer is an instant browser-
    cache hit on whatever the gallery already loaded. seg=1: the segmentation view.
    sel >= 0: the viewer's selected object, others faded -- drawn per object, not cached."""
    _d(sid)
    objs = data.object_masks(sid) if sel >= 0 else []
    if 0 <= sel < len(objs) and (p := data.sample_photo(sid)):
        b = render.thumb(p, [m for _, m in objs], seg=bool(seg), box=(480, 360), sel=sel)
    else:
        b = gallery_thumb(sid, seg)
    if b is None:
        raise HTTPException(404, "no photo")
    return Response(b, media_type="image/jpeg", headers=CACHE)


# Rendered gallery thumbs, on disk. Rendering one is ~34ms, ~26ms of it just decoding the
# 1.4MB overhead PNG -- and the browser's own cache doesn't cover it well: every new sample
# during live collection bumps rv, which re-requests every thumb in the grid. Keyed on the
# source files' mtimes (and RENDER_V), so a recaptured sample or a render change misses.
THUMB_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "viz2" / "thumbs"


def _thumb_path(p: Path, mp: Path | None, seg: int) -> Path:
    key = f"{p}|{p.stat().st_mtime_ns}|{mp}|{mp.stat().st_mtime_ns if mp else 0}|seg{seg}|{RENDER_V}"
    return THUMB_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.jpg"


def gallery_thumb(sid: str, seg: int = 0) -> bytes | None:
    p = data.sample_photo(sid)
    if not p:
        return None
    mp = data.sample_mask(sid)
    out = _thumb_path(p, mp, seg)
    try:
        return out.read_bytes()
    except OSError:
        pass
    # 480x360 covers the gallery card / viewer's on-screen size (incl. 2x retina) while
    # still a fraction of the ~1300px source photo
    b = render.thumb(p, [np.load(mp)] if mp else [], seg=bool(seg), box=(480, 360))
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_bytes(b)
    tmp.replace(out)                # atomic: a concurrent reader never sees half a JPEG
    return b


def _prewarm():
    """Background: render both thumbs (plain and segmentation) for every sample of whichever
    dataset is loaded, so scrolling and the mask toggle hit the disk cache instead of
    decoding PNGs on demand. Few threads, so interactive requests still get the CPU;
    loops to pick up samples arriving live and dataset switches."""
    with ThreadPoolExecutor(4) as ex:
        while True:
            todo = []
            for seg in (0, 1):      # the default (plain) view first
                for sid in list(data.DIRS):
                    try:
                        p = data.sample_photo(sid)
                        if p and not _thumb_path(p, data.sample_mask(sid), seg).exists():
                            todo.append((sid, seg))
                    except (KeyError, OSError):
                        pass        # sample dropped mid-walk; the next pass sees the new state
            # wait for the whole pass before re-listing, or a slow pass re-queues its own work
            for f in [ex.submit(gallery_thumb, sid, seg) for sid, seg in todo]:
                try:
                    f.result()
                except Exception:
                    pass
            time.sleep(2)


@app.get("/api/sample/{sid}")
def sample(sid: str):
    _d(sid)
    _, freqs = data.fft(sid)
    return {**data.META[sid], "freqs": [round(f, 3) for f in freqs], "stim": data.stim_params(sid),
            "stim_name": data.stim_name(sid)}


@app.get("/api/meta/{sid}")
def meta(sid: str):
    """The sample's own metadata.jsonl, as written at capture time (the viewer's raw view).
    NaN/inf (e.g. an empty sample's {"avg_com": NaN}) aren't valid JSON, so they go out as null."""
    def clean(v):
        if isinstance(v, float) and not np.isfinite(v): return None
        if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list): return [clean(x) for x in v]
        return v
    return clean(data._meta(_d(sid)))


@app.get("/api/scene/{sid}.jpg")
def scene(sid: str, mask: int = 1):
    """The cropped overhead. mask=1 (default) tints the segmentation green; mask=0 is the
    bare photo, for showing the box and its segmentation side by side."""
    _d(sid)
    p, mp = data.sample_photo(sid), data.sample_mask(sid)
    if not p:
        raise HTTPException(404, "no photo")
    m = np.load(mp) if mask and mp else None
    return Response(render.scene(Image.open(p), m),
                    media_type="image/jpeg", headers=CACHE)


@app.get("/api/mask/{sid}.png")
def mask(sid: str):
    _d(sid)
    mp = data.sample_mask(sid)
    if not mp:
        raise HTTPException(404, "no mask")
    return Response(render.mask_png(np.load(mp)),
                    media_type="image/png", headers=CACHE)


@app.get("/api/areas")
def areas():
    """{sid: segmented area px} for the loaded dataset (the gallery's "mask area" sort)."""
    return data.mask_areas(data.CURRENT, data.EPOCH)


@app.get("/api/objstats/{sid}")
def objstats(sid: str):
    """Per-object colour, volume (px) and centroid [row, col] for the viewer's mask table."""
    _d(sid)
    out = []
    for i, (name, m) in enumerate(data.object_masks(sid)):
        r, c = np.nonzero(m)
        out.append({"name": name, "color": render.OBJ_COLORS[i % len(render.OBJ_COLORS)],
                    "vol": int(m.sum()), "com": [float(r.mean()), float(c.mean())]})
    return out


@app.get("/api/objmasks/{sid}.png")
def objmasks_png(sid: str, sel: int = -1):
    _d(sid)
    ms = [m for _, m in data.object_masks(sid)]
    if not ms:
        raise HTTPException(404, "no masks")
    return Response(render.objmasks_png(ms, sel), media_type="image/png", headers=CACHE)


@app.get("/api/masks.png")
def masks(ids: str = "", colors: str = ""):
    """Several samples' masks composited into one image, one color each."""
    sids = [i for i in ids.split(",") if i]
    cols = [c for c in colors.split(",") if c]
    if not sids:
        raise HTTPException(404, "no ids")
    ms, cs = [], []
    for sid, c in zip(sids, cols):
        _d(sid)
        mp = data.sample_mask(sid)
        if mp:
            ms.append(np.load(mp))
            cs.append(tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)))
    if not ms:
        raise HTTPException(404, "no masks")
    return Response(render.masks_overlay(ms, cs), media_type="image/png", headers=CACHE)


@app.get("/api/heat/{sid}.png")
def heat(sid: str, ch: str = "avg", q: str = "logmag", kind: str = "clean"):
    """One quantity for every laser at once: rows = lasers, columns = frequency (or time).

    Signed quantities are scaled symmetrically about zero on the diverging ramp, so the
    neutral middle really is zero and the sign is readable.
    """
    _d(sid)
    v, lut, lo, hi = _plane(sid, ch, q, kind)
    return Response(render.heat(v, lo, hi, lut), media_type="image/png", headers=CACHE)


def _plane(sid, ch, q, kind):
    """The (lasers x columns) array for one quantity, plus its palette and range."""
    if q == "shifts":
        v = data.chan(data.shifts(sid, kind), ch)
    else:
        z = data.chan(data.fft(sid)[0], ch)
        v = {"logmag": lambda: data.logmag(np.abs(z)), "mag": lambda: np.abs(z),
             "phase": lambda: np.angle(z), "cosphase": lambda: np.cos(np.angle(z)),
             "re": lambda: z.real, "im": lambda: z.imag}[q]()
    # The SAME range for every sample, so two heatmaps are directly comparable and the
    # colorbar means one thing across the whole app.
    lo, hi = data.INFO["scale"][q]
    return v, ("seq" if q in ("logmag", "mag") else "div"), lo, hi


@app.get("/api/heatrange/{sid}")
def heatrange(sid: str, ch: str = "avg", q: str = "logmag", kind: str = "clean"):
    """Just the colorbar bounds, so the client can label without decoding the PNG."""
    _d(sid)
    _, lut, lo, hi = _plane(sid, ch, q, kind)
    return {"lo": lo, "hi": hi, "lut": lut}


@app.get("/api/probe/{sid}")
def probe(sid: str, ch: str = "avg", laser: str = "avg", kind: str = "clean"):
    """Everything a probe needs from one round trip."""
    _d(sid)
    f, _ = data.fft(sid)
    z = data.pick(data.chan(f, ch), laser)
    mag = np.abs(z)
    s = data.pick(data.chan(data.shifts(sid, kind), ch), laser)
    r = lambda a: [round(float(x), 5) for x in a]
    return {
        "mag": r(mag), "logmag": r(data.logmag(mag)), "phase": r(np.angle(z)),
        "re": r(z.real), "im": r(z.imag),
        "peaks": data.peaks(sid),
        "shifts": [round(float(x), 5) for x in data.envelope(s)],
        "dur": len(s) / data.INFO["fps"],
    }


@app.get("/api/mode/{sid}")
def mode(sid: str, fi: int = 0):
    """The gradient field AND the height field it integrates to -- the quiver and the
    surface are two views of one mode, so they ship together rather than costing a second
    round trip when the view is switched."""
    _d(sid)
    U, V = data.mode(sid, fi)
    Z = data.surface(U, V)
    return {"u": U.round(6).tolist(), "v": V.round(6).tolist(), "z": Z.round(6).tolist()}


@app.get("/api/audio/{sid}.wav")
def audio(sid: str, ch: str = "x", laser: str = "55"):
    _d(sid)
    pcm, sr = data.audio(sid, *data.rec_channel(ch, laser))
    return Response(_wav(pcm, sr), media_type="audio/wav", headers=CACHE)


@app.get("/api/recspec/{sid}.png")
def recspec(sid: str, ch: str = "x", laser: str = "55"):
    """Spectrogram of /api/audio's recovered signal for this laser/channel. Where the
    playback line goes rides along in an X-Playhead header (JSON), so one request serves
    both the image and its geometry."""
    import json
    _d(sid)
    png, geo = data.recspec(sid, *data.rec_channel(ch, laser))
    return Response(png, media_type="image/png",
                    headers={**CACHE, "X-Playhead": json.dumps(geo)})


def _media(path, media_type):
    if not path or not path.is_file():
        raise HTTPException(404, "no media")
    return FileResponse(path, media_type=media_type, headers=CACHE)


@app.get("/api/source_audio/{sid}.wav")
def source_audio(sid: str):
    """The played stimulus, as recorded beside the repo (data/audio/<name>/)."""
    _d(sid)
    return _media(data.source_wav(sid), "audio/wav")


@app.get("/api/stim_video/{name}.mp4")
def stim_video(name: str):
    """source_video keyed by the stimulus itself (api/sample's stim_name): every sample
    that played the same chirp shares this URL, so stepping between them never reloads it."""
    return _media(data.stim_video(name), "video/mp4")


@app.get("/api/source_video/{sid}.mp4")
def source_video(sid: str):
    """Spectrogram video of the played stimulus."""
    _d(sid)
    return _media(data.source_video(sid), "video/mp4")


@app.get("/api/recovered_audio/{sid}.wav")
def recovered_audio(sid: str):
    """The pre-rendered recovered audio (fixed laser/channel -- see recovered_video)."""
    _d(sid)
    return _media(data.recovered_wav(sid), "audio/wav")


@app.get("/api/recovered_video/{sid}.mp4")
def recovered_video(sid: str):
    """Spectrogram video of the recovered signal -- a fixed laser/x pre-render."""
    _d(sid)
    return _media(data.recovered_video(sid), "video/mp4")
