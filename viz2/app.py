"""Routes. The server does numpy; the browser draws."""

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
RENDER_V = int(max(Path(render.__file__).stat().st_mtime, Path(__file__).stat().st_mtime))


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
def samples():
    """Also picks up samples (and new experiment dirs) the watcher finished writing since
    the last call -- cheap (an iterdir), so the client just polls this to go live."""
    data.rescan_datasets()
    data.rescan()
    return _payload()


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
def sample_thumb(sid: str, mask: int = 0, bg: int = 1):
    """Row icon for the sample picker: that sample's cropped-overhead photo. With bg=1, the
    object's ground-truth outline is always traced in green -- independent of `mask`, which
    only adds the translucent fill (the gallery's "segmentation mask" checkbox); with bg=0,
    `mask` switches between the bare photo and the segmentation alone on a flat field."""
    d = _d(sid)
    p = data.sample_photo(sid)
    if not p:
        raise HTTPException(404, "no photo")
    m = None
    if (bg or mask) and data.INFO["mask"] and (d / data.INFO["mask"]).exists():
        m = np.load(d / data.INFO["mask"])
    # The gallery grid renders these wide -- box_thumb's 160x120 default was for the much
    # smaller dataset-picker row icon; used here too, it forced the browser to upscale a
    # small JPEG to fill a much bigger box, which is the blur. 480x360 (still the same 4:3,
    # still a fraction of the source photo's ~1300px) covers the card's actual on-screen
    # size, including a retina display's 2x pixel density, without the crop losing detail.
    return Response(render.thumb(p, mask=m, fill=bool(mask), bg=bool(bg), box=(480, 360)),
                     media_type="image/jpeg", headers=CACHE)


@app.get("/api/sample/{sid}")
def sample(sid: str):
    _d(sid)
    _, freqs = data.fft(sid)
    return {**data.META[sid], "freqs": [round(f, 3) for f in freqs], "stim": data.stim_params(sid)}


@app.get("/api/scene/{sid}.jpg")
def scene(sid: str, mask: int = 1):
    """The cropped overhead. mask=1 (default) tints the segmentation green; mask=0 is the
    bare photo, for showing the box and its segmentation side by side."""
    d = _d(sid)
    m = None
    if mask and data.INFO["mask"] and (d / data.INFO["mask"]).exists():
        m = np.load(d / data.INFO["mask"])
    return Response(render.scene(Image.open(d / data.INFO["photo"]), m),
                    media_type="image/jpeg", headers=CACHE)


@app.get("/api/mask/{sid}.png")
def mask(sid: str):
    d = _d(sid)
    if not data.INFO["mask"] or not (d / data.INFO["mask"]).exists():
        raise HTTPException(404, "no mask")
    return Response(render.mask_png(np.load(d / data.INFO["mask"])),
                    media_type="image/png", headers=CACHE)


@app.get("/api/masks.png")
def masks(ids: str = "", colors: str = ""):
    """Several samples' masks composited into one image, one color each."""
    sids = [i for i in ids.split(",") if i]
    cols = [c for c in colors.split(",") if c]
    if not sids:
        raise HTTPException(404, "no ids")
    ms, cs = [], []
    for sid, c in zip(sids, cols):
        d = _d(sid)
        if data.INFO["mask"] and (d / data.INFO["mask"]).exists():
            ms.append(np.load(d / data.INFO["mask"]))
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
    pcm, sr = data.audio(sid, ch, laser)
    return Response(_wav(pcm, sr), media_type="audio/wav", headers=CACHE)


def _media(path, media_type):
    if not path or not path.is_file():
        raise HTTPException(404, "no media")
    return FileResponse(path, media_type=media_type, headers=CACHE)


@app.get("/api/source_audio/{sid}.wav")
def source_audio(sid: str):
    """The played stimulus, as recorded beside the repo (data/audio/<name>/)."""
    _d(sid)
    return _media(data.source_wav(sid), "audio/wav")


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
