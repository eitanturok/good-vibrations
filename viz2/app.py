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
RENDER_V = int(Path(render.__file__).stat().st_mtime)


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


@app.get("/api/samples")
def samples():
    return {"samples": list(data.META.values()), "info": data.INFO, "rv": RENDER_V}


@app.get("/api/sample/{sid}")
def sample(sid: str):
    _d(sid)
    _, freqs = data.fft(sid)
    return {**data.META[sid], "freqs": [round(f, 3) for f in freqs]}


@app.get("/api/scene/{sid}.jpg")
def scene(sid: str):
    d = _d(sid)
    mask = None
    if data.INFO["mask"] and (d / data.INFO["mask"]).exists():
        mask = np.load(d / data.INFO["mask"])
    return Response(render.scene(Image.open(d / data.INFO["photo"]), mask),
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
             "phase": lambda: np.angle(z), "re": lambda: z.real, "im": lambda: z.imag}[q]()
    if q in ("logmag", "mag"):
        return v, "seq", float(np.percentile(v, 1)), float(np.percentile(v, 99))
    m = float(np.percentile(np.abs(v), 99)) or 1.0     # symmetric: zero stays neutral
    return v, "div", -m, m


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
    f, freqs = data.fft(sid)
    z = data.pick(data.chan(f, ch), laser)
    mag = np.abs(z)
    s = data.pick(data.chan(data.shifts(sid, kind), ch), laser)
    r = lambda a: [round(float(x), 5) for x in a]
    return {
        "mag": r(mag), "logmag": r(data.logmag(mag)), "phase": r(np.angle(z)),
        "re": r(z.real), "im": r(z.imag),
        "peaks": data.peaks(mag, freqs),
        "shifts": [round(float(x), 5) for x in data.envelope(s)],
        "dur": len(s) / data.INFO["fps"],
    }


@app.get("/api/mode/{sid}")
def mode(sid: str, fi: int = 0):
    _d(sid)
    U, V = data.mode(sid, fi)
    return {"u": U.round(6).tolist(), "v": V.round(6).tolist()}


@app.get("/api/audio/{sid}.wav")
def audio(sid: str, ch: str = "x", laser: str = "55"):
    _d(sid)
    pcm, sr = data.audio(sid, ch, laser)
    return Response(_wav(pcm, sr), media_type="audio/wav", headers=CACHE)
