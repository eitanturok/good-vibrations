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


@app.get("/api/heat/{sid}.png")
def heat(sid: str, ch: str = "avg", log: int = 1):
    _d(sid)
    f, _ = data.fft(sid)
    v = np.abs(data.chan(f, ch))
    v = data.logmag(v) if log else v
    lo, hi = data.domain(v)
    return Response(render.heat(v, lo, hi), media_type="image/png", headers=CACHE)


@app.get("/api/probe/{sid}")
def probe(sid: str, ch: str = "avg", laser: str = "avg", kind: str = "clean"):
    """Everything a probe needs from one round trip."""
    _d(sid)
    f, freqs = data.fft(sid)
    z = data.pick(data.chan(f, ch), laser)
    mag = np.abs(z)
    v = data.logmag(np.abs(data.chan(f, ch)))
    s = data.pick(data.chan(data.shifts(sid, kind), ch), laser)
    r = lambda a: [round(float(x), 5) for x in a]
    return {
        "mag": r(mag), "logmag": r(data.logmag(mag)), "phase": r(np.angle(z)),
        "re": r(z.real), "im": r(z.imag),
        "peaks": data.peaks(mag, freqs),
        "shifts": [round(float(x), 5) for x in data.envelope(s)],
        "domain": data.domain(v),
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
