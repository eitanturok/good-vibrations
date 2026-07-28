"""Mask -> PNG rendering.

Masks are soft probabilities, not binary, so both colormaps are continuous: a
two-tone render would misrepresent the data. Domains are FIXED (pred [0,1], diff
[-1,+1]) rather than per-cell autoscaled, otherwise cells would not be comparable
across the table -- which is the whole point of the tool.

Palettes come from the dataviz skill's reference instance and were validated with a
port of its checker: the diverging poles measure CVD deltaE 19.2 (target 8.0) in both
light and dark mode. Sequential is a single blue hue, light->dark; never a rainbow.
"""

import io
from functools import lru_cache

import numpy as np
from PIL import Image

from viz2 import config

# Sequential blue ramp, 100->700 (magnitude). The lightest step is allowed to recede
# toward the surface -- that is the intended reading of "near zero".
SEQ_HEX = ["#eaf2fd", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
           "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
           "#104281", "#0d366b"]

# Diverging: blue (under-prediction) <-> gray midpoint <-> red (over-prediction).
# Gray, not white, at the midpoint so zero-diff cells still read as data rather than
# dropping out to the page. blue<->aqua was rejected upstream: both cool.
DIV_NEG_HEX = "#256abf"   # truth > pred  (model missed mass)
DIV_MID_HEX = "#f0efec"
DIV_POS_HEX = "#e34948"   # pred > truth  (model hallucinated mass)


def _hex(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def _ramp(hexes: list[str], n: int = 256) -> np.ndarray:
    """Piecewise-linear interpolation through control points -> (n,3) uint8 LUT."""
    stops = np.stack([_hex(h) for h in hexes])
    x = np.linspace(0, 1, len(stops))
    xi = np.linspace(0, 1, n)
    return np.stack([np.interp(xi, x, stops[:, c]) for c in range(3)], axis=-1).astype(np.uint8)


SEQ_LUT = _ramp(SEQ_HEX)
DIV_LUT = _ramp([DIV_NEG_HEX, DIV_MID_HEX, DIV_POS_HEX])


def colorize(values: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(H,W) values -> (rgb (H,W,3) uint8, alpha (H,W) float in [0,1]).

    Alpha encodes "how much signal is here", used by the no-background view so the
    mask silhouette floats free of a filled rectangle.
    """
    if mode == "diff":
        t = np.clip((values + 1.0) / 2.0, 0.0, 1.0)      # [-1,1] -> [0,1], fixed domain
        alpha = np.abs(np.clip(values, -1.0, 1.0))
        lut = DIV_LUT
    else:
        t = np.clip(values, 0.0, 1.0)
        alpha = t
        lut = SEQ_LUT
    rgb = lut[(t * (len(lut) - 1)).round().astype(np.int32)]
    return rgb, alpha


def _png(rgb: np.ndarray, alpha: np.ndarray | None, upscale: int) -> bytes:
    if alpha is None:
        img = Image.fromarray(rgb, mode="RGB")
    else:
        a = (np.clip(alpha, 0, 1) * 255).round().astype(np.uint8)
        img = Image.fromarray(np.dstack([rgb, a]), mode="RGBA")
    # NEAREST is mandatory: interpolation would invent values between grid cells and
    # break the correspondence the hover tooltip relies on.
    img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# Floor on the mask overlay so a confident prediction still reads as solid colour over a
# photo, while near-zero cells stay transparent enough to see the scene through.
OVERLAY_GAIN = 0.85


def render_mask(values: np.ndarray, mode: str, background: bool,
                backdrop: Image.Image | None = None) -> bytes:
    """Colourize a (20,40) mask. With a backdrop, the mask is composited over the
    overhead photo so the prediction can be read against the real scene; without one it
    is drawn alone. The mask is always present -- `background` only decides whether the
    photo is behind it."""
    rgb, alpha = colorize(values, mode)
    if backdrop is None:
        return _png(rgb, None if background else alpha, config.UPSCALE)

    up = config.UPSCALE
    h, w = values.shape
    # The mask is upscaled with NEAREST (crisp grid cells); the photo is a continuous
    # image and is resampled smoothly to the same size.
    fg = Image.fromarray(rgb, mode="RGB").resize((w * up, h * up), Image.NEAREST)
    a = Image.fromarray((np.clip(alpha, 0, 1) * OVERLAY_GAIN * 255).round().astype(np.uint8),
                        mode="L").resize((w * up, h * up), Image.NEAREST)
    bg = backdrop.convert("RGB").resize((w * up, h * up), Image.LANCZOS)
    out = Image.composite(fg, bg, a)
    buf = io.BytesIO()
    # A composited cell is a photograph, so JPEG is ~10x smaller than lossless PNG at
    # indistinguishable quality -- worth it when a scroll can touch hundreds of cells.
    out.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()


def colorbar(mode: str, width: int = 256, height: int = 14) -> bytes:
    """Legend strip. Diff spans the full [-1,+1] domain; pred spans [0,1]."""
    t = np.linspace(0.0, 1.0, width)
    values = (t * 2 - 1) if mode == "diff" else t
    rgb, _ = colorize(values[None, :], mode)
    return _png(np.repeat(rgb, height, axis=0), None, 1)


# ***** cache *****
#
# ~1.4 KB per PNG and ~1.4 ms to render, so an in-process LRU plus immutable browser
# caching keeps repeated scrolling free. Deliberately not written to disk: that would
# need invalidation whenever a run resumes training and writes a new epoch.

@lru_cache(maxsize=2048)
def _backdrop(sid: int) -> Image.Image | None:
    """The cropped overhead frame the masks are aligned to, kept decoded so repeated
    composites for the same sample don't re-read the file."""
    from viz2.app import registry
    p = registry.sample_dir(sid) / config.BACKDROP_REL
    if not p.exists():
        return None
    im = Image.open(p)
    im.load()
    return im.convert("RGB")


@lru_cache(maxsize=200_000)
def _cached(kind: str, run: str, sid: int, mode: str, background: bool, epoch: int) -> bytes:
    from viz2.app import registry  # set at startup
    bd = _backdrop(sid) if background else None
    if kind == "gt":
        return render_mask(registry.gt.masks[sid], "pred", background, bd)
    rd = registry.run(run)
    row = rd.row_of[sid]
    values = rd.masks[row]
    if mode == "diff":
        values = values - registry.gt.masks[sid]
    return render_mask(values, mode, background, bd)


def cached_mask(kind: str, run: str, sid: int, mode: str, background: bool) -> bytes:
    # The run's loaded epoch is part of the cache key: a run that is still training gets
    # reloaded with new predictions, and a fixed key would keep serving the old render.
    from viz2.app import registry
    epoch = registry.run(run).epoch if kind != "gt" else 0
    return _cached(kind, run, sid, mode, background, epoch)


def media_type(data: bytes) -> str:
    """Composited cells are JPEG, bare masks are PNG; read it off the magic bytes so
    callers never have to track which branch produced the image."""
    return "image/jpeg" if data[:2] == b"\xff\xd8" else "image/png"


@lru_cache(maxsize=8)
def cached_colorbar(mode: str) -> bytes:
    return colorbar(mode)
