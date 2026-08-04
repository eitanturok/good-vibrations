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

# Sequential ramps, light->dark, one hue each (never a rainbow). The lightest step is
# allowed to recede toward the surface -- that is the intended reading of "near zero".
# A prediction is drawn on the blue ramp and a ground-truth mask on the green one, so a
# mask carries the same identity whether it is shown alone or overlaid.
# Both ramps start at white so an empty cell reads as nothing at all, and darken to the
# hue that identifies the layer.
SEQ_HEX = ["#ffffff", "#eaf2fd", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
           "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
           "#104281", "#0d366b"]

TRUE_SEQ_HEX = ["#ffffff", "#e7f4ec", "#c9e8d6", "#a9dbc0", "#86cda8", "#63bf90", "#45ad79",
                "#2f9a66", "#1f8757", "#17744a", "#0f6640", "#0d5c33", "#0a4d2b",
                "#083f23", "#06311b"]

# Diverging: green (missed truth) <-> gray midpoint <-> blue (excess prediction).
# Gray, not white, at the midpoint so zero-diff cells still read as data rather than
# dropping out to the page.
# One identity everywhere: BLUE is the prediction, GREEN is the ground truth.
#
# Prediction takes blue because it is what you look at all day -- the default view over
# a 1024-row table -- and a wall of red would read as alarm and clash with the red used
# for crashed runs and remove buttons. Ground truth is the rarer reference layer.
#
# The green is deliberately deeper than the "running" status badge (#1f7a4d, OKLab
# deltaE 9.7 away) so a ground-truth mask is never mistaken for a run-state indicator.
PRED_HEX = "#256abf"
TRUE_HEX = "#0d5c33"

# Diff arms inherit the same identity, so the views explain each other: blue means the
# prediction put mass here, green means ground-truth mass the model missed.
DIV_NEG_HEX = TRUE_HEX    # truth > pred  (model missed mass)
DIV_MID_HEX = "#f0efec"
DIV_POS_HEX = PRED_HEX    # pred > truth  (model hallucinated mass)


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
TRUE_SEQ_LUT = _ramp(TRUE_SEQ_HEX)
DIV_LUT = _ramp([DIV_NEG_HEX, DIV_MID_HEX, DIV_POS_HEX])


# Display gamma. Predictions from an undertrained run can top out around 0.4-0.7 while a
# converged run reaches 1.0, so on a linear [0,1] ramp the weak run renders almost
# entirely in the palest steps and its structure is invisible. Raising values to a power
# < 1 expands the low end where the data actually lives. This is a DISPLAY transform
# only: the domain stays fixed at [0,1] so cells remain comparable across runs, and
# hover tooltips and all metrics continue to report the true values.
GAMMA = 0.6


def colorize(values: np.ndarray, mode: str, gamma: float = GAMMA) -> tuple[np.ndarray, np.ndarray]:
    """(H,W) values -> (rgb (H,W,3) uint8, alpha (H,W) float in [0,1]).

    Alpha encodes "how much signal is here", used by the no-background view so the
    mask silhouette floats free of a filled rectangle.
    """
    if mode == "diff":
        # Gamma applied symmetrically about the neutral midpoint, so over- and
        # under-prediction brighten at the same rate. Domain stays fixed at [-1,+1].
        signed = np.clip(values, -1.0, 1.0)
        mag = np.abs(signed) ** gamma
        t = 0.5 + 0.5 * np.sign(signed) * mag
        alpha = mag
        lut = DIV_LUT
    else:
        t = np.clip(values, 0.0, 1.0) ** gamma
        alpha = t
        # "truth" draws the ground-truth ramp so a GT mask is green whether it is shown
        # on its own or as the underlay in the overlay view.
        lut = TRUE_SEQ_LUT if mode == "truth" else SEQ_LUT
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


def canvas_size(h: int, w: int, backdrop: Image.Image | None) -> tuple[int, int]:
    """Pixel size to composite an (h,w) mask over `backdrop`, in (W,H) order.

    The mask grid is a uniform box-downsample of the whole cropped frame -- verified
    corr=0.9998 against 03_smask.npy at both 20x40 and 30x30 -- so the two cover exactly
    the same region and the PHOTO defines the geometry. Sizing the canvas from the mask
    instead (w*up, h*up) forces the photo into the mask's aspect: 20x40 is 2.000 against
    the frame's 2.197 (9% horizontal squeeze) and 30x30 is 1.000 (photo crushed to a
    square). That is what made predictions sit off the scene.

    So the canvas keeps the backdrop's own aspect and the MASK is stretched onto it --
    anisotropic, but only undoing the anisotropic binning, which lands every cell back on
    the pixels it averaged. Height is pinned to h*up so cell detail is preserved, and a
    mask with no backdrop keeps the old square-cell geometry.
    """
    if backdrop is None:
        return w * config.UPSCALE, h * config.UPSCALE
    bw, bh = backdrop.size
    height = h * config.UPSCALE
    return max(1, round(height * bw / bh)), height


def render_both(pred: np.ndarray, truth: np.ndarray, background: bool,
                backdrop: Image.Image | None = None) -> bytes:
    """Ground truth and prediction in one panel: green truth underneath, blue prediction
    over it, so agreement reads as overlap and each error type stays identifiable.

    Drawn as two successive alpha composites rather than by mixing colours, because a
    blend would put purple where the two agree -- a third hue that means neither.
    """
    h, w = pred.shape
    size = canvas_size(h, w, backdrop)

    if backdrop is not None:
        base = backdrop.convert("RGB").resize(size, Image.LANCZOS)
    else:
        base = Image.new("RGB", size, (255, 255, 255))

    for values, hexcolor, gain in ((truth, TRUE_HEX, 0.72), (pred, PRED_HEX, 0.82)):
        a = np.clip(values, 0.0, 1.0) ** GAMMA
        alpha = Image.fromarray((a * gain * 255).round().astype(np.uint8), mode="L").resize(size, Image.NEAREST)
        layer = Image.new("RGB", size, tuple(int(x) for x in _hex(hexcolor)))
        base = Image.composite(layer, base, alpha)

    buf = io.BytesIO()
    if backdrop is not None:
        base.save(buf, format="JPEG", quality=82, optimize=True)
    else:
        base.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_mask(values: np.ndarray, mode: str, background: bool,
                backdrop: Image.Image | None = None) -> bytes:
    """Colourize a (20,40) mask. With a backdrop, the mask is composited over the
    overhead photo so the prediction can be read against the real scene; without one it
    is drawn alone. The mask is always present -- `background` only decides whether the
    photo is behind it."""
    rgb, alpha = colorize(values, mode)
    if backdrop is None:
        return _png(rgb, None if background else alpha, config.UPSCALE)

    h, w = values.shape
    size = canvas_size(h, w, backdrop)
    # The mask is upscaled with NEAREST (crisp grid cells); the photo is a continuous
    # image and is resampled smoothly to the same size.
    fg = Image.fromarray(rgb, mode="RGB").resize(size, Image.NEAREST)
    a = Image.fromarray((np.clip(alpha, 0, 1) * OVERLAY_GAIN * 255).round().astype(np.uint8),
                        mode="L").resize(size, Image.NEAREST)
    bg = backdrop.convert("RGB").resize(size, Image.LANCZOS)
    out = Image.composite(fg, bg, a)
    buf = io.BytesIO()
    # A composited cell is a photograph, so JPEG is ~10x smaller than lossless PNG at
    # indistinguishable quality -- worth it when a scroll can touch hundreds of cells.
    out.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()


def colorbar(mode: str, width: int = 256, height: int = 14) -> bytes:
    """Legend strip. Diff spans the full [-1,+1] domain; pred/truth span [0,1]."""
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
    p = registry.sample_dir(sid) / registry.gt.layout.backdrop
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
        return render_mask(registry.gt.masks[sid], "truth", background, bd)
    rd = registry.run(run)
    row = rd.row_of[sid]
    values = rd.masks[row]
    if mode in ("overlay", "stacked"):
        return render_both(values, registry.gt.masks[sid], background, bd)
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


@lru_cache(maxsize=2048)
def cached_backdrop(sid: int) -> bytes | None:
    """The overhead frame encoded once at cell size, for use behind canvas cells."""
    im = _backdrop(sid)
    if im is None:
        return None
    # Same canvas as the composited cells: this is drawn behind mask layers, so sizing it
    # any other way would reintroduce the misalignment at the boundary between the two.
    im = im.resize(canvas_size(config.MASK_H, config.MASK_W, im), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()
