"""Mask -> PNG rendering.

Masks are soft probabilities, not binary, so both colormaps are continuous: a
two-tone render would misrepresent the data. Domains are FIXED (pred [0,1], diff
[-1,+1]) rather than per-cell autoscaled, otherwise cells would not be comparable
across the table -- which is the whole point of the tool.

Palettes come from the dataviz skill's reference instance and were validated with a
port of its checker: the diverging poles measure CVD deltaE 19.2 (target 8.0) in both
light and dark mode. Sequential is a single blue hue, light->dark; never a rainbow.

Every `sid` argument in this module is a ROW, not a sample id -- see SPEC.md.
"""

import io
from functools import lru_cache

import numpy as np
from PIL import Image

from viz import config

# Sequential ramps, light->dark, one hue each (never a rainbow). The lightest step is
# allowed to recede toward the surface -- that is the intended reading of "near zero".
# A prediction is drawn on the blue ramp and a ground-truth mask on the green one, so a
# mask carries the same identity whether it is shown alone or overlaid.
# Both ramps start at white so an empty cell reads as nothing at all, and darken to the
# hue that identifies the layer.
# Starts at a real blue, not white. The two palest stops (#ffffff, #eaf2fd) sat within a
# few RGB units of the backdrop, so the bottom of the ramp was invisible no matter how
# opaque it was drawn -- which is exactly where an under-painting model puts its evidence.
SEQ_HEX = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
           "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
           "#104281", "#0d366b"]

TRUE_SEQ_HEX = ["#c9e8d6", "#a9dbc0", "#86cda8", "#63bf90", "#45ad79",
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


# Display gamma. A gamma < 1 expands the low end of the ramp, so an undertrained run
# that tops out around 0.4 still shows structure instead of rendering in the palest few
# steps. It pays for that at the TOP: at 0.6, the 0.9-1.0 decile got 6.1% of the ramp
# while 0.0-0.1 got 25.1%, which is why a 0.99 cell and a 0.5 cell looked alike once the
# backdrop was composited under them (deltaE 0.68 between 0.99 and 1.0 -- far below the
# ~2.3 just-noticeable threshold -- against 27.9 across the whole 0.5-1.0 span).
#
# 0.85 keeps most of the low-end lift while leaving the top decile legible. This is a
# DISPLAY transform only: hover tooltips and every metric report the true values.
# Low gamma on purpose. The model under-paints, so the cells that matter for diagnosis sit
# near 0.1-0.3, and at gamma 0.85 those land on the near-white end of the ramp: a 0.24
# prediction rendered [155,195,243] against a [238,238,235] backdrop, barely distinguishable
# even at full opacity. Gamma 0.5 pushes them onto saturated colour instead. Raising alpha
# alone could not fix this -- the colour, not the opacity, was the binding constraint.
GAMMA = 0.5

# Opacity used to be the value itself, which made a weak-but-real cell effectively
# invisible. Floor it, so anything above ALPHA_EPS is visibly present and only true zeros
# stay clear.
ALPHA_FLOOR = 0.35
ALPHA_EPS = 0.02


def _alpha(t):
    return np.where(t > ALPHA_EPS, ALPHA_FLOOR + (1.0 - ALPHA_FLOOR) * t, 0.0)


def colorize(values: np.ndarray, mode: str, gamma: float = GAMMA,
             domain: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(H,W) values -> (rgb (H,W,3) uint8, alpha (H,W) float in [0,1]).

    Alpha encodes "how much signal is here", used by the no-background view so the
    mask silhouette floats free of a filled rectangle.

    `domain` rescales before the ramp is applied, for the relative/per-sample view. The
    default (None) is the fixed domain -- [0,1] for pred/truth, [-1,+1] for diff -- which
    is the only one under which two cells can be compared, so it stays the default.
    """
    if mode == "diff":
        # Gamma applied symmetrically about the neutral midpoint, so over- and
        # under-prediction brighten at the same rate.
        signed = np.clip(values, -1.0, 1.0)
        if domain is not None:
            # Symmetric about zero: the midpoint must stay "no difference", so only the
            # magnitude is rescaled. An arm is never remapped onto the opposite colour.
            span = max(abs(domain[0]), abs(domain[1]), 1e-6)
            signed = np.clip(signed / span, -1.0, 1.0)
        mag = np.abs(signed) ** gamma
        t = 0.5 + 0.5 * np.sign(signed) * mag
        alpha = _alpha(mag)
        lut = DIV_LUT
    else:
        v = np.clip(values, 0.0, 1.0)
        if domain is not None:
            lo, hi = domain
            # A flat mask has no range to stretch; leaving it at zero renders it empty
            # rather than amplifying floating-point noise into a full-contrast image.
            v = (v - lo) / (hi - lo) if hi - lo > 1e-6 else np.zeros_like(v)
            v = np.clip(v, 0.0, 1.0)
        t = v ** gamma
        alpha = _alpha(t)
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


@lru_cache(maxsize=1)
def aspect_by_box() -> dict:
    """{box_name: width/height} of the cropped frame each box's masks tile.

    Keyed on the BOX, not the sample: downsample_mask squashes every box into the same
    (out_h,out_w), so the aspect a grid must be drawn at is a property of the enclosure and
    is identical for every sample inside it. That also makes this one PNG header read per
    box rather than per sample -- `Image.open().size` parses the header without decoding,
    unlike _backdrop, which keeps whole frames in memory.

    Cached for the process lifetime, like the backdrops it mirrors: the sample set cannot
    change under a running server.
    """
    from viz.app import registry
    out: dict = {}
    for row in range(len(registry.gt)):
        # Rows, not sample ids: sample_dir takes a row. Passing ids read the wrong photos
        # on any dataset whose ids do not start at zero.
        box = registry.gt.meta[row].get("box")
        if box in out: continue
        p = registry.sample_dir(row) / registry.gt.layout.backdrop
        if not p.exists(): continue
        with Image.open(p) as im:
            w, h = im.size
        out[box] = w / h
    return out


def row_aspect(row: int) -> float:
    """Width/height of the frame THIS row's masks tile.

    Per row rather than one global number, because a dataset holding two boxes has no
    single answer: gastronorm is 1.204 and green-plastic 1.120, and drawing both at one
    aspect slides every prediction off the features it refers to in at least one of them.
    Falls back to the dataset-wide aspect when the row's box has no backdrop on disk.
    """
    from viz.app import registry
    a = aspect_by_box().get(registry.gt.meta[row].get("box"))
    return a if a is not None else scene_aspect()


@lru_cache(maxsize=1)
def scene_aspect() -> float:
    """Dataset-wide FALLBACK aspect, for callers with no row in hand.

    Correct only while the dataset holds one box; `row_aspect` is what layout should use.
    Kept because canvas_size can be called with neither a backdrop nor an explicit aspect,
    and /api/lut needs a value to seed the client before any sample has loaded.
    """
    from viz.app import registry
    for row in range(min(25, len(registry.gt))):
        im = _backdrop(row)
        if im is not None:
            return im.size[0] / im.size[1]
    return config.MASK_W / config.MASK_H


def canvas_size(h: int, w: int, backdrop: Image.Image | None = None,
                aspect: float | None = None) -> tuple[int, int]:
    """Pixel size to draw an (h,w) mask in, in (W,H) order.

    The mask grid is a uniform box-downsample of the whole cropped frame -- verified
    corr=0.9998 against 03_smask.npy at both 20x40 and 30x30 -- so the two cover exactly
    the same region and the SCENE defines the geometry. Sizing the canvas from the mask
    instead (w*up, h*up) forces the frame into the mask's aspect: 20x40 is 2.000 against
    the frame's 2.197 (9% horizontal squeeze) and 30x30 is 1.000 (crushed to a square).
    That is what made predictions sit off the scene.

    So the canvas keeps the SCENE's aspect and the MASK is stretched onto it --
    anisotropic, but only undoing the anisotropic binning, which lands every cell back on
    the pixels it averaged.

    Size follows the PHOTO, not the grid. The photo carries all the real detail and the
    mask is a coarse overlay drawn on top, so the mask is what gets upscaled (NEAREST,
    cells stay crisp) and the photo is never downsampled to meet it. Pinning height to
    h*UPSCALE instead rendered a 1337x1110 capture into 252px behind a 21x30 mask, which
    is the blur; and it made a coarse grid produce a SMALLER image than a fine one of the
    same scene. Without a photo there is nothing to preserve, so h*UPSCALE is the floor
    that still gives a bare mask a sensible pixel size.
    """
    if aspect is None:
        aspect = backdrop.size[0] / backdrop.size[1] if backdrop is not None else scene_aspect()
    height = h * config.UPSCALE
    if backdrop is not None:
        height = min(max(height, backdrop.size[1]), config.BACKDROP_MAX_PX)
    return max(1, round(height * aspect)), height


def render_both(pred: np.ndarray, truth: np.ndarray, background: bool,
                backdrop: Image.Image | None = None, relative: bool = False) -> bytes:
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
        # Each layer scales to its OWN range: truth and prediction are separate colours
        # here, so a shared domain would let one layer's spread dictate the other's.
        _, a = colorize(values, "pred",
                        domain=domain_of(values, "pred") if relative else None)
        alpha = Image.fromarray((a * gain * 255).round().astype(np.uint8), mode="L").resize(size, Image.NEAREST)
        layer = Image.new("RGB", size, tuple(int(x) for x in _hex(hexcolor)))
        base = Image.composite(layer, base, alpha)

    buf = io.BytesIO()
    if backdrop is not None:
        base.save(buf, format="JPEG", quality=82, optimize=True)
    else:
        base.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def domain_of(values: np.ndarray, mode: str) -> tuple[float, float]:
    """Per-sample colour domain for the relative view: this mask's own range."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return (0.0, 1.0)
    return (float(v.min()), float(v.max()))


def render_mask(values: np.ndarray, mode: str, background: bool,
                backdrop: Image.Image | None = None, relative: bool = False) -> bytes:
    """Colourize a (20,40) mask. With a backdrop, the mask is composited over the
    overhead photo so the prediction can be read against the real scene; without one it
    is drawn alone. The mask is always present -- `background` only decides whether the
    photo is behind it."""
    rgb, alpha = colorize(values, mode,
                          domain=domain_of(values, mode) if relative else None)
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
    from viz.app import registry
    p = registry.sample_dir(sid) / registry.gt.layout.backdrop
    if not p.exists():
        return None
    im = Image.open(p)
    im.load()
    return im.convert("RGB")


@lru_cache(maxsize=200_000)
def _cached(kind: str, run: str, sid: int, mode: str, background: bool, epoch: int,
            relative: bool = False, shape: tuple[int, int] | None = None) -> bytes:
    from viz.app import registry  # set at startup
    bd = _backdrop(sid) if background else None
    if kind == "gt":
        # The ground-truth column follows the columns beside it, so it can be asked for a
        # grid other than the primary one. `sid` is a ROW here (callers resolve the id
        # first) and masks_at is row-aligned with gt.sample_ids, so this needs no
        # re-indexing. A missing size raises rather than quietly serving the primary
        # shape -- a silent fallback would put an unrelated resolution beside the runs.
        m = registry.gt.masks if shape is None else registry.gt.masks_at(shape)
        if m is None:
            raise KeyError(shape)
        return render_mask(m[sid], "truth", background, bd, relative)
    rd = registry.run(run)
    row = rd.row_of[sid]
    values = rd.masks[row]
    # Diff and overlay are elementwise, so the target must be at THIS run's grid -- the
    # table mixes resolutions and gt.masks holds only the primary one.
    truth = registry.gt.masks_at(rd.shape)
    if truth is None:
        return render_mask(values, mode if mode != "diff" else "pred", background, bd,
                           relative)
    if mode in ("overlay", "stacked"):
        return render_both(values, truth[sid], background, bd, relative)
    if mode == "diff":
        values = values - truth[sid]
    return render_mask(values, mode, background, bd, relative)


def cached_mask(kind: str, run: str, sid: int, mode: str, background: bool,
                relative: bool = False, shape: tuple[int, int] | None = None) -> bytes:
    # The run's loaded epoch is part of the cache key: a run that is still training gets
    # reloaded with new predictions, and a fixed key would keep serving the old render.
    # Ground truth never changes within a process, so it keeps epoch 0.
    from viz.app import registry
    epoch = registry.run(run).epoch if kind != "gt" else 0
    return _cached(kind, run, sid, mode, background, epoch, relative, shape)


def media_type(data: bytes) -> str:
    """Composited cells are JPEG, bare masks are PNG; read it off the magic bytes so
    callers never have to track which branch produced the image."""
    return "image/jpeg" if data[:2] == b"\xff\xd8" else "image/png"


@lru_cache(maxsize=8)
def cached_colorbar(mode: str) -> bytes:
    return colorbar(mode)


@lru_cache(maxsize=2048)
def cached_backdrop(sid: int) -> bytes | None:
    """The overhead frame, encoded once at full quality, for use behind canvas cells."""
    im = _backdrop(sid)
    if im is None:
        return None
    # NATIVE resolution, only shrunk if it exceeds the cap. This photo is the sharpest
    # thing on screen and the mask is a coarse grid drawn over it -- so the MASK is the
    # layer that gets upscaled (NEAREST, keeping cells crisp), never the photo downscaled
    # to meet it. Sizing this from the mask grid shipped a 1337x1110 capture at 434x360,
    # which read as badly blurred behind an otherwise sharp cell.
    if im.height > config.BACKDROP_MAX_PX:
        h = config.BACKDROP_MAX_PX
        im = im.resize((max(1, round(h * im.width / im.height)), h), Image.LANCZOS)
    # Higher quality than the composited cells: this one is fetched ONCE per sample and
    # reused across every run, epoch and repaint, so the extra bytes are paid once while
    # the sharpness is what the whole view is read against.
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()
