"""Array -> PNG. Only the laser x freq heatmaps need this: 100x1235 cells is ~1.5 MB as
JSON but ~40 KB as a PNG the browser decodes in hardware. Everything smaller ships as JSON.

Two palettes, because the quantities differ in kind. Magnitude is one-sided, so it gets
viz/render.py's sequential blue (shared with viz, so both dashboards read as one product).
Phase, real, imag and shifts are SIGNED -- on a sequential ramp zero lands mid-palette and
the sign is unreadable, so they get a diverging ramp with a neutral middle, scaled
symmetrically about zero.
"""

import io

import numpy as np
from PIL import Image

SEQ = ["#ffffff", "#eaf2fd", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
       "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
       "#104281", "#0d366b"]

DIV = ["#8c3b12", "#b8571f", "#d98635", "#eeb277", "#f7dcc0", "#f5f5f3", "#cfe0f2",
       "#93bce4", "#5595d4", "#2a70bb", "#124b8e"]


def _lut(hexes, n=256):
    stops = np.array([[int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)] for h in hexes])
    xi, x = np.linspace(0, 1, n), np.linspace(0, 1, len(stops))
    return np.stack([np.interp(xi, x, stops[:, c]) for c in range(3)], -1).astype(np.uint8)


LUTS = {"seq": _lut(SEQ), "div": _lut(DIV)}


def heat(v, lo, hi, lut="seq"):
    t = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
    img = Image.fromarray(LUTS[lut][(t * 255).astype(np.uint8)])
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def mask_png(mask, w=900):
    """The mask at its native full-frame geometry, so it aligns with the overhead photo.
    Not cropped: the framing itself is information."""
    m = np.asarray(mask) > 0.5
    img = Image.fromarray(np.where(m[..., None], np.array([24, 160, 100], np.uint8),
                                   np.array([238, 238, 235], np.uint8)))
    if img.width > w:
        img = img.resize((w, round(img.height * w / img.width)), Image.NEAREST)
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()


def masks_overlay(masks, colors, w=300):
    """Several segmentation masks in one image, each drawn in its probe's color.

    Fills are translucent and each mask also gets a hard outline, so two objects at the
    same place stay distinguishable instead of the last one hiding the rest.

    Always full frame, never cropped: the framing itself is what locates the object.
    """
    h, wd = masks[0].shape
    out = np.zeros((h, wd, 3), np.float32)
    cov = np.zeros((h, wd), np.float32)
    for m, c in zip(masks, colors):
        b = np.asarray(m) > 0.5
        rgb = np.array(c, np.float32)
        out[b] += rgb
        cov[b] += 1
        # 1px outline: the mask minus its erosion, done with plain shifts
        er = b.copy()
        for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            er &= np.roll(b, sh, axis=ax)
        out[b & ~er] = rgb
        cov[b & ~er] = 1
    img = np.full((h, wd, 3), 238, np.float32)
    hit = cov > 0
    img[hit] = out[hit] / cov[hit][:, None]
    im = Image.fromarray(img.astype(np.uint8))
    if im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.NEAREST)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def scene(photo, mask, w=900):
    """Photo with the segmentation mask as a green tint."""
    im = photo.convert("RGB")
    if im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    if mask is not None:
        m = Image.fromarray((mask > 0.5).astype(np.uint8) * 110).resize(im.size)
        im = Image.composite(Image.new("RGB", im.size, (24, 160, 100)), im, m)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=88)
    return b.getvalue()
