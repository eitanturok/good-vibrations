"""Array -> PNG. Only the laser x freq heatmaps need this: 100x1235 cells is ~1.5 MB as
JSON but ~40 KB as a PNG the browser decodes in hardware. Everything smaller ships as JSON.

Two palettes, because the quantities differ in kind. Magnitude is one-sided, so it gets
viz/render.py's sequential blue (shared with viz, so both dashboards read as one product).
Phase, real, imag and shifts are SIGNED -- on a sequential ramp zero lands mid-palette and
the sign is unreadable, so they get a diverging ramp with a neutral middle, scaled
symmetrically about zero.
"""

import io

import cv2
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


def mask_png(mask, w=900, photo=None):
    """The segmentation alone, at its native full-frame geometry so it aligns with the
    overhead: a flat grey field (as in the workbench mask composite) with the segmented
    region a solid green and a darker 1px outline. `photo` is accepted but unused."""
    m = np.asarray(mask) > 0.5
    h, wd = m.shape
    img = np.full((h, wd, 3), (238, 238, 235), np.uint8)
    img[m] = (24, 160, 100)
    er = m.copy()
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        er &= np.roll(m, sh, axis=ax)
    img[m & ~er] = (13, 92, 56)
    out = Image.fromarray(img)
    if out.width > w:
        out = out.resize((w, round(out.height * w / out.width)), Image.NEAREST)
    b = io.BytesIO()
    out.save(b, "PNG")
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


OBJ_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442"]  # Okabe-Ito


def objmasks_png(masks, sel=-1, w=640):
    """Each object filled in its own colour on a transparent field (the page sets the
    background). sel >= 0 fades every other object."""
    h, wd = masks[0].shape
    ow = min(w, wd)
    oh = round(h * ow / wd)
    img = np.zeros((oh, ow, 4), np.uint8)
    for i, m in enumerate(masks):
        # INTER_AREA + a low threshold, not NEAREST, so a small object never drops out
        hit = cv2.resize(m.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA) > 0.2
        c = OBJ_COLORS[i % len(OBJ_COLORS)]
        img[hit] = [int(c[k:k + 2], 16) for k in (1, 3, 5)] + [255 if sel < 0 or i == sel else 50]
    b = io.BytesIO()
    Image.fromarray(img).save(b, "PNG")
    return b.getvalue()


def _jpeg(im, quality=78):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=quality)
    return b.getvalue()


SEG = (40, 220, 100)   # the segmentation colour; app.js/style.css --seg must match


def thumb(path, masks=(), seg=False, box=(160, 120), sel=-1):
    """A small JPEG for the step-1 pickers and THE sample photo (gallery, viewer, sidebar).

    Each mask's outline is traced in SEG. seg=True is the segmentation view: the photo goes
    gray so the objects stand out, and their region is also filled with SEG (the
    centre-of-mass X's are drawn over it client-side). masks is the combined smask, or one
    per object; sel >= 0 fades every object's outline/fill but that one's (the viewer's
    selected table row), the photo itself untouched.
    """
    im = Image.open(path).convert("RGB")
    im.thumbnail(box)
    arr = np.array(im)
    if seg:                    # gray, and flattened toward mid-gray
        g = arr @ np.array([0.299, 0.587, 0.114])
        arr = np.repeat((0.55 * g + 0.45 * 150)[..., None], 3, axis=2).astype(np.uint8)
    for i, mask in enumerate(masks):
        # Resize the mask to the THUMBNAIL's resolution (NEAREST: a small object is only a
        # handful of pixels here, and a smooth resample shrinks it toward nothing) and
        # draw at that scale -- a contour traced at full res and then shrunk gets lost.
        m = np.asarray(Image.fromarray((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
                       .resize(im.size, Image.NEAREST)) > 127
        if not m.any():
            continue
        top = arr.copy()
        if seg:
            top[m] = (0.35 * np.array(SEG) + 0.65 * top[m]).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(top, contours, -1, SEG, 2)
        a = 1.0 if sel < 0 or i == sel else 0.2
        arr = top if a == 1 else (a * top + (1 - a) * arr).astype(np.uint8)
    return _jpeg(Image.fromarray(arr))


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
