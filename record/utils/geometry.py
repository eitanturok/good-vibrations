"""Pure coordinate math for the GUI's live-preview panels: zoom/pan (ported from
matan_main_capture.ipynb's _PreviewCamera) and the speckle ROI grid (ported from
src/record.ipynb's two-pass row/column click calibration, cells 34-50). No Tkinter, no
hardware -- these are unit-testable on their own (record/tests/test_geometry.py)."""

import math


def compute_fit_scale(canvas_w: float, canvas_h: float, sensor_w: float, sensor_h: float) -> float:
    """Scale that fits the whole sensor image inside the canvas, preserving aspect ratio."""
    return min(canvas_w / sensor_w, canvas_h / sensor_h)


def clamp_view_center(view_center: tuple[float, float], canvas_w: float, canvas_h: float,
                       sensor_w: float, sensor_h: float, scale: float) -> tuple[float, float]:
    """Keep the view center far enough from the sensor's edges that the canvas never
    shows past it; recenters on that axis once the whole sensor already fits."""
    cx, cy = view_center
    if sensor_w * scale <= canvas_w:
        cx = sensor_w / 2.0
    else:
        half = canvas_w / (2.0 * scale)
        cx = max(half, min(sensor_w - half, cx))
    if sensor_h * scale <= canvas_h:
        cy = sensor_h / 2.0
    else:
        half = canvas_h / (2.0 * scale)
        cy = max(half, min(sensor_h - half, cy))
    return (cx, cy)


def compute_transform(view_center: tuple[float, float], view_zoom: float, canvas_w: float, canvas_h: float,
                       sensor_w: float, sensor_h: float) -> tuple[float, float, float, tuple[float, float]]:
    """The affine map from sensor pixel coords to canvas coords: canvas_x = ox + sensor_x *
    scale (and the same `scale` for y -- always uniform). Returns (ox, oy, scale,
    clamped_view_center) -- the clamped center is what the caller should persist back into
    its own view_center state for the next tick."""
    scale = compute_fit_scale(canvas_w, canvas_h, sensor_w, sensor_h) * view_zoom
    cx, cy = clamp_view_center(view_center, canvas_w, canvas_h, sensor_w, sensor_h, scale)
    ox = canvas_w / 2.0 - cx * scale
    oy = canvas_h / 2.0 - cy * scale
    return ox, oy, scale, (cx, cy)


def canvas_to_sensor_coords(canvas_x: float, canvas_y: float, ox: float, oy: float, scale: float,
                             sensor_w: float, sensor_h: float, clamp: bool = True) -> tuple[float, float]:
    """Inverse of compute_transform's affine map: a canvas click/hover position back into
    sensor pixel coordinates. clamp=True (default) pins the result inside the sensor's
    bounds, for e.g. ROI click calibration where an out-of-frame click should still land
    somewhere sane; clamp=False is for hit-testing whether a point was inside the frame."""
    x, y = (canvas_x - ox) / scale, (canvas_y - oy) / scale
    if clamp:
        x, y = max(0.0, min(x, sensor_w)), max(0.0, min(y, sensor_h))
    return (x, y)


def zoom_at_point(old_zoom: float, steps: float, zoom_step: float, zoom_min: float, zoom_max: float) -> float:
    """New zoom level after `steps` wheel notches (positive = zoom in), geometric so each
    notch feels like the same relative zoom regardless of current level."""
    return max(zoom_min, min(zoom_max, old_zoom * (zoom_step ** steps)))


def readout_columns(x_start: int, x_end: int, sensor_w: int = 1920, align: int = 16, min_width: int = 128) -> tuple[int, int]:
    """The camera's horizontal readout window (OffsetX, Width) covering sensor columns
    [x_start, x_end): both multiples of 16, Width >= 128, OffsetX + Width <= sensor_w.
    Rounds OUT (offset down, width up) so no ROI is ever cut off."""
    offset = (x_start // align) * align
    width = max(min_width, -(-(x_end - offset) // align) * align)
    offset = max(0, min(offset, sensor_w - width))
    return offset, min(width, sensor_w - offset)


def compute_roi_grid(rows: list[int], cols: list[int], roi_width: int, roi_height: int,
                     crop: tuple[int, int] = (0, 1920), sensor_w: int = 1920, sensor_h: int = 1080):
    """src/record.ipynb's ROI math: one roi_width x roi_height ROI centered on every crossing of a
    horizontal line (y in `rows`) and a vertical line (x in `cols`), all in sensor px:
    - a row band starts at y - roi_height // 2, rounded up to even (the camera reads rows in pairs)
    - the camera reads the columns between the crop edges, on its 16 px grid, from offset_x
    - an ROI starts at x - roi_width // 2, i.e. x - roi_width // 2 - offset_x in the output frame
    An ROI that doesn't fit (off the sensor, or outside the crop) is an error, never moved.

    Returns (rois, row_positions, offset_x) -- the camera sends only the row bands, stacked:
    - rois: (x, y, w, h) per ROI, row-major (rows top-to-bottom, cols left-to-right), in the
      camera's OUTPUT frame: y = band * roi_height
    - row_positions: the (even) sensor row each band starts at
    - offset_x: the sensor column the output frame starts at"""
    row_positions = [y - roi_height // 2 + (y - roi_height // 2) % 2 for y in sorted(rows)]
    offset_x, width = readout_columns(*crop, sensor_w)
    xs = [x - roi_width // 2 - offset_x for x in sorted(cols)]
    if any(y0 < 0 or y0 + roi_height > sensor_h for y0 in row_positions):
        raise ValueError(f"a {roi_height} px tall ROI on rows {sorted(rows)} falls off the {sensor_h} px sensor")
    if any(x < 0 or x + roi_width > width for x in xs):
        raise ValueError(f"a {roi_width} px wide ROI on cols {sorted(cols)} falls outside the columns the camera reads "
                         f"[{offset_x}, {offset_x + width}) -- widen the crop")
    rois = [(x, band * roi_height, roi_width, roi_height) for band in range(len(row_positions)) for x in xs]
    return rois, row_positions, offset_x


def sensor_rois(rois, row_positions, offset_x: int, roi_height: int):
    """Output-frame rois -> the same boxes in full-sensor coordinates."""
    return [(x + offset_x, row_positions[y // roi_height], w, h) for x, y, w, h in rois]


def compose_sensor_view(frame, rois, row_positions, offset_x: int, roi_height: int, background=None, sensor_shape=(1080, 1920)):
    """A full-sensor image from one output frame: each row band pasted back at its sensor
    position (on `background`, e.g. a dimmed wide-open snapshot, or black)."""
    import numpy as np
    out = np.zeros(sensor_shape, dtype=frame.dtype) if background is None else background.copy()
    w = min(frame.shape[1], sensor_shape[1] - offset_x)
    for band, y0 in enumerate(row_positions):
        out[y0:y0 + roi_height, offset_x:offset_x + w] = frame[band * roi_height:(band + 1) * roi_height, :w]
    return out


def crop_rois(sensor_frame, rois, row_positions, offset_x: int, roi_height: int):
    """Inverse of compose_sensor_view: a full-sensor frame cut down to what the cropped camera
    sends -- its row bands stacked edge-to-edge, columns from offset_x."""
    import numpy as np
    width = max(x + w for x, y, w, h in rois)
    return np.concatenate([sensor_frame[y0:y0 + roi_height, offset_x:offset_x + width] for y0 in row_positions])


def roi_mosaic(frame, rois, n_cols: int):
    """The zoomed view: each ROI cell cut out of `frame` and tiled in its grid position (row-major,
    touching -- one continuous image; the tile boxes drawn over it are the grid lines).
    Returns (mosaic, tile boxes (x, y, w, h))."""
    import numpy as np
    w, h = rois[0][2], rois[0][3]
    n_rows = -(-len(rois) // n_cols)
    mosaic = np.zeros((n_rows * h, n_cols * w), dtype=frame.dtype)
    tiles = []
    for i, (x, y, _, _) in enumerate(rois):
        tx, ty = (i % n_cols) * w, (i // n_cols) * h
        mosaic[ty:ty + h, tx:tx + w] = frame[y:y + h, x:x + w]
        tiles.append((tx, ty, w, h))
    return mosaic, tiles
