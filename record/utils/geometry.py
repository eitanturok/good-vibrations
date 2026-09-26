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


def compute_roi_grid(row_clicks: list[tuple[float, float]], col_clicks: list[tuple[float, float]],
                      roi_width: int, roi_height: int, sensor_w: int = 1920, sensor_h: int = 1080):
    """ROI grid from N_rows horizontal-line clicks (only y used) and N_cols vertical-line
    clicks (only x used), in any click order: rows sort top-to-bottom, columns left-to-right,
    each ROI centered on its click and clamped inside the sensor.

    Returns (rois, row_positions, offset_x) -- the camera reads only the selected row bands,
    stacked edge-to-edge, over the column window starting at offset_x:
    - rois: (x, y, w, h) per ROI, row-major, in the camera's OUTPUT frame (what the saved raw
      frames and post-processing index): x = sensor x - offset_x, y = band * roi_height
    - row_positions: the (even) sensor row each band starts at -- the camera addresses rows in pairs
    - offset_x: the sensor column the output frame starts at"""
    xs = sorted(int(round(min(max(x - roi_width / 2, 0), sensor_w - roi_width))) for x, _ in col_clicks)
    row_positions = sorted(2 * int(round(min(max(y - roi_height / 2, 0), sensor_h - roi_height) / 2)) for _, y in row_clicks)
    offset_x, _ = readout_columns(xs[0], xs[-1] + roi_width, sensor_w)
    rois = [(x - offset_x, band * roi_height, roi_width, roi_height) for band in range(len(row_positions)) for x in xs]
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


def resize_roi_grid(rois, row_positions, offset_x: int, roi_size: int, new_size: int, sensor_w: int = 1920, sensor_h: int = 1080):
    """The same grid at a new ROI size: every ROI stays centered where it was (i.e. on the
    lines you clicked). Returns (rois, row_positions, offset_x) like compute_roi_grid."""
    col_centers = sorted({x + offset_x + roi_size / 2 for x, y, w, h in rois})
    row_clicks = [(0, y0 + roi_size / 2) for y0 in row_positions]
    return compute_roi_grid(row_clicks, [(x, 0) for x in col_centers], new_size, new_size, sensor_w, sensor_h)
