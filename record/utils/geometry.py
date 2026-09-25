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


def compute_roi_grid(row_clicks: list[tuple[float, float]], col_clicks: list[tuple[float, float]],
                      roi_width: int, roi_height: int) -> list[tuple[int, int, int, int]]:
    """Build the final (x, y, w, h) ROI boxes from N_rows horizontal-line click points and
    N_cols vertical-line click points -- ports src/record.ipynb cells 36-50's two-pass
    calibration math (row-major order, matching cell 50 exactly). Only each row click's y
    and each column click's x are used; row bands are laid out edge-to-edge starting at 0
    (row i spans y in [i*roi_height, (i+1)*roi_height)) -- the click y-positions there only
    ever fed the *hardware* row-index registers (set_rows(), a separate concern handled by
    MikrotronCamera, not this pure box layout)."""
    rois = []
    for row in range(len(row_clicks)):
        y = row * roi_height
        for x_click, _y_click in col_clicks:
            x = int(round(x_click - roi_width / 2))
            rois.append((x, y, roi_width, roi_height))
    return rois
