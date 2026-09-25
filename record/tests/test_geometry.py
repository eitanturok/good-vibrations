import pytest

from record.utils.geometry import (
    compute_fit_scale, clamp_view_center, compute_transform, canvas_to_sensor_coords,
    zoom_at_point, compute_roi_grid,
)


def test_compute_fit_scale_fits_the_smaller_dimension():
    # sensor 1920x1080 into a 960x540 canvas -> exactly 0.5 both ways
    assert compute_fit_scale(960, 540, 1920, 1080) == pytest.approx(0.5)
    # a taller/narrower canvas is width-bound
    assert compute_fit_scale(100, 1000, 1920, 1080) == pytest.approx(100 / 1920)


def test_clamp_view_center_recenters_when_whole_sensor_fits():
    cx, cy = clamp_view_center((999, 999), canvas_w=960, canvas_h=540, sensor_w=1920, sensor_h=1080, scale=0.5)
    assert (cx, cy) == (960.0, 540.0)


def test_clamp_view_center_pins_to_edge_when_zoomed():
    # zoomed in 4x: scale=2.0, so half the canvas in sensor units is 960/(2*2)=240
    cx, cy = clamp_view_center((-500, 5000), canvas_w=960, canvas_h=540, sensor_w=1920, sensor_h=1080, scale=2.0)
    assert cx == pytest.approx(240.0)
    assert cy == pytest.approx(1080 - 540 / (2 * 2.0))


def test_canvas_to_sensor_round_trip():
    ox, oy, scale, center = compute_transform((960, 540), view_zoom=2.0, canvas_w=960, canvas_h=540, sensor_w=1920, sensor_h=1080)
    canvas_x, canvas_y = 300.0, 200.0
    sx, sy = canvas_to_sensor_coords(canvas_x, canvas_y, ox, oy, scale, 1920, 1080, clamp=False)
    # forward map: canvas = ox + sensor * scale
    assert ox + sx * scale == pytest.approx(canvas_x)
    assert oy + sy * scale == pytest.approx(canvas_y)


def test_canvas_to_sensor_clamps_out_of_frame_click():
    x, y = canvas_to_sensor_coords(canvas_x=-1000, canvas_y=-1000, ox=0, oy=0, scale=1.0, sensor_w=1920, sensor_h=1080, clamp=True)
    assert (x, y) == (0.0, 0.0)


def test_zoom_at_point_clamped_to_bounds():
    assert zoom_at_point(1.0, steps=100, zoom_step=1.22, zoom_min=1.0, zoom_max=32.0) == 32.0
    assert zoom_at_point(1.0, steps=-100, zoom_step=1.22, zoom_min=1.0, zoom_max=32.0) == 1.0


def test_compute_roi_grid_shape_and_order():
    row_clicks = [(0, 10), (0, 40), (0, 70)]   # 3 rows -- only y used
    col_clicks = [(100, 0), (200, 0)]           # 2 cols -- only x used
    rois = compute_roi_grid(row_clicks, col_clicks, roi_width=40, roi_height=30)
    assert len(rois) == 3 * 2
    # row-major order (matches src/record.ipynb cell 50 exactly)
    assert rois[0] == (100 - 20, 0, 40, 30)   # row 0, col 0
    assert rois[1] == (200 - 20, 0, 40, 30)   # row 0, col 1
    assert rois[2] == (100 - 20, 30, 40, 30)  # row 1, col 0
