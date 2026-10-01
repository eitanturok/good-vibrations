import pytest

from record.utils.geometry import (
    compute_fit_scale, clamp_view_center, compute_transform, canvas_to_sensor_coords,
    zoom_at_point, compute_roi_grid, sensor_rois, roi_mosaic, crop_rois,
    compose_sensor_view,
)


def test_crop_rois_is_what_the_camera_sends():
    import numpy as np
    sensor = np.random.default_rng(0).integers(0, 256, (1080, 1920), dtype=np.uint8)
    rois, row_positions, offset_x = compute_roi_grid([300, 700], [500, 900], 32, 32)
    out = crop_rois(sensor, rois, row_positions, offset_x, 32)
    assert out.shape[0] == 2 * 32
    for (x, y, w, h), (sx, sy, _, _) in zip(rois, sensor_rois(rois, row_positions, offset_x, 32)):
        assert (out[y:y + h, x:x + w] == sensor[sy:sy + h, sx:sx + w]).all()  # each ROI's own pixels
    back = compose_sensor_view(out, rois, row_positions, offset_x, 32)
    for sx, sy, w, h in sensor_rois(rois, row_positions, offset_x, 32):
        assert (back[sy:sy + h, sx:sx + w] == sensor[sy:sy + h, sx:sx + w]).all()


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


def test_compute_roi_grid_is_src_record_math():
    """src/record.ipynb: a row band starts at y - h//2, rounded up to even (the camera reads rows in
    pairs); an ROI starts at x - w//2; the camera reads the columns between the 2 crop clicks, on its
    16 px grid -- so ROI x in the output frame = x - w//2 - offset_x. Rows top-to-bottom, cols
    left-to-right, row-major."""
    rois, row_positions, offset_x = compute_roi_grid([801, 200], [900, 500], roi_width=40, roi_height=32, crop=(405, 1300))
    assert row_positions == [184, 786]  # 200 - 16 = 184 (even); 801 - 16 = 785 -> 786
    assert offset_x == 400             # the crop's left edge, on the 16 px grid
    assert rois == [(80, 0, 40, 32), (480, 0, 40, 32), (80, 32, 40, 32), (480, 32, 40, 32)]


def test_every_roi_is_centered_on_its_line_crossing():
    rows, cols = [101, 350, 777], [123, 640, 1500]
    rois, row_positions, offset_x = compute_roi_grid(rows, cols, roi_width=80, roi_height=32, crop=(0, 1920))
    boxes = sensor_rois(rois, row_positions, offset_x, 32)
    crossings = [(x, y) for y in rows for x in cols]
    for (x, y, w, h), (cx, cy) in zip(boxes, crossings):
        assert x + w / 2 == cx and 0 <= y + h / 2 - cy <= 1  # rows land on even sensor rows


@pytest.mark.parametrize("rows, cols, crop", [([5], [500], (0, 1920)),       # off the top of the sensor
                                              ([500], [1910], (0, 1920)),    # off the right of the sensor
                                              ([500], [420], (400, 1500))])  # outside the crop the camera reads
def test_roi_that_does_not_fit_is_an_error_not_moved(rows, cols, crop):
    with pytest.raises(ValueError):
        compute_roi_grid(rows, cols, roi_width=64, roi_height=32, crop=crop)


def test_roi_mosaic_shows_only_the_roi_cells():
    """The zoomed view: every ROI cell cut out of the frame and tiled in its grid position, touching
    (one continuous image -- only the grid lines drawn over it separate the cells)."""
    import numpy as np
    rois, row_positions, offset_x = compute_roi_grid([300, 700], [500, 900, 1300], 32, 32)
    frame = np.random.default_rng(0).integers(0, 255, (64, 1920), dtype=np.uint8)
    mosaic, tiles = roi_mosaic(frame, rois, n_cols=3)
    assert mosaic.shape == (2 * 32, 3 * 32) and len(tiles) == len(rois)
    for (x, y, w, h), (tx, ty, tw, th) in zip(rois, tiles):
        assert (mosaic[ty:ty + th, tx:tx + tw] == frame[y:y + h, x:x + w]).all()
