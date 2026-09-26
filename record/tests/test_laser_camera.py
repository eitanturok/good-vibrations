"""Runs the real MikrotronCamera cell from record.ipynb against a fake grabber that models the
one hardware rule behind "GenapiError: MultiROILUTModeEn is locked": remote features are
locked while the camera is acquiring, and that acquiring state lives in the physical camera
-- shared by every grabber handle, surviving close() and kernel restarts."""
import ctypes as ct
import json
import math
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from record.utils import close_previous_instance, stop_then_close, geometry

NB = Path(__file__).resolve().parents[1] / "record.ipynb"
LOCKED_WHILE_ACQUIRING = {"MultiROILUTModeEn", "MultiROILUTIndex", "MultiROILUTValue",
                          "Width", "Height", "OffsetX", "OffsetY"}


class GenapiError(Exception):
    pass


class Camera:
    """The physical camera: its state outlives any one grabber handle."""
    def __init__(self, acquiring=False, offset_x=0, offset_y=0):
        self.acquiring = acquiring
        self.lut = {}  # multi-ROI row registers: index -> value (value v reads sensor rows 2v, 2v+1)
        self.features = {"Width": 1920 - offset_x, "Height": 1080 - offset_y, "OffsetX": offset_x, "OffsetY": offset_y,
                         "AcquisitionFrameRate": 25, "AcquisitionFrameRateMax": 3987}  # 25 = what the real camera was left at


class Remote:
    def __init__(self, cam): self.cam, self.check = cam, lambda: None

    def set(self, name, value):
        self.check()
        if self.cam.acquiring and name in LOCKED_WHILE_ACQUIRING:
            raise GenapiError(f"GenApi error code 8: {name} is locked")
        f = self.cam.features  # the sensor is 1920 wide: Width + OffsetX can never exceed it
        if name == "Width" and value > 1920 - f["OffsetX"]:
            raise GenapiError(f"GenApi error code 52: Width cannot be greater than {1920 - f['OffsetX']}")
        if name == "OffsetX" and value > 1920 - f["Width"]:
            raise GenapiError(f"GenApi error code 52: OffsetX cannot be greater than {1920 - f['Width']}")
        if name == "Height" and value > 1080 - f["OffsetY"]:  # same rule vertically
            raise GenapiError(f"GenApi error code 52: Height cannot be greater than {1080 - f['OffsetY']}")
        self.cam.features[name] = value
        if name == "MultiROILUTValue":
            self.cam.lut[self.cam.features["MultiROILUTIndex"]] = value

    def get(self, name):
        self.check()
        return self.cam.features.get(name, 0)

    def done(self, cmd): return True

    def execute(self, cmd):
        if cmd == "AcquisitionStop": self.cam.acquiring = False
        if cmd == "AcquisitionStart": self.cam.acquiring = True


class Stream:
    def __init__(self, cam): self.cam, self.values, self.check = cam, {}, lambda: None
    def set(self, name, value): self.values[name] = value

    def get(self, name):
        self.check()
        if name == "Height":  # one buffer = BufferPartCount frames stacked
            return self.cam.features["Height"] * self.values.get("BufferPartCount", 1)
        return self.cam.features[name] if name == "Width" else self.values.get(name, 1)


class ClientError(Exception):
    pass


def make_grabber_cls(cam):
    class FakeEGrabber:
        """Second real rule: while one thread is inside a grabber call (e.g. waiting in a
        Buffer pop for the camera to fill a buffer), any call from another thread fails."""
        def __init__(self, gentl):
            self.remote, self.stream, self.started = Remote(cam), Stream(cam), False
            self.owner, self.popping = None, threading.Event()
            self.produced, self.queued = 0, 0  # buffers filled so far; filled but not yet popped (FIFO)
            self.remote.check = self.stream.check = self.check

        def check(self):
            if self.owner not in (None, threading.get_ident()):
                raise ClientError("EGrabber is busy in another thread")

        def start(self):  # control_remote_device=True -> AcquisitionStart on the camera
            self.started, cam.acquiring = True, True

        def stop(self):  # only sends AcquisitionStop if THIS handle started it
            if self.started:
                self.started, cam.acquiring = False, False

        def realloc_buffers(self, n): pass
        def flush_buffers(self):
            self.check()
            self.queued = 0  # discard every filled-but-unread buffer
        def close(self): pass
    return FakeEGrabber


class FakeBuffer:
    def __init__(self, grabber, timeout):
        self.g = grabber
        w, h = grabber.stream.get("Width"), grabber.stream.get("Height")
        self.parts = grabber.stream.get("BufferPartCount")
        if getattr(grabber, "data", None) is None or len(grabber.data) != w * h:  # the board's DMA memory:
            grabber.data = (ct.c_ubyte * (w * h))()                                # allocated once, reused
        # FIFO: pop the oldest queued buffer, or wait for the next one to fill
        if grabber.queued:
            buffer_id, grabber.queued = grabber.produced - grabber.queued, grabber.queued - 1
        else:
            buffer_id, grabber.produced = grabber.produced, grabber.produced + 1
        part = w * h // self.parts
        for i in range(self.parts):  # every frame numbered: buffer_id * parts + i (mod 256)
            ct.memset(ct.addressof(grabber.data) + i * part, (buffer_id * self.parts + i) % 256, part)
        self.info = {"base": ct.addressof(grabber.data), "part_size": w * h // self.parts, "delivered": self.parts}

    def __enter__(self):
        self.g.check()
        self.g.owner = threading.get_ident()
        self.g.popping.set()
        time.sleep(0.05)  # waiting for the camera to fill the buffer
        return self

    def __exit__(self, *exc):
        self.g.owner = None
        self.g.popping.clear()

    def get_info(self, what, datatype): return self.info[what]


def load_notebook(cam):
    """Exec the ROIConfig, LaserCameraConfig and MikrotronCamera cells, as-is, with the
    fake grabber standing in for egrabber."""
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    ns = dict(np=np, dataclass=dataclass, field=field, geometry=geometry,
              close_previous_instance=close_previous_instance, stop_then_close=stop_then_close,
              EGenTL=lambda: None, EGrabber=make_grabber_cls(cam), ct=ct, threading=threading,
              Buffer=FakeBuffer, BUFFER_INFO_BASE="base", BUFFER_INFO_CUSTOM_PART_SIZE="part_size",
              BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS="delivered", INFO_DATATYPE_PTR=None, INFO_DATATYPE_SIZET=None)
    for marker in ("class ROIConfig", "class LaserCameraConfig", "class MikrotronCamera"):
        exec(next(src for src in cells if marker in src), ns)
    return ns


@pytest.mark.parametrize("left_acquiring", [False, True], ids=["fresh", "left-acquiring-by-previous-session"])
def test_construct(left_acquiring):
    ns = load_notebook(Camera(acquiring=left_acquiring))
    ns["MikrotronCamera"](ns["laser_camera_config"])


def test_construct_wide_open():
    """rois=None is what the GUI's Reset-ROIs calibration flow builds -- full sensor, no grid."""
    ns = load_notebook(Camera(acquiring=True))
    ns["MikrotronCamera"](ns["LaserCameraConfig"](roi=ns["ROIConfig"]()))


def test_default_grid_spans_sensor():
    ns = load_notebook(Camera())
    roi = ns["default_roi"]
    xs = [x for x, y, w, h in roi.rois]
    assert (roi.roi_width, roi.roi_height) == (32, 32)
    assert min(xs) == 0 and max(xs) + roi.roi_width == 1920
    assert roi.row_positions[0] == 0 and roi.row_positions[-1] + roi.roi_height == 1080


def test_full_buffer_fills_before_capture_timeout():
    """The camera only hands over a buffer once all BufferPartCount frames are in it, and
    capture_vibrations waits 5s per buffer. Real bug: the camera kept its old 25 fps, so one
    1625-frame buffer took 65s and the capture timed out."""
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](ns["laser_camera_config"])
    assert laser.grabber.stream.get("BufferPartCount") / laser.get_frame_rate() < 5


def test_construct_twice_in_a_row():
    ns = load_notebook(Camera())
    ns["MikrotronCamera"](ns["laser_camera_config"])
    ns["MikrotronCamera"](ns["laser_camera_config"])


def test_capture_while_live_preview_polls():
    """Real bug: the GUI's live-preview thread loops capture_vibrations(1); run_experiment then
    reads the same grabber from another thread -> ClientError('EGrabber is busy in another thread')."""
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](ns["LaserCameraConfig"](buffer_part_count=2))
    poller = threading.Thread(target=lambda: [laser.capture_vibrations(1) for _ in range(3)])
    poller.start()
    laser.grabber.popping.wait()  # the preview thread is mid-pop
    laser.get_frame_rate()
    assert laser.capture_vibrations(4).shape[0] == 4
    poller.join()


def test_construct_after_a_previous_horizontal_crop():
    """Real bug: an earlier ROI grid left the camera at OffsetX=448 (state survives restarts);
    the default full-width grid then set Width=1920 before resetting OffsetX ->
    'GenApi error code 52: Width cannot be greater than 1472'."""
    ns = load_notebook(Camera(offset_x=448))
    ns["MikrotronCamera"](ns["laser_camera_config"])


def test_construct_from_any_leftover_camera_state():
    """The camera keeps its settings across kernel restarts (acquiring, offsets, crop size),
    so construction must start from a known state rather than patch each leftover."""
    cam = Camera(acquiring=True, offset_x=448, offset_y=900)
    ns = load_notebook(cam)
    ns["MikrotronCamera"](ns["laser_camera_config"])
    assert cam.features["OffsetY"] == 0


@pytest.mark.parametrize("n_rows", [5, 7, 10])
def test_any_row_count_is_valid(n_rows):
    """Real rule: selected rows x 2 must be a multiple of 4 -- with 30px ROIs an odd row count
    crashed set_rows. ROI heights are kept to multiples of 4, so every row count works."""
    ns = load_notebook(Camera())
    roi = ns["ROIConfig"].spread(n_rows=n_rows, roi_width=30, roi_height=30)
    assert roi.roi_height % 4 == 0
    ns["MikrotronCamera"](ns["LaserCameraConfig"](roi=roi))


def test_preview_frame_copies_one_frame_not_the_whole_buffer():
    """Real bug: the live preview asks for 1 frame, but a buffer holds BufferPartCount (1625)
    frames and capture_vibrations allocated + copied ALL of them (~1 GB per preview frame at
    320x1920) -- the laser preview and its zoom lagged."""
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](ns["LaserCameraConfig"](buffer_part_count=50))
    laser.capture_vibrations(1)  # warm up (fake DMA memory allocated once)
    tracemalloc.start()
    frame = laser.capture_vibrations(1)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    one_frame = frame[0].nbytes
    assert frame.shape[0] == 1 and peak < 3 * one_frame, f"peak {peak / 1e6:.1f} MB for a {one_frame / 1e6:.1f} MB frame"


def test_capture_keeps_frame_order_across_buffers():
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](ns["LaserCameraConfig"](buffer_part_count=4))
    video = laser.capture_vibrations(6)  # 1.5 buffers
    assert video.shape[0] == 6
    assert [int(f[0, 0]) for f in video] == [0, 1, 2, 3, 4, 5]


def test_preview_frame_is_the_newest():
    """Real bug: a hand in front of the laser took seconds to show up. The board queues up to
    20 filled buffers (0.65 s each) and the preview popped the OLDEST one, then showed its
    OLDEST frame. The preview must show the newest frame of a fresh buffer."""
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](ns["LaserCameraConfig"](buffer_part_count=4))
    laser.grabber.produced = laser.grabber.queued = 20  # the preview fell behind: 20 buffers waiting
    frame = laser.capture_latest_frame()
    assert int(frame[0, 0]) == (20 * 4 + 3) % 256  # buffer 20 (fresh), its last frame


def test_preview_keeps_updating_during_a_recording():
    """Real bug: pressing Record froze the laser preview -- the recording holds the camera for
    the whole capture, and the preview waited on the same lock. While recording, the preview
    must show the frames the recording is capturing, without waiting."""
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](ns["LaserCameraConfig"](buffer_part_count=4))
    recording = threading.Thread(target=laser.capture_vibrations, args=(80,))  # 20 buffers, ~1 s
    recording.start()
    while laser.grabber.produced < 3:  # a few buffers into the recording
        time.sleep(0.01)
    t0 = time.perf_counter()
    frame = laser.preview_frame()
    waited = time.perf_counter() - t0
    recording.join()
    assert waited < 0.1, f"preview blocked {waited:.2f}s behind the recording"
    assert frame is not None and int(frame[0, 0]) % 4 == 3  # the last frame of a buffer the recording just read


def camera_readout(cam, sensor):
    """What the real camera sends for one frame, from the registers the notebook wrote: the
    columns [OffsetX, OffsetX + Width), and per multi-ROI register value v, sensor rows 2v, 2v+1."""
    f = cam.features
    rows = [r for i in range(f["Height"] // 2) for r in (2 * cam.lut[i], 2 * cam.lut[i] + 1)]
    return sensor[rows, f["OffsetX"]:f["OffsetX"] + f["Width"]]


def test_calibrated_rois_land_on_the_clicked_speckle():
    """Real bug: after Reset ROIs the grid missed the speckle -- the camera's output frame
    starts at OffsetX (leftmost ROI, rounded down to 16), but the stored rois kept SENSOR x.
    So the grid drawn on the frame, and the pixels post-processing crops, were shifted."""
    col_clicks, row_clicks, size = [(500, 0), (1300, 0), (900, 0)], [(0, 700), (0, 301)], 32
    sensor = np.zeros((1080, 1920), dtype=np.uint8)
    for x, _ in col_clicks:  # a bright 32x32 speckle patch at every clicked intersection
        for _, y in row_clicks:
            sensor[round(y) - 16:round(y) + 16, x - 16:x + 16] = 255

    cam = Camera()
    ns = load_notebook(cam)
    rois, row_positions, offset_x = geometry.compute_roi_grid(row_clicks, col_clicks, size, size)
    roi = ns["ROIConfig"](n_rows=2, n_cols=3, roi_width=size, roi_height=size, rois=rois,
                          row_positions=row_positions, offset_x=offset_x)
    ns["MikrotronCamera"](ns["LaserCameraConfig"](roi=roi, buffer_part_count=2))

    frame = camera_readout(cam, sensor)
    for x, y, w, h in roi.rois:
        assert frame[y:y + h, x:x + w].mean() > 200, f"ROI {(x, y, w, h)} missed its speckle"
    # the full-sensor view: the frame's bands pasted back at their sensor positions
    full = geometry.compose_sensor_view(frame, roi.rois, row_positions, offset_x, size)
    for x, y, w, h in geometry.sensor_rois(roi.rois, row_positions, offset_x, size):
        assert full[y:y + h, x:x + w].mean() > 200


def test_change_frames_per_buffer_while_streaming():
    """The live preview can refresh at most once per buffer (BufferPartCount / fps: 0.65 s at
    1625 frames). Changing it at runtime must leave the camera streaming and capturing."""
    cam = Camera()
    ns = load_notebook(cam)
    laser = ns["MikrotronCamera"](ns["LaserCameraConfig"](buffer_part_count=1625))
    laser.change_buffer_part_count(250)
    assert laser.grabber.stream.get("BufferPartCount") == 250 and cam.acquiring
    assert laser.capture_vibrations(600).shape[0] == 600


def test_live_preview_thread_delivers_frames_during_a_recording():
    """End to end, as the GUI runs it: the notebook's real _LivePoller on preview_frame, while
    a recording holds the camera exactly like run_experiment (lock -> flush -> capture).
    Frames must keep reaching the GUI's queue during the recording, not only after it."""
    import queue, sys
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    ns = load_notebook(Camera())
    ns.update(queue=queue, time=time, sys=sys)
    exec(next(src for src in cells if "class _LivePoller" in src), ns)
    laser = ns["MikrotronCamera"](ns["LaserCameraConfig"](buffer_part_count=4))
    frames = queue.Queue(maxsize=1)
    poller = ns["_LivePoller"](laser.preview_frame, frames, threading.Event())
    poller.start()

    def record():
        with laser.lock:
            laser.flush()
            laser.capture_vibrations(80)  # 20 buffers x 50 ms
    recording = threading.Thread(target=record)
    recording.start()
    delivered = 0
    while recording.is_alive():
        try:
            frames.get(timeout=0.05)
            delivered += 1
        except queue.Empty:
            pass
    poller.stop()
    assert delivered >= 5, f"only {delivered} preview frames during a ~1 s recording"


def test_default_buffer_fits_the_capture_margin():
    """First principles: a capture can start up to one buffer (B / fps) before the chirp, so
    B / fps must fit inside the capture margin -- and B = margin * fps also makes the capture
    ((chirp + margin) * fps frames) a whole number of buffers, so nothing waits after the audio."""
    ns = load_notebook(Camera())
    config = ns["LaserCameraConfig"]()
    assert config.buffer_part_count / config.fps <= config.capture_margin_s
    n_frames = math.ceil((1.0 + config.capture_margin_s) * config.fps)  # 1 s chirp
    assert n_frames % config.buffer_part_count == 0


def test_buffer_size_options_divide_the_capture():
    from record.utils import buffer_sizes_dividing
    sizes = buffer_sizes_dividing(2750)
    assert 250 in sizes and all(2750 % b == 0 and b >= 25 for b in sizes)
