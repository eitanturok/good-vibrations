"""Runs the real MikrotronCamera cell from record.ipynb against a fake grabber that models the
one hardware rule behind "GenapiError: MultiROILUTModeEn is locked": remote features are
locked while the camera is acquiring, and that acquiring state lives in the physical camera
-- shared by every grabber handle, surviving close() and kernel restarts."""
import ctypes as ct
import json
import math
import threading
import time
import tempfile
import tracemalloc
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from record2 import geometry
from record2.utils import close_previous_instance, stop_then_close

NB = Path(__file__).resolve().parents[1] / "record.ipynb"
LOCKED_WHILE_ACQUIRING = {"MultiROILUTModeEn", "MultiROILUTIndex", "MultiROILUTValue",
                          "Width", "Height", "OffsetX", "OffsetY"}


class GenapiError(Exception):
    pass


class Camera:
    """The physical camera: its state outlives any one grabber handle."""
    def __init__(self, acquiring=False, offset_x=0, offset_y=0, powered=True):
        self.acquiring, self.powered = acquiring, powered
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


class InvalidAddressException(Exception):
    pass


class ResourceInUseException(Exception):
    pass


class FakeEGenTL:
    """Real rule: one live EGenTL per process -- a second EGenTL() while an earlier one is
    still referenced (e.g. by a failed cell's traceback) fails with ResourceInUseException."""
    live = None

    def __init__(self):
        if FakeEGenTL.live is not None and FakeEGenTL.live() is not None:
            raise ResourceInUseException("GCInitLib: Requested resource is already in use")
        FakeEGenTL.live = weakref.ref(self)


def make_grabber_cls(cam):
    class FakeEGrabber:
        """Second real rule: while one thread is inside a grabber call (e.g. waiting in a
        Buffer pop for the camera to fill a buffer), any call from another thread fails."""
        def __init__(self, gentl):
            if not cam.powered:  # the grabber card has no camera on its CoaXPress link
                raise InvalidAddressException("DevGetPort: A given address is out of range or invalid")
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


class TimeoutException(Exception):
    pass


class FakeBuffer:
    def __init__(self, grabber, timeout):
        if not grabber.started:  # a stopped camera fills no buffers: the pop waits out its timeout
            raise TimeoutException("EuresysEventsGetData: Timeout expired before the operation could be completed")
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


def load_notebook(cam, gentl=lambda: None):
    """Exec the LaserCameraConfig, MikrotronCamera and ROI cells, as-is, with the
    fake grabber standing in for egrabber."""
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    ns = dict(np=np, json=json, REPO_DIR=Path("."), dataclass=dataclass, field=field, geometry=geometry, dataclasses=__import__('dataclasses'),
              close_previous_instance=close_previous_instance, stop_then_close=stop_then_close,
              EGenTL=gentl, EGrabber=make_grabber_cls(cam), ct=ct,
              InvalidAddressException=InvalidAddressException, threading=threading,
              Buffer=FakeBuffer, BUFFER_INFO_BASE="base", BUFFER_INFO_CUSTOM_PART_SIZE="part_size",
              BUFFER_INFO_CUSTOM_NUM_DELIVERED_PARTS="delivered", INFO_DATATYPE_PTR=None, INFO_DATATYPE_SIZET=None)
    for marker in ("class LaserCameraConfig", "class MikrotronCamera", "class ROIConfig", "def open_calibration_camera"):
        exec(next(src for src in cells if marker in src), ns)
    ns["ROIS_FILE"] = Path(tempfile.mkdtemp()) / "rois.json"  # never the real last-used ROIs
    return ns


def grid_config(ns, **kwargs):
    """The laser camera config with the default ROI grid (Section 4's roi_config)."""
    return ns["LaserCameraConfig"](roi=ns["ROIConfig"](), **kwargs)


@pytest.mark.parametrize("left_acquiring", [False, True], ids=["fresh", "left-acquiring-by-previous-session"])
def test_construct(left_acquiring):
    ns = load_notebook(Camera(acquiring=left_acquiring))
    ns["MikrotronCamera"](grid_config(ns))


def test_construct_wide_open():
    """rois=None is what the GUI's Reset-ROIs calibration flow builds -- full sensor, no grid."""
    ns = load_notebook(Camera(acquiring=True))
    ns["MikrotronCamera"](ns["LaserCameraConfig"]())  # roi=None: wide-open


def test_camera_turned_off_says_so_and_retry_works():
    """Real bug: with the laser camera powered off, construction failed with the misleading
    "InvalidAddressException: DevGetPort: A given address is out of range or invalid" -- and
    re-running the cell after turning the camera on then failed "GCInitLib: already in use"
    because the failed attempt's EGenTL was still alive."""
    cam = Camera(powered=False)
    ns = load_notebook(cam, gentl=FakeEGenTL)
    with pytest.raises(RuntimeError, match="turned on"):
        ns["MikrotronCamera"](grid_config(ns))
    cam.powered = True  # the user turns the camera on and re-runs the cell
    ns["MikrotronCamera"](grid_config(ns))


def test_default_grid_is_inside_the_sensor():
    ns = load_notebook(Camera())
    roi = ns["ROIConfig"]()
    assert (roi.n_rows, roi.n_cols) == (10, 10) and len(roi.rois) == 100
    for x, y, w, h in geometry.sensor_rois(roi.rois, roi.row_positions, roi.offset_x, roi.roi_height):
        assert 0 <= x and x + w <= 1920 and 0 <= y and y + h <= 1080


def test_full_buffer_fills_before_capture_timeout():
    """The camera only hands over a buffer once all BufferPartCount frames are in it, and
    capture_vibrations waits 5s per buffer. Real bug: the camera kept its old 25 fps, so one
    1625-frame buffer took 65s and the capture timed out."""
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](grid_config(ns))
    assert laser.grabber.stream.get("BufferPartCount") / laser.get_frame_rate() < 5


def test_construct_twice_in_a_row():
    ns = load_notebook(Camera())
    ns["MikrotronCamera"](grid_config(ns))
    ns["MikrotronCamera"](grid_config(ns))


def test_capture_while_live_preview_polls():
    """Real bug: the GUI's live-preview thread loops capture_vibrations(1); record_position then
    reads the same grabber from another thread -> ClientError('EGrabber is busy in another thread')."""
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](grid_config(ns, buffer_part_count=2))
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
    ns["MikrotronCamera"](grid_config(ns))


def test_construct_from_any_leftover_camera_state():
    """The camera keeps its settings across kernel restarts (acquiring, offsets, crop size),
    so construction must start from a known state rather than patch each leftover."""
    cam = Camera(acquiring=True, offset_x=448, offset_y=900)
    ns = load_notebook(cam)
    ns["MikrotronCamera"](grid_config(ns))
    assert cam.features["OffsetY"] == 0


@pytest.mark.parametrize("n_rows", [5, 7, 10])
def test_any_row_count_is_valid(n_rows):
    """Real rule: selected rows x 2 must be a multiple of 4 -- with 30px ROIs an odd row count
    crashed set_rows. ROI heights are kept to multiples of 4, so every row count works."""
    ns = load_notebook(Camera())
    with pytest.raises(ValueError):
        ns["ROIConfig"](roi_height=30)
    roi = ns["ROIConfig"](rows=list(range(60, 60 + 100 * n_rows, 100)), roi_width=30, roi_height=28)
    ns["MikrotronCamera"](ns["LaserCameraConfig"](roi=roi))


def test_preview_frame_copies_one_frame_not_the_whole_buffer():
    """Real bug: the live preview asks for 1 frame, but a buffer holds BufferPartCount (1625)
    frames and capture_vibrations allocated + copied ALL of them (~1 GB per preview frame at
    320x1920) -- the laser preview and its zoom lagged."""
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](grid_config(ns, buffer_part_count=50))
    laser.capture_vibrations(1)  # warm up (fake DMA memory allocated once)
    tracemalloc.start()
    frame = laser.capture_vibrations(1)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    one_frame = frame[0].nbytes
    assert frame.shape[0] == 1 and peak < 3 * one_frame, f"peak {peak / 1e6:.1f} MB for a {one_frame / 1e6:.1f} MB frame"


def test_capture_keeps_frame_order_across_buffers():
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](grid_config(ns, buffer_part_count=4))
    video = laser.capture_vibrations(6)  # 1.5 buffers
    assert video.shape[0] == 6
    assert [int(f[0, 0]) for f in video] == [0, 1, 2, 3, 4, 5]


def test_preview_frame_is_the_newest():
    """Real bug: a hand in front of the laser took seconds to show up. The board queues up to
    20 filled buffers (0.65 s each) and the preview popped the OLDEST one, then showed its
    OLDEST frame. The preview must show the newest frame of a fresh buffer."""
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](grid_config(ns, buffer_part_count=4))
    laser.grabber.produced = laser.grabber.queued = 20  # the preview fell behind: 20 buffers waiting
    frame = laser.capture_latest_frame()
    assert int(frame[0, 0]) == (20 * 4 + 3) % 256  # buffer 20 (fresh), its last frame


def test_preview_keeps_updating_during_a_recording():
    """Real bug: pressing Record froze the laser preview -- the recording holds the camera for
    the whole capture, and the preview waited on the same lock. While recording, the preview
    must show the frames the recording is capturing, without waiting."""
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](grid_config(ns, buffer_part_count=4))
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
    roi = ns["ROIConfig"](rows=[y for _, y in row_clicks], cols=[x for x, _ in col_clicks], roi_width=size, roi_height=size)
    row_positions, offset_x = roi.row_positions, roi.offset_x
    ns["MikrotronCamera"](ns["LaserCameraConfig"](roi=roi, buffer_part_count=2))

    frame = camera_readout(cam, sensor)
    for x, y, w, h in roi.rois:
        assert frame[y:y + h, x:x + w].mean() > 200, f"ROI {(x, y, w, h)} missed its speckle"






def test_laser_camera_starts_without_rois():
    """Section 3 builds the laser camera before any ROIs exist: wide-open, full sensor from its
    config, streaming; Section 4 then rebuilds it cropped to the ROI grid."""
    cam = Camera(offset_x=448)
    ns = load_notebook(cam)
    laser = ns["MikrotronCamera"](ns["LaserCameraConfig"]())
    assert laser.config.roi is None and cam.acquiring
    assert (cam.features["Width"], cam.features["Height"]) == (laser.config.sensor_width, laser.config.sensor_height)
    assert laser.capture_latest_frame().shape == (1080, 1920)


class FakeCv2:
    """An OpenCV window that clicks `points` (window x, y), one per frame shown."""
    EVENT_LBUTTONDOWN = 1

    def __init__(self):
        self.points, self.shown = [], []

    def setMouseCallback(self, window, on_mouse): self.on_mouse = on_mouse
    def cvtColor(self, frame, code): return np.dstack([frame] * 3)
    def waitKey(self, ms): return -1

    def imshow(self, window, frame):
        self.shown.append(frame.shape[:2])
        if self.points:
            self.on_mouse(self.EVENT_LBUTTONDOWN, *self.points.pop(0), 0, None)

    def __getattr__(self, name): return lambda *args, **kwargs: None  # namedWindow, line, putText, ...


def roi_steps(ns):
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    return lambda marker: exec(next(src for src in cells if marker in src and "def " not in src), ns)


def test_roi_steps_without_clicking_use_the_default_grid():
    """SET_ROIS = False: no clicks, and the camera still ends up cropped to the default (even) grid.
    Real bug: only the clicking path cropped it, so the first recording ran on the wide-open
    calibration camera (AttributeError: 'NoneType' object has no attribute 'rois' at the preview)."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from PIL import Image
    from record2 import viz
    ns = load_notebook(Camera())
    ns.update(cv2=FakeCv2(), Figure=Figure, Rectangle=Rectangle, Image=Image, viz=viz, display=lambda image: None, SET_ROIS=False)
    ns["laser_cam"] = ns["MikrotronCamera"](ns["LaserCameraConfig"](buffer_part_count=4))
    step = roi_steps(ns)
    for marker in ("else load_rois()", "horizontal=True", "edges = click_lines", "cols = click_lines", "roi_width="):
        step(marker)
    roi, laser = ns["roi_config"], ns["laser_cam"]
    assert laser.grabber is not None and laser.config.roi is roi
    assert (roi.n_rows, roi.n_cols, roi.roi_width, roi.roi_height) == (10, 10, 80, 32)  # ROIConfig's defaults


def test_roi_steps_rows_crop_cols_size():
    """The ROI section's cells, in order, twice (re-running must work too), clicking in the OpenCV
    window: each step updates roi_config, and the last leaves `laser_cam` OPEN and cropped to the grid.
    Real bugs: the crop step stopped the camera and its next read timed out; and the last step built
    the cropped camera without keeping it, so laser_cam was the closed calibration camera
    (AttributeError: 'NoneType' object has no attribute 'remote' on the first recording)."""
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    from PIL import Image
    from record2 import viz
    cam, cv2 = Camera(), FakeCv2()
    ns = load_notebook(cam)
    ns.update(cv2=cv2, Figure=Figure, Rectangle=Rectangle, Image=Image, viz=viz, display=lambda image: None)
    ns["laser_cam"] = ns["MikrotronCamera"](ns["LaserCameraConfig"](buffer_part_count=4))
    ns["SET_ROIS"] = True
    step = roi_steps(ns)
    step("else load_rois()")
    rows, cols = list(range(100, 1100, 100)), list(range(450, 1550, 110))
    for _ in range(2):
        cv2.points = [(5, y) for y in rows]
        step("horizontal=True")
        assert ns["roi_config"].rows == rows and ns["wide_open_frame"].shape == (1080, 1920)

        cv2.points = [(1500, 5), (400, 5)]  # RIGHT then LEFT edge
        step("edges = click_lines")
        assert ns["roi_config"].crop == (400, 1500)

        cv2.shown.clear()
        cv2.points = [(x - 400, 5) for x in cols]  # clicked in the cropped view
        step("cols = click_lines")
        assert cv2.shown[0] == (1080, 1100)  # only the cropped columns shown
        assert ns["roi_config"].cols == cols

        step("roi_width=")
        roi, laser = ns["roi_config"], ns["laser_cam"]
        assert laser.grabber is not None and laser.get_frame_rate() > 0  # open: the camera a recording will use
        assert (roi.n_rows, roi.n_cols) == (10, 10) and laser.config.roi is roi
        assert laser.config.buffer_part_count == 4  # calibration doesn't lose frames-per-buffer
        assert cam.features["Height"] == 10 * roi.roi_height  # the camera reads only the ROI rows

    ns["SET_ROIS"] = False  # the next session, without clicking: the last-used ROIs
    for marker in ("else load_rois()", "horizontal=True", "edges = click_lines", "cols = click_lines", "roi_width="):
        step(marker)
    assert ns["roi_config"].rois == roi.rois and ns["laser_cam"].config.roi.rois == roi.rois


def test_setters_keep_the_config_current():
    """The GUI's sliders change the camera live; its config must follow, so metadata and the GUI
    never show a stale exposure."""
    ns = load_notebook(Camera())
    laser = ns["MikrotronCamera"](grid_config(ns))
    laser.set_exposure(123.0)
    laser.set_gain(2.5)
    assert (laser.config.exposure_us, laser.config.gain) == (123.0, 2.5)
    assert (laser.get_exposure(), laser.get_gain()) == (123.0, 2.5)


def test_buffer_size_divides_the_speaker_padding():
    """A capture can start up to one buffer (B / fps) early; the speaker padding absorbs it only if
    B divides padding * fps -- the assert in the Hardware cell."""
    ns = load_notebook(Camera())
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    exec(next(src for src in cells if "class AudioConfig" in src), ns)
    laser, audio = ns["LaserCameraConfig"](), ns["AudioConfig"]()
    assert round(audio.speaker_padding * laser.fps) % laser.buffer_part_count == 0
