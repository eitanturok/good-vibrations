"""The notebook's recording cells, exec'd as-is, with fake hardware and segmenter -- shared by the
record_position and GUI tests."""
import dataclasses
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

NB = Path(__file__).resolve().parents[1] / "record.ipynb"
CELLS = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def cell(marker): return next(src for src in CELLS if marker in src)


class Counter:  # the position-id counter, without touching the repo's real position_id.txt
    def __init__(self): self.n = 0
    def peek(self): return self.n + 1
    def next(self):
        self.n += 1
        return self.n


class FakeLaser:
    """The laser camera: the notebook's real configs, no grabber. fail_on: the capture that times out."""
    fail_on = None

    def __init__(self, config):
        from record2 import geometry
        self.config, self.captures, self.lock = config, 0, threading.RLock()
        roi = config.roi
        self.width, self.height = ((geometry.readout_columns(*roi.crop)[1], roi.n_rows * roi.roi_height) if roi
                                   else (config.sensor_width, config.sensor_height))

    def get_frame_rate(self): return 100.0
    def get_exposure(self): return self.config.exposure_us
    def get_gain(self): return self.config.gain
    def get_max_frame_rate(self): return 4000.0
    def get_global_roi(self): return (0, 0, self.width, self.height)
    def flush(self): pass
    def capture_latest_frame(self): return np.zeros((self.height, self.width), np.uint8)
    def preview_frame(self): return self.capture_latest_frame()
    def close(self): pass

    def capture_vibrations(self, n):
        self.captures += 1
        if self.captures == self.fail_on:
            raise RuntimeError("EuresysEventsGetData: Timeout expired")
        return np.full((n, self.height, self.width), self.captures, np.uint8)

    def set_exposure(self, x): self.config = dataclasses.replace(self.config, exposure_us=x)
    def set_gain(self, x): self.config = dataclasses.replace(self.config, gain=x)


class FakeOverhead:
    def __init__(self):
        self.config = SimpleNamespace(device_id=0, exposure_ms=12.0, gain=1, exposure_bounds_ms=(0.1, 100), gain_bounds=(0, 100))
        self.width, self.height, self.frame_rate = 100, 80, 30.0

    def capture(self): return np.zeros((80, 100, 3), np.uint8)
    def get_exposure(self): return self.config.exposure_ms
    def get_gain(self): return self.config.gain
    def get_pixel_clock(self): return 86
    def set_exposure(self, x): self.config.exposure_ms = x
    def set_gain(self, x): self.config.gain = x


class FakeAudio:
    def __init__(self):
        self.config = SimpleNamespace(speaker_device_names={s: "card" for s in range(1, 9)}, speaker_channels={s: (0, False) for s in range(1, 9)},
                                      speaker_delay=0.0, speaker_padding=0.1, settle_seconds=0.25)
        self.sample_rate, self.played, self.resets = 48000, [], 0

    def play(self, audio, speakers): self.played.append(list(speakers))
    def wait(self): pass
    def reset(self): self.resets += 1


def fake_segment(segmenter, crop, prompts, objects, scale=1.0):
    mask = np.zeros(crop.shape[:2], bool)
    mask[10:20, 10:20] = True
    return [{"masks": [mask], "boxes": [[10, 10, 20, 20]], "scores": [0.9]} for _ in objects]


def fake_preview(raw, roi, fps, laser, channel, f_start, f_end, batch, use_pc):
    return {"shifts": np.zeros((1, len(raw), 2)), "fft": np.zeros((1, 5, 2)), "freqs": np.linspace(100, 1000, 5), "fps": fps,
            "laser_idx": laser, "recovered_audio": np.zeros(100, np.int16), "audio_sample_rate": 22050}


def load(tmp_path, fail_on=None):
    """The catalog, config, Experiment and recording cells, with fakes for the hardware and the segmenter."""
    import collections, dataclasses, io, math, os, shutil, socket, subprocess, time
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import dataclass, field
    from datetime import datetime, timezone
    from matplotlib.figure import Figure
    from PIL import Image
    from utils.io_utils import append, load as io_load, load_metadata, save
    from record2 import viz
    from record2.image import crop
    from record2.log import TAIL, Timing, logger, running, setup_logging
    from record2.segment import combined_smask, object_centers_of_mass
    setup_logging()
    ns = dict(collections=collections, dataclasses=dataclasses, io=io, math=math, os=os, shutil=shutil, socket=socket,
              subprocess=subprocess, time=time, threading=threading, json=json, np=np, Path=Path,
              ThreadPoolExecutor=ThreadPoolExecutor, dataclass=dataclass, field=field, datetime=datetime, timezone=timezone,
              Figure=Figure, Image=Image, append=append, load=io_load, load_metadata=load_metadata, save=save,
              viz=viz, crop=crop, TAIL=TAIL, Timing=Timing, logger=logger, running=running,
              combined_smask=combined_smask, object_centers_of_mass=object_centers_of_mass,
              segment=fake_segment, preview_vibrations=fake_preview, PositionIdCounter=Counter,
              PCLK_BATCH_SIZE=256, AUDIO_SAMPLE_RATE=22050, display=lambda x: None)
    import cv2
    from matplotlib.patches import Rectangle
    from record2 import geometry
    ns.update(cv2=cv2, Rectangle=Rectangle, geometry=geometry, MikrotronCamera=FakeLaser)
    for marker in ("class LaserCameraConfig", "class ROIConfig", "def open_calibration_camera", "BOXES = {", "class PositionConfig",
                   "default_position = ", "class PreviewConfig", "default_preview = ", "class Experiment",
                   "SAVE_POOL = ", "def job(", "def record_position", "PANELS = "):
        exec(cell(marker), ns)
    laser = FakeLaser(ns["LaserCameraConfig"](roi=ns["ROIConfig"]()))
    laser.fail_on = fail_on
    chirp = SimpleNamespace(t_sec=0.2, t_start=0.0, t_end=0.0, f_start=100.0, f_end=1000.0)
    ns["exp"] = ns["Experiment"](tmp_path / "experiment", chirp, np.zeros(10, np.float32), np.zeros(10, np.float32), hand_delay=0, min_free_gb=0)
    ns["hw"] = SimpleNamespace(overhead_cam=FakeOverhead(), laser_cam=laser, audio_engine=FakeAudio(), laser_background=None)
    ns["segmenter"] = None
    ns["position"] = ns["PositionConfig"](speakers=[1, 3, 5, 7], box="gastronorm", crop=ns["BOXES"]["gastronorm"], objects={"red-cube": 1},
                                          prompts={"red-cube": "Red cube"}, layout="one-cube")
    ns["preview"] = ns["PreviewConfig"](speaker=3, laser=5, channel=1)
    return ns


def run(ns, **kwargs):
    """record_position, then wait for every background save."""
    out = ns["record_position"](ns["hw"], ns["exp"], ns["segmenter"], ns["position"], ns["preview"], **kwargs)
    ns["SAVE_POOL"].shutdown(wait=True)
    ns["SAVE_POOL"] = ns["ThreadPoolExecutor"](2)
    return out
