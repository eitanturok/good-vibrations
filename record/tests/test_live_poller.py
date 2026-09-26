"""The GUI's _LivePoller runs forever in a background thread; Jupyter shows a background
thread's prints in whichever cell is running. Real bug: the overhead read failing on every
tick printed "failed" ~40x/s into every cell, with no error code. A persistent failure must
be reported once, with the uEye error code."""
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
from pyueye import ueye

from utils.ids_camera.pyueye_example_camera import Camera
from utils.ids_camera.pyueye_example_utils import ImageBuffer, uEyeException

NB = Path(__file__).resolve().parents[1] / "record.ipynb"


def load_cells(*markers):
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    ns = dict(threading=threading, queue=queue, time=time, sys=sys, np=np, ueye=ueye, OverheadCameraConfig=None, close_previous_instance=None,
              _PyueyeCameraBase=Camera, cv2=__import__("cv2"), uEyeException=uEyeException)
    for marker in markers:
        exec(next(src for src in cells if marker in src), ns)
    return ns


def test_persistent_overhead_failure_reported_once(monkeypatch, capsys):
    monkeypatch.setattr(ueye, "is_WaitForNextImage", lambda *a: ueye.IS_INVALID_CAMERA_HANDLE)
    ns = load_cells("class _LivePoller", "class PyueyeCamera")
    cam = object.__new__(ns["PyueyeCamera"])  # skip __init__: no hardware
    cam._cam, cam.lock = Camera(), threading.Lock()
    cam._cam.img_buffers = [ImageBuffer()]

    poller = ns["_LivePoller"](cam.capture_overhead, queue.Queue(maxsize=1), threading.Event(), poll_interval=0.001)
    poller.start()
    time.sleep(0.2)
    poller.stop()
    poller.join()
    out = capsys.readouterr()
    lines = (out.out + out.err).splitlines()
    assert len(lines) == 1 and str(ueye.IS_INVALID_CAMERA_HANDLE) in lines[0], lines[:5]
