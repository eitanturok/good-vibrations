"""Real bug: the overhead feed lagged reality by over a second. The uEye image queue (40
buffers) is FIFO -- read_frame returns the OLDEST queued frame -- and the live preview reads
slower than the camera's 30 fps, so the queue sits full and every frame shown is ~40 frames
(~1.3 s) old. That includes run_experiment's official capture_overhead()."""
import json
import threading
from collections import deque
from pathlib import Path

import numpy as np
from pyueye import ueye

from utils.ids_camera.pyueye_example_camera import Camera
from utils.ids_camera.pyueye_example_utils import uEyeException

NB = Path(__file__).resolve().parents[1] / "record.ipynb"


class FakeQueue:
    """The driver's FIFO: the camera appends frames; read_frame pops the oldest, or waits
    for the next one the camera produces."""
    def __init__(self, backlog):
        self.q, self.next_id = deque(), 0
        for _ in range(backlog):
            self.produce()

    def produce(self):
        self.q.append(self.next_id)
        self.next_id += 1

    def read_frame(self):
        if not self.q:
            self.produce()  # waits for the next exposure
        return np.full((4, 4), self.q.popleft() % 256, dtype=np.uint8), None

    def is_ImageQueue(self, h_cam, cmd, param, size):
        if cmd == ueye.IS_IMAGE_QUEUE_CMD_GET_PENDING:
            param.value = len(self.q)
        elif cmd == ueye.IS_IMAGE_QUEUE_CMD_DISCARD_N_ITEMS:
            for _ in range(param.value):
                self.q.popleft()
        return ueye.IS_SUCCESS


def test_capture_overhead_returns_fresh_frame(monkeypatch):
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    ns = dict(threading=threading, np=np, ueye=ueye, cv2=__import__("cv2"), uEyeException=uEyeException,
              OverheadCameraConfig=None, close_previous_instance=None, _PyueyeCameraBase=Camera)
    exec(next(src for src in cells if "class PyueyeCamera" in src), ns)

    queue = FakeQueue(backlog=40)  # full queue: the preview fell behind
    monkeypatch.setattr(ueye, "is_ImageQueue", queue.is_ImageQueue)
    cam = object.__new__(ns["PyueyeCamera"])  # skip __init__: no hardware
    cam._cam, cam.lock = Camera(), threading.Lock()
    cam._cam.read_frame = queue.read_frame

    frame = cam.capture_overhead()
    assert frame[0, 0, 0] == 40  # the frame exposed after the call, not frame 0 from ~1.3 s ago
