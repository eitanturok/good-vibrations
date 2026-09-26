"""Real bug: the shifts/FFT panels showed too-short plots when the GUI opened full screen. The
dry run records before the GUI exists, so its plots were drawn at the default size, and an
image drawn once can't follow its panel -- until the next recording drew new ones. Each panel
must be able to re-draw its current plot at the panel's current size."""
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib.figure import Figure

from record.utils import viz

NB = Path(__file__).resolve().parents[1] / "record.ipynb"


def load_plot_cell():
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    ns = dict(np=np, Figure=Figure, viz=viz, log=print, load_metadata=None)
    exec(next(src for src in cells if "def plot_shifts" in src), ns)
    return ns


class DoneTask:
    exception = None
    def __init__(self, result): self.result = result
    def join(self): pass


def test_plot_redraws_at_the_panel_size_after_the_gui_opens():
    ns = load_plot_cell()
    ec = SimpleNamespace(panel_sizes={}, panel_queues={"shifts": queue.Queue()}, panel_draws={},
                         laser_cam=SimpleNamespace(get_frame_rate=lambda: 2500.0))
    pclk = DoneTask({"shifts": np.random.default_rng(0).normal(size=(1, 2500, 2)), "laser_idx": 55})

    ns["plot_shifts"](ec, pclk, "Shifts Position 1 Speaker 1 (000001) Laser 55")  # dry run: no GUI yet
    assert ec.panel_queues["shifts"].get_nowait().shape[:2] == (300, 600)  # the default size

    ec.panel_sizes["shifts"] = (1400, 350)  # the GUI opens full screen
    ns["render_panel"](ec, "shifts")
    assert ec.panel_queues["shifts"].get_nowait().shape[:2] == (350, 1400)
