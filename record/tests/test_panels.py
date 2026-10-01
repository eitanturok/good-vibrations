"""Real bug: the shifts/FFT panels showed too-short plots when the GUI opened full screen -- the
dry run draws its plots before the GUI exists, when no panel size was known, so they came out
at a 600x300 default. Plots are now always drawn at their full-screen panel size."""
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib.figure import Figure

from record.utils import viz

NB = Path(__file__).resolve().parents[1] / "record.ipynb"


class DoneTask:
    exception = None
    def __init__(self, result): self.result = result
    def join(self): pass


def test_smask_plots_objects_without_a_color_in_their_name():
    """Real bug: segmenting a "mug" showed in coverage but the smask panel stayed on Loading... --
    objects with no color word got an RGBA colormap color, so (*color, alpha) had 5 values and
    plot_smask raised."""
    masks = np.zeros((1, 90, 100), bool)
    masks[0, 10:20, 10:20] = True
    fig = Figure()
    viz.draw_smask(fig.subplots(), [{"masks": masks}, {"masks": masks}], np.zeros((90, 100, 3)), ["mug", "red-cube"])
    assert viz.figure_to_array(fig).shape[2] == 4


def test_plots_are_full_panel_size_even_before_the_gui_exists():
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    ns = dict(np=np, Figure=Figure, viz=viz, log=print, load_metadata=None)
    exec(next(src for src in cells if "def plot_shifts" in src), ns)
    ec = SimpleNamespace(panel_queues={"shifts": queue.Queue()}, laser_cam=SimpleNamespace(get_frame_rate=lambda: 2500.0))
    pclk = DoneTask({"shifts": np.random.default_rng(0).normal(size=(1, 2500, 2)), "laser_idx": 55})
    ns["plot_shifts"](ec, pclk, "Shifts Position 1 Speaker 1 Laser 55")  # the dry run: no GUI yet
    assert ec.panel_queues["shifts"].get_nowait().shape[:2] == (334, 1480)  # full-screen shifts panel
