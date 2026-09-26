import numpy as np
from matplotlib.figure import Figure

from record.utils import viz


def test_draw_smask_renders():
    """Real bug: draw_smask used matplotlib.cm.get_cmap (removed in matplotlib 3.9), so the
    smask panel never rendered."""
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:10, 5:15] = True
    ax = Figure().subplots()
    viz.draw_smask(ax, [{"masks": [mask]}], np.zeros((20, 30, 3), dtype=np.uint8), ["red-cube"])
    assert viz.figure_to_array(ax.figure).shape[-1] == 4
