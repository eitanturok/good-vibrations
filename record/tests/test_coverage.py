"""Coverage heatmap: color = how many positions ago each pixel was last covered (the last 5 fade
from dark to light, older ones all the same), a black contour around each of the latest sample's
objects, and an empty box adds nothing."""
import time

import numpy as np
from matplotlib.figure import Figure

from record.utils import viz


def square(y, x, shape=(40, 60)):
    m = np.zeros(shape, dtype=bool)
    m[y:y + 4, x:x + 4] = True
    return m


def test_coverage_colors_by_recency_not_count():
    coverage = {}
    for p in range(1, 13):  # 12 positions, each a square in its own spot
        viz.add_coverage(coverage, "a", square(0, 4 * p), position_id=p)
        viz.add_coverage(coverage, "a", square(0, 4 * p), position_id=p)  # 2nd speaker, same position: not a new step
    viz.add_coverage(coverage, "a", square(0, 4), position_id=13)  # position 1's spot again: newest now, however often seen
    age = viz.coverage_age(coverage["a"])
    assert age[0, 4] == 0 and age[0, 48] == 1 and age[0, 44] == 2  # positions ago, one step per position
    assert age[0, 8] == age[0, 12] == viz.N_RECENT_POSITIONS == 5  # past the last 5: all the same
    assert np.isnan(age[30, 30])  # never covered: nothing drawn
    assert coverage["a"]["last_mask"][0, 4]  # the latest sample


def test_empty_box_plots_nothing():
    coverage = {}
    viz.add_coverage(coverage, "empty", np.zeros((40, 40), dtype=bool), position_id=1)
    entry = coverage["empty"]
    assert entry["n_samples"] == 1 and np.isnan(viz.coverage_age(entry)).all()
    fig = Figure()
    viz.draw_coverage(fig.subplots(), entry, "empty")
    assert not fig.axes[0].collections  # no object contours


def test_each_latest_object_gets_a_black_contour_on_top():
    # two overlapping objects: each keeps its own outline, over the blue (not one merged blob)
    a, b = square(10, 10), square(12, 12)
    coverage = {}
    viz.add_coverage(coverage, "a", a | b, position_id=1, object_masks=[a, b])
    fig = Figure()
    ax = fig.subplots()
    viz.draw_coverage(ax, coverage["a"], "a")
    assert len(fig.axes) == 1  # no colorbar: the plot gets the whole panel
    assert not ax.get_xticks().size and not ax.get_yticks().size
    [image] = ax.images
    assert len(ax.collections) == 2 and all(c.zorder > image.get_zorder() for c in ax.collections)
    assert all((c.get_edgecolor()[:, :3] == 0).all() for c in ax.collections)  # black


def test_draw_coverage_is_fast():
    coverage = {}
    for p in range(1, 30):
        viz.add_coverage(coverage, "a", square(p * 20 % 600, p * 13 % 600, shape=(676, 736)), position_id=p)
    t = time.perf_counter()
    viz.add_coverage(coverage, "a", square(5, 5, shape=(676, 736)), position_id=30)
    fig = Figure(figsize=(7.36, 6.76), dpi=100, layout="tight")
    viz.draw_coverage(fig.subplots(), coverage["a"], "a")
    viz.figure_to_array(fig)
    assert time.perf_counter() - t < 1.0
