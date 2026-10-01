import numpy as np

from record2.viz import add_coverage


def test_add_coverage_returns_a_new_entry():
    """A dry run shows its position on the coverage without keeping it: adding must not change the kept entry."""
    a = np.zeros((8, 8), bool); a[1:4, 1:4] = True
    b = np.zeros((8, 8), bool); b[5:8, 5:8] = True
    kept = add_coverage(None, a)
    shown = add_coverage(kept, b)
    assert kept["n_positions"] == 1 and kept["last_seen"][6, 6] == 0 and not kept["outlines"][5, 5]  # unchanged
    assert shown["n_positions"] == 2 and shown["last_seen"][2, 2] == 1 and shown["last_seen"][6, 6] == 2
    assert shown["outlines"][1, 1] and shown["outlines"][5, 5] and not shown["outlines"][2, 2]  # every position outlined, not filled
    assert shown["last_masks"][0] is b  # the latest position, outlined in red


def test_an_empty_box_adds_no_position():
    entry = add_coverage(None, np.zeros((4, 4), bool))
    assert entry["n_positions"] == 0 and not entry["last_seen"].any()
