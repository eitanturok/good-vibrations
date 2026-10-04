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
    assert (shown["last_masks"][0] == b).all()  # the latest position, outlined in red


def test_a_newer_position_hides_the_outlines_under_it():
    old = np.zeros((8, 8), bool); old[1:5, 1:5] = True
    new = np.zeros((8, 8), bool); new[3:8, 3:8] = True
    shown = add_coverage(add_coverage(None, old), new)
    assert shown["outlines"][1, 1] and not shown["outlines"][4, 4]  # old's corner: outside new shows, inside new hides
    assert shown["outlines"][3, 3]  # new's own outline


def test_a_position_cropped_to_another_size_is_kept():
    """Real bug: a new crop (another mask size) started the coverage over, dropping every earlier position."""
    a = np.zeros((8, 8), bool); a[0:4, 0:4] = True  # the box's top-left quarter
    b = np.zeros((16, 16), bool); b[8:16, 8:16] = True  # the bottom-right quarter, cropped twice as big
    shown = add_coverage(add_coverage(None, a), b)
    assert shown["n_positions"] == 2 and shown["last_seen"].shape == (8, 8)
    assert shown["last_seen"][1, 1] == 1 and shown["last_seen"][6, 6] == 2 and shown["last_masks"][0].shape == (8, 8)


def test_an_empty_box_adds_no_position():
    entry = add_coverage(None, np.zeros((4, 4), bool))
    assert entry["n_positions"] == 0 and not entry["last_seen"].any()
