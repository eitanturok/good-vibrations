"""Sample dirs are named utils.ids.sample_name(position_id, speaker): zero-padded, so a plain string
sort orders them by position, then speaker."""
from utils.ids import sample_name


def test_sample_names_sort_by_position_then_speaker():
    assert sample_name(12, 3) == "000012-3"
    names = [sample_name(p, s) for p, s in ((10, 1), (9, 8), (9, 1), (100, 2))]
    assert sorted(names) == ["000009-1", "000009-8", "000010-1", "000100-2"]
