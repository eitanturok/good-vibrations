"""Per-sample stage status: each sample (a position-speaker) goes record -> save vibration ->
save sample -> post process; the GUI shows which stage it's in, each stage's time, and the total."""
import json
import shutil

import pytest

from record.utils import status


def test_a_sample_moves_through_its_stages():
    s, t = {}, [0.0]
    now = lambda: t[0]
    status.add(s, "000012-3", now=now)
    assert status.states(s["000012-3"], now()) == [("waiting", None)] * 4 and status.total(s["000012-3"], now()) is None

    status.mark(s, "000012-3", 0, "start", now=now); t[0] = 3.0
    assert status.states(s["000012-3"], now())[0] == ("running", 3.0)
    status.mark(s, "000012-3", 0, "end", now=now)
    status.mark(s, "000012-3", 1, "start", now=now); t[0] = 4.0
    status.mark(s, "000012-3", 1, "end", now=now)
    status.mark(s, "000012-3", 2, "skip", now=now)  # save=False: no save sample
    status.mark(s, "000012-3", 3, "start", now=now); t[0] = 10.0
    assert status.states(s["000012-3"], now()) == [("done", 3.0), ("done", 1.0), ("skipped", None), ("running", 6.0)]
    assert status.total(s["000012-3"], now()) == 10.0 and not status.finished(s["000012-3"])
    status.mark(s, "000012-3", 3, "end", now=now); t[0] = 99.0
    assert status.total(s["000012-3"], now()) == 10.0 and status.finished(s["000012-3"])  # frozen once done


def test_a_failed_stage_freezes_the_row():
    s, t = {}, [0.0]
    now = lambda: t[0]
    status.add(s, "000001-1", now=now)
    with pytest.raises(OSError):
        with status.stage(s, "000001-1", 0, now=now):
            t[0] = 2.0
            raise OSError("disk full")
    t[0] = 50.0
    assert status.states(s["000001-1"], now())[0] == ("failed", 2.0)
    assert status.total(s["000001-1"], now()) == 2.0 and status.finished(s["000001-1"])


def test_marks_are_kept_in_the_samples_times_file_and_a_deleted_sample_shows(tmp_path):
    s, sample_dir = {}, tmp_path / "000002-1"
    status.add(s, "000002-1", sample_dir)
    with status.stage(s, "000002-1", status.RECORD):
        pass  # recording happens before the sample dir exists: written once it does
    assert not sample_dir.exists()
    sample_dir.mkdir()
    with status.stage(s, "000002-1", status.SAVE_VIBRATION):
        pass
    lines = [json.loads(ln) for ln in (sample_dir / "times.jsonl").read_text().splitlines()]
    assert [next(iter(d)) for d in lines] == ["record start", "record end", "save vibration start", "save vibration end"]
    assert not status.deleted(s["000002-1"])
    shutil.rmtree(sample_dir)
    assert status.deleted(s["000002-1"])
    status.mark(s, "000002-1", status.POST_PROCESS, "failed")  # e.g. post process found it gone
    assert not sample_dir.exists()  # marking never recreates a deleted sample


def test_samples_not_recorded_this_session_are_ignored():
    s = {}
    status.mark(s, "000123-1", 3, "start")  # e.g. an old sample the watcher's scan backfills
    with status.stage(s, "000123-1", 1):
        pass
    assert s == {}
