import json
import threading

import pytest

from record2.log import TAIL, Timing, logger, running, setup_logging


def test_timing_writes_times_jsonl_once_at_exit(tmp_path):
    setup_logging()
    with Timing("000001-3", "vibrate", tmp_path):
        assert not (tmp_path / "times.jsonl").exists()  # the sample dir may not exist yet when a stage starts
    lines = [json.loads(ln) for ln in (tmp_path / "times.jsonl").read_text().splitlines()]
    assert [list(ln) for ln in lines] == [["vibrate start"], ["vibrate end"]]
    end = TAIL[-1]
    assert (end["label"], end["stage"], end["failed"], end["thread"]) == ("000001-3", "vibrate", False, "MainThread")


def test_a_failed_stage_is_logged_as_an_error_and_marked_failed(tmp_path):
    setup_logging()
    with pytest.raises(OSError):
        with Timing("000001-3", "save vibration", tmp_path):
            raise OSError("disk full")
    assert TAIL[-1]["level"] == "ERROR" and TAIL[-1]["failed"]
    assert "save vibration failed" in (tmp_path / "times.jsonl").read_text()


def test_running_lists_unfinished_stages_of_one_position():
    setup_logging()
    with Timing("000007", "position"), Timing("000007-1", "vibrate"):
        assert set(running("000007")) == {("000007", "position"), ("000007-1", "vibrate")}
        assert running("000070") == []  # another position, not a prefix match
    assert running("000007") == []


def test_background_threads_never_print_into_the_cell(capsys):
    """Real bug: a background thread's print lands in whichever notebook cell ran last. Only the
    cell's own thread prints; everything still reaches the log."""
    setup_logging()
    t = threading.Thread(target=lambda: logger.info("from a worker"), name="work_0")
    t.start(); t.join()
    logger.info("from the cell")
    out = capsys.readouterr().out
    assert "from the cell" in out and "from a worker" not in out
    assert any(r["msg"] == "from a worker" and r["thread"] == "work_0" for r in TAIL)


def test_a_dying_thread_is_logged():
    setup_logging()
    t = threading.Thread(target=lambda: 1 / 0, name="bare")
    t.start(); t.join()
    assert any(r["level"] == "ERROR" and "bare" in r["msg"] for r in TAIL)
