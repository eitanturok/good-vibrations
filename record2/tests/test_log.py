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


def test_every_record_says_which_sample_it_is_about(tmp_path):
    """Anything logged inside a stage -- by Timing or by the stage's own code, on that thread -- carries the
    sample id, in record.log and in the GUI's records; outside any stage it's blank."""
    setup_logging(tmp_path / "record.log")
    with Timing("000007", "position"):
        with Timing("000007-3", "vibrate"):
            logger.info("buffer late")
        logger.info("between speakers")
    logger.info("idle")
    by_msg = {r["msg"]: r["sample"] for r in TAIL}
    assert (by_msg["buffer late"], by_msg["vibrate done in 0.00s"], by_msg["between speakers"], by_msg["idle"]) == ("000007-3", "000007-3", "000007", "")
    assert "000007-3 INFO  buffer late" in (tmp_path / "record.log").read_text()


def test_setting_up_logging_again_keeps_the_log_file(tmp_path):
    """Real bug: re-running the notebook's first cells (setup_logging() with no path) after the experiment cell dropped
    record.log's handler -- stages like a 69 s segmentation then existed only in memory."""
    setup_logging(tmp_path / "record.log")
    setup_logging()  # the notebook's second cell, re-run
    logger.info("still on disk")
    assert "still on disk" in (tmp_path / "record.log").read_text()


def test_every_stage_is_timed_on_disk(tmp_path):
    """How long everything took survives the kernel: every Timing stage -- position-level ones like segment too, which
    no sample's times.jsonl has -- is one line of the experiment's times.jsonl, next to record.log."""
    setup_logging(tmp_path / "record.log")
    with Timing("001331", "segment"):
        pass
    with pytest.raises(OSError):
        with Timing("001331-3", "vibrate"):
            raise OSError("timeout")
    lines = [json.loads(ln) for ln in (tmp_path / "times.jsonl").read_text().splitlines()]
    assert [(ln["position"], ln["speaker"], ln["stage"], ln["failed"]) for ln in lines] == [("001331", None, "segment", False), ("001331", 3, "vibrate", True)]
    assert all(ln["seconds"] >= 0 and ln["thread"] == "MainThread" for ln in lines)


def test_a_stage_carries_its_fields_to_the_gui():
    """A position's speakers are on its start record, so the timeline can draw every speaker's box before any plays."""
    setup_logging()
    with Timing("000009", "position", speakers=[1, 3]):
        start = next(r for r in reversed(TAIL) if r.get("stage") == "position" and "end" not in r)
    assert start["speakers"] == [1, 3]
