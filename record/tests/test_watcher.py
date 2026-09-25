import threading
import time
from pathlib import Path

import pytest

from record.utils.watcher import watch


def _wait_until(predicate, timeout=2.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def test_watcher_survives_a_failing_item_and_keeps_processing(tmp_path):
    """The specific case that prompted this: a sample gets deleted between being queued
    and being picked up (here simulated directly via a raising handler). The single
    persistent worker must not die -- every other queued item still needs processing, and
    it must go on accepting new work afterward."""
    processed = []
    lock = threading.Lock()

    @watch(pattern="*.marker")
    def handler(item):
        if item.name == "bad.marker":
            raise FileNotFoundError(f"simulated deletion: {item}")
        with lock:
            processed.append(item.name)

    handler.start(tmp_path)

    for name in ["a.marker", "bad.marker", "b.marker", "c.marker"]:
        handler.submit(tmp_path / name)

    assert _wait_until(lambda: len(processed) == 3)
    assert set(processed) == {"a.marker", "b.marker", "c.marker"}

    # the failing item landed in failed_samples.jsonl, not lost silently
    failed_path = tmp_path / "failed_samples.jsonl"
    assert _wait_until(failed_path.exists)
    content = failed_path.read_text(encoding="utf-8")
    assert "bad.marker" in content
    assert "simulated deletion" in content

    # the worker is still alive and accepts new work after the failure
    handler.submit(tmp_path / "d.marker")
    assert _wait_until(lambda: "d.marker" in processed)


def test_submit_before_start_raises():
    @watch(pattern="*.marker")
    def handler(item):
        pass

    with pytest.raises(RuntimeError):
        handler.submit(Path("whatever"))


def test_double_start_raises(tmp_path):
    @watch(pattern="*.marker")
    def handler(item):
        pass

    handler.start(tmp_path)
    with pytest.raises(RuntimeError):
        handler.start(tmp_path)


def test_periodic_scan_picks_up_files_not_directly_submitted(tmp_path):
    """The fallback path: a file matching `pattern` that shows up without ever going
    through .submit() (e.g. backfilling an old sample by hand) still gets processed,
    once its size has settled."""
    processed = []

    @watch(pattern="*.marker", poll_rate=0.05)
    def handler(item):
        processed.append(item.name)

    handler.start(tmp_path)
    (tmp_path / "backfill.marker").write_text("x" * (2**20 + 1), encoding="utf-8")  # above MIN_READY_BYTES

    assert _wait_until(lambda: "backfill.marker" in processed, timeout=5.0)
