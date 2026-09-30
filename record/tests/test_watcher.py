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

    for name in ["a.marker", "bad.marker", "b.marker", "c.marker"]:  # never written: submit doesn't need the file yet
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


def test_start_again_reuses_the_worker(tmp_path):
    """Real bug: re-running the notebook's ExperimentConfig cell called .start() again and
    raised "post_process.start() was already called". Re-running must just work: still one
    worker (no second GPU job in parallel), now watching the latest directory."""
    processed = []

    @watch(pattern="*.marker", poll_rate=0.05)
    def handler(item):
        processed.append(item)

    first = handler.start(tmp_path / "old")
    (tmp_path / "new").mkdir()
    assert handler.start(tmp_path / "new") is first  # same watcher, same single worker
    (tmp_path / "new" / "x.marker").write_text("x" * (2**20 + 1), encoding="utf-8")
    assert _wait_until(lambda: tmp_path / "new" / "x.marker" in processed, timeout=5.0)
    assert first.failed_path == tmp_path / "new" / "failed_samples.jsonl"


def test_submitted_and_scanned_sample_is_processed_once(tmp_path):
    """Real bug: save_raw_vibration submitted the sample DIR while the scan found its raw FILE --
    two keys, so every sample was queued twice; the 2nd run found the raw file already deleted by
    the 1st and failed ("...01_raw_vibrations.npy\\metadata.jsonl" not found). An item is the
    matched file, so the direct hand-off and the scan dedupe against each other."""
    processed = []

    @watch(pattern="**/vibration/01_raw_vibrations.npy", poll_rate=0.05)
    def handler(raw_path):
        processed.append(raw_path)
        time.sleep(0.3)  # still processing while the scan ticks
        raw_path.unlink()  # like post_process: the raw file is deleted once done

    handler.start(tmp_path)
    raw = tmp_path / "samples/000063-1/vibration/01_raw_vibrations.npy"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"x" * (2**20 + 1))
    with pytest.raises(ValueError):
        handler.submit(tmp_path / "samples/000063-1")  # not the matched file: rejected, not silently double-queued
    handler.submit(raw)

    assert _wait_until(lambda: not raw.exists(), timeout=5.0)
    time.sleep(1.5)  # several scan ticks (each waits ~1 s for the size to settle)
    assert processed == [raw] and not (tmp_path / "failed_samples.jsonl").exists()


def test_on_event_reports_when_each_item_starts_and_ends(tmp_path):
    """The GUI's status list shows when post-processing starts and ends (or fails) per sample."""
    events = []

    @watch(pattern="*.marker")
    def handler(item):
        if item.name == "bad.marker":
            raise OSError("boom")

    handler.start(tmp_path, on_event=lambda item, event: events.append((item.name, event)))
    handler.submit(tmp_path / "a.marker")
    handler.submit(tmp_path / "bad.marker")
    assert _wait_until(lambda: len(events) == 4)
    assert events == [("a.marker", "start"), ("a.marker", "end"), ("bad.marker", "start"), ("bad.marker", "failed")]


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
