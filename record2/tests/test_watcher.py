import time

from record2.log import TAIL, setup_logging
from record2.watcher import Watcher

PATTERN = "*/vibration/01_raw_vibrations.npy"


def wait_for(cond, timeout=3.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        time.sleep(0.02)
    return cond()


def raw(tmp_path, sample, name="01_raw_vibrations.npy"):
    path = tmp_path / sample / "vibration" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_each_file_once_and_never_a_half_written_one(tmp_path):
    done = []
    raw(tmp_path, "000001-1", "01_raw_vibrations.npy.tmp")  # still being written
    raw(tmp_path, "000001-1")
    w = Watcher(done.append, tmp_path, PATTERN, poll=0.05).start()
    raw(tmp_path, "000001-3")
    assert wait_for(lambda: len(done) == 2)
    time.sleep(0.2)
    w.stop()
    assert [p.parent.parent.name for p in done] == ["000001-1", "000001-3"]


def test_a_failure_is_logged_and_the_next_file_still_runs(tmp_path):
    setup_logging()
    done = []
    def fn(path):
        if path.parent.parent.name == "000001-1":
            raise ValueError("bad sample")
        done.append(path)
    raw(tmp_path, "000001-1"), raw(tmp_path, "000001-3")
    w = Watcher(fn, tmp_path, PATTERN, poll=0.05).start()
    assert wait_for(lambda: len(done) == 1)
    w.stop()
    assert any(r["level"] == "ERROR" and "000001-1" in r["msg"] for r in TAIL)
