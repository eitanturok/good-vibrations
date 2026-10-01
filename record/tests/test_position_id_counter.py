"""A position id is never handed out twice: not by concurrent threads, not by separate processes
(two kernels on the rig), not by a later session."""
import subprocess
import sys

import pytest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from record.utils.position_id import PositionIdCounter

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ids_persist_across_sessions(tmp_path):
    path = tmp_path / "ids" / "position_id.txt"
    assert PositionIdCounter(path, create=True).next() == 1
    assert [PositionIdCounter(path).next() for _ in range(2)] == [2, 3]  # a new counter object each time: the file carries it


def test_a_missing_counter_is_an_error_not_a_restart(tmp_path):
    path = tmp_path / "position_id.txt"
    with pytest.raises(FileNotFoundError):
        PositionIdCounter(path).next()  # never created
    assert PositionIdCounter(path, create=True).next() == 1
    assert PositionIdCounter(path, create=True).next() == 2  # the flag never resets an existing counter
    path.unlink()
    with pytest.raises(FileNotFoundError):
        PositionIdCounter(path).next()  # deleted: raise, don't hand out 1 again


def test_concurrent_threads_and_processes_never_share_an_id(tmp_path):
    path = tmp_path / "position_id.txt"
    script = f"from record.utils.position_id import PositionIdCounter as C; c = C(r'{path}', create=True); print(*[c.next() for _ in range(50)])"
    procs = [subprocess.Popen([sys.executable, "-c", script], cwd=REPO_ROOT, stdout=subprocess.PIPE, text=True) for _ in range(3)]
    counter = PositionIdCounter(path, create=True)
    with ThreadPoolExecutor(8) as pool:
        ids = list(pool.map(lambda _: counter.next(), range(200)))
    for p in procs:
        out, _ = p.communicate(timeout=60)
        assert p.returncode == 0
        ids += [int(x) for x in out.split()]
    assert sorted(ids) == list(range(1, 351))  # every id handed out exactly once, no gaps without failures
