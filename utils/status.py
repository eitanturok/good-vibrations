import json
import os
from pathlib import Path

from filelock import FileLock


STAGES = ["move_data", "upload"]

NOT_STARTED = "not_started"
IN_PROGRESS = "in_progress"
FINISHED    = "finished"


def _path(sample_dir): return Path(sample_dir) / "status.json"
def _lock(sample_dir): return FileLock(str(Path(sample_dir) / "status.lock"))


def read(sample_dir):
    p = _path(sample_dir)
    if not p.exists():
        return {s: NOT_STARTED for s in STAGES}
    return json.loads(p.read_text())


def _write(sample_dir, status):
    p = _path(sample_dir)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    os.replace(tmp, p)


def claim(sample_dir, stage, prerequisite=None):
    """Atomically claim a stage. Returns 'claimed', 'waiting', or 'taken'."""
    with _lock(sample_dir):
        status = read(sample_dir)
        if prerequisite and status.get(prerequisite) != FINISHED:
            return "waiting"
        if status.get(stage) == FINISHED:
            return "finished"
        if status.get(stage) == IN_PROGRESS:
            return "taken"
        status[stage] = IN_PROGRESS
        _write(sample_dir, status)
        return "claimed"


def finish(sample_dir, stage):
    with _lock(sample_dir):
        status = read(sample_dir)
        status[stage] = FINISHED
        _write(sample_dir, status)
