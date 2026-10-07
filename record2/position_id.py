"""The position-id counter, shared by every experiment on every day, and the single source of truth
for position ids: a sample is named utils.ids.sample_name(position_id, speaker), so a position id is
never handed out twice -- not across threads, not
across kernels/processes, not across experiments. Ids only go up; an id whose position failed or
was thrown away is simply skipped (a gap), never reused.

The counter is one text file holding the next id. A file lock makes read -> +1 -> write atomic,
and the new value is written to a temp file and os.replace'd in, so a crash never leaves a
half-written counter. It lives in the repo root (a local disk: file locks are unreliable on
network/synced drives), outside any experiment dir -- never delete it or restore an old copy.
A missing file is an error, never a silent restart at 1 -- unless created on purpose, with create=True.
"""
import os
from pathlib import Path

from filelock import FileLock

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "position_id.txt"  # good-vibrations/position_id.txt


class PositionIdCounter:
    def __init__(self, path=DEFAULT_PATH, create=False):
        self.path, self.create = Path(path), create
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self.path) + ".lock", timeout=30)

    def peek(self) -> int | None:
        # the id next() hands out, without taking it; None when next() would raise
        return int(self.path.read_text()) if self.path.exists() else 1 if self.create else None

    def next(self) -> int:
        with self._lock:
            if not self.path.exists() and not self.create:
                raise FileNotFoundError(f"position-id counter {self.path} is missing -- restore it, or pass "
                                        f"create=True to start a new one at 1 (only if no position id was ever handed out)")
            i = int(self.path.read_text()) if self.path.exists() else 1
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "w") as f:
                f.write(str(i + 1))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            return i
