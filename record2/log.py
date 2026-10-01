"""One log for everything: record.log on disk, TAIL in memory (what the GUI reads), and the notebook
cell only for what the cell's own thread logs -- background threads never print into a random cell."""
import collections
import itertools
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("record2")
TAIL = collections.deque(maxlen=5000)  # newest log records as dicts, numbered by "seq"; Timing records add label/stage/start/end
_seq = itertools.count(1)
_times_lock = threading.Lock()  # times.jsonl is appended from several threads


class _Tail(logging.Handler):
    def emit(self, r):
        TAIL.append(dict(seq=next(_seq), t=r.created, level=r.levelname, thread=r.threadName,
                         msg=r.getMessage(), **getattr(r, "timing", {})))


def setup_logging(log_path=None):
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    cell = logging.StreamHandler(sys.stdout)
    cell.addFilter(lambda r: r.threadName == "MainThread")
    handlers = [cell, _Tail()] + ([logging.FileHandler(log_path, encoding="utf-8")] if log_path else [])
    for h in handlers:
        h.setFormatter(logging.Formatter("%(asctime)s %(threadName)-12s %(levelname)-5s %(message)s"))
        logger.addHandler(h)
    threading.excepthook = lambda a: logger.error(f"thread {a.thread.name} died", exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
    return logger


def _iso(t): return datetime.fromtimestamp(t, timezone.utc).isoformat()


class Timing:
    """with Timing(label, stage, sample_dir=None): logs a start record and an end (or failed) record --
    the GUI timeline -- and, given a sample_dir, appends the stage's start + end|failed to its times.jsonl."""
    def __init__(self, label, stage, sample_dir=None):
        self.label, self.stage, self.sample_dir = str(label), stage, sample_dir

    def __enter__(self):
        self.start = time.time()
        logger.info(f"{self.label} {self.stage}", extra={"timing": dict(label=self.label, stage=self.stage, start=self.start)})
        return self

    def __exit__(self, exc_type, exc, tb):
        end, failed = time.time(), exc_type is not None
        timing = dict(label=self.label, stage=self.stage, start=self.start, end=end, failed=failed)
        msg = f"{self.label} {self.stage} {'FAILED' if failed else 'done'} in {end - self.start:.2f}s"
        if failed: logger.error(msg, extra={"timing": timing}, exc_info=(exc_type, exc, tb))
        else: logger.info(msg, extra={"timing": timing})
        if self.sample_dir is not None:
            lines = [{f"{self.stage} start": _iso(self.start)}, {f"{self.stage} {'failed' if failed else 'end'}": _iso(end)}]
            with _times_lock, open(Path(self.sample_dir) / "times.jsonl", "a", encoding="utf-8") as f:
                f.writelines(json.dumps(line) + "\n" for line in lines)


def running(position_id=None):
    """(label, stage) of every Timing started this session that hasn't ended -- only those of
    `position_id` (labels "000012" and "000012-3") if given."""
    started, ended = {}, set()
    for r in list(TAIL):
        if "stage" in r and position_id in (None, r["label"].split("-")[0]):
            key = (r["label"], r["stage"], r["start"])
            (ended.add(key) if "end" in r else started.__setitem__(key, (r["label"], r["stage"])))
    return [v for k, v in started.items() if k not in ended]
