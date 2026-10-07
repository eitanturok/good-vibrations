"""One log for everything: record.log on disk, TAIL in memory (what the GUI reads), and the notebook
cell only for what the cell's own thread logs -- background threads never print into a random cell.
Every Timing stage is also kept on disk as one line of times.jsonl, next to record.log: how long everything took."""
import collections
import contextvars
import itertools
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("record2")
TAIL = collections.deque(maxlen=5000)  # newest log records as dicts, numbered by "seq", plus their extra={"fields": ...}
                                      # (Timing: label/stage/start/end; a deletion: deleted) and an error's traceback
_seq = itertools.count(1)
_times_lock = threading.Lock()  # the times.jsonl files are appended from several threads
_times_path = None  # the experiment's times.jsonl (next to record.log), set by setup_logging
_sample = contextvars.ContextVar("sample", default="")  # the sample (or position) this thread is working on, set by Timing


def _tag_sample(r):
    r.sample = _sample.get()  # every record says which sample it's about: in record.log, the cell and the GUI
    return True


logger.addFilter(_tag_sample)


class _Tail(logging.Handler):
    def emit(self, r):
        exc = {"exc": logging.Formatter().formatException(r.exc_info)} if r.exc_info else {}
        TAIL.append(dict(seq=next(_seq), t=r.created, level=r.levelname, thread=r.threadName, sample=r.sample,
                         msg=r.getMessage(), **exc, **getattr(r, "fields", {})))


def setup_logging(log_path=None):
    # no log_path: keep the current file -- re-running the notebook's first cells must never silently stop record.log
    global _times_path
    old = next((h for h in logger.handlers if isinstance(h, logging.FileHandler)), None)
    file = old
    if log_path is not None and (old is None or old.baseFilename != str(Path(log_path).resolve())):
        file = logging.FileHandler(log_path, encoding="utf-8")
    _times_path = Path(file.baseFilename).parent / "times.jsonl" if file else None
    cell = logging.StreamHandler(sys.stdout)
    cell.addFilter(lambda r: r.threadName == "MainThread")
    handlers = [cell, _Tail()] + ([file] if file else [])
    for h in handlers:
        h.setFormatter(logging.Formatter("%(asctime)s %(threadName)-12s %(sample)-8s %(levelname)-5s %(message)s"))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # one swap, never cleared first: a stage ending on another thread right now still logs its end (a lost end
    # leaves that stage "running" forever -- and its position undeletable)
    logger.handlers = handlers
    if old is not None and old is not file:
        old.close()
    threading.excepthook = lambda a: logger.error(f"thread {a.thread.name} died", exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
    return logger


def _iso(t): return datetime.fromtimestamp(t, timezone.utc).isoformat()


def _append(path, lines):
    with _times_lock, open(path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(line) + "\n" for line in lines)


class Timing:
    """with Timing(label, stage, sample_dir=None): logs a start record and an end (or failed) record --
    the GUI timeline -- appends one line to the experiment's times.jsonl (position, speaker, stage, start, end,
    seconds, thread, failed), and, given a sample_dir, the stage's start + end|failed to the sample's times.jsonl.
    Everything this thread logs inside it is tagged with the label (the record's `sample`). **fields go on its start
    record for the GUI (e.g. a position's speakers, so the timeline draws a box for each before they play)."""
    def __init__(self, label, stage, sample_dir=None, **fields):
        self.label, self.stage, self.sample_dir, self.fields = str(label), stage, sample_dir, fields

    def __enter__(self):
        self.start, self.token = time.time(), _sample.set(self.label)
        logger.info(f"{self.stage}", extra={"fields": dict(label=self.label, stage=self.stage, start=self.start, **self.fields)})
        return self

    def __exit__(self, exc_type, exc, tb):
        end, failed = time.time(), exc_type is not None
        timing = dict(label=self.label, stage=self.stage, start=self.start, end=end, failed=failed)
        msg = f"{self.stage} {'FAILED' if failed else 'done'} in {end - self.start:.2f}s"
        if failed: logger.error(msg, extra={"fields": timing}, exc_info=(exc_type, exc, tb))
        else: logger.info(msg, extra={"fields": timing})
        _sample.reset(self.token)
        if _times_path is not None:
            position, _, speaker = self.label.partition("-")  # "001331" or "001331-3"
            _append(_times_path, [dict(position=position, speaker=int(speaker) if speaker else None, stage=self.stage,
                                       start=_iso(self.start), end=_iso(end), seconds=round(end - self.start, 3),
                                       thread=threading.current_thread().name, failed=failed)])
        if self.sample_dir is not None:
            _append(Path(self.sample_dir) / "times.jsonl",
                    [{f"{self.stage} start": _iso(self.start)}, {f"{self.stage} {'failed' if failed else 'end'}": _iso(end)}])


def running(position_id=None):
    """(label, stage) of every Timing started this session that hasn't ended -- only those of
    `position_id` (labels "000012" and "000012-3") if given."""
    started, ended = {}, set()
    for r in list(TAIL):
        if "stage" in r and position_id in (None, r["label"].split("-")[0]):
            key = (r["label"], r["stage"], r["start"])
            (ended.add(key) if "end" in r else started.__setitem__(key, (r["label"], r["stage"])))
    return [v for k, v in started.items() if k not in ended]
