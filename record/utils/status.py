"""Per-sample stage status for the GUI's status list: every sample ("{position_id}-{speaker}") goes
through STAGES in order, each marked from whichever thread runs it. A plain dict, sample ->
row; each row is only ever written by its own sample's stages, and the GUI tick only reads.
Every mark is also appended to the sample's times.jsonl, so its stage timings outlive the session."""

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("record", "save vibration", "save sample", "post process")
RECORD, SAVE_VIBRATION, SAVE_SAMPLE, POST_PROCESS = range(len(STAGES))


def add(status: dict, sample: str, sample_dir=None, now=time.perf_counter):
    status[sample] = {"label": sample, "dir": None if sample_dir is None else Path(sample_dir),
                      "start": [None] * len(STAGES), "end": [None] * len(STAGES),
                      "skipped": [False] * len(STAGES), "failed": None, "unwritten": []}


def mark(status: dict, sample: str, i: int, event: str, now=time.perf_counter):
    """event: "start" | "end" | "failed" | "skip". Samples not added this session are ignored
    (e.g. an old sample the post-process watcher's scan picks up)."""
    row = status.get(sample)
    if row is None:
        return
    if event == "start":
        row["start"][i] = now()
    elif event == "skip":
        row["skipped"][i] = True
    else:
        row["end"][i] = now()
        if event == "failed":
            row["failed"] = i
    # to times.jsonl -- held back until the sample dir exists (it doesn't yet while recording), and
    # a deleted sample's dir is never recreated
    row["unwritten"].append(json.dumps({f"{STAGES[i]} {event}": datetime.now(timezone.utc).isoformat()}) + "\n")
    if row["dir"] is not None and row["dir"].exists():
        with open(row["dir"] / "times.jsonl", "a", encoding="utf-8") as f:
            f.writelines(row["unwritten"])
        row["unwritten"] = []


@contextmanager
def stage(status: dict, sample: str, i: int, now=time.perf_counter):
    mark(status, sample, i, "start", now)
    try:
        yield
    except BaseException:
        mark(status, sample, i, "failed", now)
        raise
    mark(status, sample, i, "end", now)


def states(row: dict, now: float) -> list[tuple[str, float | None]]:
    """(state, seconds) per stage: waiting / running / done / failed / skipped."""
    out = []
    for i in range(len(STAGES)):
        start, end = row["start"][i], row["end"][i]
        if row["skipped"][i]:
            out.append(("skipped", None))
        elif start is None:
            out.append(("waiting", None))
        elif end is None:
            out.append(("running", now - start))
        else:
            out.append(("failed" if row["failed"] == i else "done", end - start))
    return out


def finished(row: dict) -> bool:
    return row["failed"] is not None or all(e is not None or s for e, s in zip(row["end"], row["skipped"]))


def deleted(row: dict) -> bool:
    # the sample's directory was removed by hand (only known once it had been created)
    return row["dir"] is not None and row["start"][SAVE_VIBRATION] is not None and not row["dir"].exists()


def total(row: dict, now: float) -> float | None:
    """From the start of recording to the last stage's end -- ticking while in progress."""
    if row["start"][0] is None:
        return None
    if finished(row):
        return max(e for e in row["end"] if e is not None) - row["start"][0]
    return now - row["start"][0]
