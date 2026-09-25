"""Generic "watch a directory, process what shows up" decorator.

Adapted from utils/watch_and_process.py's LocalEngine + watch_loop (proven design: a
single persistent worker thread, since GPU-bound post-processing gains nothing from
concurrency and only risks VRAM/OOM contention -- see that module's own docstring), turned
into a small reusable decorator so record/post_process.py doesn't need its own bespoke
queue/worker code.
"""

import sys
import time
import queue
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone

from utils.io_utils import append

MIN_READY_BYTES = 1 * 2**20  # reject files caught mid-write at a few KB/MB


def _is_file_ready(path: Path, check_interval: float = 1.0) -> bool:
    """True once the file's size is stable (done writing) and above a sanity floor."""
    try:
        s1 = path.stat().st_size
        time.sleep(check_interval)
        return s1 == path.stat().st_size and s1 >= MIN_READY_BYTES
    except OSError:
        return False  # vanished or locked by the writer -- retry next tick


class _Watcher:
    def __init__(self, fn, dir: Path, pattern: str, poll_rate: float):
        self.fn, self.dir, self.pattern, self.poll_rate = fn, Path(dir), pattern, poll_rate
        self.q: "queue.Queue[Path]" = queue.Queue()
        self._seen: dict[str, float] = {}  # str(path) -> mtime already queued/handled, so a
                                            # delete + re-record under the same path (new
                                            # mtime) reads as unseen and gets requeued
        self.failed_path = self.dir / "failed_samples.jsonl"
        threading.Thread(target=self._worker, daemon=True, name=f"watch-{fn.__name__}").start()
        threading.Thread(target=self._scan_loop, daemon=True, name=f"watch-{fn.__name__}-scan").start()

    def submit(self, item):
        item = Path(item)
        try:
            mtime = item.stat().st_mtime if item.exists() else time.time()
        except OSError:
            mtime = time.time()
        if self._seen.get(str(item)) == mtime:
            return
        self._seen[str(item)] = mtime
        self.q.put(item)

    def _worker(self):
        while True:
            item = self.q.get()
            # Crash isolation: one failing item (most commonly FileNotFoundError, because
            # it was deleted between being queued and being picked up) must never kill this
            # persistent worker -- every other queued item still needs processing for the
            # rest of the session.
            try:
                self.fn(item)
            except Exception as e:
                row = {"item": str(item), "error": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc(), "time": datetime.now(timezone.utc).isoformat()}
                try:
                    append(row, self.failed_path)
                except Exception:
                    pass  # even the failure ledger write failing must not kill the worker
                print(f"[watch:{self.fn.__name__}] {item} FAILED: {e}", file=sys.stderr)

    def _scan_loop(self):
        while True:
            # A whole tick is wrapped so one bad/deleted file never kills the scan loop
            # either (e.g. rglob walking into a directory deleted mid-scan raises here).
            try:
                for match in self.dir.rglob(self.pattern):
                    try:
                        mtime = match.stat().st_mtime
                    except OSError:
                        continue  # vanished since rglob listed it -- retry next tick
                    if self._seen.get(str(match)) == mtime:
                        continue
                    if not _is_file_ready(match):
                        continue
                    self.submit(match)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print(f"[watch:{self.fn.__name__}] scan tick failed, will retry: {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(self.poll_rate)


def watch(pattern: str, poll_rate: float = 2.0):
    """Decorator: gives the wrapped function a single persistent background worker (so
    concurrent items never pile up in memory/VRAM) fed two ways: (1) a `.submit(item)`
    method for immediate, zero-latency direct hand-off, and (2) a periodic scan of a
    directory for files matching `pattern` (mtime-dedup, so a delete + re-record under the
    same path is picked up again) -- a fallback that catches anything not explicitly
    submitted, e.g. after a kernel crash, or an old item you want to backfill by hand.

    `pattern` is fixed at decoration time; the directory to watch is bound later, via the
    decorated function's own `.start(dir)` (called once, e.g. at notebook startup once
    `experiment_dir` is known) -- `.submit()` raises until `.start()` has been called.

    Crash isolation: the worker wraps EACH item's call to the decorated function in its own
    try/except -- a failing item is logged to `<dir>/failed_samples.jsonl` and skipped,
    never allowed to kill the worker thread itself."""
    def decorator(fn):
        state: dict[str, _Watcher | None] = {"watcher": None}

        def start(dir):
            if state["watcher"] is not None:
                raise RuntimeError(f"{fn.__name__}.start() was already called")
            state["watcher"] = _Watcher(fn, dir, pattern, poll_rate)
            return state["watcher"]

        def submit(item):
            if state["watcher"] is None:
                raise RuntimeError(f"{fn.__name__}.start(dir) must be called before .submit()")
            state["watcher"].submit(item)

        fn.start = start
        fn.submit = submit
        return fn
    return decorator
