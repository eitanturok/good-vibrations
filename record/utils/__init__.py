"""Foundational threading/logging primitives, pulled out of the notebook (unlike GUI-canvas
drawing, these have zero Tk/hardware dependency) so they're importable and unit-testable, and
kept directly in this __init__ rather than a separate task.py so record.utils has one
canonical import surface.
"""

import time
from threading import Thread

from utils.helpers import Timing, logger  # repo-root utils/helpers.py -- reused, not duplicated

class Task(Thread):
    """Run fn(*args, **kwargs) on a daemon thread immediately; .result available after
    .join(). Extends the project's existing Task(Thread) pattern (record.ipynb) with
    thread_id/launch_time/end_time (read directly by the GUI's per-panel timing labels --
    no separate timing plumbing needed through every pipeline function) and, critically,
    with `.exception`: today's Task silently drops any exception fn raises (Python's default
    thread-exception hook just prints a traceback; .result is simply never set) -- fine in a
    cell-by-cell notebook where you'd notice the cell's own output looks wrong, dangerous
    behind a persistent GUI where a failed background Task could leave a panel silently
    stuck, or Record silently never re-enabled, with no visible cause. Callers that care
    should check `.exception` after `.join()` before trusting `.result`."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__(daemon=True)
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.result, self.exception = None, None
        self.thread_id, self.end_time = None, None
        self.launch_time = time.perf_counter()
        self.start()

    def run(self):
        self.thread_id = self.ident
        try:
            self.result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.exception = e
        finally:
            self.end_time = time.perf_counter()


def log(experiment_config, msg: str):
    """The one lifecycle-message helper -- always prints (visible in the notebook's cell
    output regardless of whether the GUI is up) AND enqueues onto
    `experiment_config.log_queue` (a plain queue.Queue, drained by the GUI's own `.after()`
    tick into its status-log widget -- the same thread-safe hand-off already used for
    frames). No callback parameter threaded through every pipeline function: callers just
    call `log(ec, ...)` directly."""
    print(msg)
    experiment_config.log_queue.put(msg)
