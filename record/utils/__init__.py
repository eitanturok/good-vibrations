"""Foundational threading/logging primitives, pulled out of the notebook (unlike GUI-canvas
drawing, these have zero Tk/hardware dependency) so they're importable and unit-testable, and
kept directly in this __init__ rather than a separate task.py so record.utils has one
canonical import surface.
"""

import sys
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
            # also report it: most tasks (every plot_*) are never joined, so a stored-only
            # exception is invisible -- a panel just silently never updates
            print(f"[task {getattr(self.fn, '__name__', self.fn)}] FAILED: {e!r}", file=sys.stderr)
        finally:
            self.end_time = time.perf_counter()


def close_previous_instance(cls):
    """If `cls` has a previously registered live instance (`cls._current`), close() it and
    clear the registration. Call this as the FIRST line of __init__, before opening any new
    hardware handle, so the previous one is released before the new one is requested;
    register the new instance yourself (`cls._current = self`) as the LAST line of
    __init__, only once construction actually succeeds. Lets a hardware-wrapper class
    enforce "at most one open handle at a time" without every construction site needing its
    own `if "x" in globals(): x.close()` guard."""
    if cls._current is not None:
        cls._current.close()
        cls._current = None


def stop_then_close(resource):
    """Calls resource.stop() then resource.close(), in that order -- the correct teardown
    sequence for a streaming hardware handle (e.g. an EGrabber): stop() tells the device to
    actually stop acquiring, which releases any SDK feature locks tied to "acquisition
    active" -- and must happen before close() tears down the local handle. Skipping
    straight to close() destroys the software object while the physical device still
    thinks it's acquiring, so the next handle opened against the same device can fail with
    a "feature is locked" error that has nothing to do with the new handle at all."""
    resource.stop()
    resource.close()


def log(experiment_config, msg: str):
    """The one lifecycle-message helper: prints to the notebook's cell output. The GUI shows
    per-sample stage progress instead (record.utils.status), not these messages."""
    print(msg)


def buffer_sizes_dividing(n_frames, smallest=25):
    """Laser frames-per-buffer choices that split a capture of n_frames into whole buffers --
    the camera only hands over full buffers, so any other size waits for (and throws away)
    a partial last buffer after the audio has ended."""
    return [b for b in range(smallest, n_frames // 2 + 1) if n_frames % b == 0]
