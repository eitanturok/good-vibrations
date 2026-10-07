"""Run fn(path) once per file matching `pattern` under `dir`, oldest first, one at a time, on one thread.
Producers must make files appear atomically (write a .tmp, then os.replace) -- a matched file is complete."""
import threading
from pathlib import Path

from record2.log import logger


class Watcher:
    def __init__(self, fn, dir, pattern, poll=2.0):
        self.fn, self.dir, self.pattern, self.poll = fn, Path(dir), pattern, poll
        self.seen, self.stop_event = set(), threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name=f"watch-{self.fn.__name__}").start()
        return self

    def stop(self): self.stop_event.set()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                for path in sorted(self.dir.glob(self.pattern)):
                    if self.stop_event.is_set(): break
                    if path in self.seen or not path.exists(): continue  # done, or deleted since the glob
                    self.seen.add(path)
                    try: self.fn(path)
                    except Exception: logger.exception(f"{self.fn.__name__}({path}) failed")
            except Exception:
                logger.exception(f"watch-{self.fn.__name__}: scan failed, retrying")
            self.stop_event.wait(self.poll)
