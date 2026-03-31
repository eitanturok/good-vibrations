import time
from pathlib import Path

def watch(directory, idle=3600, poll=5):
    """Call fn(path) for each new item in directory. Stop after idle seconds with nothing new."""
    def dec(fn):
        def wrapper():
            seen, t = set(), time.monotonic()
            while True:
                p = Path(directory)
                new = (set(p.iterdir()) - seen) if p.is_dir() else set()
                if new:
                    seen |= new; t = time.monotonic()
                    for p in sorted(new): fn(p)
                elif time.monotonic() - t >= idle: break
                time.sleep(poll)
        except KeyboardInterrupt:
            pass
        return wrapper
    return dec
