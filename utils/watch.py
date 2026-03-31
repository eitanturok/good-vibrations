import time
from pathlib import Path

def watch(directory, idle=3600, poll=5):
    """Call fn(path) for each item in directory. fn must return True (done) or False (retry later).
    Stops after idle seconds with no new items appearing."""
    def dec(fn):
        def wrapper():
            done, pending = set(), set()
            t = time.monotonic()
            try:
                while True:
                    p = Path(directory)
                    if p.is_dir():
                        new = set(p.iterdir()) - done - pending
                        if new:
                            t = time.monotonic()
                            for item in sorted(new):
                                print(f"[{time.strftime('%H:%M:%S')}] Found: {item.name}")
                            pending |= new
                    next_pending = set()
                    for item in sorted(pending):
                        try:
                            if not fn(item): next_pending.add(item)
                        except Exception as e:
                            print(f"[ERROR] {item.name}: {e}")
                            next_pending.add(item)
                    pending = next_pending
                    if not pending and time.monotonic() - t >= idle:
                        break
                    time.sleep(poll)
            except KeyboardInterrupt:
                pass
        return wrapper
    return dec
