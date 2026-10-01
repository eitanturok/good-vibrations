from record2.utils import close_previous_instance, stop_then_close


class _FakeResource:
    def __init__(self):
        self.calls = []

    def stop(self): self.calls.append("stop")
    def close(self): self.calls.append("close")


def test_stop_then_close_calls_in_order():
    """The exact bug this exists to catch: closing a streaming handle without stopping it
    first can leave the physical device thinking it's still acquiring, locking a feature
    for the next handle opened against it (e.g. egrabber's "MultiROILUTModeEn is locked")."""
    r = _FakeResource()
    stop_then_close(r)
    assert r.calls == ["stop", "close"]


class _FakeCamera:
    _current = None

    def __init__(self):
        close_previous_instance(_FakeCamera)
        self.closed = False
        _FakeCamera._current = self

    def close(self):
        self.closed = True
        if _FakeCamera._current is self:
            _FakeCamera._current = None


def test_close_previous_instance_on_reinit():
    """Constructing a second instance in a row (e.g. re-running a notebook cell) must close
    the first automatically -- the case that used to require an explicit
    `if "x" in globals(): x.close()` guard in the notebook before every construction."""
    a = _FakeCamera()
    b = _FakeCamera()
    assert a.closed
    assert not b.closed
    assert _FakeCamera._current is b


def test_close_previous_instance_noop_when_none():
    """First-ever construction (no previous instance) shouldn't try to close anything."""
    class _Cls:
        _current = None

    close_previous_instance(_Cls)  # must not raise
    assert _Cls._current is None
