"""The notebook's real AudioEngine cell, on a fake sounddevice: a sound that didn't play must fail loudly."""
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from record2.tests.fakes import cell

NAMES = ["Main Out 1-2", "Line Out 3-4", "Line Out 5-6", "Line Out 7-8"]


class FakeStream:
    fail = False  # every write raises, as a stream on a device that went away does

    def __init__(self, **kwargs): self.written = 0
    def start(self): pass
    def abort(self): pass
    def close(self): pass

    def write(self, buf):
        if FakeStream.fail:
            raise RuntimeError("Error writing to stream")
        self.written += len(buf)


def engine():
    devices = [dict(name=f"{n} (UltraLite-mk5)", hostapi=0, max_output_channels=2, default_samplerate=44100.0) for n in NAMES]
    sd = SimpleNamespace(query_hostapis=lambda: [{"name": "Windows WASAPI"}], OutputStream=FakeStream,
                         query_devices=lambda i=None: devices if i is None else devices[i])
    from record2.log import logger
    ns = dict(np=np, sd=sd, sf=None, threading=threading, time=time, Path=Path, dataclass=dataclass, field=field, logger=logger)
    exec(cell("class AudioConfig"), ns)
    exec(cell("class AudioEngine"), ns)
    FakeStream.fail = False
    return ns["AudioEngine"](ns["AudioConfig"]())


def test_a_sound_that_did_not_play_fails_the_wait():
    """Real bug: a write failing in its audio thread only logged 'thread audio died' -- record_position went on and saved
    positions with no sound and no error."""
    e = engine()
    FakeStream.fail = True
    e.play(np.zeros(441, np.float32), [3])
    with pytest.raises(RuntimeError, match="speaker"):
        e.wait()


def test_playing_on_a_closed_engine_says_so():
    """Real bug: after close() (and no reset()), every play died with a bare KeyError: 30 in its thread."""
    e = engine()
    e.close()
    with pytest.raises(RuntimeError, match="closed"):
        e.play(np.zeros(441, np.float32), [3])
    e.reset()
    e.play(np.zeros(441, np.float32), [3])
    e.wait()  # reopened: plays again
