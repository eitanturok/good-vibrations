"""Minimal repro for "the speakers go silent": the notebook's REAL AudioEngine (4 always-open MOTU streams), playing
the chirp on one speaker in a loop like a recording does -- chirp, then the done whistle on every speaker, not waited --
and after every loop iteration a check that the MOTU is still streaming: a live MOTU input delivers 1 s of samples in
~1 s with real noise in them; a stuck one takes ~3 s and returns exact zeros. Nothing else runs: no cameras, no GUI.

While it runs, do ONE candidate trigger at a time somewhere else and watch for the line that flips to STUCK:
  - in the browser: click "recovered audio" in the GUI, or play any video (Windows' default output is a MOTU output)
  - reload / open a GUI tab
  - nothing at all (does it die on its own?)

    .venv/Scripts/python.exe scripts/debug_audio_silence/repro.py --speaker 3 --gap 3

Run it with the notebook's audio engine closed (hw.audio_engine.close()), and the MOTU healthy (power-cycled).
"""
import argparse
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from record2.log import logger, setup_logging
from record2.tests.fakes import cell

a = argparse.ArgumentParser()
a.add_argument("--speaker", type=int, default=3)
a.add_argument("--gap", type=float, default=3, help="seconds between positions")
a.add_argument("--n", type=int, default=1000)
args = a.parse_args()

setup_logging()
ns = dict(np=np, sd=sd, sf=sf, threading=threading, time=time, Path=Path, dataclass=dataclass, field=field, logger=logger)
exec(cell("class AudioConfig"), ns)
exec(cell("class AudioEngine"), ns)
engine = ns["AudioEngine"](ns["AudioConfig"]())
rate = engine.sample_rate
t = np.arange(int(1.2 * rate)) / rate
chirp = (0.3 * np.sin(2 * np.pi * (100 * t + 900 / 2.4 * t ** 2))).astype(np.float32)  # 100 -> 1000 Hz in 1.2 s, like the real one
whistle = (0.2 * np.sin(2 * np.pi * 1500 * t[: rate // 2])).astype(np.float32)
api = next(i for i, h in enumerate(sd.query_hostapis()) if h["name"] == "Windows WASAPI")
motu_in = next(i for i, d in enumerate(sd.query_devices()) if d["hostapi"] == api and "Line In 3-4" in d["name"])


def motu_alive():
    t0 = time.perf_counter()
    x = sd.rec(rate // 2, samplerate=rate, channels=2, device=motu_in, dtype="float32", blocking=True)
    took = time.perf_counter() - t0
    return took < 1.0 and np.mean(x == 0) < 0.5, took, np.mean(x == 0)


print(f"speaker {args.speaker}, {rate} Hz, MOTU input {sd.query_devices(motu_in)['name']}")
for i in range(1, args.n + 1):
    t0 = time.perf_counter()
    engine.play(chirp, [args.speaker])
    engine.wait()
    play = time.perf_counter() - t0  # 1.2 s of chirp: ~1.2 s when healthy, ~1.7 s when stuck
    engine.play(whistle, list(engine.config.speaker_device_names))  # the done whistle, not waited -- as record_position
    alive, took, zeros = motu_alive()
    print(f"{time.strftime('%H:%M:%S')} #{i:4d}  chirp played in {play:.2f}s  check {took:.2f}s zeros {zeros:5.1%}  "
          f"{'healthy' if alive else '<<<<< STUCK'}", flush=True)
    if not alive:
        break
    time.sleep(args.gap)
engine.close()
