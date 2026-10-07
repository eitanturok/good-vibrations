"""A steady tone on one speaker, long enough to look at the MOTU's output meters and the amp's LEDs while it plays.
Its own process: it never touches the notebook's kernel or its streams.

    .venv/Scripts/python.exe scripts/debug_audio_silence/tone.py --speaker 3 --seconds 8 --amp 0.2
"""
import argparse
import time

import numpy as np
import sounddevice as sd

DEVICES = {1: "Main Out 1-2", 2: "Main Out 1-2", 3: "Line Out 3-4", 4: "Line Out 3-4",  # = AudioConfig's defaults
           5: "Line Out 5-6", 6: "Line Out 5-6", 7: "Line Out 7-8", 8: "Line Out 7-8"}
CHANNELS = {1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 0, 8: 1}

a = argparse.ArgumentParser()
a.add_argument("--speaker", type=int, default=3)
a.add_argument("--seconds", type=float, default=8)
a.add_argument("--amp", type=float, default=0.2, help="0..1 of full scale")
a.add_argument("--freq", type=float, default=440)
args = a.parse_args()

api = next(i for i, h in enumerate(sd.query_hostapis()) if h["name"] == "Windows WASAPI")
device = next(i for i, d in enumerate(sd.query_devices())
              if d["hostapi"] == api and DEVICES[args.speaker] in d["name"] and d["max_output_channels"] > 0)
rate = int(sd.query_devices(device)["default_samplerate"])
t = np.arange(int(rate * args.seconds)) / rate
buf = np.zeros((len(t), 2), np.float32)
buf[:, CHANNELS[args.speaker]] = args.amp * np.sin(2 * np.pi * args.freq * t)
print(f"speaker {args.speaker}: {sd.query_devices(device)['name']} channel {CHANNELS[args.speaker]}, "
      f"{args.freq:.0f} Hz at {args.amp:.2f} of full scale for {args.seconds:.0f} s -- watch the MOTU meters and the amp now")
with sd.OutputStream(samplerate=rate, device=device, channels=2) as s:
    t0 = time.perf_counter()
    s.write(buf)
    print(f"done: {args.seconds:.1f} s of audio took {time.perf_counter() - t0:.2f} s to play")
