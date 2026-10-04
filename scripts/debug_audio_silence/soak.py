"""Soak test through the real GUI: press Record via the running server's /run, over and over -- the real cameras, feeds,
recording code and threads -- and stop the moment the MOTU goes stuck. Dry runs (nothing saved) with no objects (no
segmentation). Stuck = every speaker's vibrate ~1.75 s (the 1.2 s chirp playing 1.44x too slow, silently) instead of ~1.4 s.

    .venv/Scripts/python.exe scripts/debug_audio_silence/soak.py --n 200 --gap 2

Keep the GUI tab open as you normally would. Everything that happens around the failure is in record.log (browser
playback, audio engine opens/closes, every stage) -- this prints the second it starts.
"""
import argparse
import json
import time
import urllib.request

a = argparse.ArgumentParser()
a.add_argument("--url", default="http://127.0.0.1:8765")
a.add_argument("--n", type=int, default=200)
a.add_argument("--gap", type=float, default=2, help="seconds between positions")
args = a.parse_args()


def get(path): return json.load(urllib.request.urlopen(args.url + path, timeout=10))


def post(path, body):
    req = urllib.request.Request(args.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=10).status


catalog = get("/catalog")
position = dict(catalog["position"], objects={}, prompts={})  # no objects: no segmentation call
since = max([r["seq"] for r in get("/state?since=0")["records"]], default=0)
print(f"soak: {args.n} dry positions, speakers {position['speakers']}, gap {args.gap}s")
for i in range(1, args.n + 1):
    post("/run", dict(position=position, preview=catalog["preview"], save=False, vibrate=True))
    time.sleep(0.5)
    while get("/state?since=999999999999")["recording"]:  # just the flag, no records
        time.sleep(0.3)
    records = get(f"/state?since={since}")["records"]
    since = max([r["seq"] for r in records], default=since)
    pos = next((r for r in reversed(records) if r.get("stage") == "position" and "end" in r), None)
    vib = [round(r["end"] - r["start"], 2) for r in records if r.get("stage") == "vibrate" and "end" in r]
    failed = pos is None or pos["failed"]
    stuck = len(vib) > 0 and min(vib) > 1.6
    print(f"{time.strftime('%H:%M:%S')} #{i:4d} {pos and pos['label']}  vibrate {vib}"
          f"{'  FAILED' if failed else ''}{'  <<<<< STUCK' if stuck else ''}", flush=True)
    if stuck:
        print(f"MOTU stuck at {time.strftime('%H:%M:%S')} -- see record.log just before this position")
        break
    time.sleep(args.gap)
