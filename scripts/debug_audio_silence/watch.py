"""Watch the running GUI for the MOTU going stuck -- passively: it only reads the server's /state (what the page polls),
never touches the MOTU. After every finished position (from the GUI, the soak, or the notebook), its speakers' vibrate
times: ~1.4 s healthy, all ~1.75 s = stuck (the 1.2 s chirp playing 1.44x too slow, silently). On the first stuck
position it writes everything logged in the 2 minutes before it (browser playback, audio engine opens/closes, every
stage, errors with tracebacks) to a report, and exits.

    .venv/Scripts/python.exe scripts/debug_audio_silence/watch.py --report D:/eturok/stuck_report.txt
"""
import argparse
import json
import time
import urllib.request

a = argparse.ArgumentParser()
a.add_argument("--url", default="http://127.0.0.1:8765")
a.add_argument("--report", default="stuck_report.txt")
a.add_argument("--before", type=float, default=120, help="seconds of context before the stuck position")
args = a.parse_args()


def state(since): return json.load(urllib.request.urlopen(f"{args.url}/state?since={since}", timeout=10))


def clock(t): return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 1000):03d}"


history = state(0)["records"]  # all records kept, so the report can reach back before the stuck position
since = max([r["seq"] for r in history], default=0)
done = {r["label"] for r in history if r.get("stage") == "position" and "end" in r}  # positions already finished
print(f"{time.strftime('%H:%M:%S')} watching {args.url}; {len(done)} positions already done", flush=True)
while True:
    time.sleep(2)
    try:
        new = state(since)["records"]
    except OSError as e:  # the server restarting: keep watching
        print(f"{time.strftime('%H:%M:%S')} server unreachable: {e}", flush=True)
        continue
    history += new
    since = max([r["seq"] for r in new], default=since)
    for p in [r for r in new if r.get("stage") == "position" and "end" in r and r["label"] not in done]:
        done.add(p["label"])
        vib = [round(r["end"] - r["start"], 2) for r in history if r.get("stage") == "vibrate" and "end" in r
               and r["label"].startswith(p["label"] + "-")]
        stuck = len(vib) >= 2 and min(vib) > 1.6
        print(f"{clock(p['start'])} {p['label']}  vibrate {vib}{'  FAILED' if p['failed'] else ''}{'  <<<<< STUCK' if stuck else ''}", flush=True)
        if stuck:
            context = [r for r in history if p["start"] - args.before <= r["t"] <= p["end"] + 1]
            with open(args.report, "w", encoding="utf-8") as f:
                f.write(f"MOTU stuck at position {p['label']} ({clock(p['start'])}), vibrate {vib}\n"
                        f"everything logged from {args.before:.0f} s before it:\n\n")
                for r in context:
                    f.write(f"{clock(r['t'])} {r['thread']:20s} {r.get('sample', ''):9s} {r['level']:7s} {r['msg']}\n")
                    if r.get("exc"):
                        f.write(r["exc"] + "\n")
            print(f"STUCK -- report: {args.report} ({len(context)} records)", flush=True)
            raise SystemExit(0)
