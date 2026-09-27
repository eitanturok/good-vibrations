"""The GUI's status list: one line per sample, newest on top -- "{position}-{speaker}", a 4-stage
bar (each stage's time inside it), and the total time."""
import json
import tkinter as tk
from pathlib import Path

import pytest

from record.utils import status

NB = Path(__file__).resolve().parents[1] / "record.ipynb"
CELLS = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def test_rows_newest_first_with_stage_times_and_total(tmp_path):
    ns = dict(tk=tk, status=status)
    exec(next(src for src in CELLS if "def draw_status" in src), ns)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    canvas = tk.Canvas(root, width=320, height=200)
    canvas.pack()
    root.update()

    t = [0.0]
    now = lambda: t[0]
    s = {}
    (tmp_path / "000001").mkdir()
    status.add(s, "000001", "12-3", tmp_path / "000001", now=now)
    for i in range(4):
        status.mark(s, "000001", i, "start", now=now); t[0] += 2.0
        status.mark(s, "000001", i, "end", now=now)
    status.add(s, "000002", "12-4", tmp_path / "000002", now=now)
    status.mark(s, "000002", status.RECORD, "start", now=now); t[0] += 1.5

    ns["draw_status"](canvas, s, now(), blink_on=True)
    texts = [canvas.itemcget(i, "text") for i in canvas.find_all() if canvas.type(i) == "text"]
    assert texts[0] == "12-4" and "1.5s" in texts  # newest on top, its running stage ticking
    assert texts.index("12-3") > texts.index("12-4")
    assert texts.count("2.0s") == 4 and any("8.0s" in x and "✓" in x for x in texts)  # 4 stages, total, done
    root.destroy()
