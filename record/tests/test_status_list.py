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
    (tmp_path / "000012-3").mkdir()
    status.add(s, "000012-3", tmp_path / "000012-3", now=now)
    for i in range(4):
        status.mark(s, "000012-3", i, "start", now=now); t[0] += 2.0
        status.mark(s, "000012-3", i, "end", now=now)
    status.add(s, "000012-4", tmp_path / "000012-4", now=now)
    status.mark(s, "000012-4", status.RECORD, "start", now=now); t[0] += 1.5

    ns["draw_status"](canvas, s, now(), blink_on=True)
    texts = [canvas.itemcget(i, "text") for i in canvas.find_all() if canvas.type(i) == "text"]
    assert texts[0] == "000012-4" and "1.5s" in texts  # newest on top, its running stage ticking
    assert texts.index("000012-3") > texts.index("000012-4")
    assert texts.count("2.0s") == 4 and any("8.0s" in x and "✓" in x for x in texts)  # 4 stages, total, done
    root.destroy()


def test_run_button_shows_the_next_position_and_what_is_recorded(tmp_path):
    from types import SimpleNamespace
    from record.utils.position_id import PositionIdCounter
    ns = dict(tk=tk, status=status)
    exec(next(src for src in CELLS if "def run_label" in src), ns)
    for name in ("000012-1", "000012-2", "000013-1"):
        (tmp_path / "samples" / name).mkdir(parents=True)
    ec = SimpleNamespace(experiment_dir=tmp_path, position_ids=PositionIdCounter(tmp_path / "position_id.txt"))
    assert ns["run_label"](ec) == "▶  Run  (position counter missing!)\n2 recorded · 3 samples"
    ec.position_ids = PositionIdCounter(tmp_path / "position_id.txt", create=True)
    assert [ec.position_ids.next(), ec.position_ids.next()] == [1, 2]
    assert ns["run_label"](ec) == "▶  Run position 3\n2 recorded · 3 samples"  # peeking takes no id
    assert ec.position_ids.next() == 3
