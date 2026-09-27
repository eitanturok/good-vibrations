"""Objects: 3 slots, each typed with autocomplete from the known objects, a count, and an editable
segmentation prompt that defaults to the object's known prompt. Segmentation uses the position's own
prompts; an object with no prompt at all is refused up front, not mid-recording."""
import json
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace

import pytest

NB = Path(__file__).resolve().parents[1] / "record.ipynb"
CELLS = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
PROMPTS = {"mug": "inside of the mug", "bullet": "Metal circle", "red-cube": "Red cube", "red-beans": "bag of red beans"}


class Var:
    def __init__(self, value): self.value = value
    def get(self): return self.value


def app_ns():
    ns = dict(tk=tk, ttk=ttk)
    exec(next(src for src in CELLS if "class App" in src), ns)
    return ns


def test_suggestions_narrow_as_you_type():
    ns, names = app_ns(), sorted(PROMPTS)
    assert ns["suggestions"]("", names) == names  # clicking in: every known object
    assert ns["suggestions"]("RE", names) == ["red-beans", "red-cube"]  # case-insensitive
    assert ns["suggestions"]("e", names) == ["bullet", "red-beans", "red-cube"]  # contains...
    assert ns["suggestions"]("cu", ["red-cube", "cup"]) == ["cup", "red-cube"]  # ...names starting with it first
    assert ns["suggestions"]("xyz", names) == []


def test_suggestion_list_opens_on_click_and_narrows_while_typing():
    """The list is open from the first click and the keyboard stays in the entry (a ttk.Combobox's own
    dropdown grabs the keyboard, so typing couldn't narrow it)."""
    ns = app_ns()
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.geometry("300x100+0+0")
    var = tk.StringVar(root)
    entry = ttk.Entry(root, textvariable=var)
    entry.pack()
    box = ns["SuggestionBox"](entry, var, sorted(PROMPTS))
    root.update()
    box.show()  # what clicking into the entry does
    root.update()
    assert box.top.winfo_viewable() and list(box.listbox.get(0, "end")) == sorted(PROMPTS)
    var.set("red"); box.show()  # what each keystroke does
    assert list(box.listbox.get(0, "end")) == ["red-beans", "red-cube"]
    box.move(1); box.move(1); box.pick()  # arrow down twice, Enter
    assert var.get() == "red-cube" and not box.top.winfo_viewable()
    root.destroy()


def test_objects_and_prompts_from_three_slots():
    get_objects, get_prompts = app_ns()["App"].get_objects, app_ns()["App"].get_prompts
    slots = lambda *s: SimpleNamespace(object_slots=[(Var(n), Var(c), Var(p)) for n, c, p in s])
    empty = slots(("", 1, ""), ("", 1, ""), ("", 1, ""))
    assert get_objects(empty) == {} and get_prompts(empty) == {}  # empty box
    gui = slots(("mug", 1, "white mug"), (" ", 3, ""), ("new-toy", 2, "a green plush toy"))
    assert get_objects(gui) == {"mug": 1, "new-toy": 2}  # blank slot skipped; a new object is fine
    assert get_prompts(gui) == {"mug": "white mug", "new-toy": "a green plush toy"}  # the edited prompts
    assert get_objects(slots(("bullet", 1, "p"), ("bullet", 2, "p"), ("", 1, ""))) == {"bullet": 3}  # counts add


def test_position_prompts_default_to_known_and_are_required():
    ns = dict(dataclass=dataclass, field=field, BoxConfig=object, BOXES={"gastronorm": None}, PROMPTS=PROMPTS)
    exec(next(src for src in CELLS if "class PositionConfig" in src), ns)
    PositionConfig = ns["PositionConfig"]
    assert PositionConfig(objects={"mug": 1}).prompts == {"mug": "inside of the mug"}  # the known prompt by default
    assert PositionConfig(objects={"mug": 1}, prompts={"mug": "white mug"}).prompts == {"mug": "white mug"}
    assert PositionConfig(objects={"toy": 1}, prompts={"toy": "a plush toy"}).prompts == {"toy": "a plush toy"}
    with pytest.raises(ValueError, match="toy"):
        PositionConfig(objects={"toy": 1})  # new object, no prompt typed: refused before recording
