"""A new session never reuses a sample id: after deleting samples, counting the sample dirs lands
on ids that still exist (or that positions.jsonl already names) and overwrites them."""
import json
from pathlib import Path

from utils.io_utils import load_metadata

NB = Path(__file__).resolve().parents[1] / "record.ipynb"


def test_next_sample_id_is_past_every_id_ever_used(tmp_path):
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    ns = dict(Path=Path, json=json)
    exec(next(src for src in cells if "class ExperimentConfig" in src), ns)

    assert ns["_next_sample_id"](tmp_path) == 1  # a new experiment
    for name in ("000001", "000002", "000003", "000006"):  # 000004-000005 deleted
        (tmp_path / "samples" / name).mkdir(parents=True)
    assert ns["_next_sample_id"](tmp_path) == 7  # not 5 (= 1 + 4 dirs), which would overwrite 000006
    (tmp_path / "positions.jsonl").write_text('{"1": ["000001", "000002"]}\n{"2": ["000006", "000007"]}\n')
    assert ns["_next_sample_id"](tmp_path) == 8  # 000007 was handed out (its dir since deleted)


def test_next_position_id_is_past_every_position_ever_recorded(tmp_path):
    # positions.jsonl is written at the END of a position: if the kernel dies mid-position, only
    # its samples' metadata know its id
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    ns = dict(Path=Path, json=json, load_metadata=load_metadata)
    exec(next(src for src in cells if "class ExperimentConfig" in src), ns)

    assert ns["_next_position_id"](tmp_path) == 1  # a new experiment
    (tmp_path / "positions.jsonl").write_text('{"1": ["000001"]}\n{"2": ["000002"]}\n')
    assert ns["_next_position_id"](tmp_path) == 3
    (tmp_path / "samples" / "000003").mkdir(parents=True)
    (tmp_path / "samples" / "000003" / "metadata.jsonl").write_text('{"sample_id": "000003"}\n{"position_id": 3}\n')
    (tmp_path / "samples" / "000004").mkdir()  # a sample with no metadata yet
    assert ns["_next_position_id"](tmp_path) == 4  # position 3 crashed before reaching positions.jsonl
