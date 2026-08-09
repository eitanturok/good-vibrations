"""Sample ids are not row indices. See viz/SPEC.md.

The fixture's ids deliberately start at 000007 and skip 000009, because a dataset whose
ids start at 0 makes every bug in this family invisible -- which is how the id/row skew
survived on experiment-25 and only showed up on gastronorm.
"""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from viz import config, data

H, W = 4, 6
# 000009 is absent, and 000008 ships no mask so load_gt drops it. Rows end up
# {7: 0, 10: 1, 11: 2}: not id-minus-a-constant, so an off-by-one fix cannot fake it.
IDS = [7, 8, 10, 11]
NO_MASK = 8
ROW_OF = {7: 0, 10: 1, 11: 2}


def fingerprint(row: int) -> np.ndarray:
    """A mask that names the row that produced it, via argmax."""
    m = np.zeros((H, W), dtype=np.float32)
    m.flat[row % (H * W)] = 1.0
    return m


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MASK_H", H)
    monkeypatch.setattr(config, "MASK_W", W)
    exp = tmp_path / "exp"
    row = 0
    for sid in IDS:
        d = exp / "samples" / f"{sid:06d}" / "image"
        d.mkdir(parents=True)
        # gastronorm layout: 02_cropped_overhead.png is both backdrop and overhead.
        # A distinct size per sample makes the photo identifiable from im.size alone.
        Image.new("RGB", (100 + row, 80)).save(d / "02_cropped_overhead.png")
        (d.parent / "metadata.jsonl").write_text(json.dumps({"sample_id": sid}) + "\n")
        if sid != NO_MASK:
            np.save(d / f"04_downsampled_smask_{H}h_{W}w.npy", fingerprint(row))
            row += 1
    return exp


@pytest.fixture
def runs(tmp_path):
    """One run predicting every sample that has a target, as truth * 0.9."""
    d = tmp_path / "runs" / "r" / config.OUTPUTS_SUBDIR / "train"
    d.mkdir(parents=True)
    ids = [s for s in IDS if s != NO_MASK]
    torch.save(
        {"mask_pred": torch.stack([torch.from_numpy(fingerprint(ROW_OF[s]) * 0.9) for s in ids]),
         "info": {"sample_id": torch.tensor(ids), "x_com": torch.zeros(len(ids))}},
        d / "ep0000-ba0.pt")
    return tmp_path / "runs"


@pytest.fixture
def registry(experiment, runs):
    from viz import app, render
    render._backdrop.cache_clear()
    render.scene_aspect.cache_clear()
    app.init(experiment, runs)
    return app.registry


def test_gt_rows_are_not_ids(registry):
    """The premise: without this, the rest of the suite proves nothing."""
    assert registry.gt.row_of == ROW_OF


def test_run_row_of_is_keyed_by_row(registry):
    """RunData.row_of is keyed by ROW -- that is what every consumer passes it."""
    rd = registry.run("r")
    for j, sid in enumerate(rd.sample_ids):
        assert rd.row_of[registry.sample_index(int(sid))] == j


def test_rendered_prediction_matches_its_metrics(registry):
    """The oracle: the array the renderer draws must be the one the metrics describe.

    This is the reported symptom -- iou 0.99 beside a visibly wrong picture.
    """
    rd = registry.run("r")
    for sid, row in ROW_OF.items():
        j = rd.row_of[row]                       # exactly what app.py/render.py do
        assert int(rd.sample_ids[j]) == sid
        assert rd.masks[j].argmax() == fingerprint(row).argmax()
        assert rd.iou[j] > 0.8                   # mispaired fingerprints score ~0


def test_backdrop_cache_not_poisoned(registry):
    """scene_aspect must probe rows, not ids, or it caches wrong photos under real rows."""
    from viz import render
    render.scene_aspect()
    for row in range(len(registry.gt)):
        assert render._backdrop(row).size == (100 + row, 80)
