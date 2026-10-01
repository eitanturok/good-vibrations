"""Samples recorded since record/ dropped sample ids live in utils.ids.sample_name dirs ("000012-3"),
with no sample_id in their metadata -- so a run trained on them saves info['sample_id'] == -1 (see
src/model/dataset.py) next to position_id/speaker. They must join like the old counter-named ones,
and every sample, old or new, is shown by the same universal id."""

import json

import numpy as np
import torch
from PIL import Image

from viz import config
from utils.ids import sample_name

H, W = 4, 6


def make(root, dirs, metas, info):
    """An experiment with one sample per dir (mask = one-hot at its row) and a run predicting them
    in reverse order, info as given."""
    for row, (name, meta) in enumerate(zip(dirs, metas)):
        d = root / "exp" / "samples" / name / "image"
        d.mkdir(parents=True)
        Image.new("RGB", (100, 80)).save(d / "02_cropped_overhead.png")
        m = np.zeros((H, W), np.float32); m.flat[row] = 1
        np.save(d / f"04_downsampled_smask_{H}h_{W}w.npy", m)
        (d.parent / "metadata.jsonl").write_text("".join(json.dumps({k: v}) + "\n" for k, v in meta.items()))
    runs = root / "runs" / "r" / config.OUTPUTS_SUBDIR / "train"
    runs.mkdir(parents=True)
    rows = list(range(len(dirs)))[::-1]
    pred = torch.zeros(len(rows), H, W)
    for j, r in enumerate(rows): pred[j].view(-1)[r] = 0.9
    torch.save({"mask_pred": pred, "info": {k: torch.tensor([v[r] for r in rows]) for k, v in info.items()} | {"x_com": torch.zeros(len(rows))}},
               runs / "ep0000-ba0.pt")
    from viz import app
    app.init(root / "exp", root / "runs")
    return app


def check(app, shown):
    rd = app.registry.run("r")
    assert len(rd.sample_ids) == len(shown)
    for j, gid in enumerate(rd.global_ids):  # every prediction lands on its own sample's mask
        assert rd.masks[j].argmax() == app.registry.sample_index(gid) and rd.metrics["iou"][j] > 0.8
    samples = app.api_samples()["samples"]
    assert [s["sample_id"] for s in samples] == shown
    assert app.api_detail(samples[-1]["i"])["sample_id"] == shown[-1]


def test_new_samples(tmp_path):
    ps = [(9, 1), (9, 8), (10, 1)]  # "000010-1" sorts after "000009-8"
    app = make(tmp_path, [sample_name(p, s) for p, s in ps], [{"position_id": p, "speaker": s} for p, s in ps],
               {"sample_id": [-1] * 3, "position_id": [p for p, _ in ps], "speaker": [s for _, s in ps]})
    check(app, ["000009-1", "000009-8", "000010-1"])


def test_old_samples_show_the_universal_id(tmp_path):
    ids, ps = [7, 10, 11], [(2, 1), (2, 4), (3, 1)]
    app = make(tmp_path, [f"{i:06d}" for i in ids], [{"sample_id": f"{i:06d}", "position_id": p, "speaker": s} for i, (p, s) in zip(ids, ps)],
               {"sample_id": ids, "position_id": [p for p, _ in ps], "speaker": [s for _, s in ps]})
    check(app, ["000002-1", "000002-4", "000003-1"])
