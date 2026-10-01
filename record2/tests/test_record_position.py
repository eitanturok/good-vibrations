import json
import threading

import numpy as np
import pytest

from record2.tests.fakes import load, run
from utils.io_utils import load_metadata


def test_a_saved_position(tmp_path):
    ns = load(tmp_path)
    position_id = run(ns)
    exp, samples = ns["exp"], ns["exp"].samples_dir
    names = sorted(d.name for d in samples.iterdir())
    assert position_id == "000001" and names == ["000001-1", "000001-3", "000001-5", "000001-7"]
    for name in names:
        d = samples / name
        meta = load_metadata(d / "metadata.jsonl")
        assert meta["speaker"] == int(name[-1]) and meta["layout"] == "one-cube"
        assert (meta["recovery_laser"], meta["recovery_axis"]) == (5, "y")  # post-processing listens to the previewed laser
        assert "coms" in meta  # segmentation metadata, appended at the end of the position
        assert np.load(d / "vibration/01_raw_vibrations.npy").shape == (30, 320, 1920)  # (chirp 0.2 + padding 0.1) s x 100 fps
        assert (d / "image/03_smask.npy").exists() and (d / "image/smasks/red-cube0.npy").exists()
        times = (d / "times.jsonl").read_text()
        assert all(f"{stage} {e}" in times for stage in ("speaker", "vibrate", "save vibration") for e in ("start", "end"))
    assert json.loads((exp.experiment_dir / "positions.jsonl").read_text()) == {"1": [1, 3, 5, 7]}
    assert exp.positions_per_layout["one-cube"] == 1 and exp.coverage["one-cube"]["n_positions"] == 1
    assert ns["LAST"]["preview"].result()["laser_idx"] == 5
    assert ns["hw"].audio_engine.played[-1] == list(range(1, 9))  # the whistle, on every speaker


def test_a_dry_run_writes_nothing_and_uses_no_position_id(tmp_path):
    ns = load(tmp_path)
    assert run(ns, save=False) == "dry"
    exp = ns["exp"]
    assert not any(exp.samples_dir.iterdir()) and not (exp.experiment_dir / "positions.jsonl").exists()
    assert exp.position_ids.n == 0
    assert ns["LAST"]["seg"].result()["coverage"]["n_positions"] == 1  # shown...
    assert exp.coverage == {} and not exp.positions_per_layout  # ...but not kept


def test_a_failed_capture_fails_the_position(tmp_path):
    ns = load(tmp_path, fail_on=2)
    with pytest.raises(RuntimeError, match="Timeout"):
        run(ns)
    exp = ns["exp"]
    assert ns["hw"].audio_engine.resets == 1
    assert not (exp.experiment_dir / "positions.jsonl").exists() and not exp.positions_per_layout
    assert "vibrate failed" in (exp.samples_dir / "000001-3" / "times.jsonl").read_text()
    assert not (exp.samples_dir / "000001-5").exists()  # no further speaker


def test_stop_finishes_the_current_speaker_and_skips_the_rest(tmp_path):
    ns = load(tmp_path)
    capture = ns["hw"].laser_cam.capture_vibrations
    def capture_then_stop(n):
        ns["STOP"].set()
        return capture(n)
    ns["hw"].laser_cam.capture_vibrations = capture_then_stop
    run(ns)
    assert sorted(d.name for d in ns["exp"].samples_dir.iterdir()) == ["000001-1"]


def test_a_second_run_fails_at_once(tmp_path):
    ns = load(tmp_path)
    with ns["RECORDING"]:
        with pytest.raises(RuntimeError, match="already recording"):
            run(ns)


def test_delete_position(tmp_path):
    ns = load(tmp_path)
    run(ns)
    run(ns)
    exp = ns["exp"]
    assert exp.positions_per_layout["one-cube"] == 2
    moved = ns["delete_position"](exp, 1)
    assert moved == ["000001-1", "000001-3", "000001-5", "000001-7"]
    assert sorted({d.name[:6] for d in exp.samples_dir.iterdir()}) == ["000002"]
    assert (exp.experiment_dir / "deleted" / "000001-1" / "metadata.jsonl").exists()
    assert [json.loads(ln) for ln in (exp.experiment_dir / "positions.jsonl").read_text().splitlines()] == [{"2": [1, 3, 5, 7]}]
    assert exp.positions_per_layout["one-cube"] == 1


def test_delete_is_refused_while_the_position_is_busy(tmp_path):
    ns = load(tmp_path)
    run(ns)
    with ns["Timing"]("000001-3", "post process"):
        with pytest.raises(RuntimeError, match="busy"):
            ns["delete_position"](ns["exp"], "000001")
    assert len(list(ns["exp"].samples_dir.iterdir())) == 4


def test_a_restart_counts_the_saved_positions(tmp_path):
    ns = load(tmp_path)
    run(ns)
    exp = ns["Experiment"](ns["exp"].experiment_dir, ns["exp"].chirp_config, ns["exp"].chirp_samples, ns["exp"].done_whistle_samples)
    assert exp.positions_per_layout["one-cube"] == 1 and exp.coverage["one-cube"]["n_positions"] == 1
    assert len(exp.coverage["one-cube"]["last_masks"]) == len(ns["exp"].coverage["one-cube"]["last_masks"])  # each object, from disk


def test_record_position_and_plot_draws_each_result_in_the_cell(tmp_path):
    ns = load(tmp_path)
    shown = []
    ns["display"] = shown.append
    assert ns["record_position_and_plot"](ns["hw"], ns["exp"], ns["segmenter"], ns["position"], ns["preview"], save=False) == "dry"
    assert len(shown) == 4  # smask, coverage, shifts, freqs


def test_experiment_repr(tmp_path):
    ns = load(tmp_path)
    run(ns)
    text = repr(ns["exp"])
    assert "saved positions: 1 (one-cube 1)" in text and "next position id: 2" in text and "GB free" in text
