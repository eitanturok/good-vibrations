"""The shifts/FFT preview is computed and plotted only for the preview speaker (not every
speaker in the position)."""
import json
import math
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from record.utils import Task, Timing, status

NB = Path(__file__).resolve().parents[1] / "record.ipynb"


class Lock:
    def __enter__(self): pass
    def __exit__(self, *a): pass


def test_preview_only_for_the_preview_speaker(tmp_path):
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    previewed, loading, coverage_plots = [], [], []
    ns = dict(Task=Task, Timing=Timing, status=status, math=math, time=time, dataclasses=SimpleNamespace(asdict=lambda x: {}), np=np,
              log=lambda ec, msg: None, append=lambda row, path: None,
              crop=lambda image, **kw: image, segment_mod=SimpleNamespace(segment=lambda *a: None),
              capture_metadata=lambda *a: {}, sample_metadata=lambda *a: {},
              plot_loading=lambda ec, name: loading.append(name), plot_smask=lambda *a: None,
              save_raw_vibration=lambda *a: None, save_sample=lambda *a: None, plot_coverage=lambda ec, task, sample_dir: coverage_plots.append(sample_dir.name),
              preview_vibrations=lambda raw, roi, fps, laser_idx, min_freq, max_freq, use_PC: previewed.append(int(raw[0])) or
                  {"recovered_audio": np.full(100, int(raw[0]), dtype=np.int16), "audio_sample_rate": 22050},
              plot_shifts=lambda *a: None, plot_freqs=lambda *a: None)
    exec(next(src for src in cells if "def run_experiment" in src), ns)

    class Pool:
        def submit(self, fn, *a): return None
    speaker_now, labels = [], []
    laser_cam = SimpleNamespace(lock=Lock(), flush=lambda: None, get_frame_rate=lambda: 2500.0,
                                capture_vibrations=lambda n: labels.append(ec.recording_label) or np.array([speaker_now[-1]]),
                                config=SimpleNamespace(capture_margin_s=0.1, roi=SimpleNamespace(rois=[(0, 0, 32, 32)] * 100)))
    audio = SimpleNamespace(sample_rate=48000, config=SimpleNamespace(speaker_delay=0, speaker_device_names={1: "a", 2: "a", 3: "a"}),
                            play=lambda samples, speakers: speaker_now.append(speakers[0]) or [], reset=lambda: None)
    ec = SimpleNamespace(_id_lock=threading.Lock(), next_position_id=1, next_sample_id=1, experiment_dir=tmp_path,
                         overhead_cam=SimpleNamespace(config=SimpleNamespace(hand_delay=0), capture_overhead=lambda: np.zeros((4, 4))),
                         laser_cam=laser_cam, audio_engine=audio, chirp_samples=np.zeros(4800), done_whistle_samples=np.zeros(10),
                         prompts={}, segmenter=None, stop_event=threading.Event(), raw_save_pool=Pool(), tasks={}, status={},
                         preview_config=SimpleNamespace(speaker=2, laser=55, use_pc=True), active_speaker=None, recording_label=None,
                         chirp_config=SimpleNamespace(f_start=100.0, f_end=1000.0), segment_scale=1.0)
    position = SimpleNamespace(speakers=[1, 2, 3], objects={}, prompts={}, box=SimpleNamespace(crop_params=SimpleNamespace()))

    ns["run_experiment"](ec, position, save=True, vibrate=True, verbose=False)
    for t in ec.tasks.values(): t.join()
    assert previewed == [2]  # only speaker 2's vibrations
    assert labels == ["1-1", "1-2", "1-3"] and ec.recording_label is None  # the RECORDING badge: {position}-{speaker}
    assert loading.count("shifts") == 1
    assert coverage_plots == ["000001"]  # coverage redrawn once per position, on its first speaker
    # the GUI's status list: one row per position-speaker, its record stage timed
    assert {k: r["label"] for k, r in ec.status.items()} == {"000001": "1-1", "000002": "1-2", "000003": "1-3"}
    assert all(status.states(r, time.perf_counter())[0][0] == "done" for r in ec.status.values())

    # the GUI's play button: the preview speaker's recovered audio, on the PC's default output
    played = []
    plots = dict(np=np, log=lambda ec, msg: None, sd=SimpleNamespace(play=lambda audio, sr: played.append((audio[0], sr))))
    exec(next(src for src in cells if "def play_recovered_audio" in src), plots)
    plots["play_recovered_audio"](ec)
    assert played == [(2, 22050)]

    previewed.clear(); loading.clear(); ec.tasks.clear()
    ns["run_experiment"](ec, SimpleNamespace(speakers=[1, 3], objects={}, prompts={}, box=position.box), save=False, vibrate=True, verbose=False)
    assert previewed == [] and "shifts" not in loading  # preview speaker not in this position: no plot, no "Loading..."
    assert [r["skipped"][2] for r in ec.status.values()] == [False] * 3 + [True] * 2  # save=False: no save-sample stage

    # verbose (the notebook): the images the GUI panels got, shown in the cell in two figures --
    # smask | coverage on one row, and shifts over fft as soon as the preview is ready
    shown, preview_shown = [], threading.Event()
    letters = lambda pixels: "".join(chr(p) for p in pixels)
    show = lambda im: shown.append((letters(im[0, :, 0]), letters(im[:, 0, 0]))) or (im[0, 0, 0] == ord("s") and preview_shown.set())
    plots_cell = dict(np=np, Image=SimpleNamespace(fromarray=lambda a: a), display=show)
    exec(next(src for src in cells if "def plot_smask" in src), plots_cell)
    panel = lambda letter: np.full((2, 2, 4), ord(letter), np.uint8)
    ns.update(show_plots=plots_cell["show_plots"],
              plot_smask=lambda *a: panel("m"), plot_coverage=lambda *a: panel("c"),
              plot_shifts=lambda *a: panel("s"), plot_freqs=lambda *a: panel("f"))
    started = []  # every Task run_experiment starts, to wait for them all
    ns["Task"] = lambda *a, **kw: started.append(Task(*a, **kw)) or started[-1]
    exec(next(src for src in cells if "def run_experiment" in src), ns)
    ec.laser_cam.capture_vibrations = lambda n: np.array([speaker_now[-1]])
    ns["run_experiment"](ec, position, save=False, vibrate=True, verbose=False)
    for t in started: t.join()
    assert shown == []  # the GUI: nothing in the notebook
    shown_while_recording = []
    ec.laser_cam.capture_vibrations = lambda n: (speaker_now[-1] == 3 and shown_while_recording.append(preview_shown.wait(5))) or np.array([speaker_now[-1]])
    ns["run_experiment"](ec, position, save=True, vibrate=True, verbose=True)
    for t in started: t.join()
    assert shown_while_recording == [True]  # speaker 2's preview showed while speaker 3 was recording
    assert sorted(shown) == [("mmcc", "mm"), ("ss", "ssff")]  # smask | coverage; shifts over fft

    # a failed capture shows as a failed record stage
    ec.laser_cam.capture_vibrations = lambda n: 1 / 0
    ns["run_experiment"](ec, SimpleNamespace(speakers=[1], objects={}, prompts={}, box=position.box), save=True, vibrate=True, verbose=False)
    assert status.states(ec.status[list(ec.status)[-1]], time.perf_counter())[0][0] == "failed"

    # never records over an existing sample: a reused id fails before anything is captured or written
    ec.laser_cam.capture_vibrations = lambda n: np.array([speaker_now[-1]])
    taken = tmp_path / "samples" / f"{ec.next_sample_id + 1:06d}"
    taken.mkdir(parents=True)
    n_captured, n_rows = len(speaker_now), len(ec.status)
    with pytest.raises(FileExistsError):
        ns["run_experiment"](ec, SimpleNamespace(speakers=[1, 3], objects={}, prompts={}, box=position.box), save=True, vibrate=True, verbose=False)
    assert len(speaker_now) == n_captured and len(ec.status) == n_rows and list(taken.iterdir()) == []
