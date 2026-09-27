"""Every sample's metadata.jsonl holds everything src/record.ipynb saved plus the new rig params,
one key per line (load_metadata merges them), written before the raw vibrations so post_process
can read what it needs (fps, rois, and the chirp's band as min_freq/max_freq)."""
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from record.utils import status
from utils.io_utils import load_metadata

NB = Path(__file__).resolve().parents[1] / "record.ipynb"

EXPECTED = {
    # ids
    "sample_id", "position_id", "speaker", "experiment_dir", "timestamp", "git_commit", "hostname",
    # position
    "box", "crop_left", "crop_right", "crop_top", "crop_bottom", "objects", "n_objects", "layout", "description",
    "is_empty_box", "speakers", "save", "vibrate", "prompts",
    # overhead camera
    "overhead_device_id", "overhead_exposure_ms", "overhead_gain", "overhead_frame_rate", "overhead_pixel_clock",
    "hand_delay", "raw_overhead_shape", "crop_overhead_shape",
    # laser camera
    "fps", "exposure_us", "gain", "max_frame_rate", "global_roi", "width", "height", "sensor_width", "sensor_height",
    "buffer_part_count", "capture_margin_s", "n_frames", "n_capture_seconds", "laser_power",
    # roi grid
    "rois", "row_positions", "offset_x", "n_rows", "n_cols", "roi_width", "roi_height",
    # audio + chirp
    "audio_fs", "chirp_t_sec", "chirp_t_start", "chirp_t_end", "chirp_f_start", "chirp_f_end", "min_freq", "max_freq",
    "speaker_device_name", "speaker_channel", "speaker_mono", "settle_seconds", "speaker_delay",
    # processing
    "pclk_batch_size", "use_PC", "laser_idx", "preview_speaker", "audio_sample_rate", "recovery_laser", "recovery_axis",
    "segment_scale",
    # legacy, src/record.ipynb's shape
    "run_opt", "run_opt_multiROIs",
}
SEGMENTATION = {"coms", "avg_com", "seg_boxes", "seg_scores"}


@dataclasses.dataclass
class Roi:
    n_rows: int = 2
    n_cols: int = 2
    roi_width: int = 32
    roi_height: int = 32
    rois: list = dataclasses.field(default_factory=lambda: [(0, 0, 32, 32), (100, 0, 32, 32), (0, 32, 32, 32), (100, 32, 32, 32)])
    row_positions: list = dataclasses.field(default_factory=lambda: [0, 500])
    offset_x: int = 16


def fake_ec(tmp_path):
    overhead = SimpleNamespace(config=SimpleNamespace(device_id=0, hand_delay=0.1), get_exposure=lambda: 12.0,
                               get_gain=lambda: 1, get_frame_rate=lambda: 30.0, get_pixel_clock=lambda: 86)
    laser = SimpleNamespace(config=SimpleNamespace(roi=Roi(), sensor_width=1920, sensor_height=1080, buffer_part_count=250,
                                                   capture_margin_s=0.1, laser_power=500.0),
                            get_frame_rate=lambda: 2500.0, get_exposure=lambda: 90.0, get_gain=lambda: 1.0,
                            get_max_frame_rate=np.float64(3987.0).item, get_global_roi=lambda: (16, 0, 144, 64), width=144, height=64)
    audio = SimpleNamespace(sample_rate=48000, config=SimpleNamespace(settle_seconds=0.25, speaker_delay=0.1,
                            speaker_device_names={4: "Line Out 3-4"}, speaker_channels={4: (0, False)}))
    return SimpleNamespace(experiment_dir=tmp_path, git_commit="abc123", hostname="lab-pc", segment_scale=1.0,
                           overhead_cam=overhead, laser_cam=laser, audio_engine=audio,
                           chirp_config=SimpleNamespace(t_sec=1.0, t_start=0.1, t_end=0.1, fs=44100, f_start=100.0, f_end=1000.0),
                           preview_config=SimpleNamespace(speaker=1, laser=55, use_pc=True), prompts={"red-cube": "Red cube"}, status={})


def test_every_key_saved_one_per_line_before_the_raw_vibrations(tmp_path):
    cells = ["".join(c["source"]) for c in json.loads(NB.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    submitted = []
    ns = dict(status=status, dataclasses=dataclasses, json=json, np=np, Path=Path, datetime=datetime, timezone=timezone,
              PCLK_BATCH_SIZE=256, AUDIO_SAMPLE_RATE=22050, RECOVERY_LASER=55, RECOVERY_XY=0,
              save=lambda x, path: Path(path).parent.mkdir(parents=True, exist_ok=True) or np.save(path, x),
              append=lambda row, path: None,
              full_post_process=SimpleNamespace(submit=lambda p: submitted.append((p, load_metadata(p.parent.parent / "metadata.jsonl")))))
    exec(next(src for src in cells if "def capture_metadata" in src), ns)

    ec = fake_ec(tmp_path)
    crop_params = SimpleNamespace(left=0.05, right=0.7, top=0.1, bottom=0.82)
    position = SimpleNamespace(box=SimpleNamespace(name="gastronorm", crop_params=crop_params), objects={"red-cube": 1}, prompts={"red-cube": "Red cube"},
                               layout="grid", description="a red cube", speakers=[1, 4])
    metadata = ns["capture_metadata"](ec, position, np.zeros((1200, 1600, 3)), np.zeros((900, 1000, 3)),
                                      n_frames=3250, n_capture_seconds=1.3, save=True, vibrate=True)
    metadata = ns["sample_metadata"](ec, metadata, sample_id="000007", position_id=3, speaker=4)
    sample_dir = tmp_path / "000007"
    ns["save_raw_vibration"](ec, sample_dir, np.zeros((4, 64, 144), np.uint8), metadata)

    raw_path, seen_by_post_process = submitted[0]
    assert raw_path == sample_dir / "vibration/01_raw_vibrations.npy"
    assert EXPECTED <= set(seen_by_post_process)  # all written before post_process can start
    assert (seen_by_post_process["min_freq"], seen_by_post_process["max_freq"]) == (100.0, 1000.0)  # the chirp's band

    lines = (sample_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(len(json.loads(line)) == 1 for line in lines)  # one key per line
    assert not any(isinstance(v, dict) for k, v in seen_by_post_process.items()
                   if k not in {"objects", "prompts", "run_opt", "run_opt_multiROIs"})  # nested configs unpacked
    assert seen_by_post_process["prompts"] == {"red-cube": "Red cube"}  # the prompts actually used
    assert seen_by_post_process["crop_left"] == 0.05 and seen_by_post_process["speaker_device_name"] == "Line Out 3-4"
    assert seen_by_post_process["run_opt"]["cam_params"]["camera_FPS"] == 2500.0

    # save_sample appends the segmentation keys once segmentation is done
    exec(next(src for src in cells if "def segmentation_metadata" in src), ns)
    seg = [{"masks": [np.ones((4, 4), bool)], "boxes": np.array([[1, 2, 3, 4]]), "scores": np.array([0.9])}]
    ns["write_metadata"](ns["segmentation_metadata"](ec, seg, coms=[[(1.5, 1.5)]], avg_com=[1.5, 1.5]), sample_dir / "metadata.jsonl")
    metadata = load_metadata(sample_dir / "metadata.jsonl")
    assert EXPECTED | SEGMENTATION <= set(metadata)
    assert metadata["seg_boxes"] == [[[1, 2, 3, 4]]] and metadata["seg_scores"] == [[0.9]]
