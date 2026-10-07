import json
from pathlib import Path

import numpy as np

from record2.post_process import post_process, preview_vibrations


def test_preview_uses_the_chosen_channel(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 255, (2500, 32, 32), dtype=np.uint8)  # ~1 s at 2500 fps
    x, y = (preview_vibrations(raw, (0, 0, 32, 32), 2500.0, 0, channel, 100.0, 1000.0, pclk_batch_size=64) for channel in (0, 1))
    assert x["shifts"].shape == (1, 2500, 2) and x["fps"] == 2500.0
    assert x["freqs"].min() >= 100.0 and x["freqs"].max() <= 1000.0
    assert not np.array_equal(x["recovered_audio"], y["recovered_audio"])
    assert not any(tmp_path.iterdir())  # the preview writes nothing


def test_post_process_listens_to_the_laser_and_axis_in_metadata(tmp_path, monkeypatch):
    """Which laser/axis becomes recovered_audio.wav comes from metadata (the preview's choice), not a
    constant. Also: no symlinks (Windows refuses them without admin -- [WinError 1314])."""
    monkeypatch.setattr(Path, "symlink_to", lambda *a: (_ for _ in ()).throw(OSError("no symlink privilege")))
    sample_dir = tmp_path / "000001-1"
    (sample_dir / "vibration").mkdir(parents=True)
    np.save(sample_dir / "vibration/01_raw_vibrations.npy", np.random.default_rng(0).integers(0, 255, (2500, 80, 80), dtype=np.uint8))
    rois = [(x, y, 8, 8) for y in range(0, 80, 8) for x in range(0, 80, 8)]
    metadata = {"fps": 2500.0, "rois": rois, "min_freq": 100.0, "max_freq": 1000.0, "recovery_laser": 3, "recovery_axis": "y"}
    (sample_dir / "metadata.jsonl").write_text("".join(json.dumps({k: v}) + "\n" for k, v in metadata.items()))

    post_process(sample_dir / "vibration/01_raw_vibrations.npy")
    assert (sample_dir / "vibration/05_recovered_audio_laser3_y.wav").exists()
    assert (sample_dir / "recovered_audio.wav").exists() and not (sample_dir / "recovered_audio.wav").is_symlink()
    assert not (sample_dir / "vibration/01_raw_vibrations.npy").exists()  # deleted once everything succeeded
    assert "post process end" in (sample_dir / "times.jsonl").read_text()
