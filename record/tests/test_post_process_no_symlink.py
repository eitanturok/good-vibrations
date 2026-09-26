"""Real bug: every post-processed sample failed with [WinError 1314] (Windows refuses symlinks
without admin/Developer Mode) when linking recovered_audio.wav -- after the raw vibrations were
already deleted. Post-processing must not create symlinks at all."""
import json
from pathlib import Path

import numpy as np

from record.post_process import post_process


def test_post_process_without_symlink_privilege(tmp_path, monkeypatch):
    def refuse(self, *a, **k):
        raise OSError(22, "A required privilege is not held by the client", str(self), 1314)
    monkeypatch.setattr(Path, "symlink_to", refuse)

    sample_dir = tmp_path / "000000"
    (sample_dir / "vibration").mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(sample_dir / "vibration/01_raw_vibrations.npy", rng.integers(0, 255, (2500, 80, 80), dtype=np.uint8))
    rois = [(x, y, 8, 8) for y in range(0, 80, 8) for x in range(0, 80, 8)]  # 100 ROIs, like the real grid
    (sample_dir / "metadata.jsonl").write_text(json.dumps({"fps": 2500.0, "rois": rois}) + "\n")

    post_process(sample_dir)
    recovered = sample_dir / "recovered_audio.wav"
    assert recovered.exists() and not recovered.is_symlink()
    assert any((sample_dir / "vibration").glob("06_spectrogram*.png"))
    assert any((sample_dir / "vibration").glob("06_spectrogram*.mp4"))
