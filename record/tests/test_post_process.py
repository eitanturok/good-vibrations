import numpy as np
import pytest

from record.post_process import preview_vibrations


def test_preview_vibrations_shape_and_no_disk_io(tmp_path):
    """Not re-validating pclk's own physics (already relied on throughout data/pclk.py
    elsewhere in the repo) -- just that this thin wrapper plumbs args through correctly
    and returns the right shape of result, with zero disk writes (the whole point of the
    live-preview path vs. the full save in post_process())."""
    rng = np.random.default_rng(0)
    n_frames, h, w = 2500, 32, 32  # ~1s at 2500 fps -- realistic enough that the recovered-
                                    # audio/spectrogram steps don't hit degenerate-short-signal
                                    # edge cases a tiny synthetic clip would trigger
    raw_vibrations = rng.integers(0, 255, (n_frames, h, w), dtype=np.uint8)
    roi = (0, 0, w, h)  # (x, y, w, h) spanning the whole synthetic frame
    fps = 2500.0  # matches this project's real laser-camera frame rate; MIN/MAX_FREQ (50-1000Hz)
                  # requires fps > 2*max_freq (Nyquist), so a low fps here is a bug, not a shortcut

    result = preview_vibrations(raw_vibrations, roi, fps, laser_idx=0, pclk_batch_size=64)

    assert set(result) >= {"shifts", "fft", "freqs", "n_samples", "recovered_audio", "spec_freqs", "spec_times", "Sxx"}
    assert result["shifts"].shape == (1, n_frames, 2)  # one (x, y) shift per frame
    assert result["laser_idx"] == 0
    assert len(list(tmp_path.iterdir())) == 0  # no disk I/O happened anywhere
