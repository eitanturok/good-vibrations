"""pclk's input is the raw vibration as saved: uint8 (the camera's 8-bit frames). Converting it to
float32 before the copy to the GPU (4x the bytes, cast on the CPU) or after it (on the GPU) is the
same exact integer -> float32 cast, so the shifts must be identical."""
import numpy as np
import pytest

from data.pclk import compute_shifts_for_all_rois_batched_optimized as pclk  # first: sets up CUDA before cupy loads


@pytest.mark.parametrize("use_PC", [True, False])
def test_uint8_video_gives_the_same_shifts_as_float32(use_PC):
    rng = np.random.default_rng(0)
    video = rng.integers(0, 256, (3, 65, 32, 48), dtype=np.uint8)  # (L, T, H, W), tiny
    shifts_u8 = pclk(video, batch_size=16, progress=False, use_PC=use_PC)  # 4 full batches + a partial one
    shifts_f32 = pclk(video.astype(np.float32), batch_size=16, progress=False, use_PC=use_PC)
    assert shifts_u8.shape == (3, 65, 2)
    np.testing.assert_array_equal(shifts_u8, shifts_f32)
