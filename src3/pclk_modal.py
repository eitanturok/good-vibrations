"""
Modal app for running PCLK+ (phase correlation + Lucas-Kanade) shift recovery.

Usage:
    modal run src3/pclk_modal.py --input-path experiment-20/data/samples/000000/inputs/00_raw_vibrations.npy --metadata-path experiment-20/data/samples/000000/metadata.jsonl --output-path experiment-20/data/samples/000000/inputs/01_raw_shifts.npy
"""

import modal
import numpy as np
from pathlib import Path

app = modal.App("pclk")

volume = modal.Volume.from_name("pclk-data", create_if_missing=True)
VOLUME_PATH = Path("/data")

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .pip_install("cupy-cuda12x", "numpy", "tqdm")
)

# ── PCLK helpers ──────────────────────────────────────────────────────────────
# cupy is only available inside Modal, so these are imported lazily inside each fn.

def get_pad_size(w):
    two_power = 2 ** np.ceil(np.log2(w))
    left_pad  = int((two_power - w) // 2)
    right_pad = int(two_power - w - left_pad)
    return left_pad, right_pad

def _pad(arr, up_pad, down_pad, left_pad, right_pad):
    import cupy as cp
    if arr.ndim == 2:
        return cp.pad(arr, ((up_pad, down_pad), (left_pad, right_pad)))
    return cp.pad(arr, ((0, 0), (up_pad, down_pad), (left_pad, right_pad)))

def phase_correlation_cupy(video, hannw):
    import cupy as cp
    eps = 1e-8
    N, h, w = video.shape
    video_fft = cp.fft.fft2(video * hannw, axes=(-2, -1))
    R = video_fft[:-1] * cp.conj(video_fft[1:])
    R /= (cp.abs(R) + eps)
    corr = cp.fft.ifft2(R, axes=(-2, -1)).real
    corr = cp.fft.fftshift(corr, axes=(-2, -1))
    corr_flat = corr.reshape(corr.shape[0], -1)
    max_idx   = cp.argmax(corr_flat, axis=1)
    peak_row  = max_idx // w
    peak_col  = max_idx % w
    shift_x   = peak_col - (w // 2)
    shift_y   = peak_row - (h // 2)
    return -cp.stack([shift_x, shift_y], axis=1)

def warp_roll(video, shifts):
    import cupy as cp
    aligned_video    = video.copy()
    cumulative_shift = cp.array([0, 0], dtype=cp.int32)
    for i in range(1, video.shape[0]):
        cumulative_shift += shifts[i - 1]
        dx = int(cumulative_shift[0].item())
        dy = int(cumulative_shift[1].item())
        aligned_video[i] = cp.roll(video[i], shift=(dy, dx), axis=(0, 1))
    return aligned_video

def warp_video_fft(video, shifts):
    import cupy as cp
    N, h, w = video.shape
    video_fft    = cp.fft.fft2(video, axes=(-2, -1))
    u            = cp.fft.fftfreq(h)[:, None]
    v            = cp.fft.fftfreq(w)[None, :]
    phase_shifts = cp.exp(-2j * cp.pi * (shifts[:, 0][:, None, None] * v +
                                          shifts[:, 1][:, None, None] * u))
    return cp.fft.ifft2(video_fft * phase_shifts, axes=(-2, -1)).real

def find_frame_translation_PC_cupy(video):
    import cupy as cp
    video_cp            = cp.asarray(video, dtype=cp.float32) / 255
    N, h, w             = video.shape
    hannW               = cp.outer(cp.hanning(h), cp.hanning(w))
    left_pad, right_pad = get_pad_size(w)
    up_pad,   down_pad  = get_pad_size(h)
    shifts = phase_correlation_cupy(
        _pad(video_cp, up_pad, down_pad, left_pad, right_pad),
        _pad(hannW,    up_pad, down_pad, left_pad, right_pad),
    )
    return shifts, video_cp

def find_frame_translation_LKi_cupy(video, iterations=3):
    import cupy as cp
    image1        = video[:-1]
    image2        = video[1:]
    aligned_image = image2.copy()
    shift         = cp.zeros((image1.shape[0], 2), dtype=cp.float32)
    I_x = 0.5 * (image1[:, 1:-1, 2:] - image1[:, 1:-1, :-2])
    I_y = 0.5 * (image1[:, 2:, 1:-1] - image1[:, :-2, 1:-1])
    for _ in range(iterations):
        I_t      = aligned_image[:, 1:-1, 1:-1] - image1[:, 1:-1, 1:-1]
        sum_Ix2  = cp.sum(I_x * I_x,  axis=(1, 2))
        sum_Iy2  = cp.sum(I_y * I_y,  axis=(1, 2))
        sum_IxIy = cp.sum(I_x * I_y,  axis=(1, 2))
        sum_IxIt = cp.sum(I_x * I_t,  axis=(1, 2))
        sum_IyIt = cp.sum(I_y * I_t,  axis=(1, 2))
        det      = sum_Ix2 * sum_Iy2 - sum_IxIy ** 2 + 1e-8
        delta_x  = (-sum_IxIt * sum_Iy2 + sum_IxIy * sum_IyIt) / det
        delta_y  = (-sum_IyIt * sum_Ix2 + sum_IxIy * sum_IxIt) / det
        delta    = cp.stack([delta_x, delta_y], axis=1)
        shift   += delta
        aligned_image = warp_video_fft(aligned_image, -delta)
    return shift

def find_frame_translation_PCLKi_cupy(video, iterations=3):
    shifts_PC, video_cp = find_frame_translation_PC_cupy(video)
    video_aligned_roll  = warp_roll(video_cp, -shifts_PC)
    shifts_LK           = find_frame_translation_LKi_cupy(video_aligned_roll, iterations)
    return shifts_PC + shifts_LK

def compute_shifts_for_roi(video, batch_size):
    import cupy as cp
    import tqdm
    n_ref_frames            = video.shape[0]
    all_reference_shifts    = cp.empty((n_ref_frames, 2), dtype=cp.float32)
    all_reference_shifts[0] = cp.array([0, 0], dtype=cp.float32)
    N_batches = int(np.ceil((n_ref_frames - 1) / batch_size))
    for i in tqdm.tqdm(range(N_batches)):
        start = i * batch_size
        end   = min((i + 1) * batch_size, n_ref_frames - 1)
        batch_shifts = find_frame_translation_PCLKi_cupy(video[start : end + 2])
        all_reference_shifts[start + 1 : end + 2] = batch_shifts
    return cp.asnumpy(cp.cumsum(all_reference_shifts, axis=0))


@app.function(
    gpu="L40S",
    image=cuda_image,
    timeout=3600,
    volumes={VOLUME_PATH: volume},
)
def run_pclk(
    input_path: str,
    rois: list[list[int]],
    output_path: str,
    batch_size: int = 16384,
):
    """
    Args:
        input_path:  path inside the volume to 00_raw_vibrations.npy  (T H W uint8)
        rois:        list of [x, y, w, h] ROI rectangles
        output_path: path inside the volume to write 01_raw_shifts.npy  (N_rois, T, 2)
        batch_size:  number of frame pairs per GPU batch
    """
    import tqdm

    print(f"Loading {input_path} ...")
    frame_recording = np.load(VOLUME_PATH / input_path)  # [T H W]

    all_shifts = []
    for roi in tqdm.tqdm(rois):
        x, y, w, h = roi
        print(f'Processing ROI={roi}, frame_recording={frame_recording.shape}')
        cropped_video = frame_recording[:, y:y+h, x:x+w]
        all_shifts.append(compute_shifts_for_roi(cropped_video, batch_size))

    result = np.stack(all_shifts, axis=0)  # (N_rois, T, 2)

    out = VOLUME_PATH / output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, result)
    volume.commit()
    print(f"Saved {result.shape} → {output_path}")


# ── local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    input_path: str,
    metadata_path: str,
    output_path: str,
    batch_size: int = 16384,
):
    """
    Args:
        input_path:    local path to 00_raw_vibrations.npy
        metadata_path: local path to metadata.jsonl
        output_path:   local path to write 01_raw_shifts.npy
    """
    import sys
    import json
    sys.path.insert(0, str(Path(__file__).parent))
    from io_utils import load, save

    input_path   = Path(input_path).resolve()
    metadata_path = Path(metadata_path).resolve()
    output_path  = Path(output_path).resolve()

    # upload input to volume
    volume_input = Path(input_path.name)
    print(f"Uploading {input_path} to volume ...")
    with volume.batch_upload() as batch:
        batch.put_file(input_path, str(volume_input))

    print(f"Loading {metadata_path} ...")
    meta = {k: v for d in load(metadata_path) for k, v in d.items()}
    rois = meta["roi"]  # list of [x, y, w, h]

    volume_output = Path(output_path.name)

    print(f"Submitting to Modal ({len(rois)} ROIs, batch_size={batch_size}) ...")
    run_pclk.remote(
        input_path=str(volume_input),
        rois=rois,
        output_path=str(volume_output),
        batch_size=batch_size,
    )

    print(f"Downloading result ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in volume.read_file(str(volume_output)):
            f.write(chunk)

    result = np.load(output_path)
    print(f"Saved {result.shape} → {output_path}")
