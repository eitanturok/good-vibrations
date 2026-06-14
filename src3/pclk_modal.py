"""
Modal app for running PCLK+ (phase correlation + Lucas-Kanade) shift recovery.

Usage:
    # Run on a single directory
    modal run pclk_modal.py --input-dir /path/to/data --output-dir /path/to/out

    # Override batch size (default 16384)
    modal run pclk_modal.py --input-dir /path/to/data --output-dir /path/to/out --batch-size 8192

    # Deploy as a persistent function you can call remotely
    modal deploy pclk_modal.py
"""

import modal
import numpy as np

app = modal.App("pclk")

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04", add_python="3.11")
    .pip_install("cupy-cuda12x", "numpy", "tqdm")
)


@app.function(
    gpu="L40S",
    image=cuda_image,
    timeout=3600,
)
def run_pclk(
    frame_recording_bytes: bytes,
    metadata_bytes: bytes,
    batch_size: int = 16384,
) -> bytes:
    """
    Process all ROIs from a single recording.

    Args:
        frame_recording_bytes: raw bytes of frame-recording.npy
        metadata_bytes:        raw bytes of metadata.npz
        batch_size:            number of frame pairs per GPU batch

    Returns:
        raw bytes of RECOVERY.npz
    """
    import io
    import cupy as cp
    import tqdm

    # ── deserialize inputs ──────────────────────────────────────────────────
    frame_recording = np.load(io.BytesIO(frame_recording_bytes))          # [T H W]
    meta_data       = np.load(io.BytesIO(metadata_bytes), allow_pickle=True)

    # ── PCLK helpers (inlined so the function is self-contained) ────────────

    def get_pad_size(w):
        two_power = 2 ** np.ceil(np.log2(w))
        left_pad  = int((two_power - w) // 2)
        right_pad = int(two_power - w - left_pad)
        return left_pad, right_pad

    def phase_correlation_cupy(video, hannw):
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
        aligned_video     = video.copy()
        cumulative_shift  = cp.array([0, 0], dtype=cp.int32)
        for i in range(1, video.shape[0]):
            cumulative_shift += shifts[i - 1]
            dx = int(cumulative_shift[0].item())
            dy = int(cumulative_shift[1].item())
            aligned_video[i] = cp.roll(video[i], shift=(dy, dx), axis=(0, 1))
        return aligned_video

    def warp_video_fft(video, shifts):
        N, h, w = video.shape
        video_fft    = cp.fft.fft2(video, axes=(-2, -1))
        u            = cp.fft.fftfreq(h)[:, None]
        v            = cp.fft.fftfreq(w)[None, :]
        phase_shifts = cp.exp(-2j * cp.pi * (shifts[:, 0][:, None, None] * v +
                                              shifts[:, 1][:, None, None] * u))
        return cp.fft.ifft2(video_fft * phase_shifts, axes=(-2, -1)).real

    def find_frame_translation_PC_cupy(video):
        video_cp           = cp.asarray(video, dtype=cp.float32) / 255
        N, h, w            = video.shape
        hannW              = cp.outer(cp.hanning(h), cp.hanning(w))
        left_pad, right_pad = get_pad_size(w)
        up_pad,   down_pad  = get_pad_size(h)
        def pad(arr):
            if arr.ndim == 2:
                return cp.pad(arr, ((up_pad, down_pad), (left_pad, right_pad)))
            return cp.pad(arr, ((0, 0), (up_pad, down_pad), (left_pad, right_pad)))
        shifts = phase_correlation_cupy(pad(video_cp), pad(hannW))
        return shifts, video_cp

    def find_frame_translation_LKi_cupy(video, iterations=3):
        image1         = video[:-1]
        image2         = video[1:]
        aligned_image  = image2.copy()
        shift          = cp.zeros((image1.shape[0], 2), dtype=cp.float32)
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
        shifts_PC, video_cp      = find_frame_translation_PC_cupy(video)
        video_aligned_roll       = warp_roll(video_cp, -shifts_PC)
        shifts_LK                = find_frame_translation_LKi_cupy(video_aligned_roll, iterations)
        return shifts_PC + shifts_LK

    def compute_CAM2_translations_v3_cupy(video_pp_cam2, batch_size=128):
        n_ref_frames               = video_pp_cam2.shape[0]
        all_reference_shifts       = cp.empty((n_ref_frames, 2), dtype=cp.float32)
        all_reference_shifts[0]    = cp.array([0, 0], dtype=cp.float32)
        N_batches = int(np.ceil((n_ref_frames - 1) / batch_size))
        for i in tqdm.tqdm(range(N_batches)):
            start = i * batch_size
            end   = min((i + 1) * batch_size, n_ref_frames - 1)
            batch_frames = video_pp_cam2[start : end + 2]
            reference_shifts_batch = find_frame_translation_PCLKi_cupy(batch_frames)
            all_reference_shifts[start + 1 : end + 2] = reference_shifts_batch
        all_shifts = cp.cumsum(all_reference_shifts, axis=0)
        return cp.asnumpy(all_shifts)

    # ── process each ROI ────────────────────────────────────────────────────
    all_shifts = []
    all_params = []
    for ROI in tqdm.tqdm(meta_data['run_opt'].item()['run_opt_multiROIs']['ROIs']):
        all_params.append(f'run_ROI_{ROI}')
        print(f'Processing ROI={ROI}, frame_recording={frame_recording.shape}')
        cropped_video = frame_recording[:, ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]
        shifts = compute_CAM2_translations_v3_cupy(video_pp_cam2=cropped_video, batch_size=batch_size)
        all_shifts.append(shifts)

    all_shifts = np.stack(all_shifts, axis=0)

    run_opt = {
        'run_opt_recovery': {'batch_size': batch_size},
        'run_opt_multiROIs': meta_data['run_opt'].item()['run_opt_multiROIs'],
        'cam_params': meta_data['run_opt'].item()['cam_params'],
        'run_dict': {}
    }

    buf = io.BytesIO()
    np.savez(buf,
             all_shifts      = all_shifts,
             all_params      = all_params,
             run_opt         = run_opt)
    return buf.getvalue()


# ── local entrypoint ────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    input_path: str,
    metadata_path: str,
    output_path: str,
    batch_size: int = 16384,
):
    """
    Args:
        input_path:    path to frame-recording.npy
        metadata_path: path to metadata.npz
        output_path:   path to write RECOVERY.npz
        batch_size:    GPU batch size (default 16384)
    """
    import os

    print(f"Loading {input_path} ...")
    frame_recording_bytes = open(input_path, "rb").read()

    print(f"Loading {metadata_path} ...")
    metadata_bytes = open(metadata_path, "rb").read()

    print(f"Submitting to Modal (batch_size={batch_size}) ...")
    result_bytes = run_pclk.remote(
        frame_recording_bytes=frame_recording_bytes,
        metadata_bytes=metadata_bytes,
        batch_size=batch_size,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(result_bytes)

    print(f"Saved → {output_path}")
