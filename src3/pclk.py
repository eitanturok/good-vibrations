import numpy as np

#***** PCLK algorithm *****
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
    n_ref_frames            = video.shape[0]
    all_reference_shifts    = cp.empty((n_ref_frames, 2), dtype=cp.float32)
    all_reference_shifts[0] = cp.array([0, 0], dtype=cp.float32)
    N_batches = int(np.ceil((n_ref_frames - 1) / batch_size))
    for i in range(N_batches):
        start = i * batch_size
        end   = min((i + 1) * batch_size, n_ref_frames - 1)
        batch_shifts = find_frame_translation_PCLKi_cupy(video[start : end + 2])
        all_reference_shifts[start + 1 : end + 2] = batch_shifts
    return cp.asnumpy(cp.cumsum(all_reference_shifts, axis=0))


def compute_shifts_for_all_rois_batched(videos, batch_size):
    """Process all ROIs in parallel on the GPU.

    Args:
        videos: (L, T, H, W) numpy array — all ROI crops stacked (stays on CPU)
        batch_size: number of frame pairs per GPU batch
    Returns:
        (L, T, 2) numpy array of cumulative shifts
    """
    from tqdm import tqdm
    import cupy as cp
    L, T, H, W = videos.shape

    all_shifts = np.zeros((L, T, 2), dtype=np.float32)
    N_batches  = int(np.ceil((T - 1) / batch_size))

    hannW = cp.outer(cp.hanning(H), cp.hanning(W))
    left_pad, right_pad = get_pad_size(W)
    up_pad,   down_pad  = get_pad_size(H)
    hannW_pad = _pad(hannW, up_pad, down_pad, left_pad, right_pad)   # (pH, pW), precomputed
    pH, pW = hannW_pad.shape

    for i in tqdm(range(N_batches)):
        start = i * batch_size
        end   = min((i + 1) * batch_size, T - 1)
        # transfer only this batch's frames from CPU → GPU
        clip  = cp.asarray(videos[:, start : end + 2], dtype=cp.float32) / 255  # (L, bs+1, H, W)
        n_pairs = clip.shape[1] - 1

        # flatten ROI and frame-pair dims: (L*(n_pairs+1), H, W)
        flat = clip.reshape(L * (n_pairs + 1), H, W)
        flat_pad = _pad(flat, up_pad, down_pad, left_pad, right_pad)
        flat_pad *= hannW_pad   # in-place: no extra allocation for the broadcast

        # phase correlation on all L*(n_pairs+1) frames at once
        fft   = cp.fft.fft2(flat_pad, axes=(-2, -1))
        del flat, flat_pad
        cp.get_default_memory_pool().free_all_blocks()
        fft   = fft.reshape(L, n_pairs + 1, *fft.shape[-2:])
        R     = fft[:, :-1] * cp.conj(fft[:, 1:])          # (L, n_pairs, pH, pW)
        del fft
        cp.get_default_memory_pool().free_all_blocks()
        R    /= (cp.abs(R) + 1e-8)
        corr  = cp.fft.ifft2(R, axes=(-2, -1)).real
        del R
        cp.get_default_memory_pool().free_all_blocks()
        corr  = cp.fft.fftshift(corr, axes=(-2, -1))
        corr_flat = corr.reshape(L * n_pairs, -1)
        max_idx   = cp.argmax(corr_flat, axis=1)
        del corr, corr_flat
        peak_row  = max_idx // pW
        peak_col  = max_idx % pW
        shifts_PC = -cp.stack([peak_col - pW // 2, peak_row - pH // 2], axis=1)
        shifts_PC = shifts_PC.reshape(L, n_pairs, 2).astype(cp.float32)
        cp.get_default_memory_pool().free_all_blocks()  # release PC temporaries before LK

        # Lucas-Kanade refinement — per-ROI, vectorised across L and n_pairs
        image1, image2 = clip[:, :-1], clip[:, 1:]

        # warp image2 using warp_roll per ROI — identical to pclk_old.py
        aligned = cp.empty_like(image2)                               # (L, n_pairs, H, W)
        for l in range(L):
            aligned[l] = warp_roll(clip[l], -cp.round(shifts_PC[l]).astype(cp.int32))[1:]
        del clip

        # Lucas-Kanade with 3 iterations and re-warp
        shifts_LK = cp.zeros((L, n_pairs, 2), dtype=cp.float32)
        I_x = 0.5 * (image1[:, :, 1:-1, 2:] - image1[:, :, 1:-1, :-2])  # (L,n,H-2,W-2)
        I_y = 0.5 * (image1[:, :, 2:, 1:-1] - image1[:, :, :-2, 1:-1])
        for _ in range(3):
            I_t      = aligned[:, :, 1:-1, 1:-1] - image1[:, :, 1:-1, 1:-1]
            sum_Ix2  = cp.sum(I_x * I_x,  axis=(-2, -1))
            sum_Iy2  = cp.sum(I_y * I_y,  axis=(-2, -1))
            sum_IxIy = cp.sum(I_x * I_y,  axis=(-2, -1))
            sum_IxIt = cp.sum(I_x * I_t,  axis=(-2, -1))
            sum_IyIt = cp.sum(I_y * I_t,  axis=(-2, -1))
            det      = sum_Ix2 * sum_Iy2 - sum_IxIy ** 2 + 1e-8
            delta    = cp.stack([
                (-sum_IxIt * sum_Iy2 + sum_IxIy * sum_IyIt) / det,
                (-sum_IyIt * sum_Ix2 + sum_IxIy * sum_IxIt) / det,
            ], axis=-1)
            shifts_LK += delta
            aligned_flat = aligned.reshape(L * n_pairs, H, W)
            aligned_flat = warp_video_fft(aligned_flat, -delta.reshape(L * n_pairs, 2))
            cp.get_default_memory_pool().free_all_blocks()
            aligned = aligned_flat.reshape(L, n_pairs, H, W)

        all_shifts[:, start + 1 : end + 2] = cp.asnumpy(shifts_PC + shifts_LK)

    return np.cumsum(all_shifts, axis=1)                     # (L, T, 2)
