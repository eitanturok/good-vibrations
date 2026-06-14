from argparse import ArgumentParser
import cupy as cp
import tqdm
import numpy as np
import os


# CUPY VERSION

def compute_CAM2_translations_v3_cupy(video_pp_cam2, batch_size = 128):

    n_ref_frames = video_pp_cam2.shape[0]
    # Preallocate a cupy array for all_reference_shifts: one shift per frame (first is [0,0])
    all_reference_shifts    = cp.empty((n_ref_frames, 2), dtype=cp.float32)
    all_reference_shifts[0] = cp.array([0, 0], dtype=cp.float32)

    print('----- Computing shifts -----')

    # There are (n_ref_frames - 1) shifts, process them in batches
    N_batches = int(np.ceil((n_ref_frames - 1) / batch_size))

    for i in tqdm.tqdm(range(N_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_ref_frames - 1)
        # For each batch, we need frames from index start to end+1
        batch_frames = video_pp_cam2[start : end + 2]
        # Compute shifts for this batch; expected shape: (number_of_shifts_in_batch, 2)
        reference_shifts_batch = find_frame_translation_PCLKi_cupy(batch_frames)
        # Place the computed shifts into our preallocated array
        all_reference_shifts[start + 1 : end + 2] = reference_shifts_batch

    # Compute the cumulative sum over shifts and negate to get the alignment shifts per frame
    all_shifts = cp.cumsum(all_reference_shifts, axis=0)
    # Convert the final result to a numpy array
    all_shifts = cp.asnumpy(all_shifts)

    return all_shifts

def find_frame_translation_PCLKi_cupy(video, iterations=3):

    shifts_PC,video_cp     = find_frame_translation_PC_cupy(video)
    # shifts = (127, 2), video_cp =  (128, 6, 120)
    video_aligned_roll     = warp_roll(video_cp, -shifts_PC)

    shifts_LK              = find_frame_translation_LKi_cupy(video_aligned_roll, iterations)

    return shifts_PC+shifts_LK

def find_frame_translation_PC_cupy(video):

    video_cp      = cp.asarray(video, dtype=cp.float32) / 255

    N,h,w         = video.shape
    hannW         = cp.outer(cp.hanning(h), cp.hanning(w))

    left_pad,right_pad = get_pad_size(w)
    up_pad,  down_pad  = get_pad_size(h)

    def pad(arr):
        if arr.ndim == 2:
            return cp.pad(arr, ((up_pad, down_pad), (left_pad, right_pad)))
        elif arr.ndim == 3:
            return cp.pad(arr, ((0, 0), (up_pad, down_pad), (left_pad, right_pad)))
        else:
            raise ValueError("Unsupported array dimensions for padding")

    hannW         = pad(hannW)

    shifts        = phase_correlation_cupy(pad(video_cp),hannW)

    return shifts, video_cp


def get_pad_size(w):
        two_power = 2**np.ceil(np.log2(w))
        left_pad  = int((two_power-w)//2)
        right_pad = int(two_power-w-left_pad)
        return left_pad, right_pad


def phase_correlation_cupy(video, hannw):
    # video: (N, h, w), hannw: (h, w)
    eps = 1e-8

    N, h, w = video.shape
    print(f'pc for video = {video.shape}, hannw = {hannw.shape}')

    # Apply the window and compute FFT for every frame at once.
    video_fft = cp.fft.fft2(video * hannw, axes=(-2, -1))

    # Compute the normalized cross-power spectrum between consecutive frames.
    R = video_fft[:-1] * cp.conj(video_fft[1:])
    R /= (cp.abs(R) + eps)

    # Inverse FFT to get the cross-correlation (batch mode)
    corr = cp.fft.ifft2(R, axes=(-2, -1)).real
    corr = cp.fft.fftshift(corr, axes=(-2, -1))

    # Flatten each (h, w) correlation map and find the index of its maximum.
    corr_flat = corr.reshape(corr.shape[0], -1)
    max_idx = cp.argmax(corr_flat, axis=1)
    peak_row = max_idx // w  # y-coordinate
    peak_col = max_idx % w   # x-coordinate

    # Compute shifts: if the peak is at the center, shift=0.
    shift_x = peak_col - (w // 2)
    shift_y = peak_row - (h // 2)
    shifts = cp.stack([shift_x, shift_y], axis=1)

    return -shifts

def warp_roll(video, shifts):
    aligned_video = video.copy()
    cumulative_shift = cp.array([0, 0], dtype=cp.int32)

    for i in range(1, video.shape[0]):
        cumulative_shift += shifts[i - 1]
        # Convert cumulative shift components to Python ints for cp.roll.
        dx = int(cumulative_shift[0].item())
        dy = int(cumulative_shift[1].item())
        # Note: cp.roll expects shifts as (shift_along_axis0, shift_along_axis1) which corresponds to (dy, dx)
        aligned_video[i] = cp.roll(video[i], shift=(dy, dx), axis=(0, 1))

    return aligned_video

def warp_video_fft(video, shifts):

    N, h, w = video.shape

    # Compute FFT for each frame (batch FFT over the last two axes)
    video_fft = cp.fft.fft2(video, axes=(-2, -1))

    # Create frequency grids for the spatial dimensions
    u = cp.fft.fftfreq(h)[:, None]  # shape: (h, 1)
    v = cp.fft.fftfreq(w)[None, :]   # shape: (1, w)

    # Compute phase shifts for each frame using broadcasting:
    # shifts[:, 0] is dx (applied to the horizontal frequencies v)
    # shifts[:, 1] is dy (applied to the vertical frequencies u)
    phase_shifts = cp.exp(-2j * cp.pi * (shifts[:, 0][:, None, None] * v +
                                         shifts[:, 1][:, None, None] * u))

    # Apply the phase shift to each frame in the Fourier domain
    shifted_video_fft = video_fft * phase_shifts

    # Compute the inverse FFT to obtain the shifted video
    shifted_video = cp.fft.ifft2(shifted_video_fft, axes=(-2, -1)).real

    return shifted_video

def find_frame_translation_LKi_cupy(video, iterations=3):

    # Define frame pairs: image1 and image2 (for N-1 pairs)
    image1 = video[:-1]  # shape (B, h, w)
    image2 = video[1:]   # shape (B, h, w)
    B, h, w = image1.shape

    # Initialize aligned_image (the moving image) and shift estimates.
    aligned_image = image2.copy()  # shape (B, h, w)
    shift = cp.zeros((B, 2), dtype=cp.float32)

    # Precompute spatial gradients of image1.
    # We restrict to the inner region [1:-1,1:-1] to avoid boundary issues.
    I_x = 0.5 * (image1[:, 1:-1, 2:] - image1[:, 1:-1, :-2])    # shape (B, h-2, w-2)
    I_y = 0.5 * (image1[:, 2:, 1:-1] - image1[:, :-2, 1:-1])    # shape (B, h-2, w-2)

    for _ in range(iterations):
        # Compute temporal error over the same inner region.
        I_t = aligned_image[:, 1:-1, 1:-1] - image1[:, 1:-1, 1:-1]  # shape (B, h-2, w-2)

        # Compute the sums required for the 2x2 normal equation per frame.
        sum_Ix2  = cp.sum(I_x * I_x, axis=(1, 2))
        sum_Iy2  = cp.sum(I_y * I_y, axis=(1, 2))
        sum_IxIy = cp.sum(I_x * I_y, axis=(1, 2))
        sum_IxIt = cp.sum(I_x * I_t, axis=(1, 2))
        sum_IyIt = cp.sum(I_y * I_t, axis=(1, 2))

        # Solve for the incremental shift using the closed-form solution:
        # [sum_Ix2   sum_IxIy] [delta_x] = -[sum_IxIt]
        # [sum_IxIy  sum_Iy2 ] [delta_y]   -[sum_IyIt]
        det = sum_Ix2 * sum_Iy2 - sum_IxIy ** 2 + 1e-8  # Avoid division by zero

        delta_x = (- sum_IxIt * sum_Iy2 + sum_IxIy * sum_IyIt) / det
        delta_y = (- sum_IyIt * sum_Ix2 + sum_IxIy * sum_IxIt) / det

        delta = cp.stack([delta_x, delta_y], axis=1)  # shape (B, 2)

        # Accumulate the incremental shift estimates.
        shift += delta

        # Warp the aligned_image using the negative of the computed delta shifts.
        # This warps each frame in the batch simultaneously.
        aligned_image = warp_video_fft(aligned_image, -delta)

    return shift


def main():
    parser = ArgumentParser()
    parser.add_argument('--working_dir', type=str, help='folder where frame recordings are located, and where to save recovery')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size for PCLK+')
    args = parser.parse_args()

    # load raw frame recordings
    frame_recording = np.load(os.path.join(args.working_dir, 'frame-recording.npy'))  # [T H W]
    # load metadata
    meta_data = np.load(os.path.join(args.working_dir, 'metadata.npz'), allow_pickle=True)

    # process ROIs
    all_shifts = []
    all_params = []
    for ROI in tqdm.tqdm(meta_data['run_opt'].item()['run_opt_multiROIs']['ROIs']):
        all_params.append(f'run_ROI_{ROI}')
        print(f'Processing ROI={ROI}, frame_recording={frame_recording.shape}')
        cropped_video  = frame_recording[:,ROI[1]:ROI[1]+ROI[3],ROI[0]:ROI[0]+ROI[2]]
        shifts = compute_CAM2_translations_v3_cupy(video_pp_cam2=cropped_video, batch_size=args.batch_size)
        all_shifts.append(shifts)

    all_shifts = np.stack(all_shifts, axis=0)
    # save output
    run_opt = {
        'run_opt_recovery': args,
        'run_opt_multiROIs': meta_data['run_opt'].item()['run_opt_multiROIs'],
        'cam_params': meta_data['run_opt'].item()['cam_params'],
        'run_dict': {}
    }

    np.savez(os.path.join(args.working_dir, 'RECOVERY.npz'),
             all_shifts      = all_shifts,
             all_params      = all_params,
             loaded_filename = args.working_dir,
             run_opt         = run_opt)

    print('Done.')


if __name__ == '__main__':
    main()
