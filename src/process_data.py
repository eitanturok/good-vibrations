import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt, find_peaks


def load_data(exp_dirs, grid_width=7) -> dict:
    duplicate_idx_map = defaultdict(int)
    data = {}

    for exp_dir in exp_dirs:
        with open(exp_dir / "experiment_config.json", "r") as f: config = json.load(f)
        fs = config["FPS"]

        seperator = '/' if '/' in str(exp_dir) else '\\'
        object = str(exp_dir).split(seperator)[-1].split('-')[0]
        x_position = None if object == 'empty' else int(str(exp_dir).split(seperator)[-1].split('-')[1][:2])
        y_position = None if object == 'empty' else int(str(exp_dir).split(seperator)[-1].split('-')[2][:2])

        key = (object, x_position, y_position)
        duplicate_idx_map[key] += 1
        duplicate_idx = duplicate_idx_map[key]

        name = f'{object}-{x_position}x-{y_position}y-{duplicate_idx:02}'

        recovery = np.load(exp_dir / 'RECOVERY.npz', allow_pickle=True)
        shifts_raw = recovery['all_shifts']
        time = np.arange(shifts_raw.shape[1]) / fs

        # Compute 2D position tuple and linearized index for classification
        position_xy = (x_position, y_position) if x_position is not None else None
        position_idx = (y_position * grid_width + x_position) if x_position is not None else None

        d = {
            'object': object,
            'x_position': x_position,
            'y_position': y_position,
            'position_xy': position_xy,  # tuple (x, y) for display/grouping
            'position_idx': position_idx,  # linearized index for classification
            'duplicate_idx': duplicate_idx,
            'fs': fs,
            'path': exp_dir,
            'shifts_raw': shifts_raw,
            'time': time,
            'img_path': exp_dir / 'box_overhead_image.png'
        }
        data[name] = d
    return data

def bandpass_filter(shifts, fs, lowcut=50, highcut=None, order=5):
    """Removes frequencies outside of [lowcut, highcut].
    sosfiltfilt applies the filter forward and backward, giving zero phase distortion."""
    if highcut is None: highcut = fs / 2 - 10  # Slightly below Nyquist frequency

    n_lasers, _, n_coords = shifts.shape
    filtered_shifts = np.empty_like(shifts)

    for i in range(n_lasers):
        for j in range(n_coords):
            sos = butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
            filtered_shifts[i, :, j] = sosfiltfilt(sos, shifts[i, :, j])
    return filtered_shifts

def hann_window(shifts):
    window = np.hanning(shifts.shape[1])
    return shifts * window[:, np.newaxis]


def shifts_to_fft(shifts, fs, min_freq=None, max_freq=None):
    # compute fft values and frequencies
    fft_val = np.fft.rfft(shifts, axis=1)
    n_samples = shifts.shape[1]
    freqs = np.fft.rfftfreq(n_samples, d=1.0/fs)

    # crop the frequency
    if min_freq is not None and max_freq is not None:
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        fft_val, freqs = fft_val[:, mask, :], freqs[mask]
    return fft_val, freqs

def fft_magnitude(fft_vals, return_std=False):
    """Compute fft magnitude and average it over all the lasers and x,y coordinates"""
    if len(fft_vals.shape) == 3: # (n_lasers, n_freqs, n_coords)
        fft_mag_mean = np.abs(fft_vals).mean(axis=(0,2))
        return (fft_mag_mean, np.abs(fft_vals).std(axis=(0,2))) if return_std else fft_mag_mean
    elif len(fft_vals.shape) == 2: # (n_lasers, n_coords) we already chose a specific frequency
        fft_mag_mean = np.abs(fft_vals).mean()
        return (fft_mag_mean, np.abs(fft_vals).std()) if return_std else fft_mag_mean
    else:
        raise ValueError()

def find_modes(freqs, fft_vals, n_modes=5, min_distance=100):
    """Find the top n_modes frequency peaks from mean FFT magnitude (averaged over lasers + x/y coordinate).

    Returns list of (frequency, magnitude, time) tuples sorted by frequency.
    Time is estimated assuming a log chirp from 50-2000 Hz over the recording duration.
    """

    mean_mag = fft_magnitude(fft_vals)
    peaks, _ = find_peaks(mean_mag, distance=min_distance)
    peak_mags = mean_mag[peaks]

    # Sort by magnitude and take top n_modes
    top_indices = np.argsort(peak_mags)[-n_modes:][::-1]
    mode_freqs_idx = np.array(sorted(peaks[top_indices]))

    mode_freqs = freqs[mode_freqs_idx]
    mode_fft_vals = fft_vals[:, mode_freqs_idx, :]
    return mode_freqs_idx, mode_freqs, mode_fft_vals

def sync_phases(fft_vals, laser_idx=0, xy_idx=0, eps=1e-20):
    fft_vals_synced = fft_vals.copy()             # copy
    ref = fft_vals[laser_idx, :, xy_idx]          # shape (freq,)
    phase = np.conj(ref) / (np.abs(ref)**2 + eps) # unit complex + divide by magnitude
    fft_vals_synced *= phase[None, :, None]       # broadcast over lasers and xy
    assert np.allclose(1, np.real(fft_vals_synced[laser_idx, :, xy_idx])) # check we have magnitude 1
    return fft_vals_synced # (n_lasers, n_freqs, 2)

def get_gradients(fft_vals, n_lasers):
    """Extract real-valued gradients from complex FFT values, reshaped to (n_modes, 2, grid, grid).

    Args:
        fft_vals: shape (n_lasers^2, n_modes, 2) - complex FFT values
        n_lasers: grid dimension (e.g., 10 for 10x10 grid)

    Returns:
        gradients: shape (n_modes, 2, n_lasers, n_lasers) - real-valued gradients
                   Access as gradients[mode_idx, 0] for dx, gradients[mode_idx, 1] for dy
    """
    fft_vals = np.real(fft_vals)  # shape: (100, n_modes, 2)
    # Reshape: (100, n_modes, 2) -> (10, 10, n_modes, 2) -> (n_modes, 2, 10, 10)
    gradients = fft_vals.reshape(n_lasers, n_lasers, -1, 2)
    gradients = gradients.transpose(2, 3, 0, 1)  # (n_modes, 2, n_lasers, n_lasers)
    return gradients


def process_data_all(exp_dirs, min_freq=50, max_freq=1000, n_modes=10, min_distance=10, canonical_mode_freqs=None):

    data = load_data(exp_dirs)
    print(f'Loaded {len(data)} experiments')

    for name, d in tqdm(data.items()):

        # clean data
        d['shifts'] = hann_window(bandpass_filter(d['shifts_raw'], d['fs']))

        # compute fft
        d['fft_vals'], d['freqs'] = shifts_to_fft(d['shifts'], d['fs'], min_freq, max_freq)

        # get modes
        d['mode_freqs'] = canonical_mode_freqs
        d['mode_freqs_idx'] = np.array([np.argmin(np.abs(d['freqs'] - target_freq)) for target_freq in canonical_mode_freqs])
        d['mode_fft_vals'] = d['fft_vals'][:, d['mode_freqs_idx'], :]

        # sync phases
        d['synced_fft_vals'] = sync_phases(d['fft_vals'])
        d['synced_mode_fft_vals'] = sync_phases(d['mode_fft_vals'])

        # get gradients
        n_lasers = int(np.sqrt(d['shifts'].shape[0]))
        d['synced_fft_gradients'] = get_gradients(d['synced_fft_vals'], n_lasers)  # shape: (n_freqs, 2, 10, 10)
        d['mode_fft_gradients'] = get_gradients(d['mode_fft_vals'], n_lasers)  # shape: (n_modes, 2, 10, 10)
        d['synced_mode_fft_gradients'] = get_gradients(d['synced_mode_fft_vals'], n_lasers)  # shape: (n_modes, 2, 10, 10)

    return data


def process_data(exp_dirs, min_freq=50, max_freq=1000, n_modes=10, min_distance=10, canonical_mode_freqs=None):

    data = load_data(exp_dirs)
    print(f'Loaded {len(data)} experiments')

    for name, d in tqdm(data.items()):

        # clean data
        d['shifts'] = hann_window(bandpass_filter(d['shifts_raw'], d['fs']))

        # compute fft
        d['fft_vals'], d['freqs'] = shifts_to_fft(d['shifts'], d['fs'], min_freq, max_freq)

    return data
