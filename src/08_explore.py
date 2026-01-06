import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 1. Setup

    Launch with `marimo edit --watch src/08_explore.py`
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from dataclasses import dataclass, field
    from collections import defaultdict
    from pathlib import Path
    import sys, types

    import numpy as np
    import matplotlib.pyplot as plt
    return Path, defaultdict, np, plt, sys, types


@app.cell
def _(sys, types):
    # Patch for pickle compatibility
    # if 'recover_core_lib' not in sys.modules:
    print('adding recover_core_lib to sys')
    fake = types.ModuleType('recover_core_lib')
    fake.compute_CAM2_translations_v3_cupy = lambda *a, **k: None
    sys.modules['recover_core_lib'] = fake
    return


@app.cell
def _(np):
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 2. Load Data
    """)
    return


@app.cell
def _(Path):
    BASE_DIR = Path('data/experiment_01/')
    exp_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])
    exp_dirs
    return (exp_dirs,)


@app.cell
def _(defaultdict, np):
    def load_data(exp_dirs) -> dict:
        """Load recovery and metadata from an experiment directory."""

        duplicate_idx_map = defaultdict(int)
        data = {}

        for exp_dir in exp_dirs:
            print(exp_dir)
            recovery = np.load(exp_dir / 'RECOVERY.npz', allow_pickle=True)
            # run_opt = recovery['run_opt'].item()
    
            object = str(exp_dir).split('/')[-1].split('_')[0]
            position = int(str(exp_dir).split('/')[-1].split('_')[1][3:])
    
            key = (object, position)
            duplicate_idx_map[key] += 1
            duplicate_idx = duplicate_idx_map[key]
    
            name = f'{object}-pos{position}-{duplicate_idx:02}'
    
            d = {
                'object': object,
                'position': position,
                'duplicate_idx': duplicate_idx,
                # 'fs': run_opt['cam_params']['camera_fs'],
                'fs': 5_000, # sampling frequency
                'path': exp_dir,
                'raw_shifts': recovery['all_shifts'],
            }
            data[name] = d
        return data
    return (load_data,)


@app.cell
def _(exp_dirs, load_data):
    data = load_data(exp_dirs)
    return (data,)


@app.cell
def _(data):
    name = "cube-pos1-01"
    data[name]
    return (name,)


@app.cell
def _(data, name):
    data[name]['raw_shifts'].shape
    return


@app.cell
def _(data, name, np):
    def make_timeshifts(shifts, fs):
        for name, d in data.items():
            d['timesteps'] = np.arange(d['raw_shifts'].shape[1]) / d['fs']
        return data

    data2 = make_timeshifts(data[name]['raw_shifts'])
    return (data2,)


@app.cell
def _(data2, name):
    data2[name]['timesteps'].shape
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 3. Compute FFT
    """)
    return


@app.cell
def _():
    MIN_FREQ, MAX_FREQ = 50, 2_000
    return MAX_FREQ, MIN_FREQ


@app.cell
def _(np):
    def run_fft(d, min_freq=None, max_freq=None):
        shifts, fs = d['raw_shifts'], d['']

        # compute fft values and frequencies
        fft_val = np.fft.rfft(shifts, axis=1)
        n_samples = shifts.shape[1]
        freq = np.fft.rfftfreq(n_samples, d=1.0/fs)

        # crop the frequency
        if min_freq is not None and max_freq is not None:
            mask = (freq >= min_freq) & (freq <= max_freq)
            fft_val, freq = fft_val[:, mask, :], freq[mask]
        return fft_val, freq
    return (run_fft,)


@app.cell
def _(MAX_FREQ, MIN_FREQ, data, name, run_fft):
    fft_val, freq = run_fft(data[name]['raw_shifts'], data[name]['fs'], MIN_FREQ, MAX_FREQ)
    return fft_val, freq


@app.cell
def _(fft_val):
    fft_val.shape
    return


@app.cell
def _(freq):
    freq.shape
    return


@app.cell
def _(mo):
    mo.md(r"""
    We have ~11k frequencies when we crop the frequncies to be between 50hz and 2_000 hz.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # 4. Clean Data
    """)
    return


@app.cell
def _():
    from scipy.signal import butter, sosfiltfilt
    return butter, sosfiltfilt


@app.cell
def _(butter, np, sosfiltfilt):
    def bandpass_filter(shifts, fs, lowcut=50, highcut=None, order=5):
        """Removes frequencies outside of [lowcut, highcut].
        sosfiltfilt applies the filter forward and backward, giving zero phase distortion."""
        if highcut is None: highcut = fs / 2 - 10  # Slightly below Nyquist frequency

        n_lasers, n_samples, n_coords = shifts.shape
        filtered_shifts = np.empty_like(shifts)

        for i in range(n_lasers):
            for j in range(n_coords):
                sos = butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')
                filtered_shifts[i, :, j] = sosfiltfilt(sos, shifts[i, :, j])
        return filtered_shifts
    return (bandpass_filter,)


@app.cell
def _(np):
    def hann_window(shifts):
        window = np.hanning(shifts.shape[1])
        return shifts * window[:, np.newaxis]
    return (hann_window,)


@app.cell
def _(np):
    def fft_magnitude_mean(fft_vals):
        assert len(fft_vals.shape) == 3
        return np.abs(fft_vals).mean(axis=(0,2)) # average over all lasers (dim 0), x/y coordinates (dim 2)
    return (fft_magnitude_mean,)


@app.cell
def _(bandpass_filter, fft_magnitude_mean, hann_window, plt, run_fft):
    def plot_clean_data(d):
        #unpack
        raw_shifts, fs, timesteps = d['raw_shifts'], d['fs'], d['timesteps']
    
        # fft for raw signal
        fft_vals, freqs = run_fft(raw_shifts, fs)
    
        # fft for bandpass filter
        bp_shifts = bandpass_filter(raw_shifts, fs)
        bp_fft_vals, bp_freqs = run_fft(bp_shifts, fs)
    
        # fft for bandpass filter + hann window
        wd_shifts = hann_window(bp_shifts)
        wd_fft_vals, wd_freqs = run_fft(wd_shifts, fs)
    
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs, fft_magnitude_mean(fft_vals), label='raw')
        ax.plot(bp_freqs, fft_magnitude_mean(bp_fft_vals), label='bandpass filter')
        ax.plot(wd_freqs, fft_magnitude_mean(wd_fft_vals), label='bandpass filter + hann window')
        ax.set(xlim=(45, 150), ylim=(0, 200), xlabel='Frequency (Hz)', ylabel='Mean FFT Magnitude')
        ax.legend()
        fig.suptitle('Clean the Data')
        return fig
    return (plot_clean_data,)


@app.cell
def _(data, name, plot_clean_data):
    plot_clean_data(data[name])
    return


@app.cell
def _(mo):
    mo.md(r"""
    The signal looks much cleaner with bandpass filter + hann window. Let's define a function to apply these transformations.
    """)
    return


@app.cell
def _(bandpass_filter, hann_window):
    def clean_shifts(shifts, fs):
        shifts_bp = bandpass_filter(shifts, fs)
        shifts_wd = hann_window(shifts_bp)
        return shifts_wd
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
