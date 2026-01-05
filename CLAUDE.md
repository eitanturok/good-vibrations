# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Good Vibrations is a scientific computing project for analyzing surface vibrations using high-speed camera data and laser speckle patterns. The pipeline processes video recordings of laser points on vibrating surfaces to extract and analyze sub-pixel motion signals.

## Architecture

### Pipeline Flow (Jupyter Notebooks in `src/`)

1. **00_chirp.ipynb** - Generate audio test signals (chirps, tones) as WAV files for inducing vibrations
2. **01_record.ipynb** - Record high-speed camera footage of laser speckle patterns
3. **02_analyze.ipynb** - Load recovered shift data, compute FFT frequency spectra, visualize results
4. **03_mark_visualize.ipynb** - Mark and visualize regions of interest
5. **04_dimension_reduction.ipynb** - Dimensionality reduction on vibration data
6. **05_visualize_shifts.ipynb** - Visualize computed shifts over time
7. **06_shift_fft.ipynb** - FFT analysis of shift data

### Core Libraries

- **`lib/`** - Image and video processing utilities
  - `image_processing.py` - Gamma correction, debayering, 16-bit to 8-bit conversion
  - `opencv_video_utils.py` - `videoPlayer` class for interactive video playback and export

- **`utils/recover_core_lib.py`** - Core vibration recovery algorithms:
  - Phase correlation (CPU and GPU/CuPy versions) for sub-pixel shift detection
  - Lucas-Kanade iterative refinement for sub-pixel accuracy
  - `VibrationViewer` class for interactive matplotlib visualization
  - Signal filtering and WAV export utilities

### Data Format

- Shift data shape: `(N_sensors, N_frames, 2)` where 2 represents X and Y displacement
- Typical setup: 10x10 grid of laser points (100 sensors), 5000 FPS camera
- Metadata stored in `metadata.npz` with camera parameters (`camera_FPS`, `exposure`, `ROIs`)
- Recovery results stored in `RECOVERY.npz` with `all_shifts` array

## Development

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
```

### Running Notebooks
```bash
jupyter notebook src/
```

### Key Dependencies
- NumPy, SciPy (signal processing, FFT)
- OpenCV (image/video processing)
- CuPy (GPU-accelerated computations, optional)
- Matplotlib (visualization with interactive widgets)

## Key Concepts

- **Phase correlation**: FFT-based method to find sub-pixel translations between consecutive frames
- **Hann windowing**: Applied before FFT to reduce spectral leakage
- **Lucas-Kanade refinement**: Iterative gradient-based method for sub-pixel accuracy after integer phase correlation
- **Parabolic interpolation**: Used for sub-pixel peak localization in correlation maps
