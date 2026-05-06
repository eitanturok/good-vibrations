---
configs:
  - config_name: default
    data_files:
      - data/metadata.jsonl
dataset_info:
  features:
    - name: sample_id
      dtype: int64
    - name: segmented_overhead_file_name
      dtype: image
    - name: speckle_vibrations_file_name
      dtype: video
    - name: speckle_shifts_ifft_audio_file_name
      dtype: audio
    - name: audio_file_name
      dtype: audio
    - name: experiment_id
      dtype: string
    - name: speakers
      dtype: string
    - name: x_position
      dtype: int64
    - name: y_position
      dtype: int64
    - name: x_com
      dtype: float64
    - name: y_com
      dtype: float64
    - name: object
      dtype: string
    - name: n_objects
      dtype: int64
    - name: box_material
      dtype: string
    - name: mask_file_name
      dtype: image
    - name: experiment_dir
      dtype: string
    - name: manifest
      dtype: string
---

# Laser Vibrations

Dataset of laser speckle vibration recordings used to locate objects hidden inside a cardboard box.
A 10×10 grid of lasers shines on the side of a box; as loudspeakers excite the box, each laser's
speckle pattern shifts in proportion to the local surface vibration. The goal is to reconstruct the
shape and location of an object inside the box from the vibration signals alone.

Per-sample viewer metadata lives in `data/metadata.jsonl`.
Full signal data and media files live in per-sample subdirectories under `data/`.
Shared overhead images live under `image/` and the shared audio file under `audio/`.

---

# Dataset Columns

These are the columns shown in the HuggingFace dataset viewer, sourced from `data/metadata.jsonl`.

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | int | Unique sequential identifier for this sample |
| `segmented_overhead_file_name` | image | Final per-sample overhead photo shown in the viewer; stored as `data/<sample_id>/overhead.png` and includes the speaker overlay |
| `speckle_vibrations_file_name` | video | Slow-motion preview of the laser speckle pattern while the box vibrates |
| `speckle_shifts_ifft_audio_file_name` | audio | Vibration signal of a single laser point reconstructed as audio (inverse FFT) |
| `audio_file_name` | audio | Shared excitation chirp played through the loudspeakers during recording |
| `experiment_id` | string | Source directory name from experiment-15 (unique per recording) |
| `speakers` | string | 4-bit speaker activation code — e.g. `0001` means only speaker 4 was active |
| `x_position` | int | Object grid column index (0-indexed) |
| `y_position` | int | Object grid row index (0-indexed) |
| `x_com` | float | X centre-of-mass of the segmentation mask in the cropped overhead image (pixels) |
| `y_com` | float | Y centre-of-mass of the segmentation mask in the cropped overhead image (pixels) |
| `object` | string | Object type inside the box, e.g. `cube` |
| `n_objects` | int | Number of objects inside the box |
| `box_material` | string | Box material, e.g. `cardboard` |
| `mask_file_name` | image | Binary segmentation mask of the object in the cropped overhead image |
| `experiment_dir` | string | Name of the experiment-16 target directory |
| `manifest` | string | Full JSON manifest for this sample (see [Section 3](#3-manifestjson-structure)) |

---

# File Directory Structure

```
experiment-16/
├── README.md                            # This file (dataset card)
├── audio/
│   └── chirp_50_1000_3.0sec.wav         # Shared excitation chirp (50–1000 Hz, 3 s)
├── data/
│   ├── metadata.jsonl                   # One JSON row per sample (viewer-facing)
│   ├── 0000001/                         # Per-sample directory (zero-padded 7-digit ID)
│   │   ├── manifest.json                # Full provenance + config for this sample
│   │   ├── overhead.png                 # Final overhead image with padding + speaker overlay for this sample
│   │   ├── speckle_vibration_raw.npy    # Raw laser camera frames  [100 lasers × T frames × 2 (XY)]
│   │   ├── speckle_shifts.npz           # Sub-pixel XY shifts per laser per frame
│   │   ├── speckle_shifts_clean.npz     # Bandpass-filtered + Hann-windowed shifts
│   │   ├── speckle_shifts_fft.npz       # FFT of cleaned shifts (frequency domain)
│   │   ├── speckle_shifts_ifft_audio.wav# Single-laser vibration reconstructed as audio
│   │   └── speckle_vibrations.mp4       # Slow-motion preview video of speckle motion
│   └── 0000002/
│       └── ...
└── image/
    └── <image_dir>/                     # Named: <object>-<x>x-<y>y-<n>obj-<material>-<date>
        ├── raw_overhead.png             # Full overhead photo before cropping
        ├── cropped_overhead.png         # Overhead cropped to the box region
        ├── segmented_overhead.png       # Shared overhead with mask overlay + COM marker only (no speaker overlay)
        ├── mask.png                     # Binary segmentation mask (white = object)
        └── mask.npz                     # Binary mask as compressed numpy array
```

---

# manifest.json Structure

Every sample directory contains a `manifest.json` that records full provenance, hardware config,
processing parameters, and artifact paths. The `manifest` column in `metadata.jsonl` is this
same document serialised as a JSON string. Shared image assets live under `image/<image_dir>/`,
while the sample-specific speaker-rendered overhead lives under `data/<sample_id>/overhead.png`.

## Top-Level Keys

The most important, top-level entries of `manifest.json`.

| Key | Type | Description |
|-----|------|-------------|
| `sample_id` | int | Unique sample identifier |
| `experiment_id` | string | Source recording directory name (from experiment-15) |
| `experiment_dir` | string | Target directory name (experiment-16) |
| `source_experiment_id` | string | Canonical source reference (same as `experiment_id`) |
| `source_experiment_dir` | string | Absolute NAS path to the source directory |
| `hf_repo` | string | HuggingFace repo this sample was uploaded to |
| `sample` | object | Physical setup at the time of recording |
| `segmentation` | object | Overhead-image segmentation result |
| `experiment_config` | object | Merged hardware and recording configuration |
| `experiment_output` | object | Derived statistics computed during processing |
| `processing_config` | object | Processing pipeline parameters |
| `artifacts` | object | Relative repo paths to all files produced for this sample, including both shared and sample-specific overhead images |

## `sample`

The basic attributes of a single sample of our dataset.

| Key | Type | Description |
|-----|------|-------------|
| `object` | string | Object type inside the box, e.g. `cube` or `empty` |
| `n_objects` | int | Number of objects |
| `box_material` | string | Box material, e.g. `cardboard` |
| `speakers` | string | 4-bit activation code for active speakers, e.g. `0001` |
| `x_position` | int | Object grid column (0-indexed) |
| `y_position` | int | Object grid row (0-indexed) |
| `image_dir` | string | Image subdirectory name under `data/image/` |

## `segmentation`

Values from the segmentation mask.

| Key | Type | Description |
|-----|------|-------------|
| `x_com` | float | X centre-of-mass of the mask in the cropped overhead image (pixels) |
| `y_com` | float | Y centre-of-mass of the mask in the cropped overhead image (pixels) |
| `status` | string | `completed` when segmentation succeeded |

## `experiment_config`

Processing parameters derived (and possibly overidden) from `base_experiment_config.json`

| Key | Type | Description |
|-----|------|-------------|
| `audio.file_name` | string | Path to the excitation audio file within the repo |
| `audio.sample_rate_hz` | int | Audio sample rate (Hz) |
| `audio.duration_s` | float | Total audio duration including silence padding (s) |
| `audio.total_output_channels` | int | Number of output channels on the audio interface |
| `audio.wav_channels` | int | Number of channels in the WAV file |
| `audio.sample_width_bytes` | int | Sample width in bytes |
| `audio.generation.signal` | string | Signal type, e.g. `chirp` |
| `audio.generation.method` | string | Chirp sweep method, e.g. `logarithmic` |
| `audio.generation.chirp_duration_s` | float | Duration of the chirp sweep (s) |
| `audio.generation.silence_start_s` | float | Silence padding before the chirp (s) |
| `audio.generation.silence_end_s` | float | Silence padding after the chirp (s) |
| `audio.generation.f_start_hz` | int | Chirp start frequency (Hz) |
| `audio.generation.f_end_hz` | int | Chirp end frequency (Hz) |
| `audio.generation.output_dtype` | string | Output sample dtype, e.g. `int16` |
| `audio.generation.normalization` | string | Normalization method, e.g. `peak_to_int16` |
| `recording.capture_seconds_requested` | float | Requested laser camera recording duration (s) |
| `overhead_camera.frame_rate_fps` | int | Overhead camera frame rate (fps) |
| `overhead_camera.exposure_ms` | int | Overhead camera exposure time (ms) |
| `overhead_camera.pixel_clock_mhz` | int | Overhead camera pixel clock (MHz) |
| `overhead_camera.gain` | int | Overhead camera sensor gain |
| `overhead_camera.runtime.device_id` | int | Camera device index |
| `overhead_camera.runtime.color_mode` | string | Raw sensor color mode |
| `overhead_camera.runtime.buffer_count` | int | Frame buffer count |
| `overhead_camera.runtime.rotation_degrees` | int | Image rotation applied at capture (degrees) |
| `overhead_camera.runtime.debayer_code` | string | OpenCV debayer conversion code |
| `laser_camera.runtime.info_field` | bool | Whether the camera info field overlay is enabled |
| `laser_camera.runtime.cxp_link_configuration` | string | CoaXPress link configuration |
| `laser_camera.calibration.fps` | int | Frame rate used during ROI calibration (fps) |
| `laser_camera.calibration.exposure_us` | int | Exposure time used during calibration (µs) |
| `laser_camera.calibration.gain` | int | Gain used during calibration |
| `laser_camera.capture.fps` | int | Frame rate used during recording (fps) |
| `laser_camera.capture.exposure_us` | int | Exposure time used during recording (µs) |
| `laser_camera.capture.gain` | int | Gain used during recording |
| `laser_camera.capture.buffer_part_count` | int | Number of frame buffer partitions |
| `laser_grid.n_roi_rows` | int | Number of laser rows in the grid |
| `laser_grid.n_roi_columns` | int | Number of laser columns in the grid |
| `laser_grid.roi_row_height` | int | Height of each laser ROI (pixels) |
| `laser_grid.roi_column_width` | int | Width of each laser ROI (pixels) |
| `preview.overhead_resize_factor` | float | Resize factor for the overhead preview display |
| `preview.overhead_gamma` | float | Gamma correction for the overhead preview |
| `preview.laser_preview_gamma` | float | Gamma correction for the laser preview |
| `preview.show_full_frame` | int | Whether to display the full camera frame (0/1) |
| `preview.preview_level` | int | Preview detail level |
| `preview.reset_rois` | bool | Whether to reset ROIs on startup |

## `experiment_output`

The output of the data collection phase.

| Key | Type | Description |
|-----|------|-------------|
| `overhead_camera.image_width` | int | Overhead image width (px) |
| `overhead_camera.image_height` | int | Overhead image height (px) |
| `laser_camera.global_roi` | list[int] | Full-frame ROI used `[x, y, w, h]` |
| `laser_camera.max_frame_rate_hz` | int | Maximum frame rate achieved by the laser camera |
| `laser_grid.total_image_height` | int | Total laser camera frame height (px) |
| `laser_grid.selected_row_points_image_xy` | list[list[int]] | Per-row representative laser point coordinates `[[x, y], ...]` in the camera frame |
| `laser_grid.selected_column_centers_x` | list[int] | X-centre pixel of each laser column in the cropped frame |
| `laser_grid.row_values_single_list` | list[int] | Row pixel indices belonging to all laser ROIs (flattened) |
| `laser_grid.global_crop_x` | int | X offset of the global crop in the full camera frame (px) |
| `laser_grid.global_crop_width` | int | Width of the global crop (px) |
| `laser_grid.global_crop_height` | int | Height of the global crop (px) |
| `laser_grid.row_rois_y` | list[list[int]] | Y extents of each laser row `[[y_start, y_end], ...]` in the full frame |
| `laser_grid.sensor_grid_shape` | list[int] | Laser grid dimensions `[rows, cols]` |
| `laser_grid.sensor_rois_xywh` | list[list[int]] | Per-laser bounding boxes `[[x, y, w, h], ...]` within the cropped frame |
| `speckle_vibrations.frame_count` | int | Total number of frames captured |
| `speckle_vibrations.frame_height` | int | Raw frame height (px) |
| `speckle_vibrations.frame_width` | int | Raw frame width (px) |
| `speckle_vibrations.preview_frame_count` | int | Number of frames in the preview video |
| `speckle_vibrations.preview_fps` | float | Playback frame rate of the preview video (fps) |
| `speckle_vibrations.preview_width` | int | Preview video width (px) |
| `speckle_vibrations.preview_height` | int | Preview video height (px) |
| `speckle_vibrations.capture_fps_hz` | float | Actual laser camera capture rate (Hz) |
| `speckle_vibrations.capture_seconds_requested` | float | Requested recording duration (s) |
| `speckle_vibrations.capture_seconds_observed` | float | Observed recording duration (s) |
| `speckle_vibrations.dtype` | string | Raw frame pixel dtype |
| `speckle_shifts.fs` | float | Shift signal sampling rate (Hz) |
| `speckle_shifts.shape` | list[int] | Shape of the shifts array `[n_lasers, n_frames, 2]` |
| `speckle_shifts_clean.shape` | list[int] | Shape of the filtered shifts array `[n_lasers, n_frames, 2]` |
| `speckle_shifts_clean.fs` | float | Sampling rate after filtering (Hz) |
| `speckle_shifts_fft.shape` | list[int] | Shape of the FFT array `[n_lasers, n_freqs, 2]` |
| `speckle_shifts_fft.fs` | float | Sampling rate used for the FFT (Hz) |
| `speckle_shifts_fft.n_samples` | int | Number of time-domain samples used for the FFT |
| `speckle_shifts_ifft_audio.sample_rate_hz` | int | Sample rate of the reconstructed audio WAV (Hz) |
| `speckle_shifts_ifft_audio.n_frames` | int | Number of audio samples in the output WAV |

## `processing_config`

Processing parameters derived (and possibly overidden) from `base_processing_config.json`.

| Key | Type | Description |
|-----|------|-------------|
| `speckle_vibration_raw.format` | string | Storage format: `npy` or `npz` |
| `speckle_vibration_raw.compressed` | bool | Whether the raw array is compressed |
| `speckle_vibrations_preview.max_frames` | int | Maximum number of frames in the preview video |
| `speckle_vibrations_preview.max_width` | int | Maximum preview video width (px) |
| `speckle_vibrations_preview.macro_block_size` | int | Encoder macro-block alignment size |
| `speckle_vibrations_preview.source_capture_fps_hz` | float | Source capture rate used for preview timing (Hz) |
| `speckle_vibrations_preview.preserve_physical_duration` | bool | Whether playback speed matches real time |
| `speckle_vibrations_preview.percentile_low` | float | Lower percentile for frame contrast normalisation |
| `speckle_vibrations_preview.percentile_high` | float | Upper percentile for frame contrast normalisation |
| `speckle_vibrations_preview.codec` | string | Video codec, e.g. `libx264` |
| `speckle_vibrations_preview.pixelformat` | string | Pixel format, e.g. `yuv420p` |
| `speckle_vibrations_preview.crf` | int | Constant rate factor — lower = higher quality |
| `speckle_vibrations_preview.burn_frame_index` | bool | Whether the frame index is burned into the video |
| `speckle_shifts.fs_hz` | float | Sampling rate of the shift signal (Hz) |
| `speckle_shifts_clean.filter_type` | string | Filter design, e.g. `butterworth` |
| `speckle_shifts_clean.filter_mode` | string | `bandpass`, `lowpass`, or `highpass` |
| `speckle_shifts_clean.lowcut` | float | Low cutoff frequency (Hz) |
| `speckle_shifts_clean.highcut` | float | High cutoff frequency (Hz) |
| `speckle_shifts_clean.filter_order` | int | Filter order |
| `speckle_shifts_clean.hann_applied` | bool | Whether a Hann window was applied after filtering |
| `speckle_shifts_clean.apply_order` | string | `filter_then_hann` or `hann_then_filter` |
| `speckle_shifts_fft.fft_kind` | string | FFT variant, e.g. `rfft` |
| `speckle_shifts_fft.fft_axis` | int | Axis along which the FFT is computed |
| `speckle_shifts_fft.min_freq` | float | Minimum frequency retained after crop (Hz) |
| `speckle_shifts_fft.max_freq` | float | Maximum frequency retained after crop (Hz) |
| `speckle_shifts_fft.dtype` | string | Complex dtype of the output, e.g. `complex64` |
| `speckle_shifts_fft.crop_after_fft` | bool | Whether to crop to `[min_freq, max_freq]` after FFT |
| `speckle_shifts_ifft_audio.laser_idx` | int | Which laser (0-indexed) to use for the audio preview |
| `speckle_shifts_ifft_audio.xy_idx` | int | Which shift channel: `0` = X, `1` = Y |
| `speckle_shifts_ifft_audio.method` | string | Reconstruction method, e.g. `ifft` |
| `speckle_shifts_ifft_audio.output_sample_rate_hz` | int | Output WAV sample rate (Hz) |
| `speckle_shifts_ifft_audio.normalization` | string | Normalization method, e.g. `peak_to_int16` |
| `speckle_shifts_ifft_audio.output_dtype` | string | Output sample dtype, e.g. `int16` |
| `speckle_shifts_ifft_audio.zero_fill_uncropped_bins` | bool | Whether to zero-fill bins outside the crop range |
| `segmentation.left` | float | Left crop fraction of the overhead image |
| `segmentation.right` | float | Right crop fraction of the overhead image |
| `segmentation.up` | float | Top crop fraction of the overhead image |
| `segmentation.down` | float | Bottom crop fraction of the overhead image |
| `segmentation.prompt` | string | Text prompt passed to the segmentation model |

## `artifacts`

The final artifacts we end up with in our dataset.

| Key | Type | Description |
|-----|------|-------------|
| `raw_overhead` | string | `data/image/<dir>/raw_overhead.png` — full overhead photo before cropping |
| `cropped_overhead` | string | `data/image/<dir>/cropped_overhead.png` — overhead cropped to box region |
| `segmented_overhead` | string | `data/image/<dir>/segmented_overhead.png` — overhead with mask + speaker overlay |
| `mask_png` | string | `data/image/<dir>/mask.png` — binary segmentation mask (PNG) |
| `mask_npz` | string | `data/image/<dir>/mask.npz` — binary mask as compressed numpy array |
| `audio` | string | `data/audio/chirp_50_1000_3.0sec.wav` — shared excitation chirp (all samples) |
| `speckle_vibration_raw` | string | `data/<id>/speckle_vibration_raw.npy` — raw laser camera frames |
| `speckle_vibrations` | string | `data/<id>/speckle_vibrations.mp4` — slow-motion speckle preview video |
| `speckle_shifts` | string | `data/<id>/speckle_shifts.npz` — sub-pixel XY shifts per laser per frame |
| `speckle_shifts_clean` | string | `data/<id>/speckle_shifts_clean.npz` — filtered + windowed shifts |
| `speckle_shifts_fft` | string | `data/<id>/speckle_shifts_fft.npz` — FFT of cleaned shifts |
| `speckle_shifts_ifft_audio` | string | `data/<id>/speckle_shifts_ifft_audio.wav` — single-laser audio reconstruction |
| `manifest` | string | `data/<id>/manifest.json` — this manifest file |
