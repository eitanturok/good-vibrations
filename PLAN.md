# Laser Vibrations Plan

## Goal

Create a new Hugging Face dataset called `laser-vibrations` that preserves the full processing pipeline, exposes the right assets in the Hugging Face dataset viewer, remains efficient for model training, and gives us a richer custom viewer (`viz2`) without breaking the currently used `viz`.

This plan is the source of truth for the next implementation steps.

## Current Status

- [x] `PLAN.md` created and kept as the live spec
- [x] `viz2/` created as a copy of `viz/`
- [x] `viz2` pointed at `eturok-weizmann/laser-vibrations`
- [x] `laser-vibrations` created as a fresh HF dataset
- [x] initial dataset scaffold upload works
- [x] first real sample (`sample_id=2`) uploaded in the new schema
- [ ] `viz2` adapted to read the new dataset schema

### Working Notes

- direct writes to `laser-vibrations` require PR-style commits, so upload scripts should default to `create_pr=True`
- the first local backfill attempt was slow because it copied the large raw `frame-recording.npy` from `mcluster11` back to the local machine over SSH
- transfer benchmarking showed that local SSH streaming is a real bottleneck:
  - plain SSH byte streaming moved 512 MB in about 87s, roughly 5.9 MB/s
  - chunked `dd` streaming was slower in this environment
  - `rsync` started reasonably but failed mid-transfer and was not reliable enough to trust yet
- the practical split is now:
  - remote on `mcluster11`: upload `speckle_vibration_raw.npz` and generate/upload `speckle_vibrations.mp4`
  - local: upload mask assets, shifts, cleaned shifts, FFT, IFFT audio, manifest, and the `metadata.jsonl` row
- this avoids pulling the 2.9 GB raw recording to the local machine while still keeping the richer Python/dataframe logic local
- on CentOS, the remote stage should use cluster modules plus a small venv instead of the full project environment:
  - `module load GCC/12.2.0`
  - `module load Python/3.10.8-GCCcore-12.2.0`
  - `python -m venv ~/venvs/laser-vibrations`
  - `pip install --only-binary=:all: numpy==1.26.4 opencv-python-headless==4.10.0.84 huggingface_hub==0.31.2`

What did not work:
- the first remote attempt used `uv` with a transient environment
- that pulled in packages that tried to build from source on the cluster
- builds failed because the default system toolchain was too old and lacked a working `g++`
- the next remote attempt used the cluster's newer Python modules
- those Python module builds crashed on this node with `Illegal instruction`, so they are not usable here
- the next remote attempt used `micromamba`, but environment solving failed in this cluster setup

New option we are taking instead:
- install `uv`
- use `uv python install 3.10` to fetch a standalone compatible interpreter
- create a dedicated remote venv from that Python
- install wheel-only packages into that env
- use that env only for the remote raw/mp4 uploader
- source experiment discovery should use the old dataset's continuous `x_position` and `y_position` as center-of-mass-style signals, not as user-entered placement coordinates
- backfill should infer the new canonical user `x_position` and `y_position` from the experiment directory naming / discrete filename grid, e.g. `cube-00x01y_0001--...`
- overhead-image matching should remain only as a fallback if position bucketing is ambiguous or not available
- `source_experiment_id` should be saved in the manifest in addition to `source_experiment_dir`
- manual `source_experiment_dir` override should still be supported as a fallback

Confirmed working path and timing baseline for `sample_id=2`:
- remote uploader script is synced to `mcluster11` and run there with a standalone `uv`-managed Python 3.10 environment
- raw `.npy` never comes back to the local machine
- remote timings:
  - preview generation: about `1.33s`
  - raw `.npy` + `.mp4` upload to HF: about `8.03s`
  - total remote stage: about `19.91s`
- local timings:
  - old dataset row load: about `2.59s`
  - remote config fetch + raw header inspection: about `1.41s`
  - tensor/manifest/metadata generation: all sub-second individually
  - upload remaining local assets to HF: about `2.91s`
  - total local stage: about `11.38s`

## Why We Are Doing This

The current dataset structure is too narrow for the workflow we now want.

Current limitations:
- the HF dataset is optimized mostly for final training tensors and overhead images
- intermediate assets like the audio file and speckle vibration video are not first-class dataset assets
- the current viewer can only show what was already packed into the old payload format
- the raw experiment pipeline spans multiple machines and stages, but that structure is not represented explicitly in the dataset
- some preprocessing still happens inside the dataloader, which makes inspection and reproducibility harder

Desired properties of the new design:
- HF dataset viewer should directly play the chirp audio used for a sample
- HF dataset viewer should directly play a compressed speckle vibration preview video
- `viz2` should show the same uploaded assets, without triggering any processing on click
- model training should stay fast and selective, without downloading media assets it does not need
- the pipeline should preserve intermediate stages so we can debug, inspect, and later migrate training to precomputed FFT inputs
- duplicate shared assets, especially audio, should not be re-uploaded per sample if they are identical

## High-Level Decisions

### 1. Use a New HF Dataset

We will create a new Hugging Face dataset named `laser-vibrations`.

Reasoning:
- the new structure is materially different from the current one
- the new schema includes multiple new media columns and per-sample asset directories
- we want to iterate safely without breaking the current dataset and current training flow
- the old dataset can remain stable while the new dataset becomes the canonical pipeline dataset over time

This means:
- old dataset remains available for reference and fallback
- new dataset is where we implement the richer pipeline and viewer integration

Implementation note:
- we will not bulk-copy the existing `vibrations` dataset into `laser-vibrations`
- instead, `laser-vibrations` starts as a fresh dataset with the new schema
- we will upload one dummy sample first, validate the design, and only then backfill historical samples selectively

Reasoning:
- full client-side copy is slow and wasteful for a large dataset
- the new dataset layout is intentionally different, so a repo clone would copy the wrong structure
- starting fresh keeps iteration fast and reduces migration risk

### 2. Separate Viewer Assets From Training Tensors

We will not store everything inside one tabular artifact.

Instead:
- `data/metadata.jsonl` will store one row per sample with HF-viewable file references
- shared audio files will live under `data/audio/`
- shared overhead image groups will live under `data/image/`
- per-sample assets will live under `data/<sample_id>/`

Reasoning:
- HF viewer works well with referenced audio/video assets
- storing large tensors and raw recordings inside a tabular metadata file is not ideal for append-only uploads or selective model loading
- training only needs a small subset of the sample assets
- we want the media visible in HF, but we do not want model training to download those media files

### 3. Raw Speckle Recording Is Preserved But Not Displayed In HF Viewer

We will keep the full raw recording as a NumPy file, but only the compressed MP4 will be used for viewer display.

Reasoning:
- the raw recording is essential for scientific reproducibility and reprocessing
- the raw recording is too large and not appropriate for direct dataset-viewer playback
- the MP4 is a display artifact, not a scientific source artifact

### 4. Preserve Intermediate Pipeline Stages As First-Class Assets

We want the dataset to represent the processing pipeline explicitly.

That pipeline is:
1. `speckle_vibration_raw.npz`
2. `speckle_vibrations.mp4`
3. `speckle_shifts.npz`
4. `speckle_shifts_clean.npz`
5. `speckle_shifts_fft.npz`
6. `speckle_shifts_ifft_audio.wav`
7. `mask.png`
8. `mask.npz`

Reasoning:
- intermediate stages are valuable for debugging and validation
- future training will likely consume `speckle_shifts_fft.npz` directly
- until then, we still want raw and cleaned stages saved
- `viz2` should eventually be able to visualize multiple stages of the pipeline

### 5. FFT Audio Preview Is A Derived Inspection Asset

We want an audio sanity check derived from `speckle_shifts_fft.npz`.

The preview file will be:
- `speckle_shifts_ifft_audio.wav`

Generation method:
- use inverse FFT on one chosen laser and one chosen axis

Defaults:
- `laser_idx = 50`
- `xy_idx = 0` meaning the x channel

Configurable:
- the actual parameters used will be recorded in the sample manifest

Reasoning:
- we want a simple way to ask “does this sound right?”
- using IFFT is the simplest and most faithful first implementation
- this audio preview is for inspection only, not model input

## Final Target Repository Layout

```text
README.md
data/
  metadata.jsonl
  audio/
    chirp_50_1000_3.0sec.wav
  image/
    <object>-<x-position>-<y-position>-<n-objects>-<box-material>-<timestamp>/
      raw_overhead.png
      cropped_overhead.png
      segmented_overhead.png
      mask.png
      mask.npz
  0000001/
    speckle_vibration_raw.npz
    speckle_vibrations.mp4
    speckle_shifts.npz
    speckle_shifts_clean.npz
    speckle_shifts_fft.npz
    speckle_shifts_ifft_audio.wav
    manifest.json
  0000002/
    ...
```

Naming rules:
- sample directories use zero-padded integer ids such as `0000001`
- shared image directories use `<object>-<x-position>-<y-position>-<n-objects>-<box-material>-<timestamp>`
- `object`, `n-objects`, and `box-material` should be lowercased/normalized before path generation
- `x-position` and `y-position` should be zero-padded to width 3
- `timestamp` should be ordered `YYYY-MM-DD-HH-MM-SS` so lexical sort matches time order

Coordinate rules:
- `x_position` and `y_position` are user-provided integer placement coordinates recorded at acquisition time from the physical grid drawn in the box
- image directory names must continue to use `x_position` and `y_position`
- `x_com` and `y_com` are pixel-space values derived from the segmentation mask and must be stored separately in `manifest.json`
- `x_com` and `y_com` may be floats because the mask can be discretized, binned, or averaged before center-of-mass computation
- never treat `x_com` / `y_com` as interchangeable with `x_position` / `y_position`
- historical datasets may have older `x_position` / `y_position` fields that really represent center-of-mass-style values; backfill should not copy those directly into the new canonical placement fields

**metadata.jsonl structure:**

```json
{
  "sample_id": 1,
  "segmented_overhead_file_name": "https://huggingface.co/datasets/.../resolve/main/data/image/cube-003-007-1-cardboard-2026-04-21-10-05-33/segmented_overhead.png",
  "speckle_vibrations_file_name": "https://huggingface.co/datasets/.../resolve/main/data/0000001/speckle_vibrations.mp4",
  "speckle_shifts_ifft_audio_file_name": "https://huggingface.co/datasets/.../resolve/main/data/0000001/speckle_shifts_ifft_audio.wav",
  "audio_file_name": "https://huggingface.co/datasets/.../resolve/main/data/audio/chirp_50_1000_3.0sec.wav",
  "experiment_id": "cube-00x01y_0001--31-03-18-21-24",
  "speakers": "0100",
  "x_position": 3,
  "y_position": 7,
  "x_com": 3.1,
  "y_com": 2.7,
  "n_objects": 1,
  "box_material": "cardboard",
  "mask_file_name": "https://huggingface.co/datasets/.../resolve/main/data/image/cube-003-007-1-cardboard-2026-04-21-10-05-33/mask.png",
  "experiment_dir": "experiment_15",
  "manifest": "{\"sample_id\":1,...}"
}
```

**Sharing strategy:**
- Audio files (`data/audio/`) shared across ALL samples → stored once
- Overhead images and masks (`data/image/`) shared across the same physical setup → stored once and referenced by many samples
- Per-sample files (raw speckle capture, shifts, FFT, preview video, IFFT audio, manifest) stored in sample-specific directories

**Key principles:**
1. Each sample directory is named by `sample_id` (e.g., `0000001/`)
2. FFT data (`speckle_shifts_fft.npz`) is stored per-sample as model input
3. Shared assets use HF URLs in metadata, enabling viewer rendering without duplication
4. JSONL approach enables append-only updates (add new sample directory + append row)

## Sample Asset Semantics

### `speckle_vibration_raw.npz`

Meaning:
- raw camera recording of the vibrating laser speckles

Role:
- scientific source artifact
- input to recovery / PCKL
- not directly displayed in HF viewer

### `speckle_vibrations.mp4`

Meaning:
- compressed visual preview derived from the raw recording

Role:
- viewer/display artifact only
- shown in HF dataset viewer
- shown in `viz2`

### `speckle_shifts.npz`

Meaning:
- recovered sub-pixel speckle shifts from PCKL or equivalent recovery stage

Expected contents:
- `shifts`
- `fs`

### `speckle_shifts_clean.npz`

Meaning:
- shifts after the current cleaning steps that we now do in the dataloader

Expected processing:
- Butterworth filter
- Hann window

Expected contents:
- `shifts_clean`
- `fs`
- `lowcut`
- `highcut`
- `filter_order`
- `hann_applied`

### `speckle_shifts_fft.npz`

Meaning:
- cleaned shifts converted to FFT and cropped to the frequency range we care about

Storage choice:
- store complex FFT values directly

Reasoning:
- this is likely the fastest path for the future dataloader
- if we later discover real/imag split is faster in practice, we can benchmark and change it intentionally

Expected contents:
- `fft`
- `freqs`
- `fs`
- `min_freq`
- `max_freq`
- `n_samples`

Reasoning:
- the FFT audio preview uses IFFT reconstruction from the stored cropped FFT
- to reconstruct the signal shape reliably, we need the original sample count

### `speckle_shifts_ifft_audio.wav`

Meaning:
- audio preview generated from one selected recovered-shift trace using IFFT

Role:
- sanity check / inspection asset
- playable in HF viewer and `viz2`

### `mask.png`

Meaning:
- rendered image form of the segmentation mask for direct HF viewer display

Role:
- viewer/display artifact only
- shown alongside `segmented_overhead_file_name`

### `mask.npz`

Meaning:
- segmentation target plus the exact crop/prompt configuration used to create it

Expected contents:
- `mask`
- `left`
- `right`
- `up`
- `down`
- `prompt`

Reasoning:
- we want the target mask to remain minimal but reproducible
- crop parameters and prompt are required provenance for how the mask was created
- the mask is also the source for derived geometry such as `x_com` and `y_com`, which should be recorded in the manifest

## Manifest Schema

Each sample directory will contain `manifest.json` describing the canonical paths and processing choices for that sample.

Design principle:
- keep the manifest as simple as possible while still fully reproducible
- prefer one field with a clear meaning over multiple "requested" / "actual" variants unless the distinction is truly necessary
- organize the manifest around real pipeline stages and artifacts, not around implementation details
- build `manifest.json` from one shared helper used by both:
  - future runtime capture / processing
  - historical backfill
- this is required so manifest contents stay consistent across all samples

Proposed structure:

```json
{
  "sample_id": 1,
  "experiment_id": "cube-00x01y_0001--31-03-18-21-24",
  "experiment_dir": "experiment_15",
  "hf_repo": "eturok-weizmann/laser-vibrations",
  "sample": {
    "object": "cube",
    "n_objects": 1,
    "box_material": "cardboard",
    "speakers": "0100",
    "x_position": 3,
    "y_position": 7,
    "image_dir": "cube-003-007-1-cardboard-2026-04-21-10-05-33"
  },
  "segmentation": {
    "x_com": 18.4,
    "y_com": 9.7
  },
  "experiment_config": {
    "audio": {
      "file_name": "data/audio/chirp_50_1000_3.0sec.wav",
      "sample_rate_hz": 44100,
      "duration_s": 3.0,
      "total_output_channels": 8
    },
    "recording": {
      "capture_seconds": 3.1
    },
    "overhead_camera": {
      "frame_rate_fps": 30,
      "exposure_ms": 30,
      "pixel_clock_mhz": 86,
      "gain": 60,
      "runtime": {
        "device_id": 0,
        "color_mode": "IS_CM_SENSOR_RAW8",
        "buffer_count": 40,
        "rotation_degrees": 180,
        "debayer_code": "COLOR_BAYER_BG2BGR"
      }
    },
    "laser_camera": {
      "runtime": {
        "info_field": false,
        "cxp_link_configuration": "CXP12_X4"
      },
      "calibration": {
        "fps": 500,
        "exposure_us": 30,
        "gain": 3
      },
      "capture": {
        "fps": 2500,
        "exposure_us": 150,
        "gain": 1,
        "buffer_part_count": 3000
      }
    },
    "laser_grid": {
      "n_roi_rows": 10,
      "n_roi_columns": 10,
      "roi_row_height": 30,
      "roi_column_width": 70
    },
    "preview": {
      "overhead_resize_factor": 0.75,
      "overhead_gamma": 1.0,
      "laser_preview_gamma": 2.5,
      "show_full_frame": 0,
      "preview_level": 1,
      "reset_rois": true
    }
  },
  "experiment_output": {
    "overhead_camera": {
      "image_width": 2056,
      "image_height": 1542
    },
    "laser_camera": {
      "global_roi": [352, 0, 1152, 300],
      "max_frame_rate_hz": 7970
    },
    "laser_grid": {
      "total_image_height": 300,
      "selected_row_points_image_xy": [[978, 65], [975, 171]],
      "selected_column_centers_x": [387, 457],
      "row_values_single_list": [25, 26, 27],
      "global_crop_x": 352,
      "global_crop_width": 1152,
      "global_crop_height": 300,
      "row_rois_y": [[50, 80], [156, 186]],
      "sensor_grid_shape": [10, 10],
      "sensor_rois_xywh": [[352, 0, 70, 30]]
    },
    "speckle_vibrations": {
      "frame_count": 9000,
      "frame_height": 300,
      "frame_width": 1152,
      "capture_seconds": 3.1,
      "preview_fps": 30.0,
      "dtype": "uint8"
    }
  },
  "processing_config": {
    "speckle_vibrations_preview": {
      "max_frames": 300,
      "max_width": 960,
      "percentile_low": 5,
      "percentile_high": 99.5,
      "codec": "libx264",
      "pixelformat": "yuv420p",
      "crf": 23,
      "burn_frame_index": true
    },
    "speckle_shifts": {
      "fs_hz": 2500
    },
    "speckle_shifts_clean": {
      "filter_type": "butterworth",
      "filter_mode": "bandpass",
      "lowcut": 50,
      "highcut": 1000,
      "filter_order": 5,
      "hann_applied": true,
      "apply_order": "filter_then_hann"
    },
    "speckle_shifts_fft": {
      "fft_kind": "rfft",
      "fft_axis": 1,
      "min_freq": 50,
      "max_freq": 1000,
      "dtype": "complex64",
      "crop_after_fft": true
    },
    "speckle_shifts_ifft_audio": {
      "laser_idx": 50,
      "xy_idx": 0,
      "method": "ifft",
      "output_sample_rate_hz": 22050,
      "normalization": "peak_to_int16",
      "output_dtype": "int16",
      "zero_fill_uncropped_bins": true
    }
  },
  "artifacts": {
    "shared": {
      "raw_overhead": "data/image/cube-003-007-1-cardboard-2026-04-21-10-05-33/raw_overhead.png",
      "cropped_overhead": "data/image/cube-003-007-1-cardboard-2026-04-21-10-05-33/cropped_overhead.png",
      "segmented_overhead": "data/image/cube-003-007-1-cardboard-2026-04-21-10-05-33/segmented_overhead.png",
      "mask_png": "data/image/cube-003-007-1-cardboard-2026-04-21-10-05-33/mask.png",
      "mask_npz": "data/image/cube-003-007-1-cardboard-2026-04-21-10-05-33/mask.npz",
      "audio": "data/audio/chirp_50_1000_3.0sec.wav"
    },
    "sample": {
      "speckle_vibration_raw": "data/0000001/speckle_vibration_raw.npz",
      "speckle_vibrations": "data/0000001/speckle_vibrations.mp4",
      "speckle_shifts": "data/0000001/speckle_shifts.npz",
      "speckle_shifts_clean": "data/0000001/speckle_shifts_clean.npz",
      "speckle_shifts_fft": "data/0000001/speckle_shifts_fft.npz",
      "speckle_shifts_ifft_audio": "data/0000001/speckle_shifts_ifft_audio.wav",
      "manifest": "data/0000001/manifest.json"
    }
  }
}
```

Notes on simplification:
- do not store a separate `source_notebook`
- in `audio`, keep only `file_name`; do not duplicate it with a recorded source path
- speaker routing is already represented by `sample.speakers`, so do not duplicate it inside `audio`
- for camera settings, log the values that were requested by the experiment code
- put preview-only values in a dedicated `preview` block so they do not get mixed with acquisition settings
- keep concrete observed outputs in a dedicated `experiment_output` block
- `experiment_output` should contain outputs of setup or capture rather than direct requested inputs, for example:
  - final global ROI
  - final per-row and per-sensor ROI geometry
  - frame count and saved array shape
  - preview fps actually used for the saved MP4
- `experiment_config` and `processing_config` together should be sufficient to rerun the acquisition and derivation logic without hidden notebook constants
- `experiment_output` is still required because some critical values are produced by setup/capture rather than chosen up front, especially ROI geometry and saved frame statistics
- do not duplicate near-identical requested/actual camera settings in the manifest unless we later discover a hardware mismatch that materially affects reproducibility
- keep `calibration` nested under `laser_camera` because it is a distinct stage that affects how the acquisition was configured
- `calibration` means the low-frame-rate laser-camera setup used before the final high-speed capture in order to find/set ROIs and verify alignment; it is separate from the final vibration capture settings
- keep normal high-speed laser acquisition settings under `laser_camera.capture`
- `sensor_rois_xywh` means the full list of final per-sensor `(x, y, w, h)` ROIs after combining row and column choices
- group file paths under an explicit `artifacts` block
- keep processing parameters together under a `processing_config` block

Important saved-data constraint:
- `metadata.npz` preserves selected row points and final ROIs, but does not preserve the original column click `(x, y)` points directly
- therefore the manifest should store `selected_column_centers_x`, which can be reconstructed reliably from the saved final ROIs

Additional ROI guidance:
- ROI information is important enough to log in detail because it defines how the 10x10 laser sensor grid was constructed from the raw camera image
- the manifest should keep both the human-selected inputs and the derived final geometry:
  - selected row points
  - selected column points
  - the row value LUT / selected row indices used by the camera
  - the global horizontal crop
  - the per-row Y ranges
  - the final per-sensor `(x, y, w, h)` ROIs

Fields still required for full replay of `notebooks/11_multispeaker_record_data.ipynb` and backfill processing:
- `experiment_config.recording.capture_seconds`
- `experiment_config.overhead_camera.runtime`
- `experiment_config.laser_camera.runtime`
- `processing_config.speckle_vibrations_preview`
- `processing_config.speckle_shifts_ifft_audio.output_sample_rate_hz`
- `processing_config.speckle_shifts_ifft_audio.normalization`
- `processing_config.speckle_shifts_ifft_audio.output_dtype`
- `processing_config.speckle_shifts_ifft_audio.zero_fill_uncropped_bins`

### Why Keep A Manifest If We Already Have `metadata.jsonl`?

Reasoning:
- it gives each sample a local, self-contained description
- it helps debugging if a `metadata.jsonl` row and a sample directory ever get out of sync
- it lets downstream scripts inspect a sample directory without first querying `metadata.jsonl`
- it captures nested structure more naturally than flattening everything into columns

## `metadata.jsonl` Schema

We use JSONL instead of Parquet for the reasons outlined in the "Dataset upload structure discovery" section.

Final columns (in order, with `_file_name` suffix for media columns):
1. `sample_id`
2. `segmented_overhead_file_name` - HF URL to segmented overhead image
3. `speckle_vibrations_file_name` - HF URL to speckle vibration video
4. `speckle_shifts_ifft_audio_file_name` - HF URL to audio derived from recovered shifts
5. `audio_file_name` - HF URL to the shared chirp audio
6. `experiment_id`
7. `speakers` - bitstring such as `0100`
8. `x_position` - integer user-provided placement coordinate
9. `y_position` - integer user-provided placement coordinate
10. `x_com` - float X center of mass in pixel space derived from the segmentation mask
11. `y_com` - float Y center of mass in pixel space derived from the segmentation mask
12. `n_objects`
13. `box_material`
14. `mask_file_name` - HF URL to `mask.png`
15. `experiment_dir` - relative experiment directory label such as `experiment_15`
16. `manifest` - JSON string form of `manifest.json`

Manifest requirement:
- `manifest.json` should include `hf_repo` so artifact URLs and upload targets are reproducible from the start of the experiment
- `manifest.json` should include derived segmentation center-of-mass fields `x_com` and `y_com`
- `manifest.json` should make the unit distinction clear: grid coordinates for `x_position`/`y_position`, pixel coordinates for `x_com`/`y_com`

We explicitly do NOT include:
- `raw_overhead_file_name`, `cropped_overhead_file_name` - keep these in the shared image directory and manifest, not as flat viewer columns
- tensor paths such as `speckle_shifts_fft.npz` - available in the manifest and sample directory
- `manifest_file_name` - replaced by the embedded `manifest` string

### Why These Columns?

Reasoning:
- HF viewer auto-detects media types ONLY when column names end with `_file_name`
- We store HF URLs (not relative paths) so the viewer can render directly
- Training and tooling need URL references to fetch artifacts
- Simple scalar metadata for filtering and grouping in the viewer
- The full manifest still appears in the row via the `manifest` string column
- therefore, only high-value viewer/filter fields should stay flattened in `metadata.jsonl`, while detailed configuration should live in the manifest

### Viewer-Specific Notes

For HF viewer compatibility, the media fields should be full HF URLs:
- `audio_file_name = https://huggingface.co/datasets/<HF_REPO>/resolve/main/data/audio/chirp_50_1000_3.0sec.wav`
- `segmented_overhead_file_name = https://huggingface.co/datasets/<HF_REPO>/resolve/main/data/image/<IMAGE_DIR>/segmented_overhead.png`
- `mask_file_name = https://huggingface.co/datasets/<HF_REPO>/resolve/main/data/image/<IMAGE_DIR>/mask.png`
- `speckle_vibrations_file_name = https://huggingface.co/datasets/<HF_REPO>/resolve/main/data/<SAMPLE_ID>/speckle_vibrations.mp4`
- `speckle_shifts_ifft_audio_file_name = https://huggingface.co/datasets/<HF_REPO>/resolve/main/data/<SAMPLE_ID>/speckle_shifts_ifft_audio.wav`

## Multi-Machine Pipeline Model

The real pipeline spans several machines and storage locations.

### Current Physical Flow

1. Local machine records data
2. files are copied to `/net/mraid20`
3. recovery runs on `mcluster11` using data on `mraid20`
4. overhead crop + segmentation runs separately and saves back to shared storage
5. HF upload happens from shared processed outputs

### Desired Logical Pipeline

1. take overhead image
2. play audio clip
3. record raw speckle vibrations
4. recover shifts from raw recording
5. clean shifts
6. convert cleaned shifts to FFT
7. crop overhead image
8. generate segmentation mask
9. generate display assets
10. upload all structured outputs to the new HF dataset

### Notebook And Script Split

Most acquisition logic should live in `scripts/record.py`.

We should create a notebook that imports from `scripts/record.py` and is parameterized by:
- `experiment_config.json`
- `processing_config.json`
- `experiment_dir`
- `hf_repo`
- `x_position`
- `y_position`
- `n_speakers`
- `box_material`
- `object`
- `audio`

The notebook should stay minimal and have five cells:
1. init overhead camera and preview
2. init laser camera, load or define ROIs, and preview them for manual inspection
3. init audio and preview playback
4. run acquisition and save artifacts incrementally
5. inspect the generated manifest and metadata row before continuing

Runtime expectation:
- set `hf_repo` in the manifest at the beginning of the experiment, at the same time as `experiment_dir` and the sample position metadata
- set user-supplied `x_position` and `y_position` in the manifest before recording starts
- save the shared overhead image group into `data/image/<IMAGE_DIR>/` if it does not already exist
- save all sample-specific artifacts into `data/<sample_id>/`
- compute `x_com` and `y_com` from the segmentation mask once the mask exists and append them to `manifest.json`
- update `manifest.json` incrementally as `experiment_config`, `processing_config`, outputs, and artifact paths become available

Backfill expectation:
- when backfilling old samples, derive canonical `x_position` and `y_position` from the experiment directory naming convention
- treat the old dataset's `x_position` and `y_position` values as center-of-mass-style hints only

### Final Pipeline Artifacts By Stage

#### Stage A: Capture
- `raw_overhead.png`
- `speckle_vibration_raw.npz`
- experiment config including exact audio path used

#### Stage B: Recovery
- `speckle_shifts.npz`

#### Stage C: Derived Processing
- `speckle_vibrations.mp4`
- `speckle_shifts_clean.npz`
- `speckle_shifts_fft.npz`
- `speckle_shifts_ifft_audio.wav`

#### Stage D: Segmentation
- `mask.png`
- `mask.npz`

#### Stage E: Packaging
- `manifest.json`
- `metadata.jsonl` row
- upload shared audio if needed

## Why Audio Should Be Shared

Most or all current samples use the same chirp.

So we will store shared audio files once under `data/audio/` and reference them from `metadata.jsonl` and the manifest.

Reasoning:
- avoids duplicate uploads
- avoids bloating the repository
- keeps provenance explicit
- still allows HF viewer playback because each JSONL row points to the shared HF URL

## Why We Keep `viz` Frozen And Build `viz2`

The existing `viz` is already in use and should not be destabilized.

So:
- `viz/` remains untouched
- `viz2/` will start as a copy of `viz/`
- all new dataset-specific work will happen in `viz2/`

Reasoning:
- we want a safe sandbox to iterate quickly
- we expect schema changes and viewer behavior changes
- we do not want ongoing work to break the currently used tool

## `viz2` Requirements

`viz2` should only display assets that already exist in the uploaded dataset.

It should not:
- run Modal
- trigger segmentation
- trigger recovery
- derive assets on click

It should:
- show chirp audio from uploaded sample metadata
- show `speckle_vibrations.mp4`
- show `speckle_shifts_ifft_audio.wav`
- later show visual summaries of clean shifts and FFTs

Interaction model:
- dataset cell click: sample/media popup
- run sample double click: run/sample detail popup

## Training Migration Strategy

### Near Term

Training should continue to use the current loader logic until the new dataset is validated.

### Medium Term

Training metadata load should come from `metadata.jsonl`, selecting only needed columns.

Training tensor download should fetch only:
- `mask.npz`
- `speckle_shifts_fft.npz`

It should not fetch:
- audio
- MP4 preview
- IFFT audio preview
- raw speckle recording

### Long Term

The dataloader should stop performing these preprocessing steps in `__getitem__`:
- Butterworth filter
- Hann window
- frequency cropping
- FFT conversion

Instead, those steps become pipeline outputs saved in HF.

Reasoning:
- faster dataloader
- easier reproducibility
- easier debugging
- easier asset inspection in the viewer

## Performance Logging Requirement

We want timings for every meaningful stage we implement.

This is a hard requirement, not a nice-to-have.

We should print timings for:
- raw file inspection
- media generation
- recovery
- cleaning
- FFT conversion
- FFT audio generation
- segmentation
- `metadata.jsonl` row creation
- upload step per asset
- viewer payload build stages

### Why This Matters

Reasoning:
- we want to know the bottleneck before optimizing
- the pipeline spans multiple machines and remote storage
- media conversion and upload may dominate runtime
- training migration decisions should be based on measured cost, not guesses

### Logging Style

Use explicit per-stage timing logs like:
- stage name
- elapsed time
- relevant counts or shapes
- possibly output sizes when relevant

Example:
- `Generated speckle_vibrations.mp4 in 3.2s (9000 frames -> 300 preview frames)`
- `Uploaded sample assets in 8.4s (mp4=4.2MB, fft=2.1MB)`

## Implementation Strategy

We will implement the simplest version possible, step by step.

### Phase 1: Foundation

Deliverables:
- create `PLAN.md`
- create `viz2/` as a copy of `viz/`
- point `viz2` at the new dataset `laser-vibrations`
- preserve stage timing logs in the copied code and add more as needed
- initialize the empty `laser-vibrations` dataset scaffold

Why start here:
- this gives us a safe sandbox without touching the currently used viewer

Status:
- done

### Phase 2: Dummy Sample End-To-End

Deliverables:
- create one sample in the new structure
- upload one shared audio file
- upload one MP4 preview
- upload one IFFT audio preview
- upload the tensor files and manifest
- append one `metadata.jsonl` row

Validation:
- HF viewer plays the chirp audio
- HF viewer plays the speckle vibration MP4
- HF viewer plays the IFFT audio preview

Why this phase matters:
- it validates the schema before we scale up or backfill old data
- it avoids paying the cost of copying the entire old dataset before the new schema is proven

Implementation note:
- the backfill script should auto-discover the source experiment directory by bucketing the old dataset's continuous `x_position` and `y_position` onto the discrete filename grid
- after discovery, the new canonical `x_position` and `y_position` should come from the experiment directory naming rather than being copied directly from the old row
- image matching should only be used as a fallback when the grid-based lookup is ambiguous
- manual `source_experiment_dir` override should still be supported as a fallback
- the single-sample orchestrator should take only `sample_id` and run the remote raw/mp4 stage first, then the local metadata/tensor stage

### Phase 3: `viz2` Reads New Dataset Assets

Deliverables:
- `viz2` reads `metadata.jsonl` rows from `laser-vibrations`
- dataset popup shows uploaded sample audio
- run/sample popup shows uploaded MP4 and IFFT audio preview

Why this phase matters:
- confirms that our custom viewer is consuming the same canonical uploaded assets as HF viewer

### Phase 4: Future Upload Pipeline

Deliverables:
- update upload scripts to generate and upload the new asset set for future samples
- add per-stage timing logs to all major upload and packaging steps

**New upload pattern:**
```python
from huggingface_hub import HfApi, create_repo

api = HfApi()
REPO_ID = "eturok-weizmann/laser-vibrations"

# Create repo if it doesn't exist
create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

# Upload folder (contains data/ with subdirs, audio/, image/, README.md)
api.upload_folder(
    folder_path="data",
    repo_id=REPO_ID,
    repo_type="dataset",
)
```

The folder structure to upload:
```
data/
  metadata.jsonl    → appended with new sample row (HF URLs)
  audio/            → upload new audio if different from existing
  image/            → upload if different from existing
    <IMAGE_DIR>/
      raw_overhead.png
      cropped_overhead.png
      segmented_overhead.png
      mask.png
      mask.npz
  0000001/          → sample-specific files
    manifest.json
    speckle_vibration_raw.npz
    speckle_vibrations.mp4
    speckle_shifts_ifft_audio.wav
    speckle_shifts_fft.npz
    ...
```

**For backfill:** Use the same pattern - upload shared assets first (audio/image), then per-sample directories, then update metadata.jsonl.

Why this phase matters:
- new data should land in the final structure automatically

### Phase 5: Backfill Existing Samples

Deliverables:
- map existing samples to their source experiment directories
- generate missing display and derived artifacts
- upload them into the new dataset structure

Why this phase matters:
- we want the new dataset to contain the historical data too, not just future samples
- we will backfill incrementally after the fresh dataset and dummy sample path are confirmed working

Performance note:
- raw video transfer should stay remote; only light metadata should cross SSH
- timings should be printed for discovery, config load, raw file access, MP4 generation, IFFT audio generation, `metadata.jsonl`/manifest creation, and upload

### Phase 6: Migrate Training

Deliverables:
- make training read `speckle_shifts_fft.npz` directly
- benchmark loading and verify correctness against the old path

Why this phase matters:
- this is the point where preprocessing has truly moved out of the dataloader and into the pipeline

## Non-Goals For The First Iterations

We explicitly do not want to do these immediately:
- rewrite the current `viz`
- redesign all training code at once
- store raw recordings in HF viewer-facing form
- derive assets lazily on click in the viewer
- batch-backfill the full historical dataset before a single dummy sample works cleanly
- fully clone the old `vibrations` HF dataset into `laser-vibrations` before validating the new schema

## Source Of Truth Summary

This plan fixes the target design as follows:

- new HF dataset name: `laser-vibrations`
- keep `viz` frozen; build new work in `viz2`
- use `data/metadata.jsonl` for metadata and viewer-facing file references
- keep shared audio under `data/audio/`
- keep shared overhead image groups under `data/image/<IMAGE_DIR>/`
- keep per-sample structured assets under `data/<sample_id>/`
- keep viewer-facing overhead image columns in `metadata.jsonl`
- preserve the full intermediate processing pipeline
- add `manifest.json` per sample
- set `hf_repo` in the manifest at experiment start
- keep `x_position`/`y_position` as user-supplied placement coordinates and store derived mask center of mass as `x_com`/`y_com` in the manifest
- for backfill, infer canonical `x_position`/`y_position` from experiment directory naming and treat old dataset coordinates as center-of-mass hints
- replace the vague flat `experiment_config` shape with a structured manifest built around acquisition and artifact stages
- store the full manifest again as a `manifest` string in `metadata.jsonl` for viewer visibility
- store both `source_experiment_id` and `source_experiment_dir` in the manifest
- save crop params and prompt in `mask.npz`
- save complex FFT directly in `speckle_shifts_fft.npz`
- create IFFT audio preview via IFFT, defaulting to `laser_idx=50`, `xy_idx=0`
- log timings for every stage we implement
- use PR-style HF uploads by default because direct commits to the dataset are blocked
- use a split pipeline: remote raw/mp4 upload, local metadata/tensor upload
- use one shared manifest builder for runtime capture and backfill so all samples are consistent

### Rebuild Requirement

The manifest should be rich enough that we can recreate the acquisition and packaging flow without relying on hidden notebook constants.

That means the manifest should contain enough information to:
- configure the overhead camera
- configure the laser camera
- reconstruct the ROI geometry
- play the correct audio to the correct speakers
- compute the expected number of laser frames
- reproduce the downstream processing artifacts

Fields still required for that goal:
- `sample_id` at the top level of the manifest
- laser camera row LUT values actually sent to the camera, not just the derived row ranges
- final selected column points used to build the 10x10 ROI grid
- final full sensor ROI list `(x, y, w, h)` for all sensors
- `n_capture_seconds`
- audio output metadata needed by the playback function if it is not fixed in code:
  - total output channels
  - any non-default speaker mapping convention if applicable
- any capture-mode flags that materially affect the saved raw recording, if they are not fixed by code defaults

Fields that do not need to be in the manifest if they stay fixed in code and are not expected to vary between runs:
- preview-only UI parameters
- notebook file path
- debug display settings

This document should be updated whenever we intentionally change the design.

---

## Implementation Progress Log

### Backfill pipeline (Phase 5)

- [x] End-to-end backfill working via `backfill_laser_vibrations_one.py --sample-id N`
- [x] `--create-pr` → `--no-create-pr`: direct commits work once using the correct org-scoped HF token
- [x] HF token fix: local `get_token()` was returning a stale fine-grained token scoped to the `eturok` user, not the `eturok-weizmann` org. Fix: read `~/.cache/huggingface/token` directly in both backfill scripts.
- [x] Stray `remote/experiment_config.json` fix: `load_remote_experiment_config` used to write the config to `root/"remote"/` inside the upload temp dir, which then got swept up by `upload_folder`. Fixed by parsing JSON in-memory via SSH `cat` with no disk write.
- [x] `README.md` pushed to HF dataset with dataset card, column table, sample file table, and full `manifest.json` field descriptions

### Dataset viewer

- [x] Images render when `*_file_name` image columns point to full HF URLs
- [x] Audio plays for `audio_file_name` and `speckle_shifts_ifft_audio_file_name`
- [x] Video plays for `speckle_vibrations_file_name`
- [x] Video codec fix: OpenCV writes `mp4v` (MPEG-4 Part 2) which browsers cannot play. Fixed by re-encoding to H.264 (`libx264`) via `imageio-ffmpeg`, which bundles its own ffmpeg binary so no system install needed on the cluster.
- [x] README YAML must declare media dtypes explicitly and the media columns must end with `_file_name`

### Speed optimizations

Starting point: ~110s per sample.

- [x] **Remove remote repo clone** (~40s saved): every run was doing `rm -rf` + `git clone` + `git pull` just to have a directory to run the uploader script from. The script is already piped over SSH, so the clone was entirely unnecessary. Now writes the script to `$HOME/tmp/upload_remote_speckle_assets.py` and runs it directly.
- [x] **Parallel HF uploads** (~10s saved): replaced two sequential `upload_file` calls (raw npy + mp4) with a single `create_commit(num_threads=2, operations=[...])`. The HF API parallelizes LFS uploads internally — no custom threading needed.

Current total: ~49s per sample.

### Remaining bottlenecks (in priority order)

- [x] **`uv venv --clear`** (~5-8s): dropped `--clear` so remote venv is reused across runs. `uv venv` prints a non-fatal "already exists" error but `uv pip install` still succeeds on the existing env.
- [x] **Duplicate dataset loads** (~6-12s): merged `load_old_row` + position-grid scan in `_one.py` into a single dataset pass. Skipped redundant `load_old_position_grid` in `backfill_laser_vibrations.py` when `--source-experiment-dir` is already provided. `backfill_laser_vibrations.py` still loads once for images.
- [ ] **MP4 round-trip through HF** (~2s): remote step uploads MP4, local step downloads it again while preparing metadata. Fix: keep the metadata pass URL-only.

### Current HF Upload Contract

**Problem:** HF dataset viewer does not reliably resolve relative file paths for shared media assets in JSONL rows.

**Current rule:** use full Hugging Face URLs for every media field in `metadata.jsonl`.

This is the current recommended approach because it:
- keeps shared files stored once in the repo
- lets HF render shared audio and image assets directly
- keeps sample rows append-only
- avoids path-resolution ambiguity in both HF viewer and Python tooling

### Full `README.md` Front Matter

This exact `README.md` front matter is part of the source of truth and should be copied into the dataset repo:

```yaml
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
```

Rules:
- every renderable media column must end with `_file_name`
- `x_com` and `y_com` should appear immediately after `x_position` and `y_position`
- `mask_file_name` should appear after `box_material` to match the intended viewer column order
- `manifest` is a plain string column and should contain the full serialized `manifest.json`
- `manifest` must include the `artifacts.shared` and `artifacts.sample` blocks so the row alone tells us where all files live

### Canonical `metadata.jsonl` Example

```json
{
  "sample_id": 1,
  "segmented_overhead_file_name": "https://huggingface.co/datasets/<HF_REPO>/resolve/main/data/image/<IMAGE_DIR>/segmented_overhead.png",
  "speckle_vibrations_file_name": "https://huggingface.co/datasets/<HF_REPO>/resolve/main/data/0000001/speckle_vibrations.mp4",
  "speckle_shifts_ifft_audio_file_name": "https://huggingface.co/datasets/<HF_REPO>/resolve/main/data/0000001/speckle_shifts_ifft_audio.wav",
  "audio_file_name": "https://huggingface.co/datasets/<HF_REPO>/resolve/main/data/audio/chirp_50_1000_3.0sec.wav",
  "experiment_id": "cube-00x01y_0001--31-03-18-21-24",
  "speakers": "0100",
  "x_position": 3,
  "y_position": 7,
  "x_com": 3.1,
  "y_com": 2.7,
  "n_objects": 1,
  "box_material": "cardboard",
  "mask_file_name": "https://huggingface.co/datasets/<HF_REPO>/resolve/main/data/image/<IMAGE_DIR>/mask.png",
  "experiment_dir": "experiment_15",
  "manifest": "{\"sample_id\":1,\"artifacts\":{\"shared\":{...},\"sample\":{...}}}"
}
```

Rule:
- `experiment_dir` should be a relative experiment identifier such as `experiment_15`, not an absolute filesystem path
- `hf_repo` should be set when the experiment starts, alongside `experiment_dir` and sample position metadata, not inferred later during packaging
- `x_position` and `y_position` are integer grid coordinates entered by the user; `x_com` and `y_com` are float pixel-space values derived later from the segmentation mask
