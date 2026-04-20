# Laser Vibrations Plan

## Goal

Create a new Hugging Face dataset called `laser-vibrations` that preserves the full processing pipeline, exposes the right assets in the Hugging Face dataset viewer, remains efficient for model training, and gives us a richer custom viewer (`viz2`) without breaking the currently used `viz`.

This plan is the source of truth for the next implementation steps.

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

### 2. Separate Viewer Assets From Training Tensors

We will not store everything inside Parquet.

Instead:
- `data/train-xxxx.parquet` will store metadata plus HF-viewable file references
- shared audio files will live under `audio/`
- per-sample assets will live under `samples/sample_xxxxxx/`

Reasoning:
- HF viewer works well with referenced audio/video assets
- storing large tensors and raw recordings inside parquet is not ideal for append-only uploads or selective model loading
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
1. `speckle_vibrations_raw.npy`
2. `speckle_vibrations.mp4`
3. `speckle_shifts.npz`
4. `speckle_shifts_clean.npz`
5. `speckle_shifts_fft.npz`
6. `mask.npz`

Reasoning:
- intermediate stages are valuable for debugging and validation
- future training will likely consume `speckle_shifts_fft.npz` directly
- until then, we still want raw and cleaned stages saved
- `viz2` should eventually be able to visualize multiple stages of the pipeline

### 5. FFT Audio Preview Is A Derived Inspection Asset

We want an audio sanity check derived from `speckle_shifts_fft.npz`.

The preview file will be:
- `speckle_shifts_fft_audio.wav`

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
data/
  train-0001.parquet
  train-0002.parquet

audio/
  chirp_50_1000_3.0sec.wav
  ...

samples/
  sample_000001/
    manifest.json
    mask.npz
    speckle_vibrations_raw.npy
    speckle_vibrations.mp4
    speckle_shifts.npz
    speckle_shifts_clean.npz
    speckle_shifts_fft.npz
    speckle_shifts_fft_audio.wav
  sample_000002/
    ...
```

## Sample Asset Semantics

### `speckle_vibrations_raw.npy`

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

### `speckle_shifts_fft_audio.wav`

Meaning:
- audio preview generated from one selected complex FFT trace using IFFT

Role:
- sanity check / inspection asset
- playable in HF viewer and `viz2`

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

## Manifest Schema

Each sample directory will contain `manifest.json` describing the canonical paths and processing choices for that sample.

Proposed structure:

```json
{
  "sample_idx": 1,
  "audio_file_name": "audio/chirp_50_1000_3.0sec.wav",
  "sample_dir": "samples/sample_000001",
  "speckle_vibrations": {
    "file_name": "samples/sample_000001/speckle_vibrations.mp4",
    "raw_path": "samples/sample_000001/speckle_vibrations_raw.npy",
    "frame_count": 9000,
    "frame_height": 300,
    "frame_width": 1152,
    "fps": 2500
  },
  "speckle_shifts": {
    "path": "samples/sample_000001/speckle_shifts.npz"
  },
  "speckle_shifts_clean": {
    "path": "samples/sample_000001/speckle_shifts_clean.npz",
    "lowcut": 50,
    "highcut": 1000,
    "filter_order": 5,
    "hann_applied": true
  },
  "speckle_shifts_fft": {
    "path": "samples/sample_000001/speckle_shifts_fft.npz",
    "min_freq": 50,
    "max_freq": 1000,
    "dtype": "complex64"
  },
  "speckle_shifts_fft_audio": {
    "file_name": "samples/sample_000001/speckle_shifts_fft_audio.wav",
    "laser_idx": 50,
    "xy_idx": 0,
    "method": "ifft"
  },
  "mask": {
    "path": "samples/sample_000001/mask.npz"
  },
  "source_experiment_dir": "..."
}
```

### Why Keep A Manifest If We Already Have Parquet?

Reasoning:
- it gives each sample a local, self-contained description
- it helps debugging if a parquet row and a sample directory ever get out of sync
- it lets downstream scripts inspect a sample directory without first querying parquet
- it captures nested structure more naturally than flattening everything into columns

## Parquet Schema

Parquet is for sample-level metadata and HF-viewable asset references.

We explicitly do not want frame shape/count fields in parquet. Those belong under `manifest.json -> speckle_vibrations`.

Final parquet columns:
- `sample_idx`
- `object`
- `n_objects`
- `box_material`
- `speakers`
- `x_position`
- `y_position`
- `fps`
- `experiment_config`
- `audio_file_name`
- `speckle_vibrations_file_name`
- `speckle_shifts_fft_audio_file_name`
- `sample_dir`
- `mask_path`
- `speckle_vibrations_raw_path`
- `speckle_shifts_path`
- `speckle_shifts_clean_path`
- `speckle_shifts_fft_path`

### Why These Columns?

Reasoning:
- HF viewer should be able to play the audio and MP4 by using the `*_file_name` fields
- training and tooling need string paths to the tensor artifacts
- we still want simple scalar metadata for filtering and model setup
- this keeps parquet narrow and avoids storing large binary tensors inline

### Viewer-Specific Notes

For HF viewer compatibility, the media fields should be relative repo paths:
- `audio_file_name = audio/chirp_50_1000_3.0sec.wav`
- `speckle_vibrations_file_name = samples/sample_000001/speckle_vibrations.mp4`
- `speckle_shifts_fft_audio_file_name = samples/sample_000001/speckle_shifts_fft_audio.wav`

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

### Final Pipeline Artifacts By Stage

#### Stage A: Capture
- `box_overhead_image.png`
- `speckle_vibrations_raw.npy`
- experiment config including exact audio path used

#### Stage B: Recovery
- `speckle_shifts.npz`

#### Stage C: Derived Processing
- `speckle_vibrations.mp4`
- `speckle_shifts_clean.npz`
- `speckle_shifts_fft.npz`
- `speckle_shifts_fft_audio.wav`

#### Stage D: Segmentation
- `mask.npz`

#### Stage E: Packaging
- `manifest.json`
- parquet row
- upload shared audio if needed

## Why Audio Should Be Shared

Most or all current samples use the same chirp.

So we will store shared audio files once under `audio/` and reference them from parquet and manifest.

Reasoning:
- avoids duplicate uploads
- avoids bloating the repository
- keeps provenance explicit
- still allows HF viewer playback if the parquet row points to the shared relative file path

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
- show `speckle_shifts_fft_audio.wav`
- later show visual summaries of clean shifts and FFTs

Interaction model:
- dataset cell click: sample/media popup
- run sample double click: run/sample detail popup

## Training Migration Strategy

### Near Term

Training should continue to use the current loader logic until the new dataset is validated.

### Medium Term

Training metadata load should come from parquet only, selecting only needed columns.

Training tensor download should fetch only:
- `mask.npz`
- `speckle_shifts_fft.npz`

It should not fetch:
- audio
- MP4 preview
- FFT audio preview
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
- parquet row creation
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

Why start here:
- this gives us a safe sandbox without touching the currently used viewer

### Phase 2: Dummy Sample End-To-End

Deliverables:
- create one sample in the new structure
- upload one shared audio file
- upload one MP4 preview
- upload one FFT audio preview
- upload the tensor files and manifest
- append one parquet row

Validation:
- HF viewer plays the chirp audio
- HF viewer plays the speckle vibration MP4
- HF viewer plays the FFT audio preview

Why this phase matters:
- it validates the schema before we scale up or backfill old data

### Phase 3: `viz2` Reads New Dataset Assets

Deliverables:
- `viz2` reads parquet columns from `laser-vibrations`
- dataset popup shows uploaded sample audio
- run/sample popup shows uploaded MP4 and FFT audio preview

Why this phase matters:
- confirms that our custom viewer is consuming the same canonical uploaded assets as HF viewer

### Phase 4: Future Upload Pipeline

Deliverables:
- update upload scripts to generate and upload the new asset set for future samples
- add per-stage timing logs to all major upload and packaging steps

Why this phase matters:
- new data should land in the final structure automatically

### Phase 5: Backfill Existing Samples

Deliverables:
- map existing samples to their source experiment directories
- generate missing display and derived artifacts
- upload them into the new dataset structure

Why this phase matters:
- we want the new dataset to contain the historical data too, not just future samples

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

## Source Of Truth Summary

This plan fixes the target design as follows:

- new HF dataset name: `laser-vibrations`
- keep `viz` frozen; build new work in `viz2`
- use parquet for metadata and viewer-facing file references
- keep shared audio under `audio/`
- keep per-sample structured assets under `samples/sample_xxxxxx/`
- preserve the full intermediate processing pipeline
- add `manifest.json` per sample
- save crop params and prompt in `mask.npz`
- save complex FFT directly in `speckle_shifts_fft.npz`
- create FFT audio preview via IFFT, defaulting to `laser_idx=50`, `xy_idx=0`
- log timings for every stage we implement

This document should be updated whenever we intentionally change the design.
