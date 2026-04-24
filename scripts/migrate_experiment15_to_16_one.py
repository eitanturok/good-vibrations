import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import textwrap
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


REMOTE_HOST = "ethantu@mcluster11.wisdom.weizmann.ac.il"
REMOTE_UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"
REMOTE_UV = "$HOME/.local/bin/uv"
REMOTE_VENV = "$HOME/venvs/experiment16-migrate"
REMOTE_SCRIPT_PATH = "/home/ethantu/tmp/migrate_experiment15_to_16_one.py"
REMOTE_AUDIO_PATH = "/home/ethantu/tmp/migrate_audio.wav"
REMOTE_HF_TOKEN_PATH = "/home/ethantu/.cache/huggingface/token"
REMOTE_MODAL_CONFIG_PATH = "/home/ethantu/.modal.toml"

DEFAULT_OLD_DIR = "/net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA/experiment-15"
DEFAULT_NEW_DIR = "/net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA/experiment-16"
DEFAULT_HF_REPO = "eturok-weizmann/laser-vibrations"

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_AUDIO_ROOT = REPO_ROOT / "data" / "audio_samples"
CANONICAL_AUDIO_LOCAL_PATH = LOCAL_AUDIO_ROOT / "chirp_50_1000_3.0sec.wav"
CANONICAL_AUDIO_FILE_NAME = "chirp_50_1000_3.0sec.wav"
CANONICAL_AUDIO_REL = f"audio/{CANONICAL_AUDIO_FILE_NAME}"
CANONICAL_AUDIO_SAMPLE_RATE_HZ = 44100
CANONICAL_AUDIO_DURATION_S = 3.2
CANONICAL_AUDIO_TOTAL_OUTPUT_CHANNELS = 8
CANONICAL_AUDIO_WAV_CHANNELS = 1
CANONICAL_AUDIO_SAMPLE_WIDTH_BYTES = 2
CANONICAL_AUDIO_CHIRP_DURATION_S = 3.0
CANONICAL_AUDIO_SILENCE_START_S = 0.1
CANONICAL_AUDIO_SILENCE_END_S = 0.1
CANONICAL_AUDIO_F_START_HZ = 50
CANONICAL_AUDIO_F_END_HZ = 1000
CANONICAL_AUDIO_GENERATION_METHOD = "logarithmic"
CANONICAL_AUDIO_OUTPUT_DTYPE = "int16"
CANONICAL_AUDIO_NORMALIZATION = "peak_to_int16"
ASSETS_DIR = REPO_ROOT / "assets"
SPEAKER_DIR = ASSETS_DIR / "speakers"
SPEAKER_FILES = ("1000", "0100", "0010", "0001")
OVERHEAD_SIZE = (220, 196)
PADDED_SIZE = (286, 255)
PADDED_BG = (232, 232, 232)

DEFAULT_SEGMENT_PROMPT = "A black metal cube sitting on the floor of an open cardboard box from a bird's eye view."
DEFAULT_SEGMENT_LEFT = 0.15
DEFAULT_SEGMENT_RIGHT = 0.67
DEFAULT_SEGMENT_UP = 0.08
DEFAULT_SEGMENT_DOWN = 0.7
METADATA_COLUMN_ORDER = [
    "sample_id",
    "segmented_overhead_file_name",
    "speckle_vibrations_file_name",
    "speckle_shifts_ifft_audio_file_name",
    "audio_file_name",
    "experiment_id",
    "speakers",
    "x_position",
    "y_position",
    "x_com",
    "y_com",
    "n_objects",
    "box_material",
    "mask_file_name",
    "experiment_dir",
    "manifest",
]


@contextmanager
def stage(label: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        print(f"[timing] {label}: {time.perf_counter() - t0:.2f}s", flush=True)


def run(label: str, fn):
    with stage(label):
        return fn()


def normalize_token(value: str) -> str:
    return "-".join(str(value).strip().lower().split())


def parse_source_coordinates(source_dir_name: str) -> tuple[int | None, int | None]:
    match = re.search(r"(?P<x>\d{2})x(?P<y>\d{2})y", source_dir_name)
    if match is None:
        return None, None
    return int(match.group("x")), int(match.group("y"))


def parse_speakers_from_source_dir(source_dir_name: str) -> str:
    match = re.search(r"_(?P<speakers>[01]{4})--", source_dir_name)
    return match.group("speakers") if match else ""


def parse_source_timestamp(source_dir_name: str) -> str:
    match = re.search(r"--(?P<ts>\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$", source_dir_name)
    if match is None:
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    day, month, hour, minute, second = match.group("ts").split("-")
    year = datetime.now().year
    return f"{year:04d}-{int(month):02d}-{int(day):02d}-{int(hour):02d}-{int(minute):02d}-{int(second):02d}"


def ensure_safe_paths(old_dir: Path, new_dir: Path) -> None:
    old_resolved = old_dir.resolve()
    new_resolved = new_dir.resolve()
    if old_resolved == new_resolved:
        raise ValueError("old_dir and new_dir must be different")
    if str(new_resolved).startswith(str(old_resolved)):
        raise ValueError("new_dir must not be inside old_dir")


def notebook_default_experiment_config() -> dict:
    return {
        "audio": {
            "file_name": CANONICAL_AUDIO_REL,
            "sample_rate_hz": CANONICAL_AUDIO_SAMPLE_RATE_HZ,
            "duration_s": CANONICAL_AUDIO_DURATION_S,
            "total_output_channels": CANONICAL_AUDIO_TOTAL_OUTPUT_CHANNELS,
            "wav_channels": CANONICAL_AUDIO_WAV_CHANNELS,
            "sample_width_bytes": CANONICAL_AUDIO_SAMPLE_WIDTH_BYTES,
            "generation": {
                "signal": "chirp",
                "method": CANONICAL_AUDIO_GENERATION_METHOD,
                "chirp_duration_s": CANONICAL_AUDIO_CHIRP_DURATION_S,
                "silence_start_s": CANONICAL_AUDIO_SILENCE_START_S,
                "silence_end_s": CANONICAL_AUDIO_SILENCE_END_S,
                "f_start_hz": CANONICAL_AUDIO_F_START_HZ,
                "f_end_hz": CANONICAL_AUDIO_F_END_HZ,
                "output_dtype": CANONICAL_AUDIO_OUTPUT_DTYPE,
                "normalization": CANONICAL_AUDIO_NORMALIZATION,
            },
        },
        "recording": {
            "capture_seconds_requested": 3.1,
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
                "debayer_code": "COLOR_BAYER_BG2BGR",
            },
        },
        "laser_camera": {
            "runtime": {
                "info_field": False,
                "cxp_link_configuration": "CXP12_X4",
            },
            "calibration": {
                "fps": 500,
                "exposure_us": 30,
                "gain": 3,
            },
            "capture": {
                "fps": 2500,
                "exposure_us": 150,
                "gain": 1,
                "buffer_part_count": 3000,
            },
        },
        "laser_grid": {
            "n_roi_rows": 10,
            "n_roi_columns": 10,
            "roi_row_height": 30,
            "roi_column_width": 70,
        },
        "preview": {
            "overhead_resize_factor": 0.75,
            "overhead_gamma": 1.0,
            "laser_preview_gamma": 2.5,
            "show_full_frame": 0,
            "preview_level": 1,
            "reset_rois": True,
        },
    }


def default_processing_config() -> dict:
    return {
        "speckle_vibration_raw": {
            "format": "npy",
            "compressed": False,
        },
        "speckle_vibrations_preview": {
            "max_frames": 300,
            "max_width": 960,
            "macro_block_size": 16,
            "source_capture_fps_hz": 2500.0,
            "preserve_physical_duration": True,
            "percentile_low": 5.0,
            "percentile_high": 99.5,
            "codec": "libx264",
            "pixelformat": "yuv420p",
            "crf": 23,
            "burn_frame_index": True,
        },
        "speckle_shifts": {
            "fs_hz": 2500.0,
        },
        "speckle_shifts_clean": {
            "filter_type": "butterworth",
            "filter_mode": "bandpass",
            "lowcut": 50.0,
            "highcut": 1000.0,
            "filter_order": 5,
            "hann_applied": True,
            "apply_order": "filter_then_hann",
        },
        "speckle_shifts_fft": {
            "fft_kind": "rfft",
            "fft_axis": 1,
            "min_freq": 50.0,
            "max_freq": 1000.0,
            "dtype": "complex64",
            "crop_after_fft": True,
        },
        "speckle_shifts_ifft_audio": {
            "laser_idx": 50,
            "xy_idx": 0,
            "method": "ifft",
            "output_sample_rate_hz": 22050,
            "normalization": "peak_to_int16",
            "output_dtype": "int16",
            "zero_fill_uncropped_bins": True,
        },
    }


def merge_experiment_config(base: dict, loaded: dict, metadata: dict | None) -> dict:
    merged = json.loads(json.dumps(base))
    capture_fps = loaded.get("FPS") or loaded.get("fps")
    capture_seconds = loaded.get("N_CAPTURE_SECONDS") or loaded.get("capture_seconds")
    if capture_fps is not None:
        merged["laser_camera"]["capture"]["fps"] = capture_fps
    if capture_seconds is not None:
        merged["recording"]["capture_seconds_requested"] = capture_seconds

    if metadata is not None and "run_opt" in metadata:
        run_opt = metadata["run_opt"].item()
        cam_params = run_opt.get("cam_params", {})
        laser_capture = merged["laser_camera"]["capture"]
        laser_capture["fps"] = cam_params.get("camera_FPS", laser_capture["fps"])
        laser_capture["exposure_us"] = cam_params.get("exposure", laser_capture["exposure_us"])
        laser_capture["gain"] = cam_params.get("gain", laser_capture["gain"])
    return merged


def extract_run_opt(metadata: dict | None) -> dict:
    if metadata is None or "run_opt" not in metadata:
        return {}
    return metadata["run_opt"].item()


def hf_file_url(hf_repo: str, relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    return f"https://huggingface.co/datasets/{hf_repo}/resolve/main/{relative_path}"


def build_metadata_row(
    *,
    hf_repo: str,
    sample_id: int | None = None,
    experiment_id: str | None = None,
    speakers: str | None = None,
    x_position: int | None = None,
    y_position: int | None = None,
    x_com=None,
    y_com=None,
    n_objects: int | None = None,
    box_material: str | None = None,
    experiment_dir: str | None = None,
    segmented_overhead_path: str | None = None,
    speckle_vibrations_path: str | None = None,
    speckle_shifts_ifft_audio_path: str | None = None,
    audio_path: str | None = None,
    mask_path: str | None = None,
    manifest: dict | None = None,
) -> dict:
    row = {
        "sample_id": int(sample_id) if sample_id is not None else -1,
        "segmented_overhead_file_name": hf_file_url(hf_repo, segmented_overhead_path),
        "speckle_vibrations_file_name": hf_file_url(hf_repo, speckle_vibrations_path),
        "speckle_shifts_ifft_audio_file_name": hf_file_url(hf_repo, speckle_shifts_ifft_audio_path),
        "audio_file_name": hf_file_url(hf_repo, audio_path),
        "experiment_id": experiment_id or "",
        "speakers": speakers or "",
        "x_position": int(x_position) if x_position is not None else None,
        "y_position": int(y_position) if y_position is not None else None,
        "x_com": x_com,
        "y_com": y_com,
        "n_objects": int(n_objects) if n_objects is not None else 0,
        "box_material": box_material or "",
        "mask_file_name": hf_file_url(hf_repo, mask_path),
        "experiment_dir": experiment_dir or "",
        "manifest": json.dumps(manifest or {}, ensure_ascii=True),
    }
    return {key: row[key] for key in METADATA_COLUMN_ORDER}


def normalize_metadata_row(row: dict) -> dict:
    defaults = build_metadata_row(hf_repo="")
    merged = {**defaults, **row}
    return {key: merged[key] for key in METADATA_COLUMN_ORDER}


def build_dataset_readme() -> str:
    return textwrap.dedent(
        """\
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
        | `segmented_overhead_file_name` | image | Overhead photo with segmentation mask overlay and speaker-angle annotations |
        | `speckle_vibrations_file_name` | video | Slow-motion preview of the laser speckle pattern while the box vibrates |
        | `speckle_shifts_ifft_audio_file_name` | audio | Vibration signal of a single laser point reconstructed as audio (inverse FFT) |
        | `audio_file_name` | audio | Shared excitation chirp played through the loudspeakers during recording |
        | `experiment_id` | string | Source directory name from experiment-15 (unique per recording) |
        | `speakers` | string | 4-bit speaker activation code — e.g. `0001` means only speaker 4 was active |
        | `x_position` | int | Object grid column index (0-indexed) |
        | `y_position` | int | Object grid row index (0-indexed) |
        | `x_com` | float | X centre-of-mass of the segmentation mask in the cropped overhead image (pixels) |
        | `y_com` | float | Y centre-of-mass of the segmentation mask in the cropped overhead image (pixels) |
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
                ├── segmented_overhead.png       # Overhead with mask overlay + speaker annotations
                ├── mask.png                     # Binary segmentation mask (white = object)
                └── mask.npz                     # Binary mask as compressed numpy array
        ```

        ---

        # manifest.json Structure

        Every sample directory contains a `manifest.json` that records full provenance, hardware config,
        processing parameters, and artifact paths. The `manifest` column in `metadata.jsonl` is this
        same document serialised as a JSON string.

        ## Top-Level Keys

        | Key | Type | Description |
        |-----|------|-------------|
        | `sample_id` | int | Unique sample identifier |
        | `experiment_id` | string | Source recording directory name (from experiment-15) |
        | `experiment_dir` | string | Target directory name (experiment-16) |
        | `source_experiment_id` | string | Canonical source reference (same as `experiment_id`) |
        | `source_experiment_dir` | string | Absolute NAS path to the source directory |
        | `hf_repo` | string | HuggingFace repo this sample was uploaded to |
        | `sample` | object | Physical setup — see [3.2](#32-sample) |
        | `segmentation` | object | Segmentation result — see [3.3](#33-segmentation) |
        | `experiment_config` | object | Merged hardware + recording config — see [3.4](#34-experiment_config) |
        | `experiment_output` | object | Derived signal statistics — see [3.5](#35-experiment_output) |
        | `processing_config` | object | Processing pipeline parameters — see [3.6](#36-processing_config) |
        | `artifacts` | object | Relative repo paths to all files — see [3.7](#37-artifacts) |

        ## `sample`

        Physical setup at the time of recording.

        | Key | Type | Description |
        |-----|------|-------------|
        | `object` | string | Object type inside the box, e.g. `cube` or `empty` |
        | `n_objects` | int | Number of objects |
        | `box_material` | string | Box material, e.g. `cardboard` |
        | `speakers` | string | 4-bit code for active speakers, e.g. `0001` |
        | `x_position` | int | Object grid column (0-indexed) |
        | `y_position` | int | Object grid row (0-indexed) |
        | `image_dir` | string | Image subdirectory name under `image/` |

        ## `segmentation`

        Result of the overhead-image segmentation step.

        | Key | Type | Description |
        |-----|------|-------------|
        | `x_com` | float | X centre-of-mass of the mask in the cropped overhead image (pixels) |
        | `y_com` | float | Y centre-of-mass of the mask in the cropped overhead image (pixels) |
        | `status` | string | `completed` when segmentation succeeded |

        ## `experiment_config`

        Merged from the source `experiment_config.json` and hardware defaults.

        | Key | Type | Description |
        |-----|------|-------------|
        | `audio.file_name` | string | Excitation audio file name |
        | `audio.sample_rate_hz` | int | Audio sample rate (Hz) |
        | `audio.duration_s` | float | Chirp duration (s) |
        | `audio.generation.signal` | string | Signal type, e.g. `chirp` |
        | `audio.generation.f_start_hz` | int | Chirp start frequency (Hz) |
        | `audio.generation.f_end_hz` | int | Chirp end frequency (Hz) |
        | `recording.capture_seconds_requested` | float | Requested recording duration (s) |
        | `overhead_camera.frame_rate_fps` | int | Overhead camera frame rate (fps) |
        | `overhead_camera.exposure_ms` | int | Overhead camera exposure time (ms) |
        | `overhead_camera.gain` | int | Overhead camera sensor gain |
        | `laser_camera.capture.fps` | float | Laser camera frame rate (fps) |
        | `laser_camera.global_roi` | list[int] | Full-frame ROI `[x, y, w, h]` |
        | `laser_grid.sensor_rois_xywh` | list[list[int]] | Per-laser bounding boxes `[[x,y,w,h], ...]` |

        ## `experiment_output`

        Derived statistics computed during processing.

        | Key | Type | Description |
        |-----|------|-------------|
        | `overhead_camera.image_width` | int | Overhead image width (px) |
        | `overhead_camera.image_height` | int | Overhead image height (px) |
        | `laser_camera.max_frame_rate_hz` | int | Actual frame rate achieved |
        | `laser_camera.global_roi` | list[int] | Actual ROI used `[x, y, w, h]` |
        | `laser_grid.total_image_height` | int | Total laser camera frame height (px) |
        | `laser_grid.n_lasers` | int | Number of laser points detected |
        | `speckle_shifts.n_frames` | int | Number of frames captured |
        | `speckle_shifts.duration_s` | float | Actual recording duration (s) |

        ### 3.6 `processing_config`

        Parameters used for each processing stage.

        #### 3.6.1 Raw Data

        | Key | Type | Description |
        |-----|------|-------------|
        | `speckle_vibration_raw.format` | string | Storage format: `npy` or `npz` |
        | `speckle_vibration_raw.compressed` | bool | Whether the raw array is compressed |

        #### 3.6.2 Shift Extraction

        | Key | Type | Description |
        |-----|------|-------------|
        | `speckle_shifts.fs_hz` | float | Sampling rate of the shift signal (Hz) |

        #### 3.6.3 Filtering (`speckle_shifts_clean`)

        | Key | Type | Description |
        |-----|------|-------------|
        | `filter_type` | string | Filter design, e.g. `butterworth` |
        | `filter_mode` | string | `bandpass`, `lowpass`, or `highpass` |
        | `lowcut` | float | Low cutoff frequency (Hz) |
        | `highcut` | float | High cutoff frequency (Hz) |
        | `filter_order` | int | Filter order |
        | `hann_applied` | bool | Whether a Hann window was applied after filtering |
        | `apply_order` | string | `filter_then_hann` or `hann_then_filter` |

        #### 3.6.4 FFT (`speckle_shifts_fft`)

        | Key | Type | Description |
        |-----|------|-------------|
        | `fft_kind` | string | FFT variant, e.g. `rfft` |
        | `fft_axis` | int | Axis along which FFT is computed |
        | `min_freq` | float | Minimum frequency retained after crop (Hz) |
        | `max_freq` | float | Maximum frequency retained after crop (Hz) |
        | `dtype` | string | Complex dtype, e.g. `complex64` |
        | `crop_after_fft` | bool | Whether to crop to `[min_freq, max_freq]` |

        #### 3.6.5 Audio Preview (`speckle_shifts_ifft_audio`)

        | Key | Type | Description |
        |-----|------|-------------|
        | `laser_idx` | int | Which laser (0-indexed) to use for the audio preview |
        | `xy_idx` | int | Which shift channel: `0` = X, `1` = Y |
        | `method` | string | Reconstruction method, e.g. `ifft` |
        | `output_sample_rate_hz` | int | Output WAV sample rate (Hz) |
        | `normalization` | string | Normalization method, e.g. `peak_to_int16` |
        | `output_dtype` | string | Output sample dtype, e.g. `int16` |
        | `zero_fill_uncropped_bins` | bool | Whether to zero-fill frequency bins outside crop range |

        #### 3.6.6 Video Preview (`speckle_vibrations_preview`)

        | Key | Type | Description |
        |-----|------|-------------|
        | `max_frames` | int | Maximum frames in the preview video |
        | `max_width` | int | Maximum video width (px) |
        | `codec` | string | Video codec, e.g. `libx264` |
        | `crf` | int | Constant rate factor — lower = higher quality |
        | `pixelformat` | string | Pixel format, e.g. `yuv420p` |
        | `burn_frame_index` | bool | Whether the frame index is burned into the video |
        | `preserve_physical_duration` | bool | Whether playback speed matches real time |

        #### 3.6.7 Segmentation

        | Key | Type | Description |
        |-----|------|-------------|
        | `segmentation.left` | float | Left crop fraction of the overhead image |
        | `segmentation.right` | float | Right crop fraction |
        | `segmentation.up` | float | Top crop fraction |
        | `segmentation.down` | float | Bottom crop fraction |
        | `segmentation.prompt` | string | Text prompt passed to the segmentation model |

        ### 3.7 `artifacts`

        Relative paths within the repo to every file produced for this sample.

        | Key | File | Description |
        |-----|------|-------------|
        | `raw_overhead` | `image/<dir>/raw_overhead.png` | Full overhead photo before cropping |
        | `cropped_overhead` | `image/<dir>/cropped_overhead.png` | Overhead cropped to box region |
        | `segmented_overhead` | `image/<dir>/segmented_overhead.png` | Overhead with mask + speaker overlay |
        | `mask_png` | `image/<dir>/mask.png` | Binary segmentation mask (PNG) |
        | `mask_npz` | `image/<dir>/mask.npz` | Binary mask as compressed numpy array |
        | `audio` | `audio/chirp_50_1000_3.0sec.wav` | Shared excitation chirp (all samples) |
        | `speckle_vibration_raw` | `data/<id>/speckle_vibration_raw.npy` | Raw laser camera frames |
        | `speckle_vibrations` | `data/<id>/speckle_vibrations.mp4` | Slow-motion speckle preview video |
        | `speckle_shifts` | `data/<id>/speckle_shifts.npz` | Sub-pixel XY shifts per laser per frame |
        | `speckle_shifts_clean` | `data/<id>/speckle_shifts_clean.npz` | Filtered + windowed shifts |
        | `speckle_shifts_fft` | `data/<id>/speckle_shifts_fft.npz` | FFT of cleaned shifts |
        | `speckle_shifts_ifft_audio` | `data/<id>/speckle_shifts_ifft_audio.wav` | Single-laser audio reconstruction |
        | `manifest` | `data/<id>/manifest.json` | This manifest file |
        """
    )


def to_int_list(values) -> list[int]:
    return [int(v) for v in values]


def to_int_nested_list(values) -> list[list[int]]:
    return [to_int_list(item) for item in values]


def reconstruct_selected_column_centers_x(sensor_rois_xywh: list[list[int]]) -> list[int]:
    if not sensor_rois_xywh:
        return []
    centers = []
    seen = set()
    for x, y, w, h in sensor_rois_xywh:
        center_x = int(x + w // 2)
        if center_x not in seen:
            seen.add(center_x)
            centers.append(center_x)
    return centers


def reconstruct_row_values_single_list(row_rois_y: list[list[int]]) -> list[int]:
    values = []
    for start, end in row_rois_y:
        values.extend(range(int(start // 2), int(end // 2)))
    return values


def read_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return int(image.width), int(image.height)


def build_experiment_output(source_dir: Path, metadata: dict | None, preview_info: dict, recovery_info: dict, clean_info: dict, fft_info: dict, ifft_audio_info: dict) -> dict:
    import numpy as np

    run_opt = extract_run_opt(metadata)
    cam_params = run_opt.get("cam_params", {})
    multi = run_opt.get("run_opt_multiROIs", {})

    raw_overhead_path = source_dir / "box_overhead_image.png"
    overhead_width, overhead_height = read_image_size(raw_overhead_path)

    global_roi = multi.get("global_ROI") or cam_params.get("get_global_roi")
    row_rois_y = multi.get("ROI_list") or []
    sensor_rois_xywh = multi.get("ROIs") or []
    total_image_height = multi.get("total_image_height")
    if total_image_height is None and global_roi:
        total_image_height = int(global_roi[3])

    raw_shape, raw_dtype_str, _ = inspect_npy_array(source_dir / "frame-recording.npy")
    frame_count, frame_height, frame_width = raw_shape
    preview_cfg = default_processing_config()["speckle_vibrations_preview"]
    preview_step = max(1, frame_count // int(preview_cfg["max_frames"]))
    preview_frame_count = int(len(list(range(0, frame_count, preview_step))))
    capture_duration_s = float(frame_count) / float(recovery_info["fs"])
    preview_fps = float(preview_frame_count) / capture_duration_s
    preview_width = int(preview_cfg["max_width"])
    scaled_preview_height = int(round(frame_height * (preview_width / frame_width))) if frame_width > preview_width else frame_height
    macro_block_size = int(preview_cfg["macro_block_size"])
    preview_height = int(np.ceil(scaled_preview_height / macro_block_size) * macro_block_size)
    preview_defaults = {
        "frame_count": int(frame_count),
        "frame_height": int(frame_height),
        "frame_width": int(frame_width),
        "preview_frame_count": preview_frame_count,
        "preview_fps": preview_fps,
        "preview_width": preview_width,
        "preview_height": preview_height,
    }
    preview_payload = {**preview_defaults, **preview_info}
    capture_seconds_requested = run_opt.get("N_CAPTURE_SECONDS")
    if capture_seconds_requested is None:
        capture_seconds_requested = run_opt.get("capture_seconds")
    if capture_seconds_requested is None:
        capture_seconds_requested = 3.1

    output = {
        "overhead_camera": {
            "image_width": overhead_width,
            "image_height": overhead_height,
        },
        "laser_camera": {
            "global_roi": to_int_list(global_roi) if global_roi else None,
            "max_frame_rate_hz": int(cam_params["get_max_frame_rate"]) if cam_params.get("get_max_frame_rate") is not None else None,
        },
        "laser_grid": {
            "total_image_height": int(total_image_height) if total_image_height is not None else None,
            "selected_row_points_image_xy": to_int_nested_list(multi.get("selected_row_points") or []),
            "selected_column_centers_x": reconstruct_selected_column_centers_x(to_int_nested_list(sensor_rois_xywh)),
            "row_values_single_list": reconstruct_row_values_single_list(to_int_nested_list(row_rois_y)),
            "global_crop_x": int(global_roi[0]) if global_roi else None,
            "global_crop_width": int(global_roi[2]) if global_roi else None,
            "global_crop_height": int(global_roi[3]) if global_roi else None,
            "row_rois_y": to_int_nested_list(row_rois_y),
            "sensor_grid_shape": [len(row_rois_y), len(reconstruct_selected_column_centers_x(to_int_nested_list(sensor_rois_xywh)))],
            "sensor_rois_xywh": to_int_nested_list(sensor_rois_xywh),
        },
        "speckle_vibrations": {
            **preview_payload,
            "capture_fps_hz": float(recovery_info["fs"]),
            "capture_seconds_requested": float(capture_seconds_requested),
            "capture_seconds_observed": float(preview_defaults["frame_count"]) / float(recovery_info["fs"]),
            "dtype": np.dtype(raw_dtype_str).name,
        },
        "speckle_shifts": recovery_info,
        "speckle_shifts_clean": clean_info,
        "speckle_shifts_fft": fft_info,
        "speckle_shifts_ifft_audio": ifft_audio_info,
    }
    return output


def infer_audio_local_path(experiment_config: dict) -> Path | None:
    return CANONICAL_AUDIO_LOCAL_PATH if CANONICAL_AUDIO_LOCAL_PATH.exists() else None


def remote_read_json(source_dir: str, file_name: str) -> dict:
    source_path = f"{source_dir}/{file_name}"
    cmd = f"bash -lc {shlex.quote(f'cat {shlex.quote(source_path)}')}"
    out = subprocess.run(["ssh", REMOTE_HOST, cmd], check=True, text=True, capture_output=True)
    return json.loads(out.stdout)


def remote_read_bytes(remote_path: str) -> bytes:
    cmd = f"bash -lc {shlex.quote(f'cat {shlex.quote(remote_path)}')}"
    out = subprocess.run(["ssh", REMOTE_HOST, cmd], check=True, capture_output=True)
    return out.stdout


def _local_md5(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _remote_md5(remote_path: str) -> str | None:
    cmd = "bash -lc " + shlex.quote(f'md5sum {shlex.quote(remote_path)} 2>/dev/null || true')
    out = subprocess.run(["ssh", REMOTE_HOST, cmd], capture_output=True, text=True)
    line = out.stdout.strip()
    return line.split()[0] if line else None


def sync_file_to_remote(local_path: Path, remote_path: str, label: str) -> None:
    t0 = time.perf_counter()
    local_hash = _local_md5(local_path)
    remote_hash = _remote_md5(remote_path)
    if local_hash == remote_hash:
        print(f"[timing] {label}: skipped (unchanged)", flush=True)
        return
    with local_path.open("rb") as handle:
        remote_cmd = "bash -lc " + shlex.quote(f'mkdir -p "$(dirname {remote_path})" && cat > "{remote_path}"')
        subprocess.run(["ssh", REMOTE_HOST, remote_cmd], check=True, stdin=handle)
    print(f"[timing] {label}: {time.perf_counter() - t0:.2f}s", flush=True)


def maybe_sync_modal_config_to_remote() -> bool:
    modal_config_path = Path("~/.modal.toml").expanduser()
    if not modal_config_path.exists():
        return False
    sync_file_to_remote(modal_config_path, REMOTE_MODAL_CONFIG_PATH, "sync Modal config to mcluster11")
    return True


def fetch_remote_sample_context(new_dir: str, experiment_id: str, sample_id: int | None) -> dict:
    lookup_expr = f"int(row.get('sample_id', -1)) == {sample_id}" if sample_id is not None else f"row.get('experiment_id') == {experiment_id!r}"
    code = textwrap.dedent(
        f"""
        import json
        from pathlib import Path

        new_dir = Path({new_dir!r})
        rows = [json.loads(line) for line in (new_dir / 'data' / 'metadata.jsonl').read_text().splitlines() if line.strip()]
        matches = [row for row in rows if {lookup_expr}]
        if not matches:
            raise SystemExit('No matching sample found in metadata.jsonl')
        row = sorted(matches, key=lambda item: int(item['sample_id']))[-1]
        sample_dir = "%07d" % int(row['sample_id'])
        manifest_path = new_dir / 'data' / sample_dir / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        image_dir = manifest['sample']['image_dir']
        print(json.dumps({{
            'sample_id': int(row['sample_id']),
            'sample_dir': sample_dir,
            'image_dir': image_dir,
            'image_root': str(new_dir / 'image' / image_dir),
            'manifest_path': str(manifest_path),
            'metadata_path': str(new_dir / 'data' / 'metadata.jsonl'),
            'raw_overhead_path': str(new_dir / 'image' / image_dir / 'raw_overhead.png'),
            'object': manifest['sample']['object'],
            'box_material': manifest['sample']['box_material'],
        }}))
        """
    )
    cmd = "bash -lc " + shlex.quote(f"python3 -c {shlex.quote(code)}")
    out = subprocess.run(["ssh", REMOTE_HOST, cmd], check=True, text=True, capture_output=True)
    return json.loads(out.stdout)


def crop(left: float, right: float, up: float, down: float, raw_overhead_path: Path, cropped_overhead_path: Path) -> Path:
    from PIL import Image

    with Image.open(raw_overhead_path) as image:
        width, height = image.size
        cropped = image.crop((int(width * left), int(height * up), int(width * right), int(height * down)))
        cropped.save(cropped_overhead_path)
    return cropped_overhead_path


def segment(cropped_overhead_path: Path, object_name: str, prompt: str | None, mask_path: Path) -> Path:
    code = textwrap.dedent(
        f"""
        import numpy as np
        from PIL import Image
        from utils.segment import app, segment

        cropped_overhead_path = {str(cropped_overhead_path)!r}
        object_name = {object_name!r}
        prompt = {prompt!r}
        mask_path = {str(mask_path)!r}
        mask_png_path = {str(mask_path.with_suffix('.png'))!r}

        image_array = np.asarray(Image.open(cropped_overhead_path).convert('RGB'), dtype=np.uint8)
        with app.run():
            mask, _ = segment.remote(image_array, object_name, 'cardboard', prompt)
        mask = np.asarray(mask, dtype=np.float32)
        np.savez_compressed(mask_path, mask=mask)
        Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8), mode='L').save(mask_png_path)
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(REPO_ROOT))
    return mask_path


def center_of_mass(mask_path: Path) -> tuple[float | None, float | None]:
    import numpy as np

    payload = np.load(mask_path, allow_pickle=True)
    mask = np.asarray(payload["mask"], dtype=np.float64)
    total = float(mask.sum())
    if total <= 0:
        return None, None
    ys, xs = np.indices(mask.shape)
    return float((xs * mask).sum() / total), float((ys * mask).sum() / total)


def overlay(cropped_overhead_path: Path, mask_path: Path, x_com: float | None, y_com: float | None, segmented_overhead_path: Path) -> Path:
    import numpy as np
    from PIL import Image, ImageDraw

    cropped = np.asarray(Image.open(cropped_overhead_path).convert("RGB"), dtype=np.uint8)
    payload = np.load(mask_path, allow_pickle=True)
    mask = np.asarray(payload["mask"], dtype=np.float32)
    alpha = np.clip(mask, 0.0, 1.0)[..., None] * 0.5
    tint = np.zeros_like(cropped, dtype=np.float32)
    tint[..., 1] = 204.0
    blended = (cropped.astype(np.float32) * (1.0 - alpha) + tint * alpha).astype(np.uint8)
    image = Image.fromarray(blended)
    if x_com is not None and y_com is not None:
        draw = ImageDraw.Draw(image)
        radius = 20
        draw.line([(x_com - radius, y_com), (x_com + radius, y_com)], fill=(255, 0, 0), width=4)
        draw.line([(x_com, y_com - radius), (x_com, y_com + radius)], fill=(255, 0, 0), width=4)
    image.save(segmented_overhead_path)
    return segmented_overhead_path


def apply_speaker_overlay(img_path: Path, speakers: str) -> None:
    from PIL import Image

    inner = Image.open(img_path).convert("RGB")
    inner_width, inner_height = inner.size

    target_icon_height = int(inner_height * 0.40)
    speaker_icon = Image.open(SPEAKER_DIR / "speaker.png").convert("RGBA")
    orig_w, orig_h = speaker_icon.size
    target_icon_width = int(orig_w * target_icon_height / orig_h)

    padded_width = inner_width + target_icon_width
    padded_height = inner_height + target_icon_height
    canvas = Image.new("RGB", (padded_width, padded_height), PADDED_BG)

    x = target_icon_width // 2
    y = target_icon_height // 2
    canvas.paste(inner, (x, y))

    if "1" in speakers:
        composite = canvas.convert("RGBA")
        for bit, key, idx in zip(speakers, SPEAKER_FILES, range(4)):
            if bit == "1":
                icon_orig = Image.open(SPEAKER_DIR / "speaker.png").convert("RGBA")
                icon_scaled = icon_orig.resize((target_icon_width, target_icon_height), Image.LANCZOS)
                if key == "1000":
                    px, py = 0, padded_height // 2 - icon_scaled.height // 2
                elif key == "0100":
                    px, py = padded_width // 3 - icon_scaled.width // 2, padded_height - icon_scaled.height
                elif key == "0010":
                    px, py = 2 * padded_width // 3 - icon_scaled.width // 2, padded_height - icon_scaled.height
                elif key == "0001":
                    px, py = padded_width - icon_scaled.width, padded_height // 2 - icon_scaled.height // 2
                else:
                    px, py = 0, 0
                composite.alpha_composite(icon_scaled, (px, py))
        canvas = composite.convert("RGB")

    canvas.save(img_path)


def finalize_remote_segmentation(new_dir: str, sample_context: dict, hf_repo: str, x_com: float | None, y_com: float | None) -> None:
    manifest = remote_read_bytes(sample_context["manifest_path"]).decode("utf-8")
    manifest_payload = json.loads(manifest)
    image_dir = sample_context["image_dir"]
    sample_dir = sample_context["sample_dir"]
    manifest_payload["segmentation"] = {
        "x_com": x_com,
        "y_com": y_com,
        "status": "completed",
    }
    manifest_payload["artifacts"]["cropped_overhead"] = f"image/{image_dir}/cropped_overhead.png"
    manifest_payload["artifacts"]["segmented_overhead"] = f"image/{image_dir}/segmented_overhead.png"
    manifest_payload["artifacts"]["mask_png"] = f"image/{image_dir}/mask.png"
    manifest_payload["artifacts"]["mask_npz"] = f"image/{image_dir}/mask.npz"

    metadata_rows = [json.loads(line) for line in remote_read_bytes(sample_context["metadata_path"]).decode("utf-8").splitlines() if line.strip()]
    target_sample_id = int(sample_context["sample_id"])
    for row in metadata_rows:
        if int(row.get("sample_id", -1)) != target_sample_id:
            continue
        row["x_com"] = x_com
        row["y_com"] = y_com
        row["segmented_overhead_file_name"] = hf_file_url(hf_repo, f"image/{image_dir}/segmented_overhead.png")
        row["mask_file_name"] = hf_file_url(hf_repo, f"image/{image_dir}/mask.png")
        row["manifest"] = json.dumps(manifest_payload, ensure_ascii=True)

    with tempfile.TemporaryDirectory(prefix="segment-finalize-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        manifest_path = tmp_root / "manifest.json"
        metadata_path = tmp_root / "metadata.jsonl"
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        metadata_path.write_text("\n".join(json.dumps(normalize_metadata_row(row), ensure_ascii=True) for row in metadata_rows) + "\n", encoding="utf-8")
        sync_file_to_remote(manifest_path, sample_context["manifest_path"], "sync manifest with segmentation")
        sync_file_to_remote(metadata_path, sample_context["metadata_path"], "sync metadata with segmentation")


def run_segmentation_pipeline(args: argparse.Namespace) -> None:
    sample_context = run(
        "discover remote sample context",
        lambda: fetch_remote_sample_context(args.new_dir, args.source_dir_name, args.sample_id),
    )
    with tempfile.TemporaryDirectory(prefix="segment-sample-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        raw_overhead_path = tmp_root / "raw_overhead.png"
        raw_overhead_path.write_bytes(run("fetch raw_overhead.png", lambda: remote_read_bytes(sample_context["raw_overhead_path"])))
        cropped_overhead_path = run(
            "crop overhead image",
            lambda: crop(args.left, args.right, args.up, args.down, raw_overhead_path, tmp_root / "cropped_overhead.png"),
        )
        mask_path = run(
            "segment cropped overhead",
            lambda: segment(cropped_overhead_path, sample_context["object"], args.prompt, tmp_root / "mask.npz"),
        )
        x_com, y_com = run("compute mask center of mass", lambda: center_of_mass(mask_path))
        segmented_overhead_path = run(
            "render segmentation overlay",
            lambda: overlay(cropped_overhead_path, mask_path, x_com, y_com, tmp_root / "segmented_overhead.png"),
        )
        remote_image_root = sample_context["image_root"]
        sync_file_to_remote(cropped_overhead_path, f"{remote_image_root}/cropped_overhead.png", "sync cropped_overhead.png")
        sync_file_to_remote(mask_path.with_suffix(".png"), f"{remote_image_root}/mask.png", "sync mask.png")
        sync_file_to_remote(mask_path, f"{remote_image_root}/mask.npz", "sync mask.npz")
        sync_file_to_remote(segmented_overhead_path, f"{remote_image_root}/segmented_overhead.png", "sync segmented_overhead.png")
        run(
            "finalize manifest and metadata with segmentation",
            lambda: finalize_remote_segmentation(args.new_dir, sample_context, args.hf_repo, x_com, y_com),
        )


def maybe_sync_hf_token_to_remote() -> bool:
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if not token_path.exists():
        return False
    run(
        "prepare remote Hugging Face token path",
        lambda: subprocess.run(
            ["ssh", REMOTE_HOST, f'bash -lc {shlex.quote(f"mkdir -p {shlex.quote(str(Path(REMOTE_HF_TOKEN_PATH).parent))}") }'],
            check=True,
        ),
    )
    sync_file_to_remote(token_path, REMOTE_HF_TOKEN_PATH, "sync Hugging Face token to mcluster11")
    return True


def launch_remote_upload(args: argparse.Namespace) -> None:
    sync_file_to_remote(Path(__file__), REMOTE_SCRIPT_PATH, "sync migration script to mcluster11")
    maybe_sync_hf_token_to_remote()
    remote_cmd = (
        "bash -lc "
        + shlex.quote(
            f"set -euo pipefail; "
            f"{REMOTE_UV_INSTALL}; "
            f"{REMOTE_UV} python install 3.12; "
            f"if [ ! -x {REMOTE_VENV}/bin/python ]; then {REMOTE_UV} venv --python 3.12 {REMOTE_VENV}; fi; "
            f"{REMOTE_UV} pip install --python {REMOTE_VENV}/bin/python huggingface_hub >/dev/null; "
            f"{REMOTE_VENV}/bin/python {REMOTE_SCRIPT_PATH} --new-dir {shlex.quote(args.new_dir)} --hf-repo {shlex.quote(args.hf_repo)} --upload-to-hf --remote-worker"
        )
    )
    run("remote upload experiment-16 to Hugging Face", lambda: subprocess.run(["ssh", REMOTE_HOST, remote_cmd], check=True))


def launch_remote(args: argparse.Namespace) -> None:
    source_dir = f"{args.old_dir.rstrip('/')}/{args.source_dir_name}"
    experiment_config = run("fetch remote experiment_config.json", lambda: remote_read_json(source_dir, "experiment_config.json"))
    audio_local_path = infer_audio_local_path(experiment_config)
    if audio_local_path is not None:
        print(f"[info] local audio file found: {audio_local_path}", flush=True)
    else:
        print("[info] local audio file not found; audio field will stay blank unless provided later", flush=True)

    sync_file_to_remote(Path(__file__), REMOTE_SCRIPT_PATH, "sync migration script to mcluster11")
    sync_file_to_remote(REPO_ROOT / "pyproject.toml", "/home/ethantu/pyproject.toml", "sync pyproject.toml to mcluster11")
    sync_file_to_remote(REPO_ROOT / "uv.lock", "/home/ethantu/uv.lock", "sync uv.lock to mcluster11")
    sync_file_to_remote(REPO_ROOT / "utils" / "segment.py", "/home/ethantu/utils/segment.py", "sync utils/segment.py to mcluster11")
    for speaker_file in SPEAKER_FILES:
        sync_file_to_remote(SPEAKER_DIR / f"{speaker_file}.png", f"/home/ethantu/assets/speakers/{speaker_file}.png", f"sync speaker asset {speaker_file}.png to mcluster11")
    maybe_sync_modal_config_to_remote()
    if audio_local_path is not None:
        sync_file_to_remote(audio_local_path, REMOTE_AUDIO_PATH, "sync shared audio to mcluster11")

    remote_args = [
        "--remote-worker",
        "--old-dir", args.old_dir,
        "--new-dir", args.new_dir,
        "--hf-repo", args.hf_repo,
        "--source-dir-name", args.source_dir_name,
        "--stage", args.stage,
    ]
    if args.sample_id is not None:
        remote_args.extend(["--sample-id", str(args.sample_id)])
    if audio_local_path is not None:
        remote_args.extend(["--remote-audio-path", REMOTE_AUDIO_PATH])
    if args.overwrite:
        remote_args.append("--overwrite")
    if args.compress_raw:
        remote_args.append("--compress-raw")

    remote_cmd = (
        "bash -lc "
        + shlex.quote(
            f"set -euo pipefail; "
            f"{REMOTE_UV_INSTALL}; "
            f"{REMOTE_UV} python install 3.12; "
            f"if [ ! -x {REMOTE_VENV}/bin/python ]; then {REMOTE_UV} venv --python 3.12 {REMOTE_VENV}; fi; "
            f"{REMOTE_UV} pip install --python {REMOTE_VENV}/bin/python numpy==2.2.6 scipy==1.16.2 opencv-python-headless==4.12.0.88 pillow==11.3.0 imageio-ffmpeg==0.6.0 modal>=1.3.5 >/dev/null; "
            f"{REMOTE_VENV}/bin/python {REMOTE_SCRIPT_PATH} {' '.join(shlex.quote(a) for a in remote_args)}"
        )
    )
    run("remote migrate sample into experiment-16", lambda: subprocess.run(["ssh", REMOTE_HOST, remote_cmd], check=True))


def _filtered_metadata_jsonl(experiment_dir: Path, hf_repo: str, api) -> bytes | None:
    """Return metadata.jsonl content filtered to rows whose image dirs are on HF.

    Called after upload_large_folder so the committed file list is current.
    Returns None if nothing to upload.
    """
    metadata_path = experiment_dir / "data" / "metadata.jsonl"
    if not metadata_path.exists():
        return None

    existing = set(api.list_repo_files(hf_repo, repo_type="dataset"))

    def repo_path_from_url(url: str) -> str | None:
        marker = "/resolve/main/"
        if not isinstance(url, str) or marker not in url:
            return None
        return url.split(marker, 1)[1]

    def row_paths(row: dict) -> list[str]:
        paths = []
        for key in [
            "segmented_overhead_file_name",
            "mask_file_name",
            "speckle_vibrations_file_name",
            "speckle_shifts_ifft_audio_file_name",
            "audio_file_name",
        ]:
            path = repo_path_from_url(row.get(key, ""))
            if path:
                paths.append(path)
        return paths

    rows = []
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        required_paths = row_paths(row)
        if required_paths and all(path in existing for path in required_paths):
            rows.append(line.strip())

    if not rows:
        return None
    print(f"[hf] filtered metadata.jsonl: {len(rows)} rows with committed images", flush=True)
    return ("\n".join(rows) + "\n").encode()


def _partial_sample_ignore_patterns(experiment_dir: Path) -> list[str]:
    """Return ignore patterns for sample dirs that exist on disk but are not in metadata.jsonl."""
    meta_path = experiment_dir / "data" / "metadata.jsonl"
    if not meta_path.exists():
        return []
    completed = {"%07d" % int(json.loads(l)["sample_id"]) for l in meta_path.read_text().splitlines() if l.strip()}
    partial = [
        d.name for d in (experiment_dir / "data").iterdir()
        if d.is_dir() and d.name.isdigit() and d.name not in completed
    ]
    if partial:
        print(f"[hf] ignoring {len(partial)} partial sample dirs not in metadata.jsonl: {partial}", flush=True)
    return [f"data/{d}/**" for d in partial]


def upload_experiment_dir_to_hf(experiment_dir: Path, hf_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(hf_repo, repo_type="dataset", exist_ok=True)

    partial_ignore = _partial_sample_ignore_patterns(experiment_dir)

    # Phase 1a: upload everything except the large raw .npy files first
    api.upload_large_folder(
        folder_path=str(experiment_dir),
        repo_id=hf_repo,
        repo_type="dataset",
        num_workers=8,
        ignore_patterns=["**/speckle_vibration_raw.npy"] + partial_ignore,
    )

    # Phase 1b: upload raw .npy files in a second pass
    api.upload_large_folder(
        folder_path=str(experiment_dir),
        repo_id=hf_repo,
        repo_type="dataset",
        num_workers=8,
        allow_patterns=["**/speckle_vibration_raw.npy"],
        ignore_patterns=partial_ignore,
    )

    # Phase 2: overwrite metadata.jsonl with a version filtered to only rows
    # whose image dirs are committed on HF, so the viewer never hits missing files.
    filtered = _filtered_metadata_jsonl(experiment_dir, hf_repo, api)
    if filtered:
        api.upload_file(
            path_or_fileobj=filtered,
            path_in_repo="data/metadata.jsonl",
            repo_id=hf_repo,
            repo_type="dataset",
            commit_message="Update metadata.jsonl to committed samples only",
        )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_load_npz(path: Path):
    import numpy as np

    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def next_sample_id(metadata_path: Path, sample_root: Path) -> int:
    max_id = 0
    if metadata_path.exists():
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            max_id = max(max_id, int(row.get("sample_id", 0)))
    if sample_root.exists():
        for child in sample_root.iterdir():
            if child.is_dir() and child.name.isdigit():
                max_id = max(max_id, int(child.name))
    return max_id + 1


def sample_dir_name(sample_id: int) -> str:
    return f"{sample_id:07d}"


def load_saved_artifact_infos(sample_root: Path) -> tuple[dict, dict, dict, dict]:
    import numpy as np
    import scipy.io.wavfile

    shifts = np.load(sample_root / "speckle_shifts.npz")
    recovery_info = {"fs": float(shifts["fs"]), "shape": list(shifts["shifts"].shape)}

    clean = np.load(sample_root / "speckle_shifts_clean.npz")
    clean_info = {"shape": list(clean["shifts_clean"].shape), "fs": float(clean["fs"])}

    fft = np.load(sample_root / "speckle_shifts_fft.npz")
    fft_info = {"shape": list(fft["fft"].shape), "fs": float(fft["fs"]), "n_samples": int(fft["n_samples"])}

    sr, wav_data = scipy.io.wavfile.read(sample_root / "speckle_shifts_ifft_audio.wav")
    ifft_audio_info = {"sample_rate_hz": int(sr), "n_frames": int(len(wav_data))}

    return recovery_info, clean_info, fft_info, ifft_audio_info


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def ensure_scaffold(new_dir: Path) -> None:
    (new_dir / "audio").mkdir(parents=True, exist_ok=True)
    (new_dir / "image").mkdir(parents=True, exist_ok=True)
    (new_dir / "data").mkdir(parents=True, exist_ok=True)

    readme = new_dir / "README.md"
    readme.write_text(build_dataset_readme(), encoding="utf-8")

    metadata_path = new_dir / "data" / "metadata.jsonl"
    if not metadata_path.exists():
        metadata_path.write_text("", encoding="utf-8")

    exp_cfg = new_dir / "data" / "base_experiment_config.json"
    exp_cfg.write_text(json.dumps(notebook_default_experiment_config(), indent=2), encoding="utf-8")

    proc_cfg = new_dir / "data" / "base_processing_config.json"
    proc_cfg.write_text(json.dumps(default_processing_config(), indent=2), encoding="utf-8")

    legacy_exp_cfg = new_dir / "base_experiment_config.json"
    if legacy_exp_cfg.exists():
        legacy_exp_cfg.unlink()

    legacy_proc_cfg = new_dir / "base_processing_config.json"
    if legacy_proc_cfg.exists():
        legacy_proc_cfg.unlink()


def inspect_source(source_dir: Path) -> dict:
    files = {
        "experiment_config": source_dir / "experiment_config.json",
        "frame_recording_npy": source_dir / "frame-recording.npy",
        "frame_recording_npz": source_dir / "frame-recording.npz",
        "recovery": source_dir / "RECOVERY.npz",
        "metadata": source_dir / "metadata.npz",
        "raw_overhead": source_dir / "box_overhead_image.png",
    }
    summary = {name: path.exists() for name, path in files.items()}
    summary["source_dir"] = str(source_dir)
    return summary


def save_raw_recording_artifact(source_npy: Path, dest_path: Path, compress_raw: bool, overwrite: bool) -> None:
    import shutil
    import zipfile

    if dest_path.exists() and not overwrite:
        return

    if compress_raw:
        with zipfile.ZipFile(dest_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(source_npy, arcname="frames.npy")
        return

    if dest_path.exists():
        dest_path.unlink()
    try:
        os.link(source_npy, dest_path)
    except OSError:
        # Cross-device: fall back to copy
        shutil.copy2(source_npy, dest_path)


def inspect_npy_array(path: Path) -> tuple[tuple[int, ...], str, int]:
    import numpy as np
    from numpy.lib import format as npy_format

    with path.open("rb") as handle:
        version = npy_format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported .npy version {version} for {path}")
        if fortran_order:
            raise ValueError(f"Fortran-order arrays are not supported for preview generation: {path}")
        return shape, np.dtype(dtype).str, handle.tell()


def mux_audio_into_video(video_path: Path, audio_path: Path, out_path: Path, target_duration_s: float) -> None:
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            f"[1:a]apad,atrim=0:{target_duration_s:.6f}[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def generate_speckle_preview(raw_npy_path: Path, out_path: Path, fps: float, overwrite: bool, audio_path: Path | None = None) -> dict:
    import cv2
    import imageio_ffmpeg
    import numpy as np
    import tempfile

    if out_path.exists() and not overwrite:
        return {"skipped": True}

    shape, dtype_str, data_offset = inspect_npy_array(raw_npy_path)
    if len(shape) != 3:
        raise ValueError(f"Expected raw recording shape (frames, height, width), got {shape}")
    frame_count, frame_height, frame_width = shape
    dtype = np.dtype(dtype_str)
    frame_values = frame_height * frame_width
    frame_bytes = frame_values * dtype.itemsize
    max_frames = 300
    max_width = 960
    macro_block_size = 16
    step = max(1, frame_count // max_frames)
    selected_indices = list(range(0, frame_count, step))
    capture_duration_s = float(frame_count) / float(fps)
    preview_fps = float(len(selected_indices)) / capture_duration_s
    preview_h, preview_w = frame_height, frame_width
    if frame_width > max_width:
        scale = max_width / frame_width
        preview_w = int(round(frame_width * scale))
        preview_h = int(round(frame_height * scale))
    preview_w_aligned = int(np.ceil(preview_w / macro_block_size) * macro_block_size)
    preview_h_aligned = int(np.ceil(preview_h / macro_block_size) * macro_block_size)

    def load_frame(handle, index: int) -> np.ndarray:
        handle.seek(data_offset + index * frame_bytes)
        raw = handle.read(frame_bytes)
        if len(raw) != frame_bytes:
            raise ValueError(f"Short read while loading frame {index} from {raw_npy_path}")
        return np.frombuffer(raw, dtype=dtype, count=frame_values).reshape((frame_height, frame_width))

    with raw_npy_path.open("rb") as handle:
        probe = np.asarray([load_frame(handle, idx) for idx in selected_indices[: min(len(selected_indices), 50)]])
    lo = float(np.percentile(probe, 5.0))
    hi = float(np.percentile(probe, 99.5))
    with tempfile.TemporaryDirectory(prefix="speckle-preview-") as tmp_dir:
        video_only_path = Path(tmp_dir) / "speckle_vibrations_video_only.mp4"
        writer = imageio_ffmpeg.write_frames(
        str(video_only_path),
        (preview_w_aligned, preview_h_aligned),
        fps=preview_fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        output_params=["-crf", "23"],
        )
        writer.send(None)
        try:
            with raw_npy_path.open("rb") as handle:
                for frame_index in selected_indices:
                    frame = load_frame(handle, frame_index)
                    frame_u8 = np.clip((frame.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
                    frame_u8 = (frame_u8 * 255).astype(np.uint8)
                    frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
                    if (preview_w, preview_h) != (frame_width, frame_height):
                        frame_bgr = cv2.resize(frame_bgr, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
                    if (preview_w_aligned, preview_h_aligned) != (preview_w, preview_h):
                        padded = np.zeros((preview_h_aligned, preview_w_aligned, 3), dtype=np.uint8)
                        padded[:preview_h, :preview_w] = frame_bgr
                        frame_bgr = padded
                    cv2.putText(frame_bgr, f"frame {frame_index}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    writer.send(frame_rgb.tobytes())
        finally:
            writer.close()

        if audio_path is not None and audio_path.exists():
            mux_audio_into_video(video_only_path, audio_path, out_path, capture_duration_s)
        else:
            shutil.move(str(video_only_path), str(out_path))

    return {
        "frame_count": int(frame_count),
        "frame_height": int(frame_height),
        "frame_width": int(frame_width),
        "preview_frame_count": int(len(selected_indices)),
        "preview_fps": float(preview_fps),
        "preview_width": int(preview_w_aligned),
        "preview_height": int(preview_h_aligned),
        "audio_embedded": bool(audio_path is not None and audio_path.exists()),
        "audio_start_seconds": 0.0 if audio_path is not None and audio_path.exists() else None,
        "audio_end_seconds": float(capture_duration_s) if audio_path is not None and audio_path.exists() else None,
    }


def normalize_recovery_npz(recovery_path: Path, metadata_npz_path: Path | None, dest_path: Path, overwrite: bool) -> dict:
    import numpy as np

    if dest_path.exists() and not overwrite:
        data = np.load(dest_path, allow_pickle=True)
        return {"fs": float(data["fs"]), "shape": list(data["shifts"].shape), "skipped": True}

    recovery = np.load(recovery_path, allow_pickle=True)
    shifts = None
    for key in ["shifts", "all_shifts"]:
        if key in recovery.files:
            shifts = np.asarray(recovery[key])
            break
    if shifts is None:
        raise KeyError(f"Could not find shifts array in {recovery_path}; keys={recovery.files}")

    fs = None
    for key in ["fs", "camera_FPS", "fps"]:
        if key in recovery.files:
            fs = float(np.asarray(recovery[key]).item())
            break
    if fs is None and metadata_npz_path is not None and metadata_npz_path.exists():
        metadata = np.load(metadata_npz_path, allow_pickle=True)
        if "run_opt" in metadata.files:
            run_opt = metadata["run_opt"].item()
            cam_params = run_opt.get("cam_params", {})
            fs = float(cam_params.get("get_frame_rate") or cam_params.get("camera_FPS") or 0)
    if not fs:
        fs = 2500.0

    np.savez_compressed(dest_path, shifts=shifts, fs=fs)
    return {"fs": float(fs), "shape": list(shifts.shape)}


def clean_shifts_file(source_path: Path, dest_path: Path, processing_config: dict, overwrite: bool) -> dict:
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    if dest_path.exists() and not overwrite:
        data = np.load(dest_path, allow_pickle=True)
        return {"shape": list(data["shifts_clean"].shape), "fs": float(data["fs"]), "skipped": True}

    payload = np.load(source_path, allow_pickle=True)
    shifts = np.asarray(payload["shifts"], dtype=np.float32)
    fs = float(np.asarray(payload["fs"]).item())
    cfg = processing_config["speckle_shifts_clean"]
    lowcut = float(cfg["lowcut"])
    highcut = float(cfg["highcut"])
    filter_order = int(cfg["filter_order"])
    hann_applied = bool(cfg["hann_applied"])

    sos = butter(filter_order, [lowcut, highcut], fs=fs, btype="band", output="sos")
    shifts_clean = np.empty_like(shifts, dtype=np.float32)
    for laser_idx in range(shifts.shape[0]):
        for xy_idx in range(shifts.shape[2]):
            shifts_clean[laser_idx, :, xy_idx] = sosfiltfilt(sos, shifts[laser_idx, :, xy_idx]).astype(np.float32, copy=False)

    if hann_applied:
        window = np.hanning(shifts.shape[1]).astype(np.float32, copy=False)
        shifts_clean *= window[None, :, None]

    np.savez_compressed(
        dest_path,
        shifts_clean=shifts_clean,
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        filter_order=filter_order,
        hann_applied=hann_applied,
    )
    return {"shape": list(shifts_clean.shape), "fs": fs}


def fft_shifts_file(source_path: Path, dest_path: Path, processing_config: dict, overwrite: bool) -> dict:
    import numpy as np

    if dest_path.exists() and not overwrite:
        data = np.load(dest_path, allow_pickle=True)
        return {"shape": list(data["fft"].shape), "fs": float(data["fs"]), "skipped": True}

    payload = np.load(source_path, allow_pickle=True)
    shifts_clean = np.asarray(payload["shifts_clean"], dtype=np.float32)
    fs = float(np.asarray(payload["fs"]).item())
    cfg = processing_config["speckle_shifts_fft"]
    min_freq = float(cfg["min_freq"])
    max_freq = float(cfg["max_freq"])
    n_samples = int(shifts_clean.shape[1])
    full_fft = np.fft.rfft(shifts_clean, axis=1).astype(np.complex64)
    full_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (full_freqs >= min_freq) & (full_freqs <= max_freq)
    fft = full_fft[:, mask, :]
    freqs = full_freqs[mask].astype(np.float32)
    np.savez_compressed(
        dest_path,
        fft=fft,
        freqs=freqs,
        fs=fs,
        min_freq=min_freq,
        max_freq=max_freq,
        n_samples=n_samples,
    )
    return {"shape": list(fft.shape), "fs": fs, "n_samples": n_samples}


def ifft_audio_preview_file(source_path: Path, dest_path: Path, processing_config: dict, overwrite: bool) -> dict:
    import wave
    import numpy as np
    from scipy.signal import resample

    if dest_path.exists() and not overwrite:
        with wave.open(str(dest_path), "rb") as wav_file:
            return {
                "sample_rate_hz": wav_file.getframerate(),
                "n_frames": wav_file.getnframes(),
                "skipped": True,
            }

    payload = np.load(source_path, allow_pickle=True)
    fft = np.asarray(payload["fft"], dtype=np.complex64)
    freqs = np.asarray(payload["freqs"], dtype=np.float32)
    fs = float(np.asarray(payload["fs"]).item())
    min_freq = float(np.asarray(payload["min_freq"]).item())
    max_freq = float(np.asarray(payload["max_freq"]).item())
    n_samples = int(np.asarray(payload["n_samples"]).item())

    cfg = processing_config["speckle_shifts_ifft_audio"]
    laser_idx = int(cfg["laser_idx"])
    xy_idx = int(cfg["xy_idx"])
    output_sample_rate_hz = int(cfg["output_sample_rate_hz"])

    full_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (full_freqs >= min_freq) & (full_freqs <= max_freq)
    if mask.sum() != fft.shape[1]:
        raise ValueError(f"FFT frequency mask mismatch: expected {fft.shape[1]} bins, got {int(mask.sum())}")

    spectrum = np.zeros(full_freqs.shape[0], dtype=np.complex64)
    spectrum[mask] = fft[laser_idx, :, xy_idx]
    signal = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)
    audio = resample(signal, int(round(output_sample_rate_hz * len(signal) / fs))).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    audio_i16 = (audio * 32767.0).astype(np.int16)

    with wave.open(str(dest_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(output_sample_rate_hz)
        wav_file.writeframes(audio_i16.tobytes())

    return {"sample_rate_hz": output_sample_rate_hz, "n_frames": int(len(audio_i16))}


DEFAULT_N_OBJECTS = 1
DEFAULT_BOX_MATERIAL = "cardboard"


def build_image_dir_name(
    source_dir_name: str,
    object_name: str | None = None,
    x_position: int | str | None = None,
    y_position: int | str | None = None,
    n_objects: int | None = None,
    box_material: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Canonical image directory name.

    Format: {object}-{x}x-{y}y-{n}obj-{material}[-tag...]-{timestamp}
    Sentinel placeholders when values are absent: OBJECT, POSx, POSy, Xobj, MATERIAL.
    """
    obj = normalize_token(object_name) if object_name else "OBJECT"
    mat = normalize_token(box_material) if box_material else "MATERIAL"
    ts = parse_source_timestamp(source_dir_name)

    if x_position is not None:
        x_tok = f"{int(x_position):03d}x" if isinstance(x_position, int) else f"{x_position}x"
    else:
        x_tok = "POSx"
    if y_position is not None:
        y_tok = f"{int(y_position):03d}y" if isinstance(y_position, int) else f"{y_position}y"
    else:
        y_tok = "POSy"

    n_tok = f"{int(n_objects)}obj" if n_objects is not None else "Xobj"
    tag_tokens = [normalize_token(t) for t in (tags or []) if t]

    return "-".join([obj, x_tok, y_tok, n_tok, mat, *tag_tokens, ts])


def append_metadata_row(metadata_path: Path, row: dict) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(normalize_metadata_row(row), ensure_ascii=True) + "\n")


def remote_worker(args: argparse.Namespace) -> None:
    import shutil

    old_dir = Path(args.old_dir)
    new_dir = Path(args.new_dir)
    ensure_safe_paths(old_dir, new_dir)

    source_dir = old_dir / args.source_dir_name
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    inventory = run("inspect source sample", lambda: inspect_source(source_dir))
    print(json.dumps(inventory, indent=2), flush=True)
    if args.stage == "discover":
        return

    run("init experiment-16 scaffold", lambda: ensure_scaffold(new_dir))
    if args.stage == "init":
        return

    source_x, source_y = parse_source_coordinates(args.source_dir_name)
    experiment_config = run("load experiment_config.json", lambda: read_json(source_dir / "experiment_config.json"))
    metadata_npz = run("load metadata.npz", lambda: maybe_load_npz(source_dir / "metadata.npz"))
    merged_experiment_config = run(
        "build experiment_config defaults + overrides",
        lambda: merge_experiment_config(notebook_default_experiment_config(), experiment_config, metadata_npz),
    )
    run_opt = extract_run_opt(metadata_npz)
    processing_config = default_processing_config()
    processing_config["speckle_shifts"]["fs_hz"] = float(
        run_opt.get("cam_params", {}).get("get_frame_rate")
        or merged_experiment_config["laser_camera"]["capture"]["fps"]
        or 2500.0
    )
    processing_config["speckle_vibrations_preview"]["source_capture_fps_hz"] = processing_config["speckle_shifts"]["fs_hz"]
    processing_config["speckle_vibration_raw"] = {
        "format": "npz" if args.compress_raw else "npy",
        "compressed": bool(args.compress_raw),
    }
    processing_config["segmentation"] = {
        "left": float(args.left),
        "right": float(args.right),
        "up": float(args.up),
        "down": float(args.down),
        "prompt": args.prompt,
    }

    sample_id = args.sample_id or run(
        "allocate sample_id",
        lambda: next_sample_id(new_dir / "data" / "metadata.jsonl", new_dir / "data"),
    )
    sample_dir = sample_dir_name(sample_id)
    sample_root = new_dir / "data" / sample_dir
    sample_root.mkdir(parents=True, exist_ok=True)

    speakers = parse_speakers_from_source_dir(args.source_dir_name)
    object_name = normalize_token(args.source_dir_name.split("-")[0])
    n_objects = DEFAULT_N_OBJECTS
    box_material = DEFAULT_BOX_MATERIAL
    image_dir = build_image_dir_name(
        args.source_dir_name,
        object_name=object_name,
        x_position=source_x,
        y_position=source_y,
        n_objects=n_objects,
        box_material=box_material,
    )
    image_root = new_dir / "image" / image_dir
    image_root.mkdir(parents=True, exist_ok=True)

    audio_rel = CANONICAL_AUDIO_REL
    audio_path_for_preview = None
    if args.remote_audio_path:
        audio_name = CANONICAL_AUDIO_FILE_NAME
        audio_dest = new_dir / "audio" / audio_name
        run(
            "copy shared audio",
            lambda: shutil.copy2(args.remote_audio_path, audio_dest) if (args.overwrite or not audio_dest.exists()) else None,
        )
        audio_rel = relative_posix(audio_dest, new_dir)
        audio_path_for_preview = audio_dest
    merged_experiment_config["audio"]["file_name"] = audio_rel
    merged_experiment_config["audio"]["sample_rate_hz"] = CANONICAL_AUDIO_SAMPLE_RATE_HZ
    merged_experiment_config["audio"]["duration_s"] = CANONICAL_AUDIO_DURATION_S
    merged_experiment_config["audio"]["total_output_channels"] = CANONICAL_AUDIO_TOTAL_OUTPUT_CHANNELS
    merged_experiment_config["audio"]["wav_channels"] = CANONICAL_AUDIO_WAV_CHANNELS
    merged_experiment_config["audio"]["sample_width_bytes"] = CANONICAL_AUDIO_SAMPLE_WIDTH_BYTES
    merged_experiment_config["audio"]["generation"] = {
        "signal": "chirp",
        "method": CANONICAL_AUDIO_GENERATION_METHOD,
        "chirp_duration_s": CANONICAL_AUDIO_CHIRP_DURATION_S,
        "silence_start_s": CANONICAL_AUDIO_SILENCE_START_S,
        "silence_end_s": CANONICAL_AUDIO_SILENCE_END_S,
        "f_start_hz": CANONICAL_AUDIO_F_START_HZ,
        "f_end_hz": CANONICAL_AUDIO_F_END_HZ,
        "output_dtype": CANONICAL_AUDIO_OUTPUT_DTYPE,
        "normalization": CANONICAL_AUDIO_NORMALIZATION,
    }

    # Determine which raw artifact name exists (needed for manifest in both stages)
    raw_artifact_name = (
        "speckle_vibration_raw.npz"
        if (sample_root / "speckle_vibration_raw.npz").exists()
        else "speckle_vibration_raw.npy"
    )

    if args.stage != "post_segment":
        raw_npy_path = source_dir / "frame-recording.npy"
        raw_artifact_name = "speckle_vibration_raw.npz" if args.compress_raw else "speckle_vibration_raw.npy"
        raw_artifact_path = sample_root / raw_artifact_name
        legacy_raw_artifact_path = sample_root / ("speckle_vibration_raw.npy" if args.compress_raw else "speckle_vibration_raw.npz")
        if args.overwrite and legacy_raw_artifact_path.exists():
            run("remove legacy raw artifact", lambda: legacy_raw_artifact_path.unlink())
        run(
            f"package {raw_artifact_name}",
            lambda: save_raw_recording_artifact(raw_npy_path, raw_artifact_path, args.compress_raw, args.overwrite),
        )

        preview_info = run(
            "generate speckle_vibrations.mp4",
            lambda: generate_speckle_preview(
                raw_npy_path,
                sample_root / "speckle_vibrations.mp4",
                fps=float(processing_config["speckle_shifts"]["fs_hz"]),
                overwrite=args.overwrite,
                audio_path=audio_path_for_preview,
            ),
        )

        recovery_info = run(
            "normalize RECOVERY.npz -> speckle_shifts.npz",
            lambda: normalize_recovery_npz(
                source_dir / "RECOVERY.npz",
                source_dir / "metadata.npz",
                sample_root / "speckle_shifts.npz",
                args.overwrite,
            ),
        )
        processing_config["speckle_shifts"]["fs_hz"] = recovery_info["fs"]

        clean_info = run(
            "compute speckle_shifts_clean.npz",
            lambda: clean_shifts_file(
                sample_root / "speckle_shifts.npz",
                sample_root / "speckle_shifts_clean.npz",
                processing_config,
                args.overwrite,
            ),
        )

        fft_info = run(
            "compute speckle_shifts_fft.npz",
            lambda: fft_shifts_file(
                sample_root / "speckle_shifts_clean.npz",
                sample_root / "speckle_shifts_fft.npz",
                processing_config,
                args.overwrite,
            ),
        )

        ifft_audio_info = run(
            "generate speckle_shifts_ifft_audio.wav",
            lambda: ifft_audio_preview_file(
                sample_root / "speckle_shifts_fft.npz",
                sample_root / "speckle_shifts_ifft_audio.wav",
                processing_config,
                args.overwrite,
            ),
        )

        run(
            "copy raw_overhead.png",
            lambda: shutil.copy2(source_dir / "box_overhead_image.png", image_root / "raw_overhead.png")
            if (args.overwrite or not (image_root / "raw_overhead.png").exists())
            else None,
        )

        run(
            "crop overhead image",
            lambda: crop(args.left, args.right, args.up, args.down, image_root / "raw_overhead.png", image_root / "cropped_overhead.png"),
        )

    if args.stage == "pre_segment":
        print(f"[done] pre_segment sample_id={sample_id} image_root={image_root}", flush=True)
        return

    if args.stage == "post_segment":
        recovery_info, clean_info, fft_info, ifft_audio_info = run(
            "load artifact infos from saved files",
            lambda: load_saved_artifact_infos(sample_root),
        )
        preview_info = {}

    cropped_overhead_path = image_root / "cropped_overhead.png"
    mask_path = image_root / "mask.npz"

    if args.stage not in ("post_segment",):
        mask_path = run(
            "segment cropped overhead",
            lambda: segment(cropped_overhead_path, object_name, args.prompt, image_root / "mask.npz"),
        )

    x_com, y_com = run("compute mask center of mass", lambda: center_of_mass(mask_path))
    run(
        "render segmentation overlay",
        lambda: overlay(cropped_overhead_path, mask_path, x_com, y_com, image_root / "segmented_overhead.png"),
    )
    run(
        "apply speaker overlay to segmented_overhead.png",
        lambda: apply_speaker_overlay(image_root / "segmented_overhead.png", speakers),
    )

    experiment_output = run(
        "build experiment_output payload",
        lambda: build_experiment_output(
            source_dir,
            metadata_npz,
            preview_info,
            recovery_info,
            clean_info,
            fft_info,
            ifft_audio_info,
        ),
    )

    manifest = run(
        "build manifest.json payload",
        lambda: {
            "sample_id": int(sample_id),
            "experiment_id": args.source_dir_name,
            "experiment_dir": new_dir.name,
            "source_experiment_id": args.source_dir_name,
            "source_experiment_dir": str(source_dir),
            "hf_repo": args.hf_repo,
            "sample": {
                "object": object_name,
                "n_objects": n_objects,
                "box_material": box_material,
                "speakers": speakers,
                "x_position": int(source_x) if source_x is not None else None,
                "y_position": int(source_y) if source_y is not None else None,
                "image_dir": image_dir,
            },
            "segmentation": {
                "x_com": x_com,
                "y_com": y_com,
                "status": "completed",
            },
            "experiment_config": merged_experiment_config,
            "experiment_output": experiment_output,
            "processing_config": processing_config,
            "artifacts": {
                "raw_overhead": f"image/{image_dir}/raw_overhead.png",
                "cropped_overhead": f"image/{image_dir}/cropped_overhead.png",
                "segmented_overhead": f"image/{image_dir}/segmented_overhead.png",
                "mask_png": f"image/{image_dir}/mask.png",
                "mask_npz": f"image/{image_dir}/mask.npz",
                "audio": audio_rel or None,
                "speckle_vibration_raw": f"data/{sample_dir}/{raw_artifact_name}",
                "speckle_vibrations": f"data/{sample_dir}/speckle_vibrations.mp4",
                "speckle_shifts": f"data/{sample_dir}/speckle_shifts.npz",
                "speckle_shifts_clean": f"data/{sample_dir}/speckle_shifts_clean.npz",
                "speckle_shifts_fft": f"data/{sample_dir}/speckle_shifts_fft.npz",
                "speckle_shifts_ifft_audio": f"data/{sample_dir}/speckle_shifts_ifft_audio.wav",
                "manifest": f"data/{sample_dir}/manifest.json",
            },
        },
    )

    run(
        "write manifest.json",
        lambda: (sample_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8"),
    )

    metadata_row = run(
        "build metadata.jsonl row",
        lambda: build_metadata_row(
            hf_repo=args.hf_repo,
            sample_id=sample_id,
            experiment_id=args.source_dir_name,
            speakers=speakers,
            x_position=source_x,
            y_position=source_y,
            x_com=x_com,
            y_com=y_com,
            n_objects=n_objects,
            box_material=box_material,
            experiment_dir=new_dir.name,
            segmented_overhead_path=f"image/{image_dir}/segmented_overhead.png",
            speckle_vibrations_path=f"data/{sample_dir}/speckle_vibrations.mp4",
            speckle_shifts_ifft_audio_path=f"data/{sample_dir}/speckle_shifts_ifft_audio.wav",
            audio_path=audio_rel,
            mask_path=f"image/{image_dir}/mask.png",
            manifest=manifest,
        ),
    )
    run("append metadata.jsonl", lambda: append_metadata_row(new_dir / "data" / "metadata.jsonl", metadata_row))

    print(f"[done] sample_id={sample_id} source_dir={source_dir}", flush=True)
    print(f"[info] sample_root={sample_root}", flush=True)
    print(f"[info] image_root={image_root}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate one sample from experiment-15 into experiment-16 via mcluster11.")
    parser.add_argument("--old-dir", default=DEFAULT_OLD_DIR)
    parser.add_argument("--new-dir", default=DEFAULT_NEW_DIR)
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--source-dir-name", default=None, help="Example: cube-00x01y_0001--31-03-18-21-24")
    parser.add_argument("--sample-id", type=int, default=None)
    parser.add_argument("--stage", choices=["all", "discover", "init", "pre_segment", "post_segment"], default="all")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compress-raw", action="store_true")
    parser.add_argument("--left", type=float, default=DEFAULT_SEGMENT_LEFT)
    parser.add_argument("--right", type=float, default=DEFAULT_SEGMENT_RIGHT)
    parser.add_argument("--up", type=float, default=DEFAULT_SEGMENT_UP)
    parser.add_argument("--down", type=float, default=DEFAULT_SEGMENT_DOWN)
    parser.add_argument("--prompt", default=DEFAULT_SEGMENT_PROMPT)
    parser.add_argument("--upload-to-hf", action="store_true")
    parser.add_argument("--remote-worker", action="store_true")
    parser.add_argument("--remote-audio-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.upload_to_hf:
        if not Path(args.new_dir).exists() and not args.remote_worker:
            launch_remote_upload(args)
            return
        upload_experiment_dir_to_hf(Path(args.new_dir), args.hf_repo)
        return
    if not args.source_dir_name:
        raise SystemExit("--source-dir-name is required unless --upload-to-hf is set")
    if args.remote_worker:
        remote_worker(args)
        return
    launch_remote(args)


if __name__ == "__main__":
    main()
