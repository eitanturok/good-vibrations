import argparse
import io
import json
import re
import wave
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image as PILImage
from PIL import ImageDraw
from scipy.io.wavfile import write as wav_write
from scipy.signal import butter, resample, sosfiltfilt


OLD_REPO_ID = "eturok-weizmann/vibrations"
NEW_REPO_ID = "eturok-weizmann/laser-vibrations"
REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
SPEAKER_DIR = ASSETS_DIR / "speakers"
SPEAKER_FILES = ("1000", "0100", "0010", "0001")
PADDED_BG = (232, 232, 232)
DEFAULT_STAGING_ROOT = REPO_ROOT / "tmp"
DEFAULT_INVENTORY_CSV = REPO_ROOT / "recovery_inventory_from_failures.csv"
BASE_EXPERIMENT_CONFIG_PATH = REPO_ROOT / "hf_data" / "base_experiment_config.json"
BASE_PROCESSING_CONFIG_PATH = REPO_ROOT / "hf_data" / "base_processing_config.json"
DEFAULT_AUDIO_ROOT = REPO_ROOT / "data" / "audio_samples"


class SkipSample(Exception):
    pass


def log(message: str) -> None:
    print(f"[info] {message}", flush=True)


def sample_dir_rel(sample_id: int) -> str:
    return f"data/{sample_id:07d}"


def hf_file_url(repo_id: str, repo_path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{repo_path}"


def repo_path_from_hf_url(url: str | None) -> str | None:
    marker = "/resolve/main/"
    if not isinstance(url, str) or marker not in url:
        return None
    return url.split(marker, 1)[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_jsonl_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n"
    path.write_text(body, encoding="utf-8")


def load_or_init_metadata_rows(staging_root: Path, repo_id: str) -> list[dict]:
    metadata_path = staging_root / "data" / "metadata.jsonl"
    if metadata_path.exists():
        return load_jsonl_rows(metadata_path)
    remote_path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="data/metadata.jsonl"))
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_bytes(remote_path.read_bytes())
    return load_jsonl_rows(metadata_path)


def load_old_sample_row(sample_id: int, repo_id: str) -> dict:
    columns = [
        "sample_idx",
        "object",
        "n_objects",
        "speakers",
        "box_material",
        "x_position",
        "y_position",
        "fps",
        "raw_image",
        "cropped_image",
        "overlay_image",
    ]
    ds = load_dataset(repo_id, split="train", columns=columns, verification_mode="no_checks")
    matches = [dict(row) for row in ds if int(row["sample_idx"]) == int(sample_id)]
    if len(matches) != 1:
        raise SkipSample(
            f"expected exactly 1 row for sample_id={sample_id} in {repo_id}, found {len(matches)}"
        )
    return matches[0]


def download_old_sample_npz(sample_id: int, repo_id: str) -> dict:
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f"data/sample_{sample_id:06d}.npz")
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def read_inventory_row(path: Path, sample_id: int) -> dict:
    df = pd.read_csv(path)
    row_df = df.loc[pd.to_numeric(df["sample_id"], errors="coerce") == int(sample_id)]
    if len(row_df) != 1:
        raise SkipSample(f"expected exactly 1 inventory row for sample_id={sample_id} in {path}, found {len(row_df)}")
    return row_df.iloc[0].to_dict()


def parse_source_dir_name(source_dir_name: str) -> dict:
    name = str(source_dir_name)
    lower = name.lower()

    xy_match = re.search(r"(?P<x>\d{1,2})x(?P<y>\d{1,2})y", lower)
    spk_match = re.search(r"_(?P<speakers>[01]{4})--", name)
    ts_match = re.search(r"--(?P<timestamp>\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$", name)

    x_position = int(xy_match.group("x")) if xy_match else None
    y_position = int(xy_match.group("y")) if xy_match else None
    speakers = spk_match.group("speakers") if spk_match else None
    source_timestamp = ts_match.group("timestamp") if ts_match else None
    prefix = re.sub(r"_[01]{4}--.*$", "", name)
    prefix = re.sub(r"--.*$", "", prefix)

    is_empty = lower.startswith("empty")
    is_offgrid = "offrid" in lower or "offgrid" in lower
    is_rotated = "rotated" in lower

    if is_empty:
        sample_type = "empty"
        n_objects = 0
        object_name = "empty"
    elif "four-" in lower:
        sample_type = "multi_object_four"
        n_objects = 4
        object_name = "cube"
    elif "two-" in lower or "stacked" in lower or "together" in lower or "apart" in lower:
        sample_type = "multi_object_two"
        n_objects = 2
        object_name = "cube"
    elif is_offgrid and is_rotated:
        sample_type = "offgrid_rotated"
        n_objects = 1
        object_name = "cube"
    elif is_offgrid:
        sample_type = "offgrid"
        n_objects = 1
        object_name = "cube"
    elif "cube-l" in lower or lower.startswith("cube-l_"):
        sample_type = "shape_l"
        n_objects = 5
        object_name = "cube"
    elif "zigzag" in lower:
        sample_type = "zigzag"
        n_objects = 1
        object_name = "cube"
    elif x_position is not None and y_position is not None:
        sample_type = "grid_single"
        n_objects = 1
        object_name = "cube"
    else:
        sample_type = "unknown"
        n_objects = None
        object_name = "cube"

    return {
        "source_dir_name": source_dir_name,
        "source_prefix": prefix,
        "source_timestamp": source_timestamp,
        "x_position": x_position,
        "y_position": y_position,
        "speakers": speakers,
        "sample_type": sample_type,
        "n_objects": n_objects,
        "object": object_name,
    }


def normalize_token(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return text.strip("-") or "unknown"


def image_dir_from_match(row: dict) -> str | None:
    repo_path = repo_path_from_hf_url(row.get("segmented_overhead_file_name"))
    if not repo_path:
        return None
    parts = Path(repo_path).parts
    if len(parts) < 3 or parts[0] != "image":
        return None
    return parts[1]


def timestamp_from_source_dir(source_dir_name: str, fallback_year: int) -> str:
    match = re.search(r"--(?P<d>\d{2})-(?P<m>\d{2})-(?P<h>\d{2})-(?P<mi>\d{2})-(?P<s>\d{2})$", str(source_dir_name))
    if not match:
        return f"{fallback_year:04d}-01-01-00-00-00"
    day = int(match.group("d"))
    month = int(match.group("m"))
    hour = int(match.group("h"))
    minute = int(match.group("mi"))
    second = int(match.group("s"))
    return f"{fallback_year:04d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}"


def build_image_dir_name(object_: str, x_position: int | None, y_position: int | None, n_objects: int, box_material: str, source_dir_name: str, fallback_year: int) -> str:
    x_part = f"{int(x_position):03d}x" if x_position is not None else "POSx"
    y_part = f"{int(y_position):03d}y" if y_position is not None else "POSy"
    return "-".join(
        [
            normalize_token(object_),
            x_part,
            y_part,
            f"{int(n_objects)}obj",
            normalize_token(box_material),
            timestamp_from_source_dir(source_dir_name, fallback_year),
        ]
    )


def find_existing_image_dir(metadata_rows: list[dict], object_: str, n_objects: int, box_material: str, x_position: int | None, y_position: int | None) -> str | None:
    if x_position is None or y_position is None:
        return None
    for row in metadata_rows:
        try:
            if int(row.get("x_position")) != int(x_position) or int(row.get("y_position")) != int(y_position):
                continue
            if int(row.get("n_objects")) != int(n_objects):
                continue
        except Exception:
            continue
        if str(row.get("object", "")).lower() != str(object_).lower():
            continue
        if str(row.get("box_material", "")).lower() != str(box_material).lower():
            continue
        image_dir = image_dir_from_match(row)
        if image_dir:
            return image_dir
    return None


def image_to_pil(image) -> PILImage.Image:
    if isinstance(image, PILImage.Image):
        return image
    if isinstance(image, np.ndarray):
        return PILImage.fromarray(image)
    if isinstance(image, dict):
        if image.get("bytes"):
            return PILImage.open(io.BytesIO(image["bytes"]))
        if image.get("path"):
            return PILImage.open(image["path"])
    raise TypeError(f"unsupported image type: {type(image)!r}")


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = image_to_pil(image)
    if pil.mode not in ("RGB", "RGBA", "L"):
        pil = pil.convert("RGB")
    pil.save(path)


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mask)
    if arr.dtype != np.uint8:
        arr = (arr > 0).astype(np.uint8) * 255
    PILImage.fromarray(arr, mode="L").save(path)


def build_segmented_overhead(cropped_image, mask: np.ndarray, x_com: float | None, y_com: float | None) -> PILImage.Image:
    cropped = np.asarray(image_to_pil(cropped_image).convert("RGB"), dtype=np.uint8)
    mask_arr = np.asarray(mask, dtype=np.float32)
    alpha = np.clip(mask_arr, 0.0, 1.0)[..., None] * 0.5
    tint = np.zeros_like(cropped, dtype=np.float32)
    tint[..., 1] = 204.0
    blended = (cropped.astype(np.float32) * (1.0 - alpha) + tint * alpha).astype(np.uint8)
    image = PILImage.fromarray(blended)
    if x_com is not None and y_com is not None:
        draw = ImageDraw.Draw(image)
        radius = 20
        draw.line([(x_com - radius, y_com), (x_com + radius, y_com)], fill=(255, 0, 0), width=4)
        draw.line([(x_com, y_com - radius), (x_com, y_com + radius)], fill=(255, 0, 0), width=4)
    return image


def apply_speaker_overlay(img: PILImage.Image, speakers: str) -> PILImage.Image:
    inner = img.convert("RGB")
    inner_width, inner_height = inner.size

    speaker_icon = PILImage.open(SPEAKER_DIR / "1000.png").convert("RGBA")
    target_icon_height = int(inner_height * 0.40)
    orig_w, orig_h = speaker_icon.size
    target_icon_width = int(orig_w * target_icon_height / orig_h)

    padded_width = inner_width + target_icon_width
    padded_height = inner_height + target_icon_height
    canvas = PILImage.new("RGB", (padded_width, padded_height), PADDED_BG)

    x = target_icon_width // 2
    y = target_icon_height // 2
    canvas.paste(inner, (x, y))

    if "1" in speakers:
        composite = canvas.convert("RGBA")
        for bit, key in zip(speakers, SPEAKER_FILES):
            if bit != "1":
                continue
            icon_path = SPEAKER_DIR / f"{key}.png"
            icon = PILImage.open(icon_path).convert("RGBA")
            icon = icon.resize((target_icon_width, target_icon_height), PILImage.LANCZOS)
            if key == "1000":
                px, py = 0, padded_height // 2 - icon.height // 2
            elif key == "0100":
                px, py = padded_width // 3 - icon.width // 2, padded_height - icon.height
            elif key == "0010":
                px, py = 2 * padded_width // 3 - icon.width // 2, padded_height - icon.height
            elif key == "0001":
                px, py = padded_width - icon.width, padded_height // 2 - icon.height // 2
            else:
                px, py = 0, 0
            composite.alpha_composite(icon, (px, py))
        canvas = composite.convert("RGB")
    return canvas


def save_mask_npz(
    path: Path,
    mask: np.ndarray,
    left: float,
    right: float,
    up: float,
    down: float,
    prompt: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mask=np.asarray(mask),
        left=float(left),
        right=float(right),
        up=float(up),
        down=float(down),
        prompt=prompt,
    )


def mask_center_of_mass(mask: np.ndarray) -> tuple[float | None, float | None]:
    arr = np.asarray(mask, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:
        return None, None
    ys, xs = np.indices(arr.shape)
    x_com = float((xs * arr).sum() / total)
    y_com = float((ys * arr).sum() / total)
    return x_com, y_com


def save_shifts_npz(path: Path, shifts: np.ndarray, fs: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, shifts=shifts, fs=fs)


def butterworth_filter(shifts: np.ndarray, fs: float, lowcut: float, highcut: float, filter_order: int) -> np.ndarray:
    if highcut is None:
        highcut = fs / 2 - 10
    sos = butter(filter_order, [lowcut, highcut], fs=fs, btype="band", output="sos")
    out = np.empty_like(shifts)
    for laser_idx in range(shifts.shape[0]):
        for xy_idx in range(shifts.shape[2]):
            out[laser_idx, :, xy_idx] = sosfiltfilt(sos, shifts[laser_idx, :, xy_idx])
    return out


def apply_hann_window(shifts: np.ndarray) -> np.ndarray:
    window = np.hanning(shifts.shape[1]).astype(shifts.dtype, copy=False)
    return shifts * window[None, :, None]


def clean_shifts(shifts: np.ndarray, fs: float, lowcut: float, highcut: float, filter_order: int, hann_applied: bool) -> np.ndarray:
    cleaned = butterworth_filter(shifts, fs=fs, lowcut=lowcut, highcut=highcut, filter_order=filter_order)
    if hann_applied:
        cleaned = apply_hann_window(cleaned)
    return cleaned


def save_clean_shifts_npz(path: Path, shifts_clean: np.ndarray, fs: float, lowcut: float, highcut: float, filter_order: int, hann_applied: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shifts_clean=shifts_clean,
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        filter_order=filter_order,
        hann_applied=hann_applied,
    )


def shifts_to_fft(shifts_clean: np.ndarray, fs: float, min_freq: float, max_freq: float) -> tuple[np.ndarray, np.ndarray, int]:
    n_samples = shifts_clean.shape[1]
    full_fft = np.fft.rfft(shifts_clean, axis=1).astype(np.complex64)
    full_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (full_freqs >= min_freq) & (full_freqs <= max_freq)
    return full_fft[:, mask, :], full_freqs[mask], n_samples


def save_fft_npz(path: Path, fft: np.ndarray, freqs: np.ndarray, fs: float, min_freq: float, max_freq: float, n_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fft=fft, freqs=freqs, fs=fs, min_freq=min_freq, max_freq=max_freq, n_samples=n_samples)


def generate_fft_audio_preview(fft: np.ndarray, fs: float, n_samples: int, min_freq: float, max_freq: float, out_path: Path, laser_idx: int, xy_idx: int, out_sr: int) -> None:
    full_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (full_freqs >= min_freq) & (full_freqs <= max_freq)
    spectrum = np.zeros(full_freqs.shape[0], dtype=np.complex64)
    spectrum[mask] = fft[laser_idx, :, xy_idx]
    signal = np.fft.irfft(spectrum, n=n_samples)
    audio = resample(signal, int(out_sr * len(signal) / fs))
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    wav_write(out_path, out_sr, (audio * 32767).astype(np.int16))


def read_wav_metadata(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as wav_f:
        return int(wav_f.getframerate()), int(wav_f.getnframes())


def build_recovery_metadata(metadata_rows: list[dict], repo_id: str) -> dict:
    first_audio = next((row.get("audio_file_name") for row in metadata_rows if row.get("audio_file_name")), None)
    if first_audio:
        audio_repo_path = repo_path_from_hf_url(first_audio)
        if audio_repo_path:
            return {
                "audio_file_name": first_audio,
                "audio_repo_path": audio_repo_path,
            }
    base_config = load_json(BASE_EXPERIMENT_CONFIG_PATH)
    audio_rel = str(base_config["audio"]["file_name"])
    return {
        "audio_file_name": hf_file_url(repo_id, audio_rel),
        "audio_repo_path": audio_rel,
    }


def build_manifest(sample_id: int, repo_id: str, experiment_dir: str, experiment_id: str, source_dir_name: str, object_: str, n_objects: int, box_material: str, speakers: str, x_position: int | None, y_position: int | None, image_dir: str, x_com: float | None, y_com: float | None, fs: float, shifts: np.ndarray, shifts_clean: np.ndarray, fft: np.ndarray, n_samples: int, wav_path: Path, audio_repo_path: str, raw_overhead_available: bool, cropped_overhead_available: bool, segmented_overhead_available: bool, mask_available: bool, processing_config: dict, experiment_config: dict, raw_image_size: tuple[int, int] | None, cropped_image_size: tuple[int, int] | None, mask_shape: tuple[int, ...]) -> dict:
    wav_sr, wav_frames = read_wav_metadata(wav_path)
    sample_dir = sample_dir_rel(sample_id)
    return {
        "sample_id": int(sample_id),
        "experiment_id": experiment_id,
        "experiment_dir": experiment_dir,
        "source_experiment_id": experiment_id,
        "source_experiment_dir": None,
        "hf_repo": repo_id,
        "recovery_mode": "historical_from_vibrations_only",
        "recovery_notes": {
            "source_dataset": OLD_REPO_ID,
            "missing_original_assets": ["speckle_vibration_raw", "speckle_vibrations"],
            "canonical_fields_from": "recovery_inventory_source_dir_name",
            "shift_source": f"vibrations:data/sample_{sample_id:06d}.npz[shifts]",
            "mask_source": f"vibrations:data/sample_{sample_id:06d}.npz[mask]",
        },
        "sample": {
            "object": object_,
            "n_objects": int(n_objects),
            "box_material": box_material,
            "speakers": speakers,
            "x_position": int(x_position) if x_position is not None else None,
            "y_position": int(y_position) if y_position is not None else None,
            "image_dir": image_dir,
        },
        "segmentation": {
            "x_com": x_com,
            "y_com": y_com,
            "status": "recovered_from_old_mask",
        },
        "experiment_config": experiment_config,
        "experiment_output": {
            "overhead_camera": {
                "image_width": int(raw_image_size[0]) if raw_image_size else None,
                "image_height": int(raw_image_size[1]) if raw_image_size else None,
                "cropped_image_width": int(cropped_image_size[0]) if cropped_image_size else None,
                "cropped_image_height": int(cropped_image_size[1]) if cropped_image_size else None,
            },
            "laser_camera": {
                "global_roi": None,
                "max_frame_rate_hz": None,
            },
            "laser_grid": {
                "total_image_height": None,
                "selected_row_points_image_xy": None,
                "selected_column_centers_x": None,
                "row_values_single_list": None,
                "global_crop_x": None,
                "global_crop_width": None,
                "global_crop_height": None,
                "row_rois_y": None,
                "sensor_grid_shape": None,
                "sensor_rois_xywh": None,
            },
            "speckle_vibrations": {
                "available": False,
                "frame_count": None,
                "frame_height": None,
                "frame_width": None,
                "capture_seconds": None,
                "preview_fps": None,
                "dtype": None,
            },
            "speckle_shifts": {
                "fs": float(fs),
                "shape": [int(v) for v in shifts.shape],
            },
            "speckle_shifts_clean": {
                "fs": float(fs),
                "shape": [int(v) for v in shifts_clean.shape],
            },
            "speckle_shifts_fft": {
                "fs": float(fs),
                "shape": [int(v) for v in fft.shape],
                "n_samples": int(n_samples),
            },
            "speckle_shifts_ifft_audio": {
                "sample_rate_hz": wav_sr,
                "n_frames": wav_frames,
            },
            "segmentation": {
                "mask_shape": [int(v) for v in mask_shape],
            },
        },
        "processing_config": processing_config,
        "artifacts": {
            "raw_overhead": f"image/{image_dir}/raw_overhead.png" if raw_overhead_available else None,
            "cropped_overhead": f"image/{image_dir}/cropped_overhead.png" if cropped_overhead_available else None,
            "segmented_overhead": f"image/{image_dir}/segmented_overhead.png" if segmented_overhead_available else None,
            "mask_png": f"image/{image_dir}/mask.png" if mask_available else None,
            "mask_npz": f"image/{image_dir}/mask.npz" if mask_available else None,
            "audio": audio_repo_path,
            "speckle_vibration_raw": None,
            "speckle_vibrations": None,
            "speckle_shifts": f"{sample_dir}/speckle_shifts.npz",
            "speckle_shifts_clean": f"{sample_dir}/speckle_shifts_clean.npz",
            "speckle_shifts_fft": f"{sample_dir}/speckle_shifts_fft.npz",
            "speckle_shifts_ifft_audio": f"{sample_dir}/speckle_shifts_ifft_audio.wav",
            "manifest": f"{sample_dir}/manifest.json",
        },
    }


def build_metadata_row(sample_id: int, repo_id: str, experiment_id: str, speakers: str, x_position: int | None, y_position: int | None, x_com: float | None, y_com: float | None, object_: str, n_objects: int, box_material: str, experiment_dir: str, image_dir: str, audio_file_name: str, manifest: dict) -> dict:
    sample_dir = sample_dir_rel(sample_id)
    return {
        "sample_id": int(sample_id),
        "segmented_overhead_file_name": hf_file_url(repo_id, f"image/{image_dir}/segmented_overhead.png") if image_dir else None,
        "speckle_vibrations_file_name": None,
        "speckle_shifts_ifft_audio_file_name": hf_file_url(repo_id, f"{sample_dir}/speckle_shifts_ifft_audio.wav"),
        "audio_file_name": audio_file_name,
        "experiment_id": experiment_id,
        "speakers": speakers,
        "x_position": int(x_position) if x_position is not None else None,
        "y_position": int(y_position) if y_position is not None else None,
        "x_com": x_com,
        "y_com": y_com,
        "object": object_,
        "n_objects": int(n_objects),
        "box_material": box_material,
        "mask_file_name": hf_file_url(repo_id, f"image/{image_dir}/mask.png") if image_dir else None,
        "experiment_dir": experiment_dir,
        "manifest": json.dumps(manifest, ensure_ascii=True),
    }


def upsert_metadata_row(rows: list[dict], row: dict) -> list[dict]:
    kept = [payload for payload in rows if int(payload.get("sample_id", -1)) != int(row["sample_id"])]
    kept.append(row)
    kept.sort(key=lambda item: int(item["sample_id"]))
    return kept


def validate_recovery_target(sample_id: int, metadata_rows: list[dict], inventory_row: dict, old_row: dict, allow_unsafe: bool) -> None:
    matches = [row for row in metadata_rows if int(row.get("sample_id", -1)) == int(sample_id)]
    if len(matches) > 1:
        raise SkipSample(
            f"expected at most 1 metadata row for sample_id={sample_id} in laser-vibrations, found {len(matches)}"
        )
    if len(matches) == 1:
        raise SkipSample(f"sample_id={sample_id} is already present in laser-vibrations metadata")
    if allow_unsafe:
        return
    if inventory_row is None:
        raise SkipSample(f"sample_id={sample_id} is not present in backfill failures inventory")
    if old_row is None:
        raise SkipSample(f"sample_id={sample_id} is not present in vibrations")


def parse_boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage one historical laser-vibrations recovery sample from vibrations only.")
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument(
        "--staging-root",
        default=str(DEFAULT_STAGING_ROOT),
        help="Local directory to accumulate staged files before a later batched upload",
    )
    parser.add_argument("--inventory-csv", default=str(DEFAULT_INVENTORY_CSV))
    parser.add_argument("--old-repo-id", default=OLD_REPO_ID)
    parser.add_argument("--new-repo-id", default=NEW_REPO_ID)
    parser.add_argument("--experiment-dir", default="experiment-16")
    parser.add_argument("--left", type=float, default=0.15)
    parser.add_argument("--right", type=float, default=0.67)
    parser.add_argument("--up", type=float, default=0.08)
    parser.add_argument("--down", type=float, default=0.7)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--lowcut", type=float, default=50.0)
    parser.add_argument("--highcut", type=float, default=1000.0)
    parser.add_argument("--filter-order", type=int, default=5)
    parser.add_argument("--hann-applied", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-freq", type=float, default=50.0)
    parser.add_argument("--max-freq", type=float, default=1000.0)
    parser.add_argument("--fft-audio-laser-idx", type=int, default=50)
    parser.add_argument("--fft-audio-xy-idx", type=int, default=0)
    parser.add_argument("--fft-audio-out-sr", type=int, default=22050)
    parser.add_argument("--allow-unsafe", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    staging_root = Path(args.staging_root)
    log(f"starting historical recovery staging for sample_id={args.sample_id}")
    log(f"staging_root={staging_root}")
    try:
        log(f"loading inventory row from {args.inventory_csv}")
        inventory_row = read_inventory_row(Path(args.inventory_csv), args.sample_id)
        inventory_row["already_in_new_dataset"] = parse_boolish(inventory_row.get("already_in_new_dataset"))
        inventory_row["old_row_found_by_sample_id"] = parse_boolish(
            inventory_row.get("old_row_found_by_sample_id", inventory_row.get("old_row_found"))
        )

        source_dir_name = str(inventory_row.get("source_dir_name"))
        log(f"inventory source_dir_name={source_dir_name}")
        parsed = parse_source_dir_name(source_dir_name)
        if not parsed["speakers"]:
            raise SkipSample(f"could not parse speakers from source_dir_name={source_dir_name!r}")

        log(f"loading old sample row from {args.old_repo_id}")
        old_row = load_old_sample_row(args.sample_id, args.old_repo_id)
        log(f"downloading old sample npz from {args.old_repo_id}")
        old_npz = download_old_sample_npz(args.sample_id, args.old_repo_id)
        log(f"loading or initializing staged metadata under {staging_root / 'data' / 'metadata.jsonl'}")
        metadata_rows = load_or_init_metadata_rows(staging_root, args.new_repo_id)
        log(f"loaded {len(metadata_rows)} metadata rows for duplicate/reuse checks")
        validate_recovery_target(
            args.sample_id,
            metadata_rows,
            inventory_row=inventory_row,
            old_row=old_row,
            allow_unsafe=args.allow_unsafe,
        )
        log("sample passed recovery target validation")
        recovery_meta = build_recovery_metadata(metadata_rows, args.new_repo_id)
        log(f"resolved shared audio path={recovery_meta['audio_repo_path']}")
    except SkipSample as exc:
        print(f"[skip] sample_id={args.sample_id}: {exc}")
        return

    object_ = str(parsed["object"] or old_row.get("object") or "cube").lower()
    n_objects = int(parsed["n_objects"] or old_row.get("n_objects") or 1)
    box_material = str(old_row.get("box_material") or "cardboard").lower()
    x_position = int(parsed["x_position"]) if parsed["x_position"] is not None else None
    y_position = int(parsed["y_position"]) if parsed["y_position"] is not None else None
    speakers = str(parsed["speakers"])
    experiment_id = source_dir_name
    fallback_year = int(str(inventory_row.get("timestamp", datetime.now().isoformat()))[:4])
    log(
        "parsed canonical fields: "
        f"object={object_} n_objects={n_objects} speakers={speakers} "
        f"x_position={x_position} y_position={y_position}"
    )

    shifts = np.asarray(old_npz["shifts"])
    mask = np.asarray(old_npz["mask"])
    fs = float(old_row.get("fps") or load_json(BASE_EXPERIMENT_CONFIG_PATH)["laser_camera"]["capture"]["fps"])
    log(f"loaded shifts shape={shifts.shape} dtype={shifts.dtype} fs={fs}")
    log(f"loaded mask shape={mask.shape} dtype={mask.dtype}")

    image_dir = find_existing_image_dir(metadata_rows, object_, n_objects, box_material, x_position, y_position)
    image_was_reused = image_dir is not None
    if image_dir is None:
        image_dir = build_image_dir_name(object_, x_position, y_position, n_objects, box_material, source_dir_name, fallback_year)
        log(f"no reusable image dir found; will generate new image assets at image/{image_dir}")
    else:
        log(f"reusing existing image dir image/{image_dir}")

    x_com, y_com = mask_center_of_mass(mask)
    log(f"computed mask center of mass x_com={x_com} y_com={y_com}")
    image_root = staging_root / "image" / image_dir
    if not image_was_reused:
        log("generating segmented_overhead.png from cropped_image + mask + speakers")
        segmented_overhead = apply_speaker_overlay(
            build_segmented_overhead(old_row["cropped_image"], mask, x_com, y_com),
            speakers,
        )
        log(f"writing shared image assets under {image_root}")
        save_image(image_root / "raw_overhead.png", old_row["raw_image"])
        save_image(image_root / "cropped_overhead.png", old_row["cropped_image"])
        save_image(image_root / "segmented_overhead.png", segmented_overhead)
        save_mask_png(image_root / "mask.png", mask)
        save_mask_npz(image_root / "mask.npz", mask, args.left, args.right, args.up, args.down, args.prompt)

    raw_image_size = image_to_pil(old_row["raw_image"]).size if old_row.get("raw_image") is not None else None
    cropped_image_size = image_to_pil(old_row["cropped_image"]).size if old_row.get("cropped_image") is not None else None
    log("computing cleaned shifts")
    shifts_clean = clean_shifts(shifts, fs, args.lowcut, args.highcut, args.filter_order, args.hann_applied)
    log(f"cleaned shifts shape={shifts_clean.shape} dtype={shifts_clean.dtype}")
    log("computing FFT")
    fft, freqs, n_samples = shifts_to_fft(shifts_clean, fs, args.min_freq, args.max_freq)
    log(f"fft shape={fft.shape} dtype={fft.dtype} n_samples={n_samples} n_freqs={len(freqs)}")

    sample_root = staging_root / sample_dir_rel(args.sample_id)
    sample_root.mkdir(parents=True, exist_ok=True)
    log(f"writing sample artifacts under {sample_root}")
    save_shifts_npz(sample_root / "speckle_shifts.npz", shifts, fs)
    save_clean_shifts_npz(sample_root / "speckle_shifts_clean.npz", shifts_clean, fs, args.lowcut, args.highcut, args.filter_order, args.hann_applied)
    save_fft_npz(sample_root / "speckle_shifts_fft.npz", fft, freqs, fs, args.min_freq, args.max_freq, n_samples)
    log("generating IFFT audio preview")
    generate_fft_audio_preview(
        fft=fft,
        fs=fs,
        n_samples=n_samples,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        out_path=sample_root / "speckle_shifts_ifft_audio.wav",
        laser_idx=args.fft_audio_laser_idx,
        xy_idx=args.fft_audio_xy_idx,
        out_sr=args.fft_audio_out_sr,
    )

    experiment_config = deepcopy(load_json(BASE_EXPERIMENT_CONFIG_PATH))
    processing_config = deepcopy(load_json(BASE_PROCESSING_CONFIG_PATH))
    log("building historical recovery manifest")
    experiment_config["historical_recovery"] = {
        "mode": "vibrations_only",
        "source_experiment_dir": None,
        "source_dir_name": source_dir_name,
    }
    processing_config["segmentation"] = {
        "source": "old_vibrations_mask",
        "status": "reused",
    }

    manifest = build_manifest(
        sample_id=args.sample_id,
        repo_id=args.new_repo_id,
        experiment_dir=args.experiment_dir,
        experiment_id=experiment_id,
        source_dir_name=source_dir_name,
        object_=object_,
        n_objects=n_objects,
        box_material=box_material,
        speakers=speakers,
        x_position=x_position,
        y_position=y_position,
        image_dir=image_dir,
        x_com=x_com,
        y_com=y_com,
        fs=fs,
        shifts=shifts,
        shifts_clean=shifts_clean,
        fft=fft,
        n_samples=n_samples,
        wav_path=sample_root / "speckle_shifts_ifft_audio.wav",
        audio_repo_path=recovery_meta["audio_repo_path"],
        raw_overhead_available=True,
        cropped_overhead_available=True,
        segmented_overhead_available=True,
        mask_available=True,
        processing_config=processing_config,
        experiment_config=experiment_config,
        raw_image_size=raw_image_size,
        cropped_image_size=cropped_image_size,
        mask_shape=mask.shape,
    )
    write_json(sample_root / "manifest.json", manifest)
    log("wrote manifest.json")

    metadata_row = build_metadata_row(
        sample_id=args.sample_id,
        repo_id=args.new_repo_id,
        experiment_id=experiment_id,
        speakers=speakers,
        x_position=x_position,
        y_position=y_position,
        x_com=x_com,
        y_com=y_com,
        object_=object_,
        n_objects=n_objects,
        box_material=box_material,
        experiment_dir=args.experiment_dir,
        image_dir=image_dir,
        audio_file_name=recovery_meta["audio_file_name"],
        manifest=manifest,
    )
    metadata_rows = upsert_metadata_row(metadata_rows, metadata_row)
    write_jsonl_rows(staging_root / "data" / "metadata.jsonl", metadata_rows)
    log("updated staged data/metadata.jsonl")

    summary = {
        "sample_id": args.sample_id,
        "source_dir_name": source_dir_name,
        "image_dir": image_dir,
        "image_reused_from_existing_metadata": image_was_reused,
        "sample_root": str(sample_root),
        "files_written": [
            str(sample_root / "speckle_shifts.npz"),
            str(sample_root / "speckle_shifts_clean.npz"),
            str(sample_root / "speckle_shifts_fft.npz"),
            str(sample_root / "speckle_shifts_ifft_audio.wav"),
            str(sample_root / "manifest.json"),
            str(staging_root / "data" / "metadata.jsonl"),
        ],
        "missing_artifacts": ["speckle_vibration_raw.npy", "speckle_vibrations.mp4"],
    }
    write_json(sample_root / "recovery_summary.json", summary)
    log("wrote recovery_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
