import argparse
import ast
import io
import json
import re
import shlex
import shutil
import subprocess
import textwrap
import time
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from datasets import Audio as HFAudio, Dataset, Image as HFImage, Video as HFVideo, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from scipy.io.wavfile import write as wav_write
from scipy.signal import butter, resample, sosfiltfilt


OLD_REPO_ID = "eturok-weizmann/vibrations"
NEW_REPO_ID = "eturok-weizmann/laser-vibrations"
REMOTE_HOST = "ethantu@mcluster11.wisdom.weizmann.ac.il"
REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_AUDIO_ROOT = REPO_ROOT / "data" / "audio_samples"
DEFAULT_SOURCE_DATA_ROOT = "/net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA"
OLD_IMAGE_COLS = ["raw_image", "cropped_image", "overlay_image"]
IMAGE_COLS = ["overhead_image", "cropped_overhead_image", "segmented_overhead_image"]
IMAGE_NAMES = {
    "overhead_image": ("raw_image", "raw.webp"),
    "cropped_overhead_image": ("cropped_image", "cropped.webp"),
    "segmented_overhead_image": ("overlay_image", "overlay.webp"),
}


def stage(label, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"[timing] {label}: {dt:.2f}s")
    return result


def sample_dir_name(sample_id: int) -> str:
    return f"sample_{sample_id:06d}"


def sample_dir_rel(sample_id: int) -> str:
    return f"samples/{sample_dir_name(sample_id)}"


def old_sample_npz_path(sample_id: int) -> str:
    return f"data/sample_{sample_id:06d}.npz"


def load_old_sample_row(sample_id: int, repo_id: str) -> dict:
    columns = [
        "sample_idx",
        "object",
        "n_objects",
        "speakers",
        "box_material",
        "x_position",
        "y_position",
        *OLD_IMAGE_COLS,
    ]
    ds = load_dataset(repo_id, split="train", columns=columns, verification_mode="no_checks")
    for row in ds:
        if int(row["sample_idx"]) == int(sample_id):
            return dict(row)
    raise ValueError(f"Sample {sample_id} not found in {repo_id}")


def load_old_position_grid(repo_id: str) -> tuple[np.ndarray, np.ndarray]:
    cols = ["object", "x_position", "y_position"]
    ds = load_dataset(repo_id, split="train", columns=cols, verification_mode="no_checks")
    xs, ys = [], []
    for row in ds:
        if row.get("object") == "empty":
            continue
        x = float(row.get("x_position") or -1)
        y = float(row.get("y_position") or -1)
        if x >= 0:
            xs.append(x)
        if y >= 0:
            ys.append(y)
    unique_x = np.unique(np.round(np.asarray(xs, dtype=np.float64), 6))
    unique_y = np.unique(np.round(np.asarray(ys, dtype=np.float64), 6))
    return np.sort(unique_x), np.sort(unique_y)


def download_old_sample_npz(sample_id: int, repo_id: str) -> dict:
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=old_sample_npz_path(sample_id))
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def stream_remote_file(remote_path: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    remote_command = f"cat {shlex.quote(remote_path)}"
    with local_path.open("wb") as f:
        subprocess.run(["ssh", REMOTE_HOST, remote_command], check=True, stdout=f)
    return local_path


def fetch_remote_bytes(remote_path: str, count: int, skip: int = 0) -> bytes:
    remote_command = (
        'sh -lc ' + shlex.quote(
            f'dd if={shlex.quote(remote_path)} bs=1 skip={skip} count={count} 2>/dev/null'
        )
    )
    result = subprocess.run(["ssh", REMOTE_HOST, remote_command], check=True, capture_output=True)
    return result.stdout


def inspect_remote_npy(remote_path: str) -> tuple[int, int, int]:
    header_bytes = fetch_remote_bytes(remote_path, count=4096)
    if header_bytes[:6] != b"\x93NUMPY":
        raise ValueError(f"Remote file is not a .npy file: {remote_path}")
    major = header_bytes[6]
    if major == 1:
        header_len = int.from_bytes(header_bytes[8:10], "little")
        start = 10
    else:
        header_len = int.from_bytes(header_bytes[8:12], "little")
        start = 12
    header = header_bytes[start:start + header_len].decode("latin1")
    payload = ast.literal_eval(header)
    shape = payload["shape"]
    if len(shape) != 3:
        raise ValueError(f"Expected 3D raw recording, got shape={shape}")
    return int(shape[0]), int(shape[1]), int(shape[2])


def copy_any_file(src: str | Path, dst: Path) -> Path:
    src_path = Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src_path.exists():
        shutil.copy2(src_path, dst)
        return dst
    return stream_remote_file(str(src), dst)


def load_remote_experiment_config(remote_experiment_dir: str) -> dict:
    cmd = f"cat {shlex.quote(remote_experiment_dir + '/experiment_config.json')}"
    t0 = time.perf_counter()
    result = subprocess.run(["ssh", REMOTE_HOST, cmd], check=True, text=True, capture_output=True)
    print(f"[timing] download experiment_config.json: {time.perf_counter() - t0:.2f}s")
    return json.loads(result.stdout)


def load_remote_run_opt(remote_experiment_dir: str) -> dict:
    with TemporaryDirectory(prefix="laser-vibrations-meta-") as tmp:
        local_path = Path(tmp) / "metadata.npz"
        stream_remote_file(f"{remote_experiment_dir}/metadata.npz", local_path)
        data = np.load(local_path, allow_pickle=True)
        return data["run_opt"].item()


def resolve_audio_file(experiment_config: dict) -> tuple[Path, str]:
    raw = (
        experiment_config.get("AUDIO_FILE")
        or experiment_config.get("audio_file")
        or experiment_config.get("audio")
        or experiment_config.get("wav")
    )
    if not raw:
        raise ValueError("Experiment config does not include an audio file path")
    basename = Path(str(raw).replace("\\", "/")).name
    local_path = LOCAL_AUDIO_ROOT / basename
    if not local_path.exists():
        raise FileNotFoundError(f"Audio file not found locally: {local_path}")
    return local_path, f"audio/{basename}"


def speaker_code(speakers) -> str:
    return "".join(str(int(x)) for x in (speakers or []))


def downsample_grayscale(img: Image.Image, size: tuple[int, int] = (96, 96)) -> np.ndarray:
    return np.asarray(img.convert("L").resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def discover_source_experiment_dir(row: dict, source_data_root: str) -> str:
    root = Path(source_data_root)
    if not root.exists():
        raise FileNotFoundError(f"Source data root does not exist locally: {root}")

    obj = row.get("object", "")
    spk = speaker_code(row.get("speakers"))
    pattern = f"**/{obj}-*_{spk}--*" if obj else f"**/*_{spk}--*"

    t0 = time.perf_counter()
    candidates = sorted(p for p in root.glob(pattern) if p.is_dir())
    print(f"[timing] discover candidates: {time.perf_counter() - t0:.2f}s (n={len(candidates)})")
    if not candidates:
        raise FileNotFoundError(f"No candidate experiment dirs found for object={obj!r} speakers={spk!r} under {root}")
    if len(candidates) == 1:
        print(f"[info] single candidate source dir: {candidates[0]}")
        return str(candidates[0])

    target = downsample_grayscale(row["raw_image"])
    scores = []
    t0 = time.perf_counter()
    for cand in candidates:
        img_path = cand / "box_overhead_image.png"
        if not img_path.exists():
            continue
        with Image.open(img_path) as img:
            arr = downsample_grayscale(img)
        mad = float(np.mean(np.abs(arr - target)))
        scores.append((mad, cand))
    print(f"[timing] discover image matching: {time.perf_counter() - t0:.2f}s (n={len(scores)})")
    if not scores:
        raise FileNotFoundError("No candidates had box_overhead_image.png for image matching")

    scores.sort(key=lambda x: x[0])
    for mad, cand in scores[:5]:
        print(f"[debug] source candidate MAD={mad:.4f} dir={cand}")
    return str(scores[0][1])


def discover_source_experiment_dir_from_grid(row: dict, source_data_root: str, unique_x: np.ndarray, unique_y: np.ndarray) -> str | None:
    obj = row.get("object", "")
    if obj == "empty":
        return None

    x_val = float(row.get("x_position") or -1)
    y_val = float(row.get("y_position") or -1)
    if x_val < 0 or y_val < 0 or len(unique_x) == 0 or len(unique_y) == 0:
        return None

    x_idx = int(np.argmin(np.abs(unique_x - x_val)))
    y_idx = int(np.argmin(np.abs(unique_y - y_val))) + 1
    spk = speaker_code(row.get("speakers"))
    basename_pattern = f"{obj}-{x_idx:02d}x{y_idx:02d}y_{spk}--*"
    t0 = time.perf_counter()
    cmd = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        f"root = Path({source_data_root!r})\n"
        f"pattern = {basename_pattern!r}\n"
        "for p in sorted(root.rglob(pattern)):\n"
        "    if p.is_dir():\n"
        "        print(p)\n"
        "PY"
    )
    result = subprocess.run(["ssh", REMOTE_HOST, cmd], check=True, text=True, capture_output=True)
    candidates = [Path(line) for line in result.stdout.splitlines() if line.strip()]
    print(
        f"[timing] discover grid candidates: {time.perf_counter() - t0:.2f}s "
        f"(pattern={basename_pattern}, n={len(candidates)}, x={x_val:.3f}->{x_idx:02d}, y={y_val:.3f}->{y_idx:02d})"
    )
    if len(candidates) == 1:
        return str(candidates[0])
    if len(candidates) > 1:
        print("[warn] multiple grid candidates, will fall back to image matching within this subset")
        target = downsample_grayscale(row["raw_image"])
        scores = []
        t0 = time.perf_counter()
        for cand in candidates:
            img_path = cand / "box_overhead_image.png"
            if not img_path.exists():
                continue
            with Image.open(img_path) as img:
                arr = downsample_grayscale(img)
            mad = float(np.mean(np.abs(arr - target)))
            scores.append((mad, cand))
        print(f"[timing] grid fallback image matching: {time.perf_counter() - t0:.2f}s (n={len(scores)})")
        if scores:
            scores.sort(key=lambda x: x[0])
            for mad, cand in scores[:5]:
                print(f"[debug] grid candidate MAD={mad:.4f} dir={cand}")
            return str(scores[0][1])
    return None


def get_capture_fps(experiment_config: dict, run_opt: dict | None = None) -> float:
    if run_opt is not None:
        fps = run_opt.get("cam_params", {}).get("get_frame_rate")
        if fps:
            return float(fps)
    return float(experiment_config.get("FPS") or 0)


def save_mask_npz(path: Path, mask: np.ndarray, left: float, right: float, up: float, down: float, prompt: str | None) -> None:
    np.savez_compressed(path, mask=mask, left=left, right=right, up=up, down=down, prompt=prompt)


def save_shifts_npz(path: Path, shifts: np.ndarray, fs: float) -> None:
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


def clean_shifts(
    shifts: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    filter_order: int,
    hann_applied: bool,
) -> np.ndarray:
    cleaned = butterworth_filter(shifts, fs=fs, lowcut=lowcut, highcut=highcut, filter_order=filter_order)
    if hann_applied:
        cleaned = apply_hann_window(cleaned)
    return cleaned


def save_clean_shifts_npz(
    path: Path,
    shifts_clean: np.ndarray,
    fs: float,
    lowcut: float,
    highcut: float,
    filter_order: int,
    hann_applied: bool,
) -> None:
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
    full_fft = np.fft.rfft(shifts_clean, axis=1)
    full_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (full_freqs >= min_freq) & (full_freqs <= max_freq)
    return full_fft[:, mask, :], full_freqs[mask], n_samples


def save_fft_npz(path: Path, fft: np.ndarray, freqs: np.ndarray, fs: float, min_freq: float, max_freq: float, n_samples: int) -> None:
    np.savez_compressed(path, fft=fft, freqs=freqs, fs=fs, min_freq=min_freq, max_freq=max_freq, n_samples=n_samples)


def generate_fft_audio_preview(
    fft: np.ndarray,
    fs: float,
    n_samples: int,
    min_freq: float,
    max_freq: float,
    out_path: Path,
    laser_idx: int,
    xy_idx: int,
    out_sr: int,
) -> None:
    full_freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask = (full_freqs >= min_freq) & (full_freqs <= max_freq)
    spectrum = np.zeros(full_freqs.shape[0], dtype=np.complex64)
    spectrum[mask] = fft[laser_idx, :, xy_idx]
    signal = np.fft.irfft(spectrum, n=n_samples)
    audio = resample(signal, int(out_sr * len(signal) / fs))
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    wav_write(out_path, out_sr, (audio * 32767).astype(np.int16))


def generate_speckle_preview(raw_npy_path: Path, out_path: Path, fps: float, max_frames: int = 300, max_width: int = 960) -> tuple[int, int, int]:
    frames = np.load(raw_npy_path, mmap_mode="r")
    frame_count, frame_height, frame_width = frames.shape
    step = max(1, frame_count // max_frames)
    selected = frames[::step]
    preview_h, preview_w = frame_height, frame_width
    if frame_width > max_width:
        scale = max_width / frame_width
        preview_w = int(round(frame_width * scale))
        preview_h = int(round(frame_height * scale))

    probe = np.asarray(selected[: min(len(selected), 50)])
    lo = float(np.percentile(probe, 5))
    hi = float(np.percentile(probe, 99.5))
    tmp_path = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / step), (preview_w, preview_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {tmp_path}")
    try:
        for i, frame in enumerate(selected):
            frame_u8 = np.clip((frame.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
            frame_u8 = (frame_u8 * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
            if (preview_w, preview_h) != (frame_width, frame_height):
                frame_bgr = cv2.resize(frame_bgr, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            cv2.putText(frame_bgr, f"frame {i * step}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame_bgr)
    finally:
        writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_path), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
        check=True, capture_output=True,
    )
    tmp_path.unlink()
    return frame_count, frame_height, frame_width


AUDIO_COLS = ["audio", "speckle_shifts_ifft_audio"]
VIDEO_COLS = ["speckle_vibrations"]


def to_audio(path: Path) -> dict:
    return {"bytes": path.read_bytes(), "path": path.name}


def to_video(path: Path) -> dict:
    return {"bytes": path.read_bytes(), "path": path.name}


def write_parquet_row(
    path: Path,
    sample_id: int,
    row: dict,
    audio_src: Path,
    fft_audio_src: Path,
    video_src: Path,
    manifest_payload: dict,
    image_paths: dict[str, str],
) -> None:
    rel_dir = sample_dir_rel(sample_id)
    record = {
        "sample_idx": int(sample_id),
        "object": row.get("object", ""),
        "n_objects": int(row.get("n_objects") or 1),
        "box_material": row.get("box_material", ""),
        "speakers": row.get("speakers", []),
        "x_position": row.get("x_position"),
        "y_position": row.get("y_position"),
        "audio": to_audio(audio_src),
        "speckle_vibrations": to_video(video_src),
        "speckle_shifts_ifft_audio": to_audio(fft_audio_src),
        "manifest_json": json.dumps(manifest_payload),
        "mask_path": f"{rel_dir}/mask.npz",
        "overhead_image": image_paths["overhead_image"],
        "cropped_overhead_image": image_paths["cropped_overhead_image"],
        "segmented_overhead_image": image_paths["segmented_overhead_image"],
    }
    ds = Dataset.from_list([record])
    for col in IMAGE_COLS:
        ds = ds.cast_column(col, HFImage())
    for col in AUDIO_COLS:
        ds = ds.cast_column(col, HFAudio())
    for col in VIDEO_COLS:
        ds = ds.cast_column(col, HFVideo())
    with path.open("wb") as f:
        ds.to_parquet(f)


def write_manifest(
    path: Path,
    sample_id: int,
    row: dict,
    audio_file_name: str,
    audio_src: Path,
    remote_experiment_dir: str,
    frame_count: int,
    frame_height: int,
    frame_width: int,
    lowcut: float,
    highcut: float,
    filter_order: int,
    hann_applied: bool,
    min_freq: float,
    max_freq: float,
    laser_idx: int,
    xy_idx: int,
    fft_audio_out_sr: int,
    experiment_config: dict,
    run_opt: dict,
    image_key: str,
) -> dict:
    payload = build_manifest_payload(
        sample_id=sample_id,
        row=row,
        audio_file_name=audio_file_name,
        audio_src=audio_src,
        remote_experiment_dir=remote_experiment_dir,
        experiment_config=experiment_config,
        run_opt=run_opt,
        frame_count=frame_count,
        frame_height=frame_height,
        frame_width=frame_width,
        lowcut=lowcut,
        highcut=highcut,
        filter_order=filter_order,
        hann_applied=hann_applied,
        min_freq=min_freq,
        max_freq=max_freq,
        laser_idx=laser_idx,
        xy_idx=xy_idx,
        fft_audio_out_sr=fft_audio_out_sr,
        image_key=image_key,
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def maybe_stage_audio(audio_src: Path, audio_rel: str, repo_id: str, root: Path, existing_files: set[str]) -> None:
    if audio_rel in existing_files:
        print(f"[info] audio already exists in repo, skipping upload: {audio_rel}")
        return
    dst = root / audio_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(audio_src.read_bytes())


def image_key_from_experiment_dir(source_experiment_dir: str) -> str:
    basename = Path(source_experiment_dir).name
    m = re.match(r'^(.+-\d{2}x\d{2}y)', basename)
    if m:
        return m.group(1)
    return basename.split("_")[0]


def stage_shared_images(row: dict, image_key: str, root: Path, existing_files: set[str]) -> dict[str, str]:
    """Write shared overhead images to images/{image_key}/ once; return repo-relative paths."""
    paths = {}
    for new_col, (old_col, filename) in IMAGE_NAMES.items():
        rel = f"images/{image_key}/{filename}"
        paths[new_col] = rel
        if rel in existing_files:
            print(f"[info] shared image already in repo, skipping: {rel}")
            continue
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        row[old_col].convert("RGB").save(buf, format="WEBP", quality=85)
        dst.write_bytes(buf.getvalue())
    return paths


def read_wav_metadata(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as wav_f:
        sample_rate_hz = wav_f.getframerate()
        duration_s = wav_f.getnframes() / sample_rate_hz if sample_rate_hz else 0.0
    return int(sample_rate_hz), float(duration_s)


def to_python(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_python(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_python(v) for k, v in value.items()}
    return value


def build_row_values_single_list(roi_list: list[list[int]]) -> list[int]:
    values = []
    for start, end in roi_list:
        values.extend(int(v) for v in np.arange(int(start), int(end), 2) // 2)
    return values


def infer_selected_column_centers_x(sensor_rois_xywh: list[list[int]], n_cols: int) -> list[int]:
    if not sensor_rois_xywh or n_cols <= 0:
        return []
    first_row = sensor_rois_xywh[:n_cols]
    return [int(x + w // 2) for x, _, w, _ in first_row]


def build_manifest_payload(
    sample_id: int,
    row: dict,
    audio_file_name: str,
    audio_src: Path,
    remote_experiment_dir: str,
    experiment_config: dict,
    run_opt: dict,
    frame_count: int,
    frame_height: int,
    frame_width: int,
    lowcut: float,
    highcut: float,
    filter_order: int,
    hann_applied: bool,
    min_freq: float,
    max_freq: float,
    laser_idx: int,
    xy_idx: int,
    fft_audio_out_sr: int,
    image_key: str,
) -> dict:
    rel_dir = sample_dir_rel(sample_id)
    sample_root = f"samples/sample_{sample_id:06d}"
    source_experiment_path = Path(remote_experiment_dir)
    sample_rate_hz, duration_s = read_wav_metadata(audio_src)
    cam_params = to_python(run_opt.get("cam_params", {}))
    multi_rois = to_python(run_opt.get("run_opt_multiROIs", {}))
    roi_list = multi_rois.get("ROI_list", [])
    sensor_rois = multi_rois.get("ROIs", [])
    global_roi = cam_params.get("get_global_roi") or multi_rois.get("global_ROI")
    global_crop_x = global_crop_width = global_crop_height = None
    if global_roi:
        global_crop_x = int(global_roi[0])
        global_crop_width = int(global_roi[2])
        global_crop_height = int(global_roi[3])
    n_cols = int(experiment_config.get("N_ROI_COLUMNS", 0) or 0)

    return {
        "sample_idx": int(sample_id),
        "experiment_id": source_experiment_path.name,
        "experiment_dir": remote_experiment_dir,
        "sample": {
            "object": row.get("object", ""),
            "n_objects": int(row.get("n_objects") or 1),
            "box_material": row.get("box_material", ""),
            "speakers": to_python(row.get("speakers", [])),
        },
        "experiment_config": {
            "audio": {
                "file_name": audio_file_name,
                "sample_rate_hz": sample_rate_hz,
                "duration_s": duration_s,
                "total_output_channels": 8,
            },
            "recording": {
                "capture_seconds": float(experiment_config.get("N_CAPTURE_SECONDS", 0.0)),
            },
            "overhead_camera": {
                "frame_rate_fps": to_python(experiment_config.get("FRAME_RATE")),
                "exposure_ms": to_python(experiment_config.get("EXPOSURE_MS")),
                "pixel_clock_mhz": to_python(experiment_config.get("PIXEL_CLOCK")),
                "gain": to_python(experiment_config.get("CAMERA_GAIN")),
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
                    "fps": to_python(experiment_config.get("CALIBRATION_FPS")),
                    "exposure_us": to_python(experiment_config.get("CALIBRATION_EXPOSURE")),
                    "gain": to_python(experiment_config.get("CALIBRATION_GAIN")),
                },
                "capture": {
                    "fps": to_python(experiment_config.get("FPS")),
                    "exposure_us": to_python(experiment_config.get("EXPOSURE")),
                    "gain": to_python(experiment_config.get("GAIN")),
                    "buffer_part_count": to_python(experiment_config.get("BUFFER_PART_COUNT")),
                },
            },
            "laser_grid": {
                "n_roi_rows": to_python(experiment_config.get("N_ROI_ROWS")),
                "n_roi_columns": to_python(experiment_config.get("N_ROI_COLUMNS")),
                "roi_row_height": to_python(experiment_config.get("ROI_ROW_HEIGHT")),
                "roi_column_width": to_python(experiment_config.get("ROI_COLUMN_WIDTH")),
            },
            "preview": {
                "overhead_resize_factor": to_python(experiment_config.get("RESIZE_FACTOR")),
                "overhead_gamma": to_python(experiment_config.get("GAMMA")),
                "laser_preview_gamma": 2.5,
                "show_full_frame": 0,
                "preview_level": 1,
                "reset_rois": True,
            },
        },
        "experiment_output": {
            "overhead_camera": {
                "image_width": int(row["raw_image"].size[0]),
                "image_height": int(row["raw_image"].size[1]),
            },
            "laser_camera": {
                "global_roi": to_python(global_roi),
                "max_frame_rate_hz": to_python(cam_params.get("get_max_frame_rate")),
            },
            "laser_grid": {
                "total_image_height": to_python(multi_rois.get("total_image_height")),
                "selected_row_points_image_xy": to_python(multi_rois.get("selected_row_points", [])),
                "selected_column_centers_x": infer_selected_column_centers_x(sensor_rois, n_cols),
                "row_values_single_list": build_row_values_single_list(roi_list),
                "global_crop_x": global_crop_x,
                "global_crop_width": global_crop_width,
                "global_crop_height": global_crop_height,
                "row_rois_y": to_python(roi_list),
                "sensor_grid_shape": [int(experiment_config.get("N_ROI_ROWS", 0)), int(experiment_config.get("N_ROI_COLUMNS", 0))],
                "sensor_rois_xywh": to_python(sensor_rois),
            },
            "speckle_vibrations": {
                "frame_count": int(frame_count),
                "frame_height": int(frame_height),
                "frame_width": int(frame_width),
                "capture_seconds": float(experiment_config.get("N_CAPTURE_SECONDS", 0.0)),
                "preview_fps": 30.0,
                "dtype": "uint8",
            },
        },
        "artifacts": {
            "overhead_image": f"images/{image_key}/raw.webp",
            "cropped_image": f"images/{image_key}/cropped.webp",
            "overlay_image": f"images/{image_key}/overlay.webp",
            "speckle_vibrations_raw": f"{sample_root}/speckle_vibrations_raw.npy",
            "speckle_vibrations_preview": f"{sample_root}/speckle_vibrations.mp4",
            "speckle_shifts": f"{sample_root}/speckle_shifts.npz",
            "speckle_shifts_clean": f"{sample_root}/speckle_shifts_clean.npz",
            "speckle_shifts_fft": f"{sample_root}/speckle_shifts_fft.npz",
            "speckle_shifts_ifft_audio": f"{sample_root}/speckle_shifts_ifft_audio.wav",
            "mask": f"{sample_root}/mask.npz",
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
                "burn_frame_index": True,
            },
            "speckle_shifts": {
                "fs_hz": get_capture_fps(experiment_config, run_opt),
            },
            "speckle_shifts_clean": {
                "filter_type": "butterworth",
                "filter_mode": "bandpass",
                "lowcut": float(lowcut),
                "highcut": float(highcut),
                "filter_order": int(filter_order),
                "hann_applied": bool(hann_applied),
                "apply_order": "filter_then_hann",
            },
            "speckle_shifts_fft": {
                "fft_kind": "rfft",
                "fft_axis": 1,
                "min_freq": float(min_freq),
                "max_freq": float(max_freq),
                "dtype": "complex128",
                "crop_after_fft": True,
            },
            "speckle_shifts_ifft_audio": {
                "laser_idx": int(laser_idx),
                "xy_idx": int(xy_idx),
                "method": "ifft",
                "output_sample_rate_hz": int(fft_audio_out_sr),
                "normalization": "peak_to_int16",
                "output_dtype": "int16",
                "zero_fill_uncropped_bins": True,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill one sample from vibrations + mraid20 into laser-vibrations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example:
              uv run python scripts/backfill_laser_vibrations.py \
                --sample-id 42 \
                --source-experiment-dir /net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA/experiment-15/cube-09x00y_0010--31-03-20-42-51
            """
        ),
    )
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--source-experiment-dir", default=None)
    parser.add_argument("--source-data-root", default=DEFAULT_SOURCE_DATA_ROOT)
    parser.add_argument("--auto-discover-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-remote-speckle-assets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--old-repo-id", default=OLD_REPO_ID)
    parser.add_argument("--new-repo-id", default=NEW_REPO_ID)
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
    parser.add_argument("--create-pr", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = Path("~/.cache/huggingface/token").expanduser().read_text().strip()
    api = HfApi(token=token)

    repo_files = stage("list new repo files", lambda: set(api.list_repo_files(args.new_repo_id, repo_type="dataset")))
    old_row = stage("load old dataset row", lambda: load_old_sample_row(args.sample_id, args.old_repo_id))
    old_npz = stage("download old sample npz", lambda: download_old_sample_npz(args.sample_id, args.old_repo_id))

    source_experiment_dir = args.source_experiment_dir
    if source_experiment_dir is None and args.auto_discover_source:
        unique_x, unique_y = stage("load old dataset position grid", lambda: load_old_position_grid(args.old_repo_id))
        print(f"[info] discovered position grid: n_x={len(unique_x)} n_y={len(unique_y)}")
        source_experiment_dir = stage(
            "auto-discover source experiment dir from position grid",
            lambda: discover_source_experiment_dir_from_grid(old_row, args.source_data_root, unique_x, unique_y),
        )
        if source_experiment_dir is None:
            print("[warn] position-grid discovery did not yield a unique match; falling back to image search")
            source_experiment_dir = stage(
                "auto-discover source experiment dir from image match",
                lambda: discover_source_experiment_dir(old_row, args.source_data_root),
            )
    if source_experiment_dir is None:
        raise ValueError("Either --source-experiment-dir must be provided or --auto-discover-source must remain enabled")
    print(f"[info] using source experiment dir: {source_experiment_dir}")

    with TemporaryDirectory(prefix=f"laser-vibrations-{args.sample_id:06d}-") as tmp:
        root = Path(tmp)
        sample_root = root / sample_dir_rel(args.sample_id)
        sample_root.mkdir(parents=True, exist_ok=True)
        (root / "data").mkdir(parents=True, exist_ok=True)

        experiment_config = stage(
            "load remote experiment config",
            lambda: load_remote_experiment_config(source_experiment_dir),
        )
        run_opt = stage(
            "load remote metadata.npz",
            lambda: load_remote_run_opt(source_experiment_dir),
        )
        audio_src, audio_rel = stage("resolve shared audio", lambda: resolve_audio_file(experiment_config))
        stage("stage shared audio", lambda: maybe_stage_audio(audio_src, audio_rel, args.new_repo_id, root, repo_files))
        image_key = image_key_from_experiment_dir(source_experiment_dir)
        print(f"[info] image_key={image_key}")
        image_paths = stage("stage shared images", lambda: stage_shared_images(old_row, image_key, root, repo_files))

        if args.skip_remote_speckle_assets:
            frame_count, frame_height, frame_width = stage(
                "inspect remote raw speckle recording header",
                lambda: inspect_remote_npy(f"{source_experiment_dir}/frame-recording.npy"),
            )
            video_src = Path(stage(
                "download speckle preview mp4 from HF",
                lambda: hf_hub_download(
                    repo_id=args.new_repo_id,
                    repo_type="dataset",
                    filename=f"{sample_dir_rel(args.sample_id)}/speckle_vibrations.mp4",
                ),
            ))
        else:
            raw_npy_path = stage(
                "download raw speckle recording",
                lambda: copy_any_file(Path(source_experiment_dir) / "frame-recording.npy", sample_root / "speckle_vibrations_raw.npy"),
            )

            frame_count, frame_height, frame_width = stage(
                "generate speckle preview mp4",
                lambda: generate_speckle_preview(
                    raw_npy_path=raw_npy_path,
                    out_path=sample_root / "speckle_vibrations.mp4",
                    fps=get_capture_fps(experiment_config, run_opt),
                ),
            )
            video_src = sample_root / "speckle_vibrations.mp4"

        shifts = np.asarray(old_npz["shifts"])
        mask = np.asarray(old_npz["mask"])
        fs = get_capture_fps(experiment_config, run_opt)

        stage("save mask.npz", lambda: save_mask_npz(sample_root / "mask.npz", mask, args.left, args.right, args.up, args.down, args.prompt))
        stage("save speckle_shifts.npz", lambda: save_shifts_npz(sample_root / "speckle_shifts.npz", shifts, fs))

        shifts_clean = stage(
            "compute cleaned shifts",
            lambda: clean_shifts(
                shifts=shifts,
                fs=fs,
                lowcut=args.lowcut,
                highcut=args.highcut,
                filter_order=args.filter_order,
                hann_applied=args.hann_applied,
            ),
        )
        stage(
            "save speckle_shifts_clean.npz",
            lambda: save_clean_shifts_npz(
                sample_root / "speckle_shifts_clean.npz",
                shifts_clean,
                fs,
                args.lowcut,
                args.highcut,
                args.filter_order,
                args.hann_applied,
            ),
        )

        fft, freqs, n_samples = stage(
            "compute FFT",
            lambda: shifts_to_fft(shifts_clean=shifts_clean, fs=fs, min_freq=args.min_freq, max_freq=args.max_freq),
        )
        stage(
            "save speckle_shifts_fft.npz",
            lambda: save_fft_npz(sample_root / "speckle_shifts_fft.npz", fft, freqs, fs, args.min_freq, args.max_freq, n_samples),
        )
        stage(
            "generate FFT audio preview",
            lambda: generate_fft_audio_preview(
                fft=fft,
                fs=fs,
                n_samples=n_samples,
                min_freq=args.min_freq,
                max_freq=args.max_freq,
                out_path=sample_root / "speckle_shifts_ifft_audio.wav",
                laser_idx=args.fft_audio_laser_idx,
                xy_idx=args.fft_audio_xy_idx,
                out_sr=args.fft_audio_out_sr,
            ),
        )

        manifest_payload = stage(
            "write manifest.json",
            lambda: write_manifest(
                path=sample_root / "manifest.json",
                sample_id=args.sample_id,
                row=old_row,
                audio_file_name=audio_rel,
                audio_src=audio_src,
                remote_experiment_dir=source_experiment_dir,
                frame_count=frame_count,
                frame_height=frame_height,
                frame_width=frame_width,
                lowcut=args.lowcut,
                highcut=args.highcut,
                filter_order=args.filter_order,
                hann_applied=args.hann_applied,
                min_freq=args.min_freq,
                max_freq=args.max_freq,
                laser_idx=args.fft_audio_laser_idx,
                xy_idx=args.fft_audio_xy_idx,
                fft_audio_out_sr=args.fft_audio_out_sr,
                experiment_config=experiment_config,
                run_opt=run_opt,
                image_key=image_key,
            ),
        )

        parquet_path = root / "data" / f"train-{args.sample_id:06d}.parquet"
        stage(
            "write parquet row",
            lambda: write_parquet_row(
                path=parquet_path,
                sample_id=args.sample_id,
                row=old_row,
                audio_src=audio_src,
                fft_audio_src=sample_root / "speckle_shifts_ifft_audio.wav",
                video_src=video_src,
                manifest_payload=manifest_payload,
                image_paths=image_paths,
            ),
        )

        stage(
            "upload folder to new dataset",
            lambda: api.upload_folder(
                folder_path=str(root),
                repo_id=args.new_repo_id,
                repo_type="dataset",
                commit_message=f"Backfill sample {args.sample_id:06d} into laser-vibrations",
                create_pr=args.create_pr,
            ),
        )

    print(f"[done] backfilled sample {args.sample_id:06d} into https://huggingface.co/datasets/{args.new_repo_id}")


if __name__ == "__main__":
    main()
