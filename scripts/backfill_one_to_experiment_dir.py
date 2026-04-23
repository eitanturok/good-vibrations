import argparse
import io
import json
import subprocess
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage
from scipy.io.wavfile import write as wav_write
from scipy.signal import butter, resample, sosfiltfilt


DEFAULT_SOURCE_DATA_ROOT = "/net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA"
DEFAULT_AUDIO_ROOT = Path(__file__).resolve().parent.parent / "data" / "audio_samples"


def stage(label, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"[timing] {label}: {dt:.2f}s")
    return result


def download_old_sample_npz(sample_id: int, repo_id: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f"data/sample_{sample_id:06d}.npz")
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def get_capture_fps(experiment_config: dict, run_opt: dict | None = None) -> float:
    if run_opt is not None:
        fps = run_opt.get("cam_params", {}).get("get_frame_rate")
        if fps:
            return float(fps)
    return float(experiment_config.get("FPS") or 0)


def resolve_audio_file(experiment_config: dict, audio_root: Path) -> tuple[Path, str]:
    raw = (
        experiment_config.get("AUDIO_FILE")
        or experiment_config.get("audio_file")
        or experiment_config.get("audio")
        or experiment_config.get("wav")
    )
    if not raw:
        raise ValueError("Experiment config does not include an audio file path")
    basename = Path(str(raw).replace("\\", "/")).name
    local_path = audio_root / basename
    if not local_path.exists():
        raise FileNotFoundError(f"Audio file not found locally: {local_path}")
    return local_path, f"audio/{basename}"


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
        check=True,
        capture_output=True,
    )
    tmp_path.unlink()
    return frame_count, frame_height, frame_width


def sample_dir_name(sample_id: int) -> str:
    return f"{sample_id:07d}"


def init_experiment_dir(root: Path) -> None:
    (root / "data" / "audio").mkdir(parents=True, exist_ok=True)
    (root / "data" / "image").mkdir(parents=True, exist_ok=True)
    readme_path = root / "README.md"
    if not readme_path.exists():
        readme_path.write_text(f"# {root.name}\n\nCanonical experiment root for the Laser Vibrations pipeline.\n", encoding="utf-8")
    metadata_path = root / "data" / "metadata.jsonl"
    if not metadata_path.exists():
        metadata_path.write_text("", encoding="utf-8")


def normalize_token(value: str) -> str:
    return "-".join(str(value).strip().lower().split())


def speaker_code(speakers) -> str:
    if speakers is None:
        return ""
    return "".join(str(int(x)) for x in list(speakers))


def kmeans_1d(values: np.ndarray, k: int, n_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    centers = np.linspace(values.min(), values.max(), k)
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(n_iter):
        d = np.abs(values[:, None] - centers[None, :])
        labels = d.argmin(axis=1)
        new_centers = centers.copy()
        for i in range(k):
            pts = values[labels == i]
            if len(pts):
                new_centers[i] = pts.mean()
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    order = np.argsort(centers)
    centers = centers[order]
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels], dtype=np.int64)
    return centers, labels


def load_old_row_and_cluster_data(sample_id: int, repo_id: str) -> tuple[dict, list, list, list, list]:
    light_cols = [
        "sample_idx",
        "object",
        "n_objects",
        "speakers",
        "box_material",
        "x_position",
        "y_position",
    ]
    sample_cols = [
        *light_cols,
        "raw_image",
        "cropped_image",
        "overlay_image",
    ]
    from huggingface_hub import hf_hub_download

    parquet_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="data/train-00000-of-00001.parquet")
    df = pd.read_parquet(parquet_path, columns=light_cols)
    sample_df = pd.read_parquet(parquet_path, columns=sample_cols, filters=[("sample_idx", "==", sample_id)])
    target_row = None
    xs, ys, sample_ids, objects = [], [], [], []
    for row in df.to_dict(orient="records"):
        if row.get("object") != "empty":
            xs.append(float(row["x_position"]))
            ys.append(float(row["y_position"]))
            sample_ids.append(int(row["sample_idx"]))
            objects.append(row.get("object", ""))
    if len(sample_df) == 1:
        target_row = dict(sample_df.iloc[0].to_dict())
    if target_row is None:
        raise ValueError(f"Sample {sample_id} not found in {repo_id}")
    for key in ["raw_image", "cropped_image", "overlay_image"]:
        value = target_row.get(key)
        if isinstance(value, dict) and value.get("bytes") is not None:
            target_row[key] = PILImage.open(io.BytesIO(value["bytes"]))
    return target_row, xs, ys, sample_ids, objects


def infer_discrete_position(sample_row: dict, xs: list, ys: list, sample_ids: list, objects: list) -> tuple[int, int]:
    target_obj = sample_row.get("object", "")
    obj_xs = [x for x, obj in zip(xs, objects) if obj == target_obj]
    obj_ys = [y for y, obj in zip(ys, objects) if obj == target_obj]
    obj_ids = [sid for sid, obj in zip(sample_ids, objects) if obj == target_obj]
    _, x_labels = kmeans_1d(np.asarray(obj_xs), 11)
    _, y_labels = kmeans_1d(np.asarray(obj_ys), 12)
    idx = obj_ids.index(int(sample_row["sample_idx"]))
    x_idx = int(x_labels[idx])
    y_idx = int(y_labels[idx]) + 1
    print(f"[info] inferred discrete position: x={x_idx:02d} y={y_idx:02d} from x={sample_row['x_position']:.3f} y={sample_row['y_position']:.3f}")
    return x_idx, y_idx


def discover_source_experiment_dir(sample_row: dict, source_data_root: str, x_idx: int, y_idx: int) -> Path:
    root = Path(source_data_root)
    obj = sample_row.get("object", "")
    spk = speaker_code(sample_row.get("speakers"))
    pattern = f"{obj}-{x_idx:02d}x{y_idx:02d}y_{spk}--*"
    candidates = sorted(p for p in root.rglob(pattern) if p.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one source experiment dir, got {len(candidates)} for pattern {pattern}: {candidates[:5]}")
    return candidates[0]


def load_local_experiment_config(source_experiment_dir: Path) -> dict:
    return json.loads((source_experiment_dir / "experiment_config.json").read_text())


def load_local_run_opt(source_experiment_dir: Path) -> dict:
    data = np.load(source_experiment_dir / "metadata.npz", allow_pickle=True)
    return data["run_opt"].item()


def save_image(path: Path, image: PILImage.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG")


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={arr.shape}")
    arr = np.clip(arr, 0.0, 1.0)
    PILImage.fromarray((arr * 255).astype(np.uint8), mode="L").save(path, format="PNG")


def save_raw_recording_npz(source_npy: Path, dest_npz: Path) -> None:
    with zipfile.ZipFile(dest_npz, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(source_npy, arcname="frames.npy")


def mask_center_of_mass(mask: np.ndarray) -> tuple[float | None, float | None]:
    arr = np.asarray(mask, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:
        return None, None
    ys, xs = np.indices(arr.shape)
    x_com = float((xs * arr).sum() / total)
    y_com = float((ys * arr).sum() / total)
    return x_com, y_com


def image_dir_name(row: dict, x_position: int, y_position: int) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return "-".join(
        [
            normalize_token(row.get("object", "")),
            f"{int(x_position):03d}x",
            f"{int(y_position):03d}y",
            normalize_token(str(int(row.get("n_objects") or 1))),
            normalize_token(row.get("box_material", "")),
            timestamp,
        ]
    )


def append_metadata_row(path: Path, row: dict) -> None:
    existing = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if int(payload.get("sample_id", -1)) == int(row["sample_id"]):
                continue
            existing.append(payload)
    existing.append(row)
    existing.sort(key=lambda item: int(item["sample_id"]))
    text = "\n".join(json.dumps(item, ensure_ascii=True) for item in existing) + "\n"
    path.write_text(text, encoding="utf-8")


def build_manifest(
    sample_id: int,
    experiment_dir: str,
    source_experiment_dir: Path,
    sample_row: dict,
    experiment_config: dict,
    run_opt: dict,
    x_position: int,
    y_position: int,
    x_com: float | None,
    y_com: float | None,
    audio_rel: str,
    image_dir: str,
    sample_dir: str,
    fs: float,
    lowcut: float,
    highcut: float,
    filter_order: int,
    hann_applied: bool,
    min_freq: float,
    max_freq: float,
    fft_audio_laser_idx: int,
    fft_audio_xy_idx: int,
    fft_audio_out_sr: int,
) -> dict:
    return {
        "sample_id": int(sample_id),
        "experiment_id": source_experiment_dir.name,
        "experiment_dir": experiment_dir,
        "source_experiment_id": source_experiment_dir.name,
        "source_experiment_dir": str(source_experiment_dir),
        "hf_repo": None,
        "sample": {
            "object": sample_row.get("object", ""),
            "n_objects": int(sample_row.get("n_objects") or 1),
            "box_material": sample_row.get("box_material", ""),
            "speakers": speaker_code(sample_row.get("speakers")),
            "x_position": int(x_position),
            "y_position": int(y_position),
            "x_com": x_com,
            "y_com": y_com,
        },
        "acquisition": {
            "audio_file_name": audio_rel,
            "experiment_config": experiment_config,
            "run_opt": run_opt,
        },
        "processing_config": {
            "speckle_shifts": {"fs_hz": float(fs)},
            "speckle_shifts_clean": {
                "lowcut": float(lowcut),
                "highcut": float(highcut),
                "filter_order": int(filter_order),
                "hann_applied": bool(hann_applied),
            },
            "speckle_shifts_fft": {
                "min_freq": float(min_freq),
                "max_freq": float(max_freq),
            },
            "speckle_shifts_ifft_audio": {
                "laser_idx": int(fft_audio_laser_idx),
                "xy_idx": int(fft_audio_xy_idx),
                "output_sample_rate_hz": int(fft_audio_out_sr),
            },
        },
        "artifacts": {
            "shared": {
                "audio": audio_rel,
                "raw_overhead": f"data/image/{image_dir}/raw_overhead.png",
                "cropped_overhead": f"data/image/{image_dir}/cropped_overhead.png",
                "segmented_overhead": f"data/image/{image_dir}/segmented_overhead.png",
                "mask_png": f"data/image/{image_dir}/mask.png",
                "mask_npz": f"data/image/{image_dir}/mask.npz",
            },
            "sample": {
                "speckle_vibration_raw": f"data/{sample_dir}/speckle_vibration_raw.npz",
                "speckle_vibrations": f"data/{sample_dir}/speckle_vibrations.mp4",
                "speckle_shifts": f"data/{sample_dir}/speckle_shifts.npz",
                "speckle_shifts_clean": f"data/{sample_dir}/speckle_shifts_clean.npz",
                "speckle_shifts_fft": f"data/{sample_dir}/speckle_shifts_fft.npz",
                "speckle_shifts_ifft_audio": f"data/{sample_dir}/speckle_shifts_ifft_audio.wav",
                "manifest": f"data/{sample_dir}/manifest.json",
            },
        },
    }


def build_metadata_row(
    sample_id: int,
    source_experiment_dir: Path,
    speakers: str,
    x_position: int,
    y_position: int,
    x_com: float | None,
    y_com: float | None,
    n_objects: int,
    box_material: str,
    experiment_dir: str,
    audio_rel: str,
    image_dir: str,
    sample_dir: str,
    manifest: dict,
) -> dict:
    return {
        "sample_id": int(sample_id),
        "segmented_overhead_file_name": f"data/image/{image_dir}/segmented_overhead.png",
        "speckle_vibrations_file_name": f"data/{sample_dir}/speckle_vibrations.mp4",
        "speckle_shifts_ifft_audio_file_name": f"data/{sample_dir}/speckle_shifts_ifft_audio.wav",
        "audio_file_name": audio_rel,
        "experiment_id": source_experiment_dir.name,
        "speakers": speakers,
        "x_position": int(x_position),
        "y_position": int(y_position),
        "x_com": x_com,
        "y_com": y_com,
        "n_objects": int(n_objects),
        "box_material": box_material,
        "mask_file_name": f"data/image/{image_dir}/mask.png",
        "experiment_dir": experiment_dir,
        "manifest": json.dumps(manifest, ensure_ascii=True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill one sample into a canonical experiment directory.")
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--experiment-dir", required=True, help="Destination experiment directory label, e.g. experiment-16")
    parser.add_argument("--dest-root", default=DEFAULT_SOURCE_DATA_ROOT, help="Parent directory that contains experiment folders")
    parser.add_argument("--source-data-root", default=DEFAULT_SOURCE_DATA_ROOT)
    parser.add_argument("--source-experiment-dir", default=None)
    parser.add_argument("--old-repo-id", default="eturok-weizmann/vibrations")
    parser.add_argument("--audio-root", default=str(DEFAULT_AUDIO_ROOT))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dest_experiment_root = Path(args.dest_root) / args.experiment_dir
    stage("init experiment dir", lambda: init_experiment_dir(dest_experiment_root))

    sample_row, xs, ys, sample_ids, objects = stage(
        "load old dataset row + position clusters",
        lambda: load_old_row_and_cluster_data(args.sample_id, args.old_repo_id),
    )
    x_position, y_position = stage(
        "infer canonical grid position",
        lambda: infer_discrete_position(sample_row, xs, ys, sample_ids, objects),
    )
    source_experiment_dir = Path(args.source_experiment_dir) if args.source_experiment_dir else stage(
        "discover source experiment dir",
        lambda: discover_source_experiment_dir(sample_row, args.source_data_root, x_position, y_position),
    )
    old_npz = stage("download old sample npz", lambda: download_old_sample_npz(args.sample_id, args.old_repo_id))
    experiment_config = stage("load local experiment config", lambda: load_local_experiment_config(source_experiment_dir))
    run_opt = stage("load local metadata.npz", lambda: load_local_run_opt(source_experiment_dir))
    audio_root = Path(args.audio_root)
    audio_src, audio_rel = stage("resolve shared audio", lambda: resolve_audio_file(experiment_config, audio_root))

    sample_dir = sample_dir_name(args.sample_id)
    sample_root = dest_experiment_root / "data" / sample_dir
    sample_root.mkdir(parents=True, exist_ok=True)

    image_dir = image_dir_name(sample_row, x_position, y_position)
    image_root = dest_experiment_root / "data" / "image" / image_dir
    image_root.mkdir(parents=True, exist_ok=True)

    stage(
        "copy shared audio",
        lambda: (dest_experiment_root / "data" / audio_rel).write_bytes(audio_src.read_bytes())
        if not (dest_experiment_root / "data" / audio_rel).exists()
        else None,
    )
    stage("save raw overhead image", lambda: save_image(image_root / "raw_overhead.png", sample_row["raw_image"]))
    stage("save cropped overhead image", lambda: save_image(image_root / "cropped_overhead.png", sample_row["cropped_image"]))
    stage("save segmented overhead image", lambda: save_image(image_root / "segmented_overhead.png", sample_row["overlay_image"]))

    mask = np.asarray(old_npz["mask"], dtype=np.float32)
    x_com, y_com = stage("compute mask center of mass", lambda: mask_center_of_mass(mask))
    stage("save mask.png", lambda: save_mask_png(image_root / "mask.png", mask))
    stage("save mask.npz", lambda: save_mask_npz(image_root / "mask.npz", mask, args.left, args.right, args.up, args.down, args.prompt))

    raw_npy_path = source_experiment_dir / "frame-recording.npy"
    stage("save speckle_vibration_raw.npz", lambda: save_raw_recording_npz(raw_npy_path, sample_root / "speckle_vibration_raw.npz"))
    frame_count, frame_height, frame_width = stage(
        "generate speckle preview mp4",
        lambda: generate_speckle_preview(raw_npy_path, sample_root / "speckle_vibrations.mp4", fps=get_capture_fps(experiment_config, run_opt)),
    )

    shifts = np.asarray(old_npz["shifts"])
    fs = get_capture_fps(experiment_config, run_opt)
    stage("save speckle_shifts.npz", lambda: save_shifts_npz(sample_root / "speckle_shifts.npz", shifts, fs))
    shifts_clean = stage(
        "compute cleaned shifts",
        lambda: clean_shifts(shifts, fs, args.lowcut, args.highcut, args.filter_order, args.hann_applied),
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
        lambda: shifts_to_fft(shifts_clean, fs, args.min_freq, args.max_freq),
    )
    stage(
        "save speckle_shifts_fft.npz",
        lambda: save_fft_npz(sample_root / "speckle_shifts_fft.npz", fft, freqs, fs, args.min_freq, args.max_freq, n_samples),
    )
    stage(
        "generate speckle_shifts_ifft_audio.wav",
        lambda: generate_fft_audio_preview(
            fft,
            fs,
            n_samples,
            args.min_freq,
            args.max_freq,
            sample_root / "speckle_shifts_ifft_audio.wav",
            args.fft_audio_laser_idx,
            args.fft_audio_xy_idx,
            args.fft_audio_out_sr,
        ),
    )

    manifest = stage(
        "build manifest payload",
        lambda: build_manifest(
            sample_id=args.sample_id,
            experiment_dir=args.experiment_dir,
            source_experiment_dir=source_experiment_dir,
            sample_row=sample_row,
            experiment_config=experiment_config,
            run_opt=run_opt,
            x_position=x_position,
            y_position=y_position,
            x_com=x_com,
            y_com=y_com,
            audio_rel=f"data/{audio_rel}",
            image_dir=image_dir,
            sample_dir=sample_dir,
            fs=fs,
            lowcut=args.lowcut,
            highcut=args.highcut,
            filter_order=args.filter_order,
            hann_applied=args.hann_applied,
            min_freq=args.min_freq,
            max_freq=args.max_freq,
            fft_audio_laser_idx=args.fft_audio_laser_idx,
            fft_audio_xy_idx=args.fft_audio_xy_idx,
            fft_audio_out_sr=args.fft_audio_out_sr,
        ),
    )
    stage(
        "write manifest.json",
        lambda: (sample_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8"),
    )

    metadata_row = stage(
        "build metadata row",
        lambda: build_metadata_row(
            sample_id=args.sample_id,
            source_experiment_dir=source_experiment_dir,
            speakers=speaker_code(sample_row.get("speakers")),
            x_position=x_position,
            y_position=y_position,
            x_com=x_com,
            y_com=y_com,
            n_objects=int(sample_row.get("n_objects") or 1),
            box_material=sample_row.get("box_material", ""),
            experiment_dir=args.experiment_dir,
            audio_rel=f"data/{audio_rel}",
            image_dir=image_dir,
            sample_dir=sample_dir,
            manifest=manifest,
        ),
    )
    stage(
        "append metadata.jsonl",
        lambda: append_metadata_row(dest_experiment_root / "data" / "metadata.jsonl", metadata_row),
    )

    print(f"[done] sample_id={args.sample_id} experiment_dir={args.experiment_dir} source={source_experiment_dir}")
    print(f"[info] sample_root={sample_root}")
    print(f"[info] image_root={image_root}")
    print(f"[info] frame_count={frame_count} frame_height={frame_height} frame_width={frame_width}")


if __name__ == "__main__":
    main()
