import argparse
import ast
import io
import json
import shlex
import shutil
import subprocess
import textwrap
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from datasets import Dataset, Image as HFImage, load_dataset
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
IMAGE_COLS = ["raw_image", "cropped_image", "overlay_image"]


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
        *IMAGE_COLS,
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


def load_remote_experiment_config(remote_experiment_dir: str, local_dir: Path) -> dict:
    config_path = stage(
        "download experiment_config.json",
        lambda: stream_remote_file(f"{remote_experiment_dir}/experiment_config.json", local_dir / "experiment_config.json"),
    )
    return json.loads(config_path.read_text(encoding="utf-8"))


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
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / step), (preview_w, preview_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path}")
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
    return frame_count, frame_height, frame_width


def to_webp(img) -> dict:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="WEBP", quality=85)
    return {"bytes": buf.getvalue(), "path": None}


def write_parquet_row(
    path: Path,
    sample_id: int,
    row: dict,
    audio_file_name: str,
    manifest_payload: dict,
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
        "audio_file_name": audio_file_name,
        "speckle_vibrations_file_name": f"{rel_dir}/speckle_vibrations.mp4",
        "speckle_shifts_fft_audio_file_name": f"{rel_dir}/speckle_shifts_fft_audio.wav",
        "manifest_json": json.dumps(manifest_payload),
        "sample_dir": rel_dir,
        "mask_path": f"{rel_dir}/mask.npz",
        "speckle_vibrations_raw_path": f"{rel_dir}/speckle_vibrations_raw.npy",
        "speckle_shifts_path": f"{rel_dir}/speckle_shifts.npz",
        "speckle_shifts_clean_path": f"{rel_dir}/speckle_shifts_clean.npz",
        "speckle_shifts_fft_path": f"{rel_dir}/speckle_shifts_fft.npz",
        "raw_image": to_webp(row["raw_image"]),
        "cropped_image": to_webp(row["cropped_image"]),
        "overlay_image": to_webp(row["overlay_image"]),
    }
    ds = Dataset.from_list([record])
    for col in IMAGE_COLS:
        ds = ds.cast_column(col, HFImage())
    with path.open("wb") as f:
        ds.to_parquet(f)


def write_manifest(
    path: Path,
    sample_id: int,
    audio_file_name: str,
    remote_experiment_dir: str,
    fps: float,
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
    experiment_config: dict,
) -> dict:
    rel_dir = sample_dir_rel(sample_id)
    source_experiment_path = Path(remote_experiment_dir)
    payload = {
        "sample_idx": int(sample_id),
        "audio_file_name": audio_file_name,
        "sample_dir": rel_dir,
        "experiment_config": experiment_config,
        "speckle_vibrations": {
            "file_name": f"{rel_dir}/speckle_vibrations.mp4",
            "raw_path": f"{rel_dir}/speckle_vibrations_raw.npy",
            "frame_count": int(frame_count),
            "frame_height": int(frame_height),
            "frame_width": int(frame_width),
            "fps": float(fps),
        },
        "speckle_shifts": {
            "path": f"{rel_dir}/speckle_shifts.npz",
        },
        "speckle_shifts_clean": {
            "path": f"{rel_dir}/speckle_shifts_clean.npz",
            "lowcut": float(lowcut),
            "highcut": float(highcut),
            "filter_order": int(filter_order),
            "hann_applied": bool(hann_applied),
        },
        "speckle_shifts_fft": {
            "path": f"{rel_dir}/speckle_shifts_fft.npz",
            "min_freq": float(min_freq),
            "max_freq": float(max_freq),
            "dtype": "complex128",
        },
        "speckle_shifts_fft_audio": {
            "file_name": f"{rel_dir}/speckle_shifts_fft_audio.wav",
            "laser_idx": int(laser_idx),
            "xy_idx": int(xy_idx),
            "method": "ifft",
        },
        "mask": {
            "path": f"{rel_dir}/mask.npz",
        },
        "source_experiment_id": source_experiment_path.name,
        "source_experiment_dir": remote_experiment_dir,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def maybe_stage_audio(audio_src: Path, audio_rel: str, repo_id: str, root: Path, existing_files: set[str]) -> None:
    if audio_rel in existing_files:
        print(f"[info] audio already exists in repo, skipping upload: {audio_rel}")
        return
    dst = root / audio_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(audio_src.read_bytes())


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
    api = HfApi()

    repo_files = stage("list new repo files", lambda: set(api.list_repo_files(args.new_repo_id, repo_type="dataset")))
    old_row = stage("load old dataset row", lambda: load_old_sample_row(args.sample_id, args.old_repo_id))
    unique_x, unique_y = stage("load old dataset position grid", lambda: load_old_position_grid(args.old_repo_id))
    print(f"[info] discovered position grid: n_x={len(unique_x)} n_y={len(unique_y)}")
    old_npz = stage("download old sample npz", lambda: download_old_sample_npz(args.sample_id, args.old_repo_id))

    source_experiment_dir = args.source_experiment_dir
    if source_experiment_dir is None and args.auto_discover_source:
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
            lambda: load_remote_experiment_config(source_experiment_dir, root / "remote"),
        )
        audio_src, audio_rel = stage("resolve shared audio", lambda: resolve_audio_file(experiment_config))
        stage("stage shared audio", lambda: maybe_stage_audio(audio_src, audio_rel, args.new_repo_id, root, repo_files))

        if args.skip_remote_speckle_assets:
            frame_count, frame_height, frame_width = stage(
                "inspect remote raw speckle recording header",
                lambda: inspect_remote_npy(f"{source_experiment_dir}/frame-recording.npy"),
            )
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
                    fps=float(experiment_config.get("FPS") or experiment_config.get("camera_FPS") or 1),
                ),
            )

        shifts = np.asarray(old_npz["shifts"])
        mask = np.asarray(old_npz["mask"])
        fs = float(experiment_config.get("FPS") or experiment_config.get("camera_FPS") or 0)

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
                out_path=sample_root / "speckle_shifts_fft_audio.wav",
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
                audio_file_name=audio_rel,
                remote_experiment_dir=source_experiment_dir,
                fps=fs,
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
                experiment_config=experiment_config,
            ),
        )

        parquet_path = root / "data" / f"train-{args.sample_id:06d}.parquet"
        stage(
            "write parquet row",
            lambda: write_parquet_row(
                path=parquet_path,
                sample_id=args.sample_id,
                row=old_row,
                audio_file_name=audio_rel,
                manifest_payload=manifest_payload,
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
