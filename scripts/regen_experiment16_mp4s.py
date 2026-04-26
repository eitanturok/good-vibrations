"""
Re-generate speckle_vibrations.mp4 for all experiment-16 samples at 30fps and re-upload to HF.

Run on mcluster11 (where the raw .npy files live):
    uv pip install opencv-python-headless huggingface_hub imageio imageio-ffmpeg numpy
    python regen_experiment16_mp4s.py --data-root /path/to/experiment-16/data --repo-id eturok-weizmann/laser-vibrations
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi

MAX_DISPLAY_FPS = 30.0
MAX_FRAMES = 300
MAX_WIDTH = 960


def stage(label, fn):
    t0 = time.perf_counter()
    result = fn()
    print(f"[timing] {label}: {time.perf_counter() - t0:.2f}s")
    return result


def read_npy_header(path: Path) -> tuple[tuple, np.dtype, int]:
    from numpy.lib import format as npy_format
    with path.open("rb") as f:
        major, minor = npy_format.read_magic(f)
        if (major, minor) == (1, 0):
            shape, _, dtype = npy_format.read_array_header_1_0(f)
        else:
            shape, _, dtype = npy_format.read_array_header_2_0(f)
        offset = f.tell()
    return shape, np.dtype(dtype), offset


def read_frame_at(path: Path, offset: int, dtype: np.dtype, frame_h: int, frame_w: int, idx: int) -> np.ndarray:
    frame_size = frame_h * frame_w
    byte_offset = offset + idx * frame_size * dtype.itemsize
    with path.open("rb") as f:
        f.seek(byte_offset)
        return np.fromfile(f, dtype=dtype, count=frame_size).reshape(frame_h, frame_w)


def generate_mp4(raw_npy_path: Path, out_path: Path, capture_fps: float) -> float:
    shape, dtype, offset = read_npy_header(raw_npy_path)
    frame_count, frame_height, frame_width = int(shape[0]), int(shape[1]), int(shape[2])
    step = max(1, frame_count // MAX_FRAMES)
    selected_idxs = list(range(0, frame_count, step))

    preview_w, preview_h = frame_width, frame_height
    if frame_width > MAX_WIDTH:
        scale = MAX_WIDTH / frame_width
        preview_w = int(round(frame_width * scale))
        preview_h = int(round(frame_height * scale))

    probe = np.stack([
        read_frame_at(raw_npy_path, offset, dtype, frame_height, frame_width, i)
        for i in selected_idxs[: min(len(selected_idxs), 50)]
    ])
    lo = float(np.percentile(probe, 5))
    hi = float(np.percentile(probe, 99.5))

    preview_fps = min(MAX_DISPLAY_FPS, max(1.0, capture_fps / step))

    tmp_path = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), preview_fps, (preview_w, preview_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {tmp_path}")
    try:
        for i, idx in enumerate(selected_idxs):
            frame = read_frame_at(raw_npy_path, offset, dtype, frame_height, frame_width, idx)
            frame_u8 = np.clip((frame.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
            frame_u8 = (frame_u8 * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
            if (preview_w, preview_h) != (frame_width, frame_height):
                frame_bgr = cv2.resize(frame_bgr, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            cv2.putText(frame_bgr, f"frame {idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame_bgr)
    finally:
        writer.release()

    import subprocess
    import imageio_ffmpeg
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(tmp_path), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
        check=True, capture_output=True,
    )
    tmp_path.unlink()
    return preview_fps


def update_manifest_preview_fps(manifest_path: Path, new_fps: float) -> None:
    data = json.loads(manifest_path.read_text())
    spv = data.get("experiment_output", {}).get("speckle_vibrations", {})
    if "preview_fps" in spv:
        spv["preview_fps"] = new_fps
    manifest_path.write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-generate experiment-16 speckle mp4s at 30fps and re-upload to HF.")
    parser.add_argument("--data-root", default=str(Path.home() / "mark_sheinin_lab/DATA/experiment-16/data"))
    parser.add_argument("--repo-id", default="eturok-weizmann/laser-vibrations")
    parser.add_argument("--sample-ids", nargs="*", type=int, default=None, help="Specific sample IDs to process (default: all)")
    parser.add_argument("--skip-before", type=int, default=None, help="Skip sample IDs numerically less than this value (for resuming)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    api = HfApi()

    sample_dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit())
    if args.sample_ids:
        sample_dirs = [p for p in sample_dirs if int(p.name) in args.sample_ids]
    if args.skip_before is not None:
        sample_dirs = [p for p in sample_dirs if int(p.name) >= args.skip_before]

    print(f"[info] processing {len(sample_dirs)} samples from {data_root}")

    for i, sample_dir in enumerate(sample_dirs):
        sample_id = int(sample_dir.name)
        raw_npy = sample_dir / "speckle_vibration_raw.npy"
        manifest_path = sample_dir / "manifest.json"
        hf_mp4_path = f"data/{sample_dir.name}/speckle_vibrations.mp4"
        hf_manifest_path = f"data/{sample_dir.name}/manifest.json"

        if not raw_npy.exists():
            print(f"[warn] [{i+1}/{len(sample_dirs)}] sample {sample_id:07d}: raw npy not found, skipping")
            continue

        capture_fps = 2500.0
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text())
                capture_fps = float(m.get("experiment_output", {}).get("speckle_vibrations", {}).get("capture_fps_hz") or 2500.0)
            except Exception:
                pass

        print(f"[info] [{i+1}/{len(sample_dirs)}] sample {sample_id:07d}: capture_fps={capture_fps}")

        if args.dry_run:
            step = max(1, 9000 // MAX_FRAMES)
            print(f"  dry-run: would generate at fps={min(MAX_DISPLAY_FPS, capture_fps / step):.2f}, upload to {hf_mp4_path}")
            continue

        with tempfile.TemporaryDirectory(prefix=f"regen-{sample_id:07d}-") as tmp:
            out_mp4 = Path(tmp) / "speckle_vibrations.mp4"
            actual_fps = stage(
                f"generate mp4 sample {sample_id:07d}",
                lambda: generate_mp4(raw_npy, out_mp4, capture_fps),
            )
            print(f"  preview_fps={actual_fps:.2f}")

            ops = [CommitOperationAdd(path_in_repo=hf_mp4_path, path_or_fileobj=str(out_mp4))]

            if manifest_path.exists():
                updated_manifest = Path(tmp) / "manifest.json"
                import shutil
                shutil.copy2(manifest_path, updated_manifest)
                update_manifest_preview_fps(updated_manifest, actual_fps)
                ops.append(CommitOperationAdd(path_in_repo=hf_manifest_path, path_or_fileobj=str(updated_manifest)))

            for attempt in range(10):
                try:
                    stage(
                        f"upload sample {sample_id:07d}",
                        lambda: api.create_commit(
                            repo_id=args.repo_id,
                            repo_type="dataset",
                            commit_message=f"Regen speckle mp4 at {actual_fps:.0f}fps for sample {sample_id:07d}",
                            create_pr=False,
                            operations=ops,
                        ),
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 9:
                        wait = 3600
                        print(f"[rate-limit] HF 429 on sample {sample_id:07d}, waiting {wait}s before retry {attempt+1}/9...")
                        time.sleep(wait)
                    else:
                        raise

    print(f"[done] processed {len(sample_dirs)} samples")


if __name__ == "__main__":
    main()
