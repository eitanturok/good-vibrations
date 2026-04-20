import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import HfApi
from numpy.lib import format as npy_format


DEFAULT_REPO_ID = "eturok-weizmann/laser-vibrations"


def stage(label, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"[timing] {label}: {dt:.2f}s")
    return result


def sample_dir_rel(sample_id: int) -> str:
    return f"samples/sample_{sample_id:06d}"


def read_npy_header(raw_npy_path: Path) -> tuple[tuple[int, int, int], np.dtype, int]:
    with raw_npy_path.open("rb") as f:
        major, minor = npy_format.read_magic(f)
        if (major, minor) == (1, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_1_0(f)
        elif (major, minor) == (2, 0):
            shape, fortran_order, dtype = npy_format.read_array_header_2_0(f)
        else:
            raise ValueError(f"Unsupported .npy version {(major, minor)} for {raw_npy_path}")
        if fortran_order:
            raise ValueError(f"Fortran-order arrays are not supported for preview generation: {raw_npy_path}")
        offset = f.tell()
    return shape, np.dtype(dtype), offset


def read_frame(raw_npy_path: Path, offset: int, dtype: np.dtype, frame_shape: tuple[int, int], frame_idx: int) -> np.ndarray:
    frame_size = int(np.prod(frame_shape))
    byte_offset = offset + frame_idx * frame_size * dtype.itemsize
    with raw_npy_path.open("rb") as f:
        f.seek(byte_offset)
        frame = np.fromfile(f, dtype=dtype, count=frame_size)
    return frame.reshape(frame_shape)


def generate_speckle_preview(raw_npy_path: Path, out_path: Path, fps: float, max_frames: int = 300, max_width: int = 960) -> tuple[int, int, int]:
    shape, dtype, offset = read_npy_header(raw_npy_path)
    frame_count, frame_height, frame_width = shape
    step = max(1, frame_count // max_frames)
    selected_idxs = list(range(0, frame_count, step))
    preview_h, preview_w = frame_height, frame_width
    if frame_width > max_width:
        scale = max_width / frame_width
        preview_w = int(round(frame_width * scale))
        preview_h = int(round(frame_height * scale))

    probe = np.asarray(
        [
            read_frame(raw_npy_path, offset, dtype, (frame_height, frame_width), idx)
            for idx in selected_idxs[: min(len(selected_idxs), 50)]
        ]
    )
    lo = float(np.percentile(probe, 5))
    hi = float(np.percentile(probe, 99.5))
    preview_fps = min(30.0, max(1.0, fps / step))
    tmp_path = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), preview_fps, (preview_w, preview_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {tmp_path}")
    try:
        for i, idx in enumerate(selected_idxs):
            frame = read_frame(raw_npy_path, offset, dtype, (frame_height, frame_width), idx)
            frame_u8 = np.clip((frame.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
            frame_u8 = (frame_u8 * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
            if (preview_w, preview_h) != (frame_width, frame_height):
                frame_bgr = cv2.resize(frame_bgr, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            cv2.putText(frame_bgr, f"frame {idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame_bgr)
    finally:
        writer.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(tmp_path), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
        check=True, capture_output=True,
    )
    tmp_path.unlink()
    return frame_count, frame_height, frame_width


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload raw speckle assets from mcluster11 to laser-vibrations.")
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--source-experiment-dir", required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--create-pr", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    api = HfApi()
    rel_dir = sample_dir_rel(args.sample_id)
    raw_path = Path(args.source_experiment_dir) / "frame-recording.npy"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    with tempfile.TemporaryDirectory(prefix=f"remote-speckle-{args.sample_id:06d}-") as tmp:
        preview_path = Path(tmp) / "speckle_vibrations.mp4"
        frame_count, frame_height, frame_width = stage(
            "generate speckle preview mp4",
            lambda: generate_speckle_preview(raw_path, preview_path, fps=args.fps),
        )
        stage(
            "upload raw speckle recording",
            lambda: api.upload_file(
                path_or_fileobj=str(raw_path),
                path_in_repo=f"{rel_dir}/speckle_vibrations_raw.npy",
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"Upload raw speckle assets for sample {args.sample_id:06d}",
                create_pr=args.create_pr,
            ),
        )
        stage(
            "upload speckle preview mp4",
            lambda: api.upload_file(
                path_or_fileobj=str(preview_path),
                path_in_repo=f"{rel_dir}/speckle_vibrations.mp4",
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"Upload speckle preview for sample {args.sample_id:06d}",
                create_pr=args.create_pr,
            ),
        )

    print(f"[done] uploaded raw speckle assets for sample {args.sample_id:06d}")
    print(f"[info] frame_count={frame_count} frame_height={frame_height} frame_width={frame_width}")


if __name__ == "__main__":
    main()
