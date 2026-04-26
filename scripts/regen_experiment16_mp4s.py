"""
Re-generate speckle_vibrations.mp4 for all experiment-16 samples at 30fps and re-upload to HF.

Batches multiple samples per commit to stay under HF's 128-commits/hour rate limit.

Run on mcluster11 (where the raw .npy files live):
    $HOME/venvs/laser-vibrations-uv/bin/python scripts/regen_experiment16_mp4s.py \
        --data-root /home/ethantu/mark_sheinin_lab/DATA/experiment-16/data \
        --repo-id eturok-weizmann/laser-vibrations
"""
import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi

MAX_DISPLAY_FPS = 30.0
MAX_FRAMES = 300
MAX_WIDTH = 960


def log(msg: str) -> None:
    print(msg, flush=True)


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


def update_manifest_preview_fps(src: Path, dst: Path, new_fps: float) -> None:
    data = json.loads(src.read_text())
    spv = data.get("experiment_output", {}).get("speckle_vibrations", {})
    if "preview_fps" in spv:
        spv["preview_fps"] = new_fps
    dst.write_text(json.dumps(data, indent=2))


def commit_with_retry(api: HfApi, repo_id: str, commit_message: str, ops: list) -> None:
    for attempt in range(10):
        try:
            t0 = time.perf_counter()
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
                create_pr=False,
                operations=ops,
            )
            log(f"[timing] commit ({len(ops)} ops): {time.perf_counter() - t0:.2f}s")
            return
        except Exception as e:
            if "429" in str(e) and attempt < 9:
                log(f"[rate-limit] 429 on attempt {attempt+1}/10, waiting 3600s...")
                time.sleep(3600)
            else:
                raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-generate experiment-16 speckle mp4s at 30fps and re-upload to HF.")
    parser.add_argument("--data-root", default=str(Path.home() / "mark_sheinin_lab/DATA/experiment-16/data"))
    parser.add_argument("--repo-id", default="eturok-weizmann/laser-vibrations")
    parser.add_argument("--batch-size", type=int, default=50, help="Samples per HF commit (default: 50)")
    parser.add_argument("--sample-ids", nargs="*", type=int, default=None, help="Specific sample IDs to process")
    parser.add_argument("--skip-before", type=int, default=None, help="Skip sample IDs less than this value (for resuming)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    api = HfApi()

    sample_dirs = sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit())
    if args.sample_ids:
        sample_dirs = [p for p in sample_dirs if int(p.name) in args.sample_ids]
    if args.skip_before is not None:
        sample_dirs = [p for p in sample_dirs if int(p.name) >= args.skip_before]

    n_batches = (len(sample_dirs) + args.batch_size - 1) // args.batch_size
    log(f"[info] {len(sample_dirs)} samples → {n_batches} batches of {args.batch_size}")

    for batch_num, batch_start in enumerate(range(0, len(sample_dirs), args.batch_size)):
        batch = sample_dirs[batch_start: batch_start + args.batch_size]
        first_id = int(batch[0].name)
        last_id = int(batch[-1].name)
        log(f"[batch {batch_num+1}/{n_batches}] samples {first_id:07d}–{last_id:07d} ({len(batch)} samples)")

        if args.dry_run:
            for sample_dir in batch:
                log(f"  dry-run: {sample_dir.name}/speckle_vibrations.mp4 @ {MAX_DISPLAY_FPS:.0f}fps")
            continue

        with tempfile.TemporaryDirectory(prefix=f"regen-batch{batch_num+1}-") as tmp:
            tmp_path = Path(tmp)
            ops = []

            for j, sample_dir in enumerate(batch):
                sample_id = int(sample_dir.name)
                raw_npy = sample_dir / "speckle_vibration_raw.npy"
                manifest_src = sample_dir / "manifest.json"

                if not raw_npy.exists():
                    log(f"  [warn] {sample_dir.name}: raw npy not found, skipping")
                    continue

                capture_fps = 2500.0
                if manifest_src.exists():
                    try:
                        m = json.loads(manifest_src.read_text())
                        capture_fps = float(m.get("experiment_output", {}).get("speckle_vibrations", {}).get("capture_fps_hz") or 2500.0)
                    except Exception:
                        pass

                out_mp4 = tmp_path / f"{sample_dir.name}.mp4"
                t0 = time.perf_counter()
                actual_fps = generate_mp4(raw_npy, out_mp4, capture_fps)
                log(f"  [{j+1}/{len(batch)}] {sample_dir.name}: generated at {actual_fps:.0f}fps ({time.perf_counter()-t0:.1f}s)")

                ops.append(CommitOperationAdd(
                    path_in_repo=f"data/{sample_dir.name}/speckle_vibrations.mp4",
                    path_or_fileobj=str(out_mp4),
                ))

                if manifest_src.exists():
                    updated_manifest = tmp_path / f"{sample_dir.name}_manifest.json"
                    update_manifest_preview_fps(manifest_src, updated_manifest, actual_fps)
                    ops.append(CommitOperationAdd(
                        path_in_repo=f"data/{sample_dir.name}/manifest.json",
                        path_or_fileobj=str(updated_manifest),
                    ))

            if ops:
                log(f"[batch {batch_num+1}/{n_batches}] uploading {len(ops)} files...")
                commit_with_retry(
                    api, args.repo_id,
                    f"Regen speckle mp4s at 30fps: samples {first_id:07d}–{last_id:07d}",
                    ops,
                )
                log(f"[batch {batch_num+1}/{n_batches}] done")

    log(f"[done] all {len(sample_dirs)} samples processed")


if __name__ == "__main__":
    main()
