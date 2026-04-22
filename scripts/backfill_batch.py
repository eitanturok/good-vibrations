"""
Batch backfill with a single persistent Modal segmentation container.

Opens app.run() once so the GPU container stays warm across all samples,
eliminating the ~35s cold-start penalty that the per-sample script pays.

Pipeline per sample:
  1. pre_segment  — runs on mcluster11: package raw files, FFT, crop overhead
  2. segment      — runs locally:       call segmenter.segment.remote() (warm container)
  3. push mask    — push mask.npz + mask.png back to mraid20 over SSH
  4. post_segment — runs on mcluster11: overlay, speaker overlay, manifest, metadata

Pipelining: pre_segment for sample N+1 runs in a background thread while
modal inference runs for sample N, hiding most of the ~30s I/O cost.

Usage (one sample):
    python scripts/backfill_batch.py --samples cube-00x01y_0001--31-03-18-21-24:1

Usage (multiple samples):
    python scripts/backfill_batch.py --samples cube-00x01y_0001--31-03-18-21-24:1 cube-01x02y_0001--01-04-10-00-00:2
"""
import io
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from utils.segment import app, Segmenter
from migrate_experiment15_to_16_one import build_image_dir_name

REMOTE_HOST   = "ethantu@mcluster11.wisdom.weizmann.ac.il"
REMOTE_VENV   = "$HOME/venvs/experiment16-migrate"
REMOTE_SCRIPT = "/home/ethantu/tmp/migrate_experiment15_to_16_one.py"
OLD_DIR       = "/net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA/experiment-15"
NEW_DIR       = "/net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA/experiment-16"
HF_REPO       = "eturok-weizmann/laser-vibrations"
DEFAULT_PROMPT = "A black metal cube sitting on the floor of an open cardboard box from a bird's eye view."


# ── SSH helpers ───────────────────────────────────────────────────────────────

def _ssh_run(cmd: str) -> None:
    """Run a remote command, streaming output to stdout."""
    subprocess.run(
        ["ssh", REMOTE_HOST, f"bash -lc {shlex.quote(cmd)}"],
        check=True,
    )


def _ssh_fetch(cmd: str, stdin: bytes | None = None) -> bytes:
    """Run a remote command and capture its stdout."""
    result = subprocess.run(
        ["ssh", REMOTE_HOST, f"bash -lc {shlex.quote(cmd)}"],
        check=True, capture_output=True, input=stdin,
    )
    return result.stdout


def _sync_static_files() -> None:
    """Sync the migration script and static assets to mcluster11 (checksum-skipped if unchanged)."""
    import hashlib

    def _local_md5(path: Path) -> str:
        h = hashlib.md5()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _remote_md5(remote_path: str) -> str | None:
        cmd = f"md5sum {shlex.quote(remote_path)} 2>/dev/null || true"
        line = _ssh_fetch(cmd).decode().strip()
        return line.split()[0] if line else None

    def sync(local: Path, remote: str, label: str) -> None:
        t = time.perf_counter()
        if _local_md5(local) == _remote_md5(remote):
            print(f"[sync] {label}: skipped (unchanged)", flush=True)
            return
        _ssh_fetch(f"mkdir -p $(dirname {shlex.quote(remote)}) && cat > {shlex.quote(remote)}", stdin=local.read_bytes())
        print(f"[sync] {label}: {time.perf_counter() - t:.2f}s", flush=True)

    migrate_script = REPO_ROOT / "scripts" / "migrate_experiment15_to_16_one.py"
    sync(migrate_script, REMOTE_SCRIPT, "migrate script")
    for key in ("1000", "0100", "0010", "0001"):
        sync(REPO_ROOT / "assets" / "speakers" / f"{key}.png", f"/home/ethantu/assets/speakers/{key}.png", f"speaker {key}.png")


def _run_remote_stage(stage: str, source_dir_name: str, sample_id: int) -> None:
    # Note: REMOTE_VENV contains $HOME which must NOT be single-quoted so bash expands it.
    quoted_args = " ".join(shlex.quote(a) for a in [
        REMOTE_SCRIPT, "--remote-worker",
        "--old-dir", OLD_DIR, "--new-dir", NEW_DIR, "--hf-repo", HF_REPO,
        "--source-dir-name", source_dir_name,
        "--sample-id", str(sample_id),
        "--stage", stage, "--overwrite",
    ])
    _ssh_run(f"{REMOTE_VENV}/bin/python {quoted_args}")


# ── Path helpers ──────────────────────────────────────────────────────────────

def _image_dir(
    source_dir_name: str,
    object_name: str | None = None,
    n_objects: int = 1,
    box_material: str = "cardboard",
    tags: list[str] | None = None,
) -> str:
    """Canonical image directory name matching migrate_experiment15_to_16_one.build_image_dir_name."""
    m = re.search(r"(?P<x>\d{2})x(?P<y>\d{2})y", source_dir_name)
    x = int(m.group("x")) if m else None
    y = int(m.group("y")) if m else None
    obj = object_name or source_dir_name.split("-")[0].strip().lower()
    return build_image_dir_name(
        source_dir_name,
        object_name=obj,
        x_position=x,
        y_position=y,
        n_objects=n_objects,
        box_material=box_material,
        tags=tags,
    )


# ── Per-sample pipeline ───────────────────────────────────────────────────────

def process_sample(source_dir_name: str, sample_id: int, segmenter: Segmenter,
                   pre_segment_future: Future | None = None) -> None:
    t0 = time.perf_counter()
    image_dir = _image_dir(source_dir_name)
    remote_image_root = f"{NEW_DIR}/data/image/{image_dir}"
    object_name = source_dir_name.split("-")[0].strip().lower()

    # 1. Wait for pre_segment to finish (may already be done if pipelined)
    if pre_segment_future is not None:
        t = time.perf_counter()
        pre_segment_future.result()  # raises if it failed
        print(f"[timing] pre_segment:   {time.perf_counter() - t:.2f}s (waited)", flush=True)
    else:
        t = time.perf_counter()
        _run_remote_stage("pre_segment", source_dir_name, sample_id)
        print(f"[timing] pre_segment:   {time.perf_counter() - t:.2f}s", flush=True)

    # 2. Fetch cropped overhead image for local Modal call
    t = time.perf_counter()
    cropped_bytes = _ssh_fetch(f"cat {shlex.quote(remote_image_root + '/cropped_overhead.png')}")
    cropped_arr = np.array(Image.open(io.BytesIO(cropped_bytes)).convert("RGB"), dtype=np.uint8)
    print(f"[timing] fetch cropped: {time.perf_counter() - t:.2f}s", flush=True)

    # 3. Segment via Modal (container already warm — no cold start)
    t = time.perf_counter()
    mask, _ = segmenter.segment.remote(cropped_arr, object_name, "cardboard", DEFAULT_PROMPT)
    print(f"[timing] modal segment: {time.perf_counter() - t:.2f}s", flush=True)

    # 4. Push mask back to mraid20
    t = time.perf_counter()
    mask_npz_buf = io.BytesIO()
    np.savez_compressed(mask_npz_buf, mask=mask)
    _ssh_fetch(f"cat > {shlex.quote(remote_image_root + '/mask.npz')}", stdin=mask_npz_buf.getvalue())
    mask_png_buf = io.BytesIO()
    Image.fromarray((np.clip(mask.astype(np.float32), 0, 1) * 255).astype(np.uint8)).save(mask_png_buf, format="PNG")
    _ssh_fetch(f"cat > {shlex.quote(remote_image_root + '/mask.png')}", stdin=mask_png_buf.getvalue())
    print(f"[timing] push mask:     {time.perf_counter() - t:.2f}s", flush=True)

    # 5. Post-segmentation: overlay, speaker overlay, manifest, metadata
    t = time.perf_counter()
    _run_remote_stage("post_segment", source_dir_name, sample_id)
    print(f"[timing] post_segment:  {time.perf_counter() - t:.2f}s", flush=True)

    print(f"[timing] sample total:  {time.perf_counter() - t0:.2f}s", flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--samples", nargs="+", required=True, metavar="DIR_NAME:SAMPLE_ID",
        help="e.g. cube-00x01y_0001--31-03-18-21-24:1",
    )
    return p.parse_args()


def main():
    args = parse_args()
    samples = []
    for s in args.samples:
        name, sid = s.rsplit(":", 1)
        samples.append((name, int(sid)))

    _sync_static_files()

    segmenter = Segmenter()

    with app.run(), ThreadPoolExecutor(max_workers=1) as executor:
        # Kick off pre_segment for the first sample immediately
        pre_future = executor.submit(_run_remote_stage, "pre_segment", samples[0][0], samples[0][1])

        for i, (source_dir_name, sample_id) in enumerate(samples):
            print(f"\n[batch] {source_dir_name} (sample_id={sample_id})", flush=True)

            # Start pre_segment for the next sample in background (if there is one)
            next_future = None
            if i + 1 < len(samples):
                next_name, next_id = samples[i + 1]
                next_future = executor.submit(_run_remote_stage, "pre_segment", next_name, next_id)

            process_sample(source_dir_name, sample_id, segmenter, pre_segment_future=pre_future)

            pre_future = next_future


if __name__ == "__main__":
    main()
