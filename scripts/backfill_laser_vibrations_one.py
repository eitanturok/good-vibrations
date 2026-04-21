import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_AUDIO_ROOT = REPO_ROOT / "data" / "audio_samples"
REMOTE_HOST = "ethantu@mcluster11.wisdom.weizmann.ac.il"
REMOTE_UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"
REMOTE_UV = "$HOME/.local/bin/uv"
REMOTE_VENV = "$HOME/venvs/laser-vibrations-uv"
REMOTE_PIP_PACKAGES = [
    "numpy==1.26.4",
    "opencv-python-headless==4.10.0.84",
    "huggingface_hub==0.31.2",
    "imageio==2.37.0",
    "imageio-ffmpeg==0.5.1",
]


def stage(label, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"[timing] {label}: {dt:.2f}s")
    return result


def get_hf_token() -> str:
    token_file = Path("~/.cache/huggingface/token").expanduser()
    if token_file.exists():
        return token_file.read_text().strip()
    raise RuntimeError("No local Hugging Face token found; run `huggingface-cli login` first")


def speaker_code(speakers) -> str:
    return "".join(str(int(x)) for x in (speakers or []))


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
    cols = ["sample_idx", "object", "n_objects", "speakers", "box_material", "x_position", "y_position"]
    ds = load_dataset(repo_id, split="train", columns=cols, verification_mode="no_checks")
    target_row = None
    xs, ys, sample_ids, objects = [], [], [], []
    for row in ds:
        if int(row["sample_idx"]) == sample_id:
            target_row = dict(row)
        if row.get("object") != "empty":
            xs.append(float(row["x_position"]))
            ys.append(float(row["y_position"]))
            sample_ids.append(int(row["sample_idx"]))
            objects.append(row.get("object", ""))
    if target_row is None:
        raise ValueError(f"Sample {sample_id} not found in {repo_id}")
    return target_row, xs, ys, sample_ids, objects


def infer_discrete_position(sample_row: dict, xs: list, ys: list, sample_ids: list, objects: list) -> tuple[int, int]:
    target_obj = sample_row.get("object", "")
    obj_xs = [x for x, obj in zip(xs, objects) if obj == target_obj]
    obj_ys = [y for y, obj in zip(ys, objects) if obj == target_obj]
    obj_ids = [sid for sid, obj in zip(sample_ids, objects) if obj == target_obj]
    x_centers, x_labels = kmeans_1d(np.asarray(obj_xs), 11)
    y_centers, y_labels = kmeans_1d(np.asarray(obj_ys), 12)
    idx = obj_ids.index(int(sample_row["sample_idx"]))
    x_idx = int(x_labels[idx])
    y_idx = int(y_labels[idx]) + 1
    print(f"[info] inferred discrete position: x={x_idx:02d} y={y_idx:02d} from x={sample_row['x_position']:.3f} y={sample_row['y_position']:.3f}")
    return x_idx, y_idx


def discover_source_experiment_dir(sample_row: dict, source_data_root: str, x_idx: int, y_idx: int) -> str:
    obj = sample_row.get("object", "")
    spk = speaker_code(sample_row.get("speakers"))
    basename_pattern = f"{obj}-{x_idx:02d}x{y_idx:02d}y_{spk}--*"
    t0 = time.perf_counter()
    script = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        f"root = Path({source_data_root!r})\n"
        f"pattern = {basename_pattern!r}\n"
        "for p in sorted(root.rglob(pattern)):\n"
        "    if p.is_dir():\n"
        "        print(p)\n"
        "PY\n"
    )
    result = subprocess.run(["ssh", REMOTE_HOST, "bash", "-s"], input=script, check=True, text=True, capture_output=True)
    candidates = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    print(
        f"[timing] discover source dir: {time.perf_counter() - t0:.2f}s "
        f"(pattern={basename_pattern}, n={len(candidates)})"
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one source experiment dir, got {len(candidates)}: {candidates[:5]}")
    return candidates[0]


def load_remote_experiment_config(source_experiment_dir: str) -> dict:
    cmd = f"cat {shlex.quote(source_experiment_dir + '/experiment_config.json')}"
    t0 = time.perf_counter()
    result = subprocess.run(["ssh", REMOTE_HOST, cmd], check=True, text=True, capture_output=True)
    print(f"[timing] fetch remote experiment_config.json: {time.perf_counter() - t0:.2f}s")
    return json.loads(result.stdout)


REMOTE_SCRIPT_PATH = "$HOME/tmp/upload_remote_speckle_assets.py"
REMOTE_AUDIO_PATH = "$HOME/tmp/chirp_audio.wav"


def resolve_audio_path(cfg: dict) -> Path | None:
    raw = cfg.get("AUDIO_FILE") or cfg.get("audio_file") or cfg.get("audio")
    if not raw:
        return None
    basename = Path(str(raw).replace("\\", "/")).name
    local_path = LOCAL_AUDIO_ROOT / basename
    return local_path if local_path.exists() else None


def run_remote_speckle_upload(sample_id: int, source_experiment_dir: str, fps: float, repo_id: str, audio_src: Path | None = None) -> None:
    hf_token = get_hf_token()

    with open(Path(__file__).resolve().parent / "upload_remote_speckle_assets.py", "rb") as f:
        stage(
            "sync remote uploader script",
            lambda: subprocess.run(
                ["ssh", REMOTE_HOST, f"mkdir -p $HOME/tmp && cat > {REMOTE_SCRIPT_PATH}"],
                check=True,
                stdin=f,
            ),
        )

    if audio_src is not None:
        with open(audio_src, "rb") as f:
            stage(
                "sync audio file to cluster",
                lambda: subprocess.run(
                    ["ssh", REMOTE_HOST, f"cat > {REMOTE_AUDIO_PATH}"],
                    check=True,
                    stdin=f,
                ),
            )

    remote_args = [
        "--sample-id", str(sample_id),
        "--source-experiment-dir", source_experiment_dir,
        "--repo-id", repo_id,
        "--fps", str(fps),
        "--no-create-pr",
    ]
    if audio_src is not None:
        remote_args += ["--audio-path", REMOTE_AUDIO_PATH]
    remote_cmd = (
        f"bash -lc {shlex.quote(
            f'source /etc/profile >/dev/null 2>&1 || true; '
            f'export HF_TOKEN={shlex.quote(hf_token)}; '
            f'export HUGGINGFACE_HUB_TOKEN={shlex.quote(hf_token)}; '
            f'{REMOTE_UV_INSTALL}; '
            f'{REMOTE_UV} python install 3.10; '
            f'{REMOTE_UV} venv --python 3.10 {REMOTE_VENV}; '
            f'{REMOTE_UV} pip install --python {REMOTE_VENV}/bin/python --only-binary=:all: {' '.join(REMOTE_PIP_PACKAGES)} >/dev/null; '
            f'{REMOTE_VENV}/bin/python {REMOTE_SCRIPT_PATH} ' + f"{' '.join(shlex.quote(a) for a in remote_args)}"
        )}"
    )
    stage("remote upload raw/mp4 assets", lambda: subprocess.run(["ssh", REMOTE_HOST, remote_cmd], check=True))


def run_local_backfill(sample_id: int, source_experiment_dir: str, repo_id_old: str, repo_id_new: str) -> None:
    cmd = [
        "uv", "run", "python",
        "scripts/backfill_laser_vibrations.py",
        "--sample-id", str(sample_id),
        "--source-experiment-dir", source_experiment_dir,
        "--old-repo-id", repo_id_old,
        "--new-repo-id", repo_id_new,
        "--skip-remote-speckle-assets",
        "--no-create-pr",
    ]
    stage("local backfill remaining assets", lambda: subprocess.run(cmd, check=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill one sample into laser-vibrations using remote raw/mp4 upload and local metadata/tensor upload.")
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--old-repo-id", default="eturok-weizmann/vibrations")
    parser.add_argument("--new-repo-id", default="eturok-weizmann/laser-vibrations")
    parser.add_argument("--source-data-root", default="/net/mraid20/ifs/wisdom/groups/mark_sheinin_lab/DATA")
    args = parser.parse_args()

    row, xs, ys, sample_ids, objects = stage("load old dataset row + cluster data", lambda: load_old_row_and_cluster_data(args.sample_id, args.old_repo_id))
    x_idx, y_idx = infer_discrete_position(row, xs, ys, sample_ids, objects)
    source_experiment_dir = discover_source_experiment_dir(row, args.source_data_root, x_idx, y_idx)
    print(f"[info] source_experiment_dir={source_experiment_dir}")
    cfg = load_remote_experiment_config(source_experiment_dir)
    fps = float(cfg.get("FPS") or 0)
    print(f"[info] source fps={fps}")
    audio_src = resolve_audio_path(cfg)
    print(f"[info] audio_src={audio_src}")

    run_remote_speckle_upload(args.sample_id, source_experiment_dir, fps, args.new_repo_id, audio_src=audio_src)
    run_local_backfill(args.sample_id, source_experiment_dir, args.old_repo_id, args.new_repo_id)


if __name__ == "__main__":
    main()
