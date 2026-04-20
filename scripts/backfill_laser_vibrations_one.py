import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import get_token


REMOTE_HOST = "ethantu@mcluster11.wisdom.weizmann.ac.il"
REMOTE_REPO = "mark_sheinin_lab/code/eitan/good-vibrations"
REMOTE_MAMBA = "$HOME/bin/micromamba"
REMOTE_ENV_PREFIX = "$HOME/micromamba/envs/laser-vibrations"
REMOTE_UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"
REMOTE_UV = "$HOME/.local/bin/uv"
REMOTE_VENV = "$HOME/venvs/laser-vibrations-uv"
REMOTE_PIP_PACKAGES = [
    "numpy==1.26.4",
    "opencv-python-headless==4.10.0.84",
    "huggingface_hub==0.31.2",
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


def load_old_row(sample_id: int, repo_id: str) -> dict:
    cols = ["sample_idx", "object", "n_objects", "speakers", "box_material", "x_position", "y_position"]
    ds = load_dataset(repo_id, split="train", columns=cols, verification_mode="no_checks")
    for row in ds:
        if int(row["sample_idx"]) == sample_id:
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
    return np.sort(np.unique(np.round(np.asarray(xs), 6))), np.sort(np.unique(np.round(np.asarray(ys), 6)))


def kmeans_1d(values, k: int, n_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
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


def infer_discrete_position(sample_row: dict, repo_id: str) -> tuple[int, int]:
    cols = ["sample_idx", "object", "x_position", "y_position"]
    ds = load_dataset(repo_id, split="train", columns=cols, verification_mode="no_checks")
    xs, ys, sample_ids = [], [], []
    for row in ds:
        if row.get("object") != sample_row.get("object"):
            continue
        if row.get("object") == "empty":
            continue
        xs.append(float(row["x_position"]))
        ys.append(float(row["y_position"]))
        sample_ids.append(int(row["sample_idx"]))
    x_centers, x_labels = kmeans_1d(xs, 11)
    y_centers, y_labels = kmeans_1d(ys, 12)
    idx = sample_ids.index(int(sample_row["sample_idx"]))
    x_idx = int(x_labels[idx])
    y_idx = int(y_labels[idx]) + 1
    print(f"[info] inferred discrete position: x={x_idx:02d} y={y_idx:02d} from x={sample_row['x_position']:.3f} y={sample_row['y_position']:.3f}")
    print(f"[debug] x_centers={np.round(x_centers, 2).tolist()}")
    print(f"[debug] y_centers={np.round(y_centers, 2).tolist()}")
    return x_idx, y_idx


def discover_source_experiment_dir(sample_row: dict, source_data_root: str, x_idx: int, y_idx: int) -> str:
    obj = sample_row.get("object", "")
    spk = speaker_code(sample_row.get("speakers"))
    basename_pattern = f"{obj}-{x_idx:02d}x{y_idx:02d}y_{spk}--*"
    inner = (
        "python3 -c "
        + shlex.quote(
            "from pathlib import Path; "
            f"root = Path({source_data_root!r}); "
            f"pattern = {basename_pattern!r}; "
            "[print(p) for p in sorted(root.rglob(pattern)) if p.is_dir()]"
        )
    )
    cmd = f"sh -lc {shlex.quote(inner)}"
    t0 = time.perf_counter()
    result = subprocess.run(["ssh", REMOTE_HOST, cmd], check=True, text=True, capture_output=True)
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


def run_remote_speckle_upload(sample_id: int, source_experiment_dir: str, fps: float, repo_id: str) -> None:
    hf_token = get_hf_token()
    remote_run_repo = "$HOME/tmp/laser-vibrations-run/good-vibrations"

    stage(
        "prepare fresh remote repo clone",
        lambda: subprocess.run(
            [
                "ssh",
                REMOTE_HOST,
                "bash -lc "
                + shlex.quote(
                    "mkdir -p $HOME/tmp/laser-vibrations-run && "
                    "rm -rf $HOME/tmp/laser-vibrations-run/good-vibrations && "
                    "git clone https://github.com/eitanturok/good-vibrations $HOME/tmp/laser-vibrations-run/good-vibrations && "
                    "cd $HOME/tmp/laser-vibrations-run/good-vibrations && git pull"
                ),
            ],
            check=True,
        ),
    )
    with open(Path(__file__).resolve().parent / "upload_remote_speckle_assets.py", "rb") as f:
        stage(
            "sync remote uploader script",
            lambda: subprocess.run(
                ["ssh", REMOTE_HOST, f"cat > {remote_run_repo}/scripts/upload_remote_speckle_assets.py"],
                check=True,
                stdin=f,
            ),
        )

    remote_args = [
        "python", "scripts/upload_remote_speckle_assets.py",
        "--sample-id", str(sample_id),
        "--source-experiment-dir", source_experiment_dir,
        "--repo-id", repo_id,
        "--fps", str(fps),
        "--no-create-pr",
    ]
    remote_cmd = (
        f"bash -lc {shlex.quote(
            f'source /etc/profile >/dev/null 2>&1 || true; '
            f'export HF_TOKEN={shlex.quote(hf_token)}; '
            f'export HUGGINGFACE_HUB_TOKEN={shlex.quote(hf_token)}; '
            f'{REMOTE_UV_INSTALL}; '
            f'{REMOTE_UV} python install 3.10; '
            f'{REMOTE_UV} venv --python 3.10 --clear {REMOTE_VENV}; '
            f'{REMOTE_UV} pip install --python {REMOTE_VENV}/bin/python --only-binary=:all: {' '.join(REMOTE_PIP_PACKAGES)} >/dev/null; '
            f'cd {remote_run_repo}; '
            f'{REMOTE_VENV}/bin/python ' + f"{' '.join(shlex.quote(a) for a in remote_args[1:])}"
        )}"
    )
    stage("remote upload raw/mp4 assets", lambda: subprocess.run(["ssh", REMOTE_HOST, remote_cmd], check=True))


def run_local_backfill(sample_id: int, source_experiment_dir: str, repo_id_old: str, repo_id_new: str) -> None:
    cmd = [
        "python",
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

    row = stage("load old dataset row", lambda: load_old_row(args.sample_id, args.old_repo_id))
    x_idx, y_idx = stage("infer discrete source position", lambda: infer_discrete_position(row, args.old_repo_id))
    source_experiment_dir = discover_source_experiment_dir(row, args.source_data_root, x_idx, y_idx)
    print(f"[info] source_experiment_dir={source_experiment_dir}")
    cfg = load_remote_experiment_config(source_experiment_dir)
    fps = float(cfg.get("FPS") or cfg.get("camera_FPS") or 0)
    print(f"[info] source fps={fps}")

    run_remote_speckle_upload(args.sample_id, source_experiment_dir, fps, args.new_repo_id)
    run_local_backfill(args.sample_id, source_experiment_dir, args.old_repo_id, args.new_repo_id)


if __name__ == "__main__":
    main()
