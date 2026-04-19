#!/usr/bin/env python3
"""Submit a Slurm job on the best available GPU and auto-resubmit near timeout."""

import argparse
import shlex
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Optional


PARTITIONS = [
    ("normal.q", 6, 9, "6:00:00"),
    ("long.q", 2, 4, "12:00:00"),
]

GPUS = [
    ("l40s", "gpu:L40S:1", 48, 48),
    ("quadro_rtx_8000", "gpu:quadro_rtx_8000:1", 48, 48),
    ("a10", "gpu:a10:1", 24, 48),
    ("quadro_rtx_6000", "gpu:quadro_rtx_6000:1", 24, 48),
]


def sh(cmd: str) -> str:
    result = subprocess.run(
        cmd,
        shell=True,
        check=False,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Submit a Slurm job using the best available resources"
    )
    ap.add_argument("--job-name", default="job")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpu", choices=[name for name, *_ in GPUS])
    ap.add_argument("--dependency", default=None)
    ap.add_argument("--signal-seconds", type=int, default=600)
    ap.add_argument("--time", default=None, help="Override Slurm walltime, e.g. 00:15:00")
    return ap.parse_known_args()


def strip_arg(tokens: List[str], flag: str) -> List[str]:
    cleaned: List[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token == flag:
            skip_next = True
            continue
        if token.startswith(f"{flag}="):
            continue
        cleaned.append(token)
    return cleaned


def normalize_model_args(model_args: List[str], job_name: str) -> List[str]:
    model_args = strip_arg(model_args, "--run-name")
    return model_args + ["--run-name", job_name]


def job_counts():
    counts = {}
    for line in sh("squeue -u $USER -h -o '%P %T'").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        partition, state = parts
        running, total = counts.get(partition, (0, 0))
        counts[partition] = (running + (state == "RUNNING"), total + 1)
    return counts


def best_gpu(partition: str):
    idle = set()
    for line in sh(f"sinfo -p {partition} -h -o '%G %t'").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        gres, state = parts
        if state in ("idle", "mix") and gres.startswith("gpu:"):
            idle.add(gres.split(":")[1].lower())
    for name, flag, vram, ram in GPUS:
        if name in idle:
            return flag, vram, ram
    return GPUS[-1][1], GPUS[-1][2], GPUS[-1][3]


def pick_resources(forced_gpu: Optional[str]):
    counts = job_counts()
    for partition, max_running, max_submitted, max_time in PARTITIONS:
        running, submitted = counts.get(partition, (0, 0))
        if running >= max_running or submitted >= max_submitted:
            print(
                f"  {partition}: full ({running}/{max_running} running, {submitted}/{max_submitted} queued)"
            )
            continue
        gres, vram, ram = best_gpu(partition)
        if forced_gpu:
            gres, vram, ram = next(
                (flag, gpu_vram, gpu_ram)
                for name, flag, gpu_vram, gpu_ram in GPUS
                if name == forced_gpu
            )
        print(
            f"  {partition}: {max_running - running} slot(s) free → {gres} ({vram}GB VRAM, {ram}GB RAM)"
        )
        return partition, gres, ram, max_time
    return None, None, None, None


def build_log_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def build_model_command(model_args: List[str]) -> str:
    return shell_join(["uv", "run", "python", "src/model.py", *model_args])


def build_resubmit_command(args, model_args: List[str]) -> str:
    dependency_placeholder = "__SLURM_DEPENDENCY__"
    cmd = [
        "python3",
        "smart_sbatch.py",
        "--job-name",
        args.job_name,
        "--dependency",
        dependency_placeholder,
        "--signal-seconds",
        str(args.signal_seconds),
    ]
    if args.time:
        cmd.extend(["--time", args.time])
    if args.gpu:
        cmd.extend(["--gpu", args.gpu])
    return shell_join(cmd + model_args).replace(
        dependency_placeholder, "afterany:${SLURM_JOB_ID}"
    )


def build_script(args, model_args: List[str], partition: str, gres: str, ram: int, max_time: str, log_dir: Path) -> str:
    repo_root = Path.cwd().resolve()
    model_cmd = build_model_command(model_args)
    resubmit_cmd = build_resubmit_command(args, model_args)
    walltime = args.time or max_time
    return textwrap.dedent(
        f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={ram}G
#SBATCH --gres={gres}
#SBATCH --time={walltime}
#SBATCH --signal=B:USR1@{args.signal_seconds}
#SBATCH --job-name={args.job_name}
#SBATCH --output={log_dir.resolve()}/out.log
#SBATCH --error={log_dir.resolve()}/err.log

set -euo pipefail

cd {shlex.quote(str(repo_root))}
. /usr/local/lmod/lmod/init/bash
module load CUDA/12.2.2
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
source $HOME/mark_sheinin_lab/code/eitan/.secrets

RESUBMITTED=0
RESUBMIT_JOB_ID=""
handle_timeout() {{
    if [ "$RESUBMITTED" -eq 1 ]; then
        return
    fi
    RESUBMITTED=1
    echo "[smart_sbatch] Caught SIGUSR1 for $SLURM_JOB_ID; scheduling resume job"
    RESUBMIT_OUTPUT="$({resubmit_cmd} 2>&1)"
    RESUBMIT_STATUS=$?
    printf '%s\n' "$RESUBMIT_OUTPUT"
    if [ "$RESUBMIT_STATUS" -ne 0 ]; then
        echo "[smart_sbatch] Failed to submit resume job" >&2
        return
    fi
    RESUBMIT_JOB_ID="$(printf '%s\n' "$RESUBMIT_OUTPUT" | awk '/Submitted batch job/ {{print $4}}' | tail -n 1)"
}}
trap handle_timeout USR1

TRAIN_EXIT=0
{model_cmd} || TRAIN_EXIT=$?
if [ "$TRAIN_EXIT" -eq 0 ]; then
    if [ -n "$RESUBMIT_JOB_ID" ]; then
        echo "[smart_sbatch] Training completed; cancelling queued resume job $RESUBMIT_JOB_ID"
        scancel "$RESUBMIT_JOB_ID" || true
    fi
    exit 0
fi
exit "$TRAIN_EXIT"
"""
    )


def submit_script(script_path: Path, dependency: Optional[str]):
    cmd = ["sbatch"]
    if dependency:
        cmd.extend(["--dependency", dependency])
    cmd.append(str(script_path))
    return subprocess.run(
        cmd,
        check=False,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def main():
    args, raw_model_args = parse_args()
    model_args = normalize_model_args(raw_model_args, args.job_name)

    print("Checking cluster state...")
    partition, gres, ram, max_time = pick_resources(args.gpu)
    if not partition:
        print("All slots full. Try again later.")
        sys.exit(1)

    log_dir = build_log_dir()
    script = build_script(args, model_args, partition, gres, ram, max_time, log_dir)
    script_path = log_dir / "job.sh"
    script_path.write_text(script)
    print(f"\n{script}")

    if args.dry_run:
        print(f"[dry-run] script saved to {script_path}, not submitted")
        return

    result = submit_script(script_path, args.dependency)
    print(result.stdout or result.stderr)
    print(f"Logs: tail -f {log_dir}/out.log {log_dir}/err.log")


if __name__ == "__main__":
    main()
