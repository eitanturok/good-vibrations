#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smart_sbatch.py - auto-picks the best available partition + GPU, then submits."""

import subprocess, sys, argparse, textwrap
from datetime import datetime
from pathlib import Path

# QOS limits discovered via sacctmgr/scontrol (see CLAUDE.md)
PARTITIONS = [
    # (name, max_running, max_submit, max_time)
    ("normal.q", 6, 9,  "6:00:00"),
    ("long.q",   2, 4, "12:00:00"),
]
# Preference order: best GPU first
# (name, gres_flag, vram_gb, max_ram_gb)
# max_ram_gb = node_total_ram / num_gpus_on_node, i.e. fair-share RAM per GPU slot
GPUS = [
    ("l40s",            "gpu:L40S:1",            48,  48),  # cluster hard-caps all jobs at 48G RAM
    ("quadro_rtx_8000", "gpu:quadro_rtx_8000:1", 48,  48),  # 384 GB / 8 GPUs
    ("a10",             "gpu:a10:1",             24,  48),  # 256 GB / 4 GPUs (NUMA socket split limits to ~48G)
    ("quadro_rtx_6000", "gpu:quadro_rtx_6000:1", 24,  48),  # 192 GB / 4 GPUs
]


def sh(cmd):
    return subprocess.run(cmd, shell=True, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def job_counts():
    """Returns {partition: (running, total_submitted)} for the current user."""
    counts = {}
    for line in sh("squeue -u $USER -h -o '%P %T'").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        p, state = parts
        r, t = counts.get(p, (0, 0))
        counts[p] = (r + (state == "RUNNING"), t + 1)
    return counts


def best_gpu(partition):
    """Returns (gres_flag, vram_gb, ram_gb) for the best GPU with an idle/mix node in partition."""
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
    return GPUS[-1][1], GPUS[-1][2], GPUS[-1][3]  # no idle GPU; submit anyway and wait in queue


def pick():
    """Returns (partition, gres_flag, ram_gb, max_time) or (None, None, None, None) if all full."""
    counts = job_counts()
    for part, max_run, max_sub, max_time in PARTITIONS:
        running, submitted = counts.get(part, (0, 0))
        if running >= max_run or submitted >= max_sub:
            print(f"  {part}: full ({running}/{max_run} running, {submitted}/{max_sub} queued)")
            continue
        gres, vram, ram = best_gpu(part)
        print(f"  {part}: {max_run - running} slot(s) free → {gres} ({vram}GB VRAM, {ram}GB RAM)")
        return part, gres, ram, max_time
    return None, None, None, None


def main():
    ap = argparse.ArgumentParser(description="Submit a Slurm job using best available resources")
    ap.add_argument("--job-name", default="job")
    ap.add_argument("--dry-run", action="store_true", help="Print the sbatch script without submitting")
    ap.add_argument("--gpu", default=None, choices=[name for name, *_ in GPUS], help="Force a specific GPU type instead of auto-picking")
    args, model_args = ap.parse_known_args()

    print("Checking cluster state...")
    partition, gres, ram, max_time = pick()
    if args.gpu:
        gres, _, ram = next((flag, vram, r) for name, flag, vram, r in GPUS if name == args.gpu)  # noqa
    if not partition:
        print("All slots full. Try again later.")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/{ts}")
    log_dir.mkdir(parents=True, exist_ok=True)

    model_cmd = " ".join(["uv run python src/model.py"] + model_args + [f"--run-name {args.job_name}"])

    script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --partition={partition}
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem={ram}G
        #SBATCH --gres={gres}
        #SBATCH --time={max_time}
        #SBATCH --job-name={args.job_name}
        #SBATCH --output={log_dir.resolve()}/out.log
        #SBATCH --error={log_dir.resolve()}/err.log

        . /usr/local/lmod/lmod/init/bash
        module load CUDA/12.2.2
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.local/bin/env
        source $HOME/mark_sheinin_lab/code/eitan/.secrets

        {model_cmd}
    """)

    script_path = log_dir / "job.sh"
    script_path.write_text(script)
    print(f"\n{script}")

    if args.dry_run:
        print(f"[dry-run] script saved to {script_path}, not submitted")
        return

    result = subprocess.run(["sbatch", str(script_path)], encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stdout or result.stderr)
    print(f"Logs: tail -f {log_dir}/out.log {log_dir}/err.log")


if __name__ == "__main__":
    main()
