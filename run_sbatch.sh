#!/bin/bash

job_name="model_1sp"
dir="."

timestamp=$(date +"%Y%m%d_%H%M%S")
log_dir="${dir}/logs/${timestamp}"
mkdir -p "${log_dir}"
touch "${log_dir}/out.log" "${log_dir}/err.log"

sbatch --partition=long.q --ntasks=1 --cpus-per-task=4 --mem=32G --gres=gpu:quadro_rtx_8000:1 --time=12:00:00 \
  --job-name=${job_name} --output=${log_dir}/out.log --error=${log_dir}/err.log \
  --wrap=". /usr/local/lmod/lmod/init/bash; module load CUDA/12.2.2; curl -LsSf https://astral.sh/uv/install.sh | sh; source \$HOME/.local/bin/env; source \$HOME/mark_sheinin_lab/code/eitan/.secrets; uv run python src/model.py --speakers '[1,0,0,0]' --run-name 1sp --batch-size 128 --max-duration 1ep"
