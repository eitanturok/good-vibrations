#!/bin/bash

job_name="model_1sp"
dir="."

sbatch --partition=normal.q --ntasks=1 --cpus-per-task=2 --mem=20G --gres=gpu:quadro_rtx_8000:1 --time=1:00:00 \
  --job-name=${job_name} --output=${dir}/out.log --error=${dir}/err.log \
  --wrap=". /usr/local/lmod/lmod/init/bash; module load CUDA/12.2.2; curl -LsSf https://astral.sh/uv/install.sh | sh; source \$HOME/.local/bin/env; source \$HOME/mark_sheinin_lab/code/eitan/.secrets; uv run python src/model.py --speakers '[1,0,0,0]' --run-name 1sp"
