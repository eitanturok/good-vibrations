#!/usr/bin/env bash
# Sweep src3/train.py over the 3 core axes: target (mask/photo) x start-mode
# (noise/empty-box) x condition-mode (laser-freq/spectrogram) -- 8 runs total,
# one wandb run each, run sequentially (single GPU). lr/batch-size/frozen-params
# match the AudioToken paper's own fine-tuning setup for every run (train.py's
# defaults, same as train_cardboard_photo_spectrogram.sh) -- only these 3 flags
# vary between runs. Per-target prompt is src3/dataset.py:PROMPTS' default
# (not overridden here -- this sweep is about the architecture axes, not wording).
#
# Only run this against a box with real single-cube data (--target-n-objects 1,
# the default): gastronorm (560 purple-cube samples) or wood (140 one-cube-grid1).
# cardboard/plastic have none -- see src3/dataset.py:n_object_split's docstring.
#
# Usage:
#   bash src3/sweep.sh [box] [max_steps]
#   bash src3/sweep.sh gastronorm 2000

set -euo pipefail

BOX="${1:-gastronorm}"
MAX_STEPS="${2:-2000}"

for target in mask photo; do
  for start_mode in noise empty-box; do
    for condition_mode in laser-freq spectrogram; do
      run_name="${BOX}-${target}-${condition_mode}-${start_mode}"
      echo "=== ${run_name} ==="
      python src3/train.py \
        --box "${BOX}" \
        --condition-mode "${condition_mode}" \
        --target "${target}" \
        --start-mode "${start_mode}" \
        --start-strength 0.75 \
        --lr 1.0e-05 \
        --batch-size 4 \
        --max-steps "${MAX_STEPS}" \
        --eval-interval 200 \
        --eval-samples 5 \
        --run-name "${run_name}" \
        --checkpoint-dir "src3/runs/${run_name}"
    done
  done
done
