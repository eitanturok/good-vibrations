#!/usr/bin/env bash
# Fine-tunes SonicDiffusion (src4/train.py) three times -- none, then linear, then log
# spectrogram stretch (see src4/spectrogram_stretch.py) -- on the entire gastronorm
# dataset (1-cube + 2-cube positions, split by position not sample, empty-box always
# in train -- see src4/dataset.py:multi_object_split), 100 epochs each so the three
# runs are directly comparable in wandb. Periodic eval-viz sampling (src4/callbacks.py)
# uses only 20 inference steps -- cheaper than a full-quality 30-50 step sample, since
# this is just for tracking progress, not final generation. --batch-size 10 is the
# binary-searched max for this 16GB GPU at bf16 (src4/find_max_batch_size.py).
#
# Usage:
#   bash src4/sweep_stretch.sh [epochs]

set -euo pipefail

EPOCHS="${1:-100}"
SWEEP_TAG="stretch-sweep-v5"  # shared wandb tag across all 3 runs, for grouping/filtering in the UI
# v5: gradient accumulation (--batch-size 64, effective/optimizer-step size) over
# --microbatch-size 10 (max that actually fits on-GPU at once, re-verified empirically --
# 11 fits but tight, 12+ hits a hard CUDA OOM, see src4/find_max_batch_size.py) --
# averages the per-step noise-prediction loss over more (sample, timestep) pairs, which
# is what actually reduces its step-to-step variance (unlike a lower LR, which doesn't
# change what's measured). Plus gradient clipping (--grad-clip-norm) as a safety net
# against the rising grad-norm trend observed in v4. v4: fixed a real bug in
# spectrogram_stretch.py:stretch_log (see git history / prior conversation). v3:
# --target s_mask instead of v2's --target rgb.

# Shared across all 3 runs -- only args that actually DIFFER per run (--spectrogram-stretch,
# --run-name, --checkpoint-dir, --wandb-tags) are written out explicitly below.
COMMON_ARGS=(
  --box gastronorm
  --target-n-objects 1 2
  --target s_mask
  --start-strength 0.75
  --num-inference-steps 20
  --lr 1.0e-04
  --batch-size 64
  --microbatch-size 10
  --grad-clip-norm 0.1
  --epochs "${EPOCHS}"
  --eval-interval 200
  --eval-samples 5
)

for STRETCH in none linear log; do
  TAG="gastronorm-1and2cube-smask-stretch-${STRETCH}-v5"
  python src4/train.py \
    "${COMMON_ARGS[@]}" \
    --spectrogram-stretch "${STRETCH}" \
    --run-name "${TAG}" \
    --checkpoint-dir "src4/runs/${TAG}" \
    --wandb-tags "${SWEEP_TAG}" "stretch-${STRETCH}"
done
