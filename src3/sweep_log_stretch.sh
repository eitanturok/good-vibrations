#!/usr/bin/env bash
# Trains src3's AudioToken-style model twice -- with and without --log-stretch (see
# src3/beats_input.py:to_beats_fbank and notebooks/83_spectrogram_stretch_strategies.ipynb) --
# otherwise identical to src3/train_gastronorm_photo_spectrogram.sh's config (gastronorm,
# spectrogram condition, photo target, single-cube split), for 100 epochs each so the two
# runs are directly comparable in wandb.
#
# --start-mode defaults to empty-box now (src3/train.py), so it's not passed explicitly here.
#
# Prereqs: python src3/download.py (once)
#
# Usage:
#   bash src3/sweep_log_stretch.sh [epochs]

set -euo pipefail

EPOCHS="${1:-100}"

for LOG_STRETCH in 0 1; do
  TAG="gastronorm-photo-spectrogram-emptybox-logstretch${LOG_STRETCH}"
  python src3/train.py \
    --box gastronorm \
    --condition-mode spectrogram \
    --target photo \
    --start-strength 0.75 \
    --target-n-objects 1 \
    --prompt "A photo of a <*> metal cube in a box from a bird's eye view" \
    --log-stretch "${LOG_STRETCH}" \
    --lr 1.0e-05 \
    --batch-size 4 \
    --epochs "${EPOCHS}" \
    --eval-interval 200 \
    --eval-samples 5 \
    --run-name "${TAG}" \
    --checkpoint-dir "src3/runs/${TAG}"
done
