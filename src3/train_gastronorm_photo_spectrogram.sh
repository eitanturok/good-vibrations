#!/usr/bin/env bash
# Train src3's AudioToken-style textual inversion model on gastronorm's single-cube
# split: predict the natural overhead RGB photo, conditioned on a real STFT
# spectrogram (averaged over lasers + x/y) recovered from the vibration data,
# sampling starts from the box's empty-box photo (SDEdit-style) rather than pure
# noise. gastronorm, not cardboard: cardboard's data only has empty-box (0 objects)
# and two-cubes-grid* (2 objects) layouts, no single-cube scenes at all -- see
# src3/dataset.py:n_object_split's docstring. gastronorm has 560 single-cube
# (purple-cube) samples, by far the most single-object data of any box here.
#
# --lr and the frozen/trainable split match vendor/train.py's own defaults exactly
# (see vendor/README.md's training command and
# vendor/modules/AudioToken/AudioToken.py:AudioTokenWrapper's requires_grad_(False)
# calls): learning_rate=1e-5, batch size 4, vae/unet/text_encoder/BEATs frozen,
# only the embedder MLP trained (no --finetune-beats, no LoRA).
#
# Prereqs: python src3/download.py (once)
#
# Usage:
#   bash src3/train_gastronorm_photo_spectrogram.sh [max_steps]

set -euo pipefail

MAX_STEPS="${1:-2000}"

python src3/train.py \
  --box gastronorm \
  --condition-mode spectrogram \
  --target photo \
  --start-mode empty-box \
  --start-strength 0.75 \
  --target-n-objects 1 \
  --prompt "A photo of a <*> metal cube in a box from a bird's eye view" \
  --lr 1.0e-05 \
  --batch-size 4 \
  --max-steps "${MAX_STEPS}" \
  --eval-interval 200 \
  --eval-samples 5 \
  --run-name "gastronorm-photo-spectrogram-emptybox" \
  --checkpoint-dir "src3/runs/gastronorm-photo-spectrogram-emptybox"
