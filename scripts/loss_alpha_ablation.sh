#!/usr/bin/env bash
# FP/FN weighting ablation: mse-asym x ce-pixel-asym over alpha in {0.5, 0.7, 0.9}.
# alpha weighs false negatives (missing the cube) against false positives (painting where
# there is none). alpha=0.5 is exactly plain mse / ce-pixel, so those two runs are the
# baselines and the only thing moving across the sweep is alpha.
# Runs sequentially on the single GPU. Launch under tmux:
#   tmux new -s alpha
#   ./scripts/loss_alpha_ablation.sh 2>&1 | tee runs/loss_alpha_ablation.log
# Detach with ctrl-b d; reattach with `tmux attach -t alpha`.

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the sweep

cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG=v2
GROUP=loss-alpha-ablation-attn-$TAG

# ***** mse-asym *****

# alpha 0.5: identical to plain mse -- the baseline
python src/run.py --split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn mse-asym --loss-alpha 0.5 --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name loss-alpha-mse-a05-attn-$TAG

# alpha 0.7: false negatives cost 2.33x a false positive
python src/run.py --split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn mse-asym --loss-alpha 0.7 --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name loss-alpha-mse-a07-attn-$TAG

# alpha 0.9: false negatives cost 9x a false positive
python src/run.py --split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn mse-asym --loss-alpha 0.9 --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name loss-alpha-mse-a09-attn-$TAG

# ***** ce-pixel-asym *****

# alpha 0.5: pos_weight 1, identical to plain ce-pixel -- the baseline
python src/run.py --split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-pixel-asym --loss-alpha 0.5 --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name loss-alpha-ce-pixel-a05-attn-$TAG

# alpha 0.7: pos_weight 2.33
python src/run.py --split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-pixel-asym --loss-alpha 0.7 --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name loss-alpha-ce-pixel-a07-attn-$TAG

# alpha 0.9: pos_weight 9
python src/run.py --split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-pixel-asym --loss-alpha 0.9 --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name loss-alpha-ce-pixel-a09-attn-$TAG
