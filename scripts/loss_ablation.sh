#!/usr/bin/env bash
# Loss ablation: only --loss-fn changes between runs.
set -euo pipefail
cd "$(dirname "$0")/.."

# 7x10 + augment-fft 0 match the existing MDS cache; other values force a full rebuild
PYTHONPATH=. python src/run.py --split gastronorm_one_cube --out-h 7 --out-w 10 --augment-mask 0 --augment-fft 0 \
    --loss-fn mse --max-duration 1000ep --wandb-group loss-ablation --run-name loss-ablation-mse-2

# per-cell binary CE: 70 independent yes/no questions
PYTHONPATH=. python src/run.py --split gastronorm_one_cube --out-h 7 --out-w 10 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-pixel --max-duration 1000ep --wandb-group loss-ablation --run-name loss-ablation-ce-pixel-2

# one softmax over 70 cells + empty slot; keeps total mask mass
PYTHONPATH=. python src/run.py --split gastronorm_one_cube --out-h 7 --out-w 10 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-spatial --max-duration 1000ep --wandb-group loss-ablation --run-name loss-ablation-ce-spatial-2

# same, but cube mass renormalized to 1 so the empty slot is a pure occupancy bit
PYTHONPATH=. python src/run.py --split gastronorm_one_cube --out-h 7 --out-w 10 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-spatial-normalized --max-duration 1000ep --wandb-group loss-ablation --run-name loss-ablation-ce-spatial-normalized-2
