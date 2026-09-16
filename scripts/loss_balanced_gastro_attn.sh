#!/usr/bin/env bash
# Does --loss-balanced fix ce-pixel-asym's dilution problem on gastronorm_one_cube (attn decoder)?
#
# ce_pixel_asym_loss's pos_weight only reweights each POSITIVE pixel's own term, but
# reduction='mean' still divides by ALL pixels. gastronorm_one_cube boxes cover only ~0.65% of the
# 32x32 grid on average, so balancing the two terms needs pos_weight ~(1-p)/p ~150 -- alpha=0.85
# (pos_weight~5.7) is nowhere close, and alpha=0.97 (pos_weight~32, still short of 150) already
# overshoots into the opposite (paint-everything) collapse. Between those two collapse points,
# training spends thousands of steps stuck near the background-collapse plateau before the small,
# diluted signal from the ~0.65% positive pixels accumulates enough to escape it (see
# masked_attn_gastro_m.sh's run history: attn-decoder-gastro-attn-v4 escaped by step ~1000 and hit
# iou 0.38; a same-config rerun, attn-decoder-gastro-attn-v5, stayed stuck until step ~4700 and only
# reached iou 0.10 before the cosine schedule ran the LR down to ~0 -- pure escape-timing luck).
#
# --loss-balanced (run.py, arch.py's ce_pixel_asym_loss) replaces pos_weight-inside-a-global-mean
# with separate means over positive and negative pixels, combined via alpha -- so the
# false-negative/false-positive tradeoff no longer depends on the class prior at all, and alpha=0.85
# should behave like alpha=0.85 was ALWAYS meant to (heavily favor recall), without needing to
# guess (1-p)/p first.
#
# 2x2 (alpha in {0.5, 0.85}) x (--loss-balanced in {0, 1}), same M-size attn-decoder config as
# masked_attn_gastro_m.sh's baseline:
#   1. alpha=0.5,  --loss-balanced 0  (default: pos_weight=1, same as plain ce-pixel -- baseline)
#   2. alpha=0.5,  --loss-balanced 1  (balanced-mean control: pos_mean/neg_mean weighted 50/50,
#      should behave close to #1 -- balancing shouldn't matter when alpha itself is already neutral)
#   3. alpha=0.85, --loss-balanced 0  (the diluted version that gave the v4/v5 split above)
#   4. alpha=0.85, --loss-balanced 1  (same alpha, dilution fixed)
#
# Plus a 5th, architecture baseline: --model boombox (full conv encoder + conv decoder, no
# attention -- see boombox_boxes.sh) at alpha=0.5/unbalanced (plain ce-pixel-equivalent), same
# split/grid/loss-fn/duration as the attn-decoder runs above. --loss-balanced only affects
# VibrationTransformer's ce_pixel_asym_loss path (run.py doesn't thread it into BoomboxModel), so
# it's meaningless for this run and left off -- this is purely "does a conv encoder-decoder hit the
# same background-collapse plateau on this sparse target, independent of the attn-decoder's
# escape-timing luck story."
#
#   tmux new -s loss-balanced
#   ./scripts/loss_balanced_gastro_attn.sh 2>&1 | tee runs/loss_balanced_gastro_attn.log

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TAG=v1

COMMON="--model transformer \
        --out-h 32 --out-w 32 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel-asym --max-duration 5000ep \
        --laser-dropout 0 --freq-dropout 0"

GASTRO="--split gastronorm_one_cube --wandb-group loss-balanced-gastro-attn"

# M size, matching masked_attn_gastro_m.sh
M_ENC="--d-model 128 --pnt-num-layers 3 --seq-num-layers 10 --enc-ffn-dim 256 --pnt-num-heads 4 --seq-num-heads 8 --decoder-num-heads 8"
BATCH=256
DECODER="--decoder attn --decoder-num-layers 18 --dec-ffn-dim 1024"

# --- 1. baseline: alpha=0.5, unbalanced (pos_weight=1, unweighted -- same as plain ce-pixel) ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH $DECODER \
    --loss-alpha 0.5 --loss-balanced 0 \
    --run-name loss-balanced-gastro-attn-a05-unbal-$TAG

# --- 2. alpha=0.5, balanced (control: balancing at a neutral alpha should look like #1) ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH $DECODER \
    --loss-alpha 0.5 --loss-balanced 1 \
    --run-name loss-balanced-gastro-attn-a05-bal-$TAG

# --- 3. alpha=0.85, unbalanced (pos_weight=5.7, diluted by the ~99.35% negative-pixel mean) ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH $DECODER \
    --loss-alpha 0.85 --loss-balanced 0 \
    --run-name loss-balanced-gastro-attn-a085-unbal-$TAG

# --- 4. alpha=0.85, balanced (separate pos/neg means -- dilution fixed) ---
python src/run.py $COMMON $GASTRO $M_ENC --batch-size $BATCH $DECODER \
    --loss-alpha 0.85 --loss-balanced 1 \
    --run-name loss-balanced-gastro-attn-a085-bal-$TAG

# --- 5. boombox baseline: conv encoder + conv decoder, no attention (see boombox_boxes.sh) ---
BOOMBOX="--model boombox --d-model 1024"
python src/run.py $COMMON $GASTRO $BOOMBOX --batch-size $BATCH \
    --loss-alpha 0.5 \
    --run-name loss-balanced-gastro-boombox-baseline-$TAG
