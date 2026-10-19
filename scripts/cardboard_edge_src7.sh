#!/usr/bin/env bash
# src7 outline-head comparison, cardboard four_objs only, the s6-cardboard-3k recipe (batch 1024, speakers 1 3 5 7, 3000 epochs).
#   base    -- no outline head (= s6-cardboard-3k)
#   edge    -- the original outline head: 1 extra channel on the mask head's last conv, --edge-weight 1
#   bigedge -- a separate outline head on the upsampled features: 4 convs, 64 wide (~93k params vs 288), --edge-weight 5
# each on two datasets:
#   cardboard              -- the usual splits (soap-dispenser-rotated + cube-rotated are 80/20 train/eval)
#   cardboard_no_rotations -- the same, but soap-dispenser-rotated + cube-rotated are 100% eval: no rotation is ever trained on
#   ./scripts/cardboard_edge_src7.sh 2>&1 | tee runs/cardboard_edge_src7.log
#   WANDB=0 EPOCHS=1 ...  -- smoke test without a wandb run

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-3000}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 3000 ] && echo 3k || echo "${EPOCHS}ep")

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 25ep \
        --checkpoint-interval 500ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group cardboard-edge"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/python src7/run.py $COMMON --dataset cardboard                                         --run-name s7-card-base-$TAG
.venv/bin/python src7/run.py $COMMON --dataset cardboard --edge-weight 1                         --run-name s7-card-edge-$TAG
.venv/bin/python src7/run.py $COMMON --dataset cardboard --edge-weight 5 --edge-width 64         --run-name s7-card-bigedge-$TAG

.venv/bin/python src7/run.py $COMMON --dataset cardboard_no_rotations                            --run-name s7-card-norot-base-$TAG
.venv/bin/python src7/run.py $COMMON --dataset cardboard_no_rotations --edge-weight 1            --run-name s7-card-norot-edge-$TAG
.venv/bin/python src7/run.py $COMMON --dataset cardboard_no_rotations --edge-weight 5 --edge-width 64 --run-name s7-card-norot-bigedge-$TAG
