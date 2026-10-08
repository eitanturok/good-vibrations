#!/usr/bin/env bash
# Gastro only vs plastic only vs both boxes, on identical eval sets (same held-out positions in every arm).
# Same recipe as gastro_plastic_four_objs.sh. Splits live in BOX_FOUR_OBJS (src/model/dataset.py).
#   spk137  -- speakers 1,3,7 only; speaker 4 held out at the unseen-speaker positions (as gastro-plastic-bb-v3)
#   allspk  -- every speaker in train and eval, no speaker held out
# Epochs, not steps: every sample is seen equally often in every arm, so `both` takes ~2x the steps.
# Batches/epoch at bs 1024: gastro/plastic spk137 2 each, both spk137 5; allspk gastro 4, plastic 3, both 8.
#   TAG=1 ./scripts/box_compare_four_objs.sh 2>&1 | tee runs/box_compare_four_objs.log
#   WANDB=0 EPOCHS=1 ...  -- smoke test without a wandb run

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-1}"
EPOCHS="${EPOCHS:-2000}"
NO_WANDB=$([ "${WANDB:-1}" = 0 ] && echo --no-wandb)
GROUP="box-compare-four-objs-v$TAG"
GASTRO=experiments/2026_09_27_gastronorm_four_objs
PLASTIC=experiments/2026_10_05_green_plastic_four_objs

COMMON="--model boombox --d-model 1024 --batch-size 1024 \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel \
        --laser-dropout 0 --freq-dropout 0 \
        --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 100ep --viz-interval 250ep \
        --checkpoint-interval 500ep --seed 42 --no-viz $NO_WANDB --wandb-group $GROUP"

run() { .venv/bin/python src/run.py $COMMON --data-dir "$1" --split "$2" --run-name "$3-v$TAG"; }

run $GASTRO  gastro_four_objs_spk137         box-gastro-spk137
run $PLASTIC plastic_four_objs_spk137        box-plastic-spk137
run $GASTRO  gastro_plastic_four_objs        box-both-spk137
run $GASTRO  gastro_four_objs_allspk         box-gastro-allspk
run $PLASTIC plastic_four_objs_allspk        box-plastic-allspk
run $GASTRO  gastro_plastic_four_objs_allspk box-both-allspk
