#!/usr/bin/env bash
# src6, four_objs, 3000 epochs: cardboard only, then cardboard + gastro + plastic together.
# Based on four_objs_3k_src6.sh, speakers 1 3 5 7 (speaker 4 held out at the unseen-speaker positions).
#   ./scripts/cardboard_four_objs_src6.sh 2>&1 | tee runs/cardboard_four_objs_src6.log
#   WANDB=0 EPOCHS=1 ...  -- smoke test without a wandb run

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-3000}"
WANDB="${WANDB:-1}"

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 25ep \
        --checkpoint-interval 500ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group cardboard-four-objs"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# skip a run whose final checkpoint already exists, so a relaunch picks up where the script stopped
run() {
    local name="s6-$2-3k"
    if [ -e "runs/$name/checkpoints/latest-rank0.pt" ] && [[ "$(readlink runs/$name/checkpoints/latest-rank0.pt)" == ep${EPOCHS}-* ]]; then
        echo "skip $name: finished"; return
    fi
    .venv/bin/python src6/run.py $COMMON --dataset "$1" --run-name "$name"
}

run cardboard                    cardboard
run gastronorm_plastic_cardboard gastro-plastic-cardboard
