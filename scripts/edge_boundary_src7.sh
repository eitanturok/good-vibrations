#!/usr/bin/env bash
# src7, gastro + plastic + cardboard four_objs, EPOCHS epochs (default 3000): the s6-gastro-plastic-cardboard-3k recipe (the baseline) plus
#   1. boundary loss (Kervadec) on the mask, weight 10
#   2. an outline head (one extra decoder channel, bce + dice on the mask outline), weight 1
# Both runs also log metrics/*/boundary_iou. Runs one after the other (batch 1024 needs most of the 16 GB GPU), and waits for any
# other run.py to exit first.
#   ./scripts/edge_boundary_src7.sh 2>&1 | tee runs/edge_boundary_src7.log
#   WANDB=0 EPOCHS=1 ...  -- smoke test without a wandb run

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-3000}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 3000 ] && echo 3k || echo "${EPOCHS}ep")   # run-name suffix, 3k like the baseline

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 25ep \
        --checkpoint-interval 500ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group cardboard-four-objs \
        --dataset gastronorm_plastic_cardboard"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while pgrep -f "src[0-9]*/run.py" > /dev/null; do echo "$(date +%H:%M) waiting for another run.py to exit"; sleep 120; done

# skip a run whose final checkpoint already exists, so a relaunch picks up where the script stopped
run() {
    local name="s7-gpc-$1-$TAG"; shift
    if [ -e "runs/$name/checkpoints/latest-rank0.pt" ] && [[ "$(readlink runs/$name/checkpoints/latest-rank0.pt)" == ep${EPOCHS}-* ]]; then
        echo "skip $name: finished"; return
    fi
    .venv/bin/python src7/run.py $COMMON --run-name "$name" "$@"
}

run boundary --boundary-weight 10
run edge     --edge-weight 1
