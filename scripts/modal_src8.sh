#!/usr/bin/env bash
# src8 ModalFormer (--encoder modal) vs s7-gpc-tukey-logpower-1k (target: ALL iou@0.08 .299, miss@0.08 .275 at ep750; src8/score.py).
# All three boxes (gastronorm_plastic_cardboard), the log_power_src7.sh recipe: --signal log_power --window tukey, batch 1024
# (microbatch 256: ~27 MB/sample), speakers 1 3 5 7, seed 42, 1000 epochs. Theories in src8/EXPERIMENTS.md:
#   modal         -- T1 pixel-query decoder + T2 axial LFLF laser x band tokens + T3 floor<->wall prior (geo)
#   modal-contour -- + T5 outline + sdf channels + moment (area / centroid / orientation) loss
#   modal-inv     -- + T4 laser dropout 0.2 + random spectral tilt 0.3 (unseen speakers / lasers)
#   modal-center  -- + T4 laser-centering (mean over lasers removed per freq: cancels the speaker's S(f))
#   modal-FFFF    -- T2 ablation: frequency-only axial blocks (lasers meet only after pooling, like the conv model)
#   modal-LLLL    -- T2 ablation: laser-only axial blocks (mode shapes per band, bands meet only in the pooling)
#   modal-nogeo   -- T3 ablation: no floor<->wall prior
#   modal-mag     -- input: magnitude instead of log power (the conv model preferred magnitude at ep750, .328 vs .299)
#   conv-logpower -- control: the target's recipe (conv encoder + conv decoder) in src8, so it logs the new metrics and reaches ep1000
#   conv-mag      -- control: the same with magnitude input (s7-gpc-tukey-base-1k's recipe)
# Runs whose epoch-1000 checkpoint exists are skipped.
#   ./scripts/modal_src8.sh 2>&1 | tee -a runs/modal_src8.log
#   WANDB=0 EPOCHS=1 ...  -- smoke test without a wandb run

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-1000}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 1000 ] && echo 1k || echo "${EPOCHS}ep")

COMMON="--dataset gastronorm_plastic_cardboard --window tukey \
        --batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 25ep \
        --checkpoint-interval 250ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group modal-src8"

MODAL="--encoder modal --d-model 256 --microbatch 256"  # ~27 MB/sample: 4 microbatches of 256 per 1024 batch
CONV="--encoder conv"                                   # the conv model fits a whole 1024 batch (14.6 GB), as in src7

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# runs can hang after wandb's last line (idle, 0% CPU, everything saved): kill the run once it's been there 5 min,
# and delete its MDS copy of the dataset, which a killed run can't clean up itself
kill_if_hung() {  # $1 = log, $2 = run name
    if grep -q "Find logs at: ./wandb/run-.*-$2/logs" "$1" 2>/dev/null && [ $(( $(date +%s) - $(stat -c %Y "$1") )) -gt 300 ]; then
        echo "$(date +%T) $2: hung after wandb finished, killing"; pkill -KILL -f -- "--run-name $2( |$)"
    fi
}

run() {  # $1 = run name, rest = flags
    local name=$1; shift
    if ls runs/$name/checkpoints/ep${EPOCHS}-ba*-rank0.pt >/dev/null 2>&1; then echo "$(date +%T) $name: done already, skipping"; return; fi
    echo "$(date +%T) $name: start (log runs/$name.out)"
    .venv/bin/python src8/run.py $COMMON "$@" --run-name "$name" > "runs/$name.out" 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do sleep 60; kill_if_hung "runs/$name.out" "$name"; done
    local mds=$(grep -o "wrote [0-9]* samples to [^ ]*" "runs/$name.out" | tail -1 | awk '{print $NF}')
    [ -n "$mds" ] && [ -d "$mds" ] && { echo "$(date +%T) $name: removing leftover $mds"; rm -rf "$mds"; }
    echo "$(date +%T) $name: finished"
}

run s8-modal-$TAG          $MODAL --signal log_power
run s8-conv-logpower-$TAG  $CONV  --signal log_power
run s8-conv-mag-$TAG       $CONV  --signal magnitude
run s8-modal-contour-$TAG  $MODAL --signal log_power --edge-weight 1 --sdf-weight 1 --moment-weight 1
run s8-modal-inv-$TAG      $MODAL --signal log_power --laser-drop 0.2 --color-aug 0.3
run s8-modal-center-$TAG   $MODAL --signal log_power --laser-center 1
run s8-modal-FFFF-$TAG     $MODAL --signal log_power --modal-order FFFF
run s8-modal-LLLL-$TAG     $MODAL --signal log_power --modal-order LLLL
run s8-modal-nogeo-$TAG    $MODAL --signal log_power --geo 0
run s8-modal-mag-$TAG      $MODAL --signal magnitude
run s8-conv-logpower-z-$TAG $CONV  --signal log_power --norm z   # per-sample centring: is the log-power gap its loudness offset? (user analysis, 2026-10-18)
