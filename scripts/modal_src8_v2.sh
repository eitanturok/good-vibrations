#!/usr/bin/env bash
# src8 sweep v2, 500 epochs (rankings at ep500 match ep1000: Spearman 0.8-1.0 over 25 past runs, runs/ep500_vs_1k.csv).
# All: gastronorm_plastic_cardboard, --window tukey, batch 1024, speakers 1 3 5 7, seed 42, lr 1e-4, cosine with 50 ep warmup,
# eval every 50 ep. Headline: CORE iou/miss@0.08 (vase, cube, soap, candle x 3 boxes) + HARD (cardboard rotated soap) -- score.py.
# v1 (scripts/modal_src8.sh) is retired: its pixel-query ModalFormer fit a constant mask (attention-pooling over bands washed the
# per-sample content out; fixed by gated flatten pooling, see src8/EXPERIMENTS.md). Arms:
#   hybrid           -- ModalFormer front end (freq stem + log Hz, laser/band/box embeddings, per-box input norm, gated band pooling,
#                       2 laser-attention blocks), no axial blocks, flattened laser tokens -> the conv decoder. fitdiag best (.442 vs conv .390)
#   conv-logpower    -- control: the target's recipe at 500 ep (conv encoder + conv decoder, log power)
#   hybrid-contour   -- T5: + outline channel (edge) + moment loss (area / centroid / orientation): the rotated-soap case
#   hybrid-orient    -- T5: + orientation aux head (ecc * (cos 2θ, sin 2θ) of the mask, from the embedding): a ridge probe reads
#                       rotated-soap orientation at 17° from the input, the target model gets 36.5° (chance 45°)
#   hybrid-farina    -- input: + the 2nd-harmonic impulse response's spectrum (C 2 -> 4; probe: independent position info, best speaker transfer)
#   query            -- T1: the fully attentional ModalFormer: LFLF axial blocks + 4-layer pixel-query decoder + floor<->wall prior
#   hybrid-LFLF      -- T2: + LFLF axial blocks (laser attention per band = mode shapes, band attention per laser)
#   hybrid-inv       -- T4: + laser dropout 0.2 + random spectral tilt 0.3 (unseen lasers / speakers)
#   hybrid-center    -- T4: + laser-centering (cancels the speaker's S(f))
#   conv-logpower-z  -- per-sample centring (--norm z): is the log-power gap its loudness offset? (user analysis 2026-10-18)
#   conv-mag         -- control: magnitude input
# Runs whose epoch-500 checkpoint exists are skipped.
#   ./scripts/modal_src8_v2.sh 2>&1 | tee -a runs/modal_src8_v2.log

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-500}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 500 ] && echo 500ep || echo "${EPOCHS}ep")

COMMON="--dataset gastronorm_plastic_cardboard --window tukey --batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 50ep \
        --eval-interval 50ep --image-interval 100ep --checkpoint-interval 250ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group modal-src8-v2"

HYBRID="--encoder modal --decoder conv --d-model 1024 --microbatch 512"
QUERY="--encoder modal --decoder query --d-model 256 --microbatch 256 --modal-dec-layers 4"
CONV="--encoder conv"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# runs can hang after wandb's last line (idle, 0% CPU, everything saved): kill the run once it's been there 5 min
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

run s8-hybrid-base-$TAG       $HYBRID --signal log_power --modal-order ""
run s8-conv-logpower-$TAG     $CONV   --signal log_power
run s8-hybrid-contour-$TAG    $HYBRID --signal log_power --modal-order "" --edge-weight 1 --moment-weight 1
run s8-hybrid-orient-$TAG     $HYBRID --signal log_power --modal-order "" --orient-weight 1
run s8-query-$TAG             $QUERY  --signal log_power
run s8-hybrid-LFLF-$TAG       $HYBRID --signal log_power --modal-order LFLF
run s8-hybrid-farina-$TAG     $HYBRID --signal log_power --modal-order "" --window farina --num-workers 2
run s8-hybrid-inv-$TAG        $HYBRID --signal log_power --modal-order "" --laser-drop 0.2 --color-aug 0.3
run s8-hybrid-center-$TAG     $HYBRID --signal log_power --modal-order "" --laser-center 1
run s8-conv-logpower-z-$TAG   $CONV   --signal log_power --norm z
run s8-conv-mag-$TAG          $CONV   --signal magnitude
