#!/usr/bin/env bash
# src7 preprocessing ablation, gastronorm and cardboard four_objs, each trained separately, the s6-cardboard-3k recipe
# (batch 1024, speakers 1 3 5 7, 3000 epochs, seed 42; same as scripts/window_src7.sh, so its runs are reused).
#   hann      -- the current method: the stored 04_ffts (hann over the whole capture)        [from window_src7.sh]
#   tukey     -- the fft redone from the raw shifts with 13ms tapers (nb100)                  [from window_src7.sh]
#   deconv    -- tukey / the chirp's spectrum: best com probe in nb100 (cardboard cube .79 -> .82)
#   sqrt      -- tukey, |X|^0.5: halves cardboard's later-layout share of the floor (nb101 §10, ΣS/ΣN .65 -> 1.07)
#   lasernorm -- tukey, each laser scaled to rms 1: the only nb101 §10 candidate up on both scores in both boxes
# Runs whose epoch-3000 checkpoint exists are skipped. Waits for scripts/window_src7.sh to finish first.
#   ./scripts/preproc_src7.sh 2>&1 | tee runs/preproc_src7.log
#   WANDB=0 EPOCHS=1 ...  -- smoke test without a wandb run

set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-3000}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 3000 ] && echo 3k || echo "${EPOCHS}ep")

COMMON="--batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 100ep --eval-interval 25ep \
        --checkpoint-interval 500ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group fft-window"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# cardboard runs can hang after wandb's last line (idle, 0% CPU, everything saved): kill the run once it's been there 5 min,
# and delete its MDS copy of the dataset (~6.5 GB), which a killed run can't clean up itself
kill_if_hung() {  # $1 = log, $2 = run name
    if grep -q "Find logs at: ./wandb/run-.*-$2/logs" "$1" 2>/dev/null && [ $(( $(date +%s) - $(stat -c %Y "$1") )) -gt 300 ]; then
        echo "$(date +%T) $2: hung after wandb finished, killing"; pkill -KILL -f -- "--run-name $2( |$)"
        rm -rf "$(grep -o "wrote [0-9]* samples to [^ ]*" "$1" | tail -1 | awk '{print $NF}')"
    fi
}

while pgrep -f "scripts/window_src7.sh" >/dev/null; do
    for r in $(grep -o "run-name s7-[a-z]*-[a-z]*-$TAG" <(pgrep -af "src7/run.py") | awk '{print $2}' | sort -u); do kill_if_hung runs/window_src7.log "$r"; done
    sleep 60
done

run() {  # $1 = run name, rest = flags
    local name=$1; shift
    if ls runs/$name/checkpoints/ep${EPOCHS}-ba*-rank0.pt >/dev/null 2>&1; then echo "$(date +%T) $name: done already, skipping"; return; fi
    echo "$(date +%T) $name: start (log runs/$name.out)"
    .venv/bin/python src7/run.py $COMMON "$@" --run-name "$name" > "runs/$name.out" 2>&1 &
    local pid=$!
    while kill -0 $pid 2>/dev/null; do sleep 60; kill_if_hung "runs/$name.out" "$name"; done
    echo "$(date +%T) $name: finished"
}

for DATASET in gastronorm cardboard; do
    D=${DATASET:0:4}
    run s7-$D-hann-$TAG      --dataset $DATASET --window hann
    run s7-$D-tukey-$TAG     --dataset $DATASET --window tukey
    run s7-$D-deconv-$TAG    --dataset $DATASET --window deconv
    run s7-$D-sqrt-$TAG      --dataset $DATASET --window tukey --signal sqrt_magnitude
    run s7-$D-lasernorm-$TAG --dataset $DATASET --window tukey --laser-norm 1
done
