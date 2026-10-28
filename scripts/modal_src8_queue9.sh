#!/bin/bash
# src8 queue 6 (2026-10-19 23:20): queue 5 died at 23:12 -- a global OOM (superposition.py ~20 GB of RAM next to a run's dataset build)
# took down the tmux train session, queue 5 and s8-mag-slots2-r2 at ep293. Re-planned around the priority: ANY number of objects AND
# rotation. s8-mag-nopose-r2 (no pose branch) finds both objects in .71 of two-object scenes (s8-best-mag .36) but rotated soap is 27 deg
# (17 with the pose branch). Every new arm uses MULTI = s8-best-mag's recipe minus the pose branch, with the moment loss replaced by the
# per-instance one (--imoment-weight: each object's area / centroid / covariance in its own Mahalanobis cell; = moment_loss on a single
# object), so rotation is supervised for every object in every scene, and the decoders differ only in how they draw objects:
#   mag-qpose        object queries + a pose head per query (canonical mask + angle, warped to the query's own centroid; angle loss per
#                    matched object): the count from set matching, rotation from the pose branch -- the main candidate
#   mag-nopose-imom  the direct decoder + the per-instance moment loss: does the loss alone bring rotation back?
#   mag-centernet-r2 centre heatmap + per-peak pose-branch mask + angle (angle loss per object)
#   mag-queries-r2   object queries without the pose head: isolates the pose head (vs qpose)
#   mag-qpose-os3    qpose + two-object scenes 3x per epoch
#   mag-slots2-r3    2 pose slots (DETR cost): rotation-capable but capped at 2 objects, so lower
# batch 1024 everywhere; the query arms in microbatches of 256 (gradient accumulation: the same optimization): at 512 they OOM at the first
# training step after eval-before-train (qpose 23:41, queries v1 21:01) although a bare smoke test peaked at 11.0 GB. Then the earlier analyses and arms (importance, mag-drop, mag-fjit, reruns).
#   tmux: ulimit -n 524288; ./scripts/modal_src8_queue6.sh 2>&1 | tee -a runs/modal_src8_queue6.log
set -u
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-500}"
WANDB="${WANDB:-1}"
TAG=$([ "$EPOCHS" = 500 ] && echo 500ep || echo "${EPOCHS}ep")

COMMON="--dataset gastronorm_plastic_cardboard_wedge_region --window tukey --batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 50ep \
        --eval-interval 50ep --image-interval 100ep --checkpoint-interval 250ep --seed 42 --speakers 1 3 5 7 --wandb $WANDB --wandb-group modal-src8-v2"
HYBRID="--encoder modal --decoder conv --d-model 1024 --microbatch 512"
QUERY="--encoder modal --decoder query --d-model 256 --modal-dec-layers 4"
CONV="--encoder conv"
# s8-hybrid-posesum (the best so far: CORE .662, HARD .550 / 21 deg) = the hybrid + this pose branch; later fixes build on it
POSE="--pose-branch 1 --pose-merge sum --pose-center mask --orient-weight 1"
# s8-best-jitter (best: CORE .705 / .009, HARD .603 / .000, ALL .367 / .220) = POSE + outline channel + moment loss + laser jitter;
# the arms after it (spkgain, center, film, LFLF, farina, linskip) are rebased on it (2026-10-19)
BEST="$POSE --edge-weight 1 --moment-weight 1 --laser-jitter 0.5"
# s8-best-spkgain (best: CORE .722 / .006, HARD .596 / .000, ALL .388 / .168, unseen speaker .275 > target .260) = BEST + speaker gain;
# center / film / LFLF / farina / linskip are rebased on it (2026-10-19). jitterp dropped: with spkgain the two-object splits are back
# (.323 vs .328 before jitter), so the regression it was meant to fix is gone
JIT="$BEST"
BEST="$BEST --color-aug 0.35 --color-ell 20"

NOPOSE="--orient-weight 1 --edge-weight 1 --moment-weight 1 --laser-jitter 0.5 --color-aug 0.35 --color-ell 20"   # = BEST minus the pose flags

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# runs can hang after wandb's last line (idle, 0% CPU, everything saved): kill the run once it's been there 5 min
kill_if_hung() {  # $1 = log, $2 = run name
    if grep -q "Find logs at: ./wandb/run-.*-$2/logs" "$1" 2>/dev/null && [ $(( $(date +%s) - $(stat -c %Y "$1") )) -gt 300 ]; then
        echo "$(date +%T) $2: hung after wandb finished, killing"; pkill -KILL -f -- "--run-name $2( |$)"
    fi
}
# wait for any src8 run already on the GPU (started by an earlier queue)
while pgrep -f "src8/run.py" >/dev/null; do
    for n in $(ps -eo args | grep -o "src8/run.py.*--run-name [^ ]*" | awk '{print $NF}' | sort -u); do kill_if_hung "runs/$n.out" "$n"; done
    sleep 60
done
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

# importance on GPU between runs only: concurrent GPU jobs OOM (reported as the NVML assert)
imp() {  # $1 = run, $2 = baseline
    local suf=$([ "$2" = mean ] && echo "" || echo "_$2")
    .venv/bin/python src8/importance.py "$1" --baseline "$2" > "runs/importance_$1$suf.log" 2>&1 && echo "$(date +%T) importance $1 $2" || echo "$(date +%T) importance $1 $2 FAILED"
}
cam() { .venv/bin/python src8/gradcam.py "$1" > "runs/gradcam_$1.log" 2>&1 && echo "$(date +%T) gradcam $1" || echo "$(date +%T) gradcam $1 FAILED"; }
nb106() { (cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 106_laser_freq_importance.ipynb > ../runs/nb106.log 2>&1) && echo "$(date +%T) notebook 106 executed" || echo "$(date +%T) notebook 106 FAILED"; }


MULTI="$NOPOSE --moment-weight 0 --imoment-weight 1"
NOAUX="--object-weight 0 --n-objects-weight 0 --com-weight 0 --area-weight 0 --box-weight 0"
# 2026-10-22 14:45: masks only from now on (the user dropped the photo). On the region split the masks fail on the unseen region
# (gray-region: UNSEEN-POS iou .055, miss .53 vs CORE .73): the conv decoder paints trained positions next to the region, never in it,
# while the regressed com head lands inside it 32-53% of the time. Simple-architecture arms, all mask-only, region split:
run s8-mag-nopose-imom-region-$TAG $HYBRID --signal magnitude --modal-order "" $MULTI                  # the mask-only baseline (= gray-region minus the photo head)
run s8-mag-poseonly-region-$TAG    $HYBRID --signal magnitude --modal-order "" $BEST --pose-merge only  # where + what: one canonical shape placed at a regressed pose
run s8-mag-noaux-region-$TAG       $HYBRID --signal magnitude --modal-order "" $MULTI $NOAUX --orient-weight 0   # simplest loss: mask + edge + per-instance moments, no aux heads
echo "$(date +%T) queue 9 done"
