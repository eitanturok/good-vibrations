#!/bin/bash
# src8 queue 5 (2026-10-19): starts once queue 4 (bash pid $Q4, its own copy of modal_src8_queue.sh) exits. From 13:13 to ~16:00 every
# queue-4 run died at step 1 (CUDA OOM, reported as the NVML assert): an orphaned s8-mag-slots-500ep process held the GPU. The crashed
# names' wandb ids are dead, so the reruns are "-r2".
#   mag-drop   -- s8-best-mag + laser dropout U(0,.25) + band dropout U(0,.25) of the 19 ~49 Hz bands: a dropped laser / band is
#                 masked out of the attention and the band gate and its slots zeroed, so at eval any laser, band or (laser, band) cell
#                 can be truly removed and the model stays in distribution (importance.py --remove). Also a robustness arm for unseen /
#                 missing lasers. Before it: importance (mean, resample) + Grad-CAM on spkgain and mag; after it, the same + remove on it
#   *-r2       -- reruns of the crashed arms, in priority order (reasons in modal_src8_queue.sh)
#   Q4=<pid> ./scripts/modal_src8_queue5.sh 2>&1 | tee -a runs/modal_src8_queue5.log
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

# wait for queue 4 to finish all of its runs
while kill -0 "${Q4:?pid of queue 4}" 2>/dev/null; do sleep 60; done
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

for r in s8-best-spkgain-500ep s8-best-mag-500ep; do imp $r mean; imp $r resample; cam $r; done
# augmentation ablation, no training: posesum-contour -> best-jitter adds only --laser-jitter 0.5; best-jitter -> spkgain adds only the
# speaker gain. Does jitter make the laser importance diffuse (redundant neighbours)? And how far does each model tolerate a moved grid /
# a stretched frequency axis (shift_probe.py)?
for r in s8-posesum-contour-500ep s8-best-jitter-500ep; do imp $r mean; cam $r; done
probe() { .venv/bin/python src8/shift_probe.py "$@" > runs/shift_probe_$(date +%H%M).log 2>&1 && echo "$(date +%T) shift probe $*" || echo "$(date +%T) shift probe FAILED"; }
probe s8-posesum-contour-500ep s8-best-jitter-500ep s8-best-spkgain-500ep s8-best-mag-500ep
nb106
run s8-mag-drop-$TAG        $HYBRID --signal magnitude --modal-order "" $BEST --laser-drop 0.25 --band-drop 0.25
for b in remove mean resample; do imp s8-mag-drop-$TAG $b; done; cam s8-mag-drop-$TAG
nb106
# fjit: the frequency counterpart of laser jitter: every spectrum stretched f -> f (1 + eps), eps ~ U(-.002, .002) per sample (resonances
# move together, as with a slight stiffness / temperature change). Compared with s8-best-mag (identical otherwise)
run s8-mag-fjit-$TAG        $HYBRID --signal magnitude --modal-order "" $BEST --freq-jitter 0.002
imp s8-mag-fjit-$TAG mean; probe s8-mag-fjit-$TAG s8-mag-drop-$TAG
nb106
run s8-mag-nopose-r2-$TAG    $HYBRID --signal magnitude --modal-order "" $NOPOSE
run s8-mag-poseonly-r2-$TAG  $HYBRID --signal magnitude --modal-order "" $BEST --pose-merge only
run s8-mag-spk2-r2-$TAG      $HYBRID --signal magnitude --modal-order "" $BEST --speakers 1 2 3 5 7
run s8-mag-slots2-r2-$TAG    $HYBRID --signal magnitude --modal-order "" $BEST --pose-merge slots
run s8-hybrid-mag-r2-$TAG    $HYBRID --signal magnitude --modal-order ""
run s8-hybrid-nobn-r2-$TAG   $HYBRID --signal log_power --modal-order "" --box-norm 0
run s8-hybrid-nolaser-r2-$TAG $HYBRID --signal log_power --modal-order "" --modal-laser-layers 0
echo "$(date +%T) queue 5 done"
