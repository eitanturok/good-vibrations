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

COMMON="--dataset gastronorm_plastic_cardboard --window tukey --batch-size 1024 --max-duration ${EPOCHS}ep --t-warmup 50ep \
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
# qpose v1 (killed at ep57): the union gated as logsumexp(mask + log p(exists)) let low-p blobs through (phantoms .997, CORE .06);
# v2 and later use soft_or, p = 1 - prod(1 - sigma(mask) sigma(exists)); the same fix is in --pose-merge slots (slots2-r3)
# qpose-v3 (00:50): v2 (soft_or) still drew one input-independent central blob. Toy (scratchpad toy_queries.py, objects encoded by which
# lasers carry a bump): queries alone find both objects in 1.00 of two-object scenes, + the pose head .16; the pose centre (the query
# mask's weighted centroid) let the shape loss move the centre -- detached soft-argmax centre: 1.00. Also the pixel embedding is now a
# conv head on the trunk + the direct logits (--obj-pix 1; toy: same as the 1x1). nopose-imom died at ep229 in the 01:20 global OOM
# (eval loaders' ~280 persistent workers, 42 GB): eval now loads in the main process
run s8-mag-qpose-v3-$TAG        $HYBRID --signal magnitude --modal-order "" $MULTI --obj-decoder queries --obj-pose 1 --microbatch 256
# qpeak (01:55): two-stage queries -- each starts at one of the direct head's top-5 local maxima (its features + position). v3 at ep50
# still had all 5 queries at the mean position (CORE .28 vs .37 for the baselines at ep50). Toy: iou .672 (best), both found 1.00
run s8-mag-qpeak-$TAG           $HYBRID --signal magnitude --modal-order "" $MULTI --obj-decoder queries --obj-pose 1 --obj-init peaks --microbatch 256
run s8-mag-nopose-imom-r2-$TAG  $HYBRID --signal magnitude --modal-order "" $MULTI
# qonly (04:20): qpose-v3 (CORE .745, HARD .599, ALL .415, both .61 -- the best on the goal metrics) but its rotated soap is 25.8 deg vs
# 17.1 for the single pose slot; its per-query angle loss is ~0 on train, so the free-form part of each query mask memorizes rotated
# templates. Here each query's mask IS its warped canonical mask (rotation only through the angle). Toy: iou .548 / both .80 (v3 .647 / 1.00)
run s8-mag-qonly-$TAG           $HYBRID --signal magnitude --modal-order "" $MULTI --obj-decoder queries --obj-pose 1 --obj-pose-merge only --microbatch 256
run s8-mag-centernet-r2-$TAG  $HYBRID --signal magnitude --modal-order "" $MULTI --obj-decoder centernet
run s8-mag-queries-r2-$TAG    $HYBRID --signal magnitude --modal-order "" $MULTI --obj-decoder queries --microbatch 256
run s8-mag-qpose-os3-$TAG     $HYBRID --signal magnitude --modal-order "" $MULTI --obj-decoder queries --obj-pose 1 --microbatch 256 --multi-oversample 3
# unseen-angle test (2026-10-20 14:40): the rotated soap's 30-60 deg wedge is held out of training (--dataset *_wedge, dataset.UNSEEN_ANGLE);
# HARD never tested unseen angles (every eval angle had a train angle within 0.7 deg). The candidates and their baselines, on the same data:
WEDGE="--dataset gastronorm_plastic_cardboard_wedge"
run s8-mag-qgauss-w-$TAG      $HYBRID --signal magnitude --modal-order "" $MULTI $WEDGE --obj-decoder queries --obj-shape gauss --microbatch 256
run s8-best-mag-w-$TAG        $HYBRID --signal magnitude --modal-order "" $BEST $WEDGE
run s8-mag-queries-w-$TAG     $HYBRID --signal magnitude --modal-order "" $MULTI $WEDGE --obj-decoder queries --microbatch 256
# 2026-10-21 00:50: gray first (the user's priority), centernet last (r2 under-detected)
# 2026-10-20 19:50: the simplest model (nopose-imom) also draws the grayscale photo, (1 - mask) floor + mask appearance (mask detached, so
# its mask metrics stay comparable); losses: whole-image L1 + object L1 on the true mask + gradient L1 near the object (arch.gray_losses)
run s8-mag-gray-w-$TAG        $HYBRID --signal magnitude --modal-order "" $MULTI $WEDGE --gray 1
# 2026-10-21 01:30: v1 killed at ep~60 -- its floor and appearance started at 0.5 gray and lr 1e-4 couldn't move them (gray_obj_l1 = the
# floor's). r2 starts both from the data, as differences from the empty box in logits: floor = median train photo, app = floor + contrast
run s8-mag-gray-r2-w-$TAG     $HYBRID --signal magnitude --modal-order "" $MULTI $WEDGE --gray 1
run s8-mag-nopose-imom-w-$TAG $HYBRID --signal magnitude --modal-order "" $MULTI $WEDGE
# 2026-10-21 02:40: gray-r2-w works (CORE photo L1 .098 < the .13 flat-colour oracle; masks = nopose-imom). Does the photo (sharp
# object edges) also make a better MASK target? gray-joint: the composite's mask is not detached
run s8-mag-gray-joint-w-$TAG  $HYBRID --signal magnitude --modal-order "" $MULTI $WEDGE --gray 1 --gray-joint 1
# moved up from the queue-5 block: the all-attention pixel-query decoder (s8-query crashed at ep50 on a since-fixed recompile bug)
run s8-mag-query-w-$TAG       $QUERY  --signal magnitude $MULTI $WEDGE --microbatch 256
# 2026-10-21 04:20: gray-joint = gray-r2 (masks .740 vs .743, photo .097 vs .098): the detached mask stays. The photo is now limited by
# ORIENTATION (rotated soap photo .17-.18 vs floor .20, masks at 27 deg). best-mag's pose slot gets 17-20 deg: best-mag + --gray
run s8-best-mag-gray-w-$TAG  $HYBRID --signal magnitude --modal-order "" $BEST $WEDGE --gray 1
run s8-mag-centernet-r3-w-$TAG $HYBRID --signal magnitude --modal-order "" $MULTI $WEDGE --obj-decoder centernet --obj-heat-sigma obj --obj-count-floor 1 --obj-jitter 1.5
# 2026-10-21 07:30: best-mag-gray: better masks (19 deg) but NOT a better rotated-soap photo -- the appearance head on the TRUE mask
# (.174) = the drawn photo, at the flat-colour oracle: it lacks inner detail, not orientation. Its 2 convs see 5 px; an object is
# 10-15 px. gray-wide: 4 layers, dilations 1,2,4 (17 px), so it can see which end of the object the pump is on
run s8-mag-gray-wide-w-$TAG   $HYBRID --signal magnitude --modal-order "" $MULTI $WEDGE --gray 1 --gray-layers 4
# from here on every run is on the wedge split (2026-10-20): the other eval splits are identical, HARD = the old HARD minus the wedge
COMMON="${COMMON/--dataset gastronorm_plastic_cardboard /--dataset gastronorm_plastic_cardboard_wedge }"
run s8-mag-slots2-r3-$TAG     $HYBRID --signal magnitude --modal-order "" $BEST --pose-merge slots
run s8-mag-gray-wide-detach-w-$TAG $HYBRID --signal magnitude --modal-order "" $MULTI $WEDGE --gray 1 --gray-layers 4 --gray-detach 1 --gray-floor median   # wide head on DETACHED features: does the hard-mask drop of gray-wide-w come from the photo loss training the decoder?
run s8-mag-gray-empty-w-$TAG    $HYBRID --signal magnitude --modal-order "" $MULTI $WEDGE --gray 1 --gray-floor empty   # = gray-r2-w with the floor = each box's first empty-box train photo

# --- the earlier queue-5 work, after the multi-object arms
probe() { .venv/bin/python src8/shift_probe.py "$@" > runs/shift_probe_$(date +%H%M).log 2>&1 && echo "$(date +%T) shift probe $*" || echo "$(date +%T) shift probe FAILED"; }
for r in s8-posesum-contour-500ep s8-best-jitter-500ep; do imp $r mean; cam $r; done
probe s8-posesum-contour-500ep s8-best-jitter-500ep s8-best-spkgain-500ep s8-best-mag-500ep
nb106
run s8-mag-drop-$TAG        $HYBRID --signal magnitude --modal-order "" $BEST --laser-drop 0.25 --band-drop 0.25
for b in remove mean resample; do imp s8-mag-drop-$TAG $b; done; cam s8-mag-drop-$TAG
nb106
run s8-mag-fjit-$TAG        $HYBRID --signal magnitude --modal-order "" $BEST --freq-jitter 0.002
imp s8-mag-fjit-$TAG mean; probe s8-mag-fjit-$TAG s8-mag-drop-$TAG
nb106
# s8-mag-query: moved up (s8-mag-query-w)
run s8-mag-poseonly-r2-$TAG  $HYBRID --signal magnitude --modal-order "" $BEST --pose-merge only
run s8-mag-spk2-r2-$TAG      $HYBRID --signal magnitude --modal-order "" $BEST --speakers 1 2 3 5 7
run s8-hybrid-mag-r2-$TAG    $HYBRID --signal magnitude --modal-order ""
run s8-hybrid-nobn-r2-$TAG   $HYBRID --signal log_power --modal-order "" --box-norm 0
run s8-hybrid-nolaser-r2-$TAG $HYBRID --signal log_power --modal-order "" --modal-laser-layers 0
echo "$(date +%T) queue 6 done"
