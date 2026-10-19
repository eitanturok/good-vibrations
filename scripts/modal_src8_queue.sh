#!/usr/bin/env bash
# src8 priority queue (replaces modal_src8_v2/v3/v4.sh), 500 ep, one run per question. Order = what fixes the failures found so far
# (src8/EXPERIMENTS.md section 5, analyze.py / decompose.py). Every arm is the hybrid (ModalFormer front end, no axial blocks -> conv
# decoder, s8-hybrid-base-500ep) plus one change, unless noted. Contour is built on s8-hybrid-posesum (POSE), the later fixes (s8-best-*)
# on s8-posesum-contour (BEST);
# the nobn / nolaser / mag ablations stay on the plain hybrid (what made it beat conv):
#   pose       -- STN pose-factored branch (canonical mask warped by com + angle) + angle supervision: rotated soap is drawn upright
#   posesum    -- the pose branch merged as a residual (direct + pose logits) instead of a soft OR: pose (OR) over-painted
#                 (area x1.5, phantoms .40) because it could only add pixels; a residual can add and erase. Centred at the soft-argmax of
#                 the direct mask, not the com head (2x less accurate: CORE .043 vs .024)
#   contour    -- + outline channel + moment loss (area / centroid / orientation)
#   tversky    -- bce + Tversky (FN 0.7 / FP 0.3): misses and under-painting of weak, small objects (cardboard cube)
#   orient     -- + the orientation aux head only (does the embedding carry rotation, without drawing it?)
#   nobn / nolaser / mag -- ablations: what makes the hybrid localize better than conv (box norm, laser attention, input)
#   query      -- the fully attentional ModalFormer: LFLF axial + 4-layer pixel-query decoder + floor<->wall prior (interpretable)
#   LFLF       -- + axial laser/band attention (mode shapes)
#   farina     -- + the 2nd-harmonic IR spectrum as input channels
#   jitter     -- unseen lasers: train on the laser grid resampled at random sub-cell shifts (neighbour spectra correlate .87)
#   jitterp    -- jitter on half the samples only: full jitter cost the two-object splits .33 -> .25 (com err up, misses up),
#                 a sub-cell bilinear shift averages neighbouring lasers (spatial low-pass) and blurs the pattern that separates two objects
#   spkgain    -- unseen speakers: a random log-gain with the measured speaker-difference structure (shared by all lasers, ~14 Hz
#                 correlation, rms .35); the old --color-aug (4 cosines over 900 Hz) varied on the wrong scale
#   center     -- unseen speakers: laser-centering
#   mag (best) -- the best model on magnitude input: s6-gastro-plastic-cardboard-3k (conv, magnitude) holds unseen speaker 4 in
#                 plastic at its seen-speaker level (.668 vs .66), s8 (log power) drops to .395; log boosts the valleys between peaks,
#                 which are mostly the speaker's own response
#   film       -- per-box FiLM on the frequency stem: resonance-shift detection is a matched filter tuned to the linewidth, and
#                 cardboard's linewidths differ (weak-object misses: cardboard cube)
#   linskip    -- + a rank-256 linear shortcut from the input to the embedding
#   nopose     -- the best minus the pose branch (orient head + moment loss kept: theta in the latent, no warp). Eval-time ablation of
#                 spkgain: the direct half alone is at 33.6 deg, the pose half 19.0 deg -- is the explicit warp needed, trained without it?
#   poseonly   -- draw the mask once: the warped canonical mask IS the mask; the conv decoder is only a centre heatmap (softmax CE vs
#                 the true mask). Eval-time: pose half alone .712 CORE / .573 HARD vs .722 / .596 summed. One slot: two-object scenes at risk
#   conv-logpower-z / conv-mag -- input controls on the conv model
#   ./scripts/modal_src8_queue.sh 2>&1 | tee -a runs/modal_src8_queue.log

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

run s8-hybrid-pose-$TAG        $HYBRID --signal log_power --modal-order "" --pose-branch 1 --orient-weight 1
run s8-hybrid-posesum-$TAG     $HYBRID --signal log_power --modal-order "" $POSE
run s8-posesum-contour-$TAG    $HYBRID --signal log_power --modal-order "" $POSE --edge-weight 1 --moment-weight 1
run s8-best-tversky-$TAG    $HYBRID --signal log_power --modal-order "" $BEST --loss-fn bce-tversky
run s8-best-jitter-$TAG     $HYBRID --signal log_power --modal-order "" $JIT
run s8-best-spkgain-$TAG    $HYBRID --signal log_power --modal-order "" $BEST
run s8-best-center-$TAG     $HYBRID --signal log_power --modal-order "" $BEST --laser-center 1
run s8-best-film-$TAG       $HYBRID --signal log_power --modal-order "" $BEST --box-film 1
run s8-best-mag-$TAG       $HYBRID --signal magnitude --modal-order "" $BEST
# rebased on magnitude input (s8-best-mag = the new best: CORE .738, unseen speaker .355); the reference is s8-best-mag
NOPOSE="--orient-weight 1 --edge-weight 1 --moment-weight 1 --laser-jitter 0.5 --color-aug 0.35 --color-ell 20"   # = BEST minus POSE's pose flags
# slots: two-object scenes are the worst failure -- spkgain finds only one of the two objects in 42-83% (jitter: 100%), though its count
# head says 2 on 93% of them and its com head roughly places the missed one (.12). One pose slot per object (Mask2Former-style):
# own centre heatmap + canonical mask + angle + exists, matched to the objects by centre, supervised per object; soft max over slots
run s8-mag-slots-$TAG      $HYBRID --signal magnitude --modal-order "" $BEST --pose-merge slots
# slots2: v1 matched slots by centre only -> both slots on the one object of a single-object scene, the match a coin flip, exists bce
# stuck at .63 and both slots drawn (area x2-8 at ep50). Now DETR's cost: 5 x centre L1 - p(exists)
run s8-mag-slots2-$TAG     $HYBRID --signal magnitude --modal-order "" $BEST --pose-merge slots
# spk2: gastro's 5th speaker is 2, not 5, so --speakers 1 3 5 7 trains gastro on 3 speakers (plastic / cardboard: 4) and its speaker-2
# captures were never used. Unseen gastro speaker 4 is the worst group (.143; its response deviates most: log-gain rms .66 vs .45-.57,
# a -3.5 notch); more training speakers = more speaker diversity. Gastro eval splits gain speaker-2 samples at the same positions
run s8-mag-spk2-$TAG       $HYBRID --signal magnitude --modal-order "" $BEST --speakers 1 2 3 5 7
# slots2 again: the first attempt crashed at its first step when another process took GPU memory (driver/library mismatch -> allocator NVML assert)
run s8-mag-slots2-$TAG     $HYBRID --signal magnitude --modal-order "" $BEST --pose-merge slots
# laser x frequency importance (occlusion + integrated gradients) for notebooks/106: needs the GPU free of training (16 GB)
for r in s8-best-spkgain-500ep s8-best-mag-500ep; do
    [ -f runs/$r/importance.pt ] || { echo "$(date +%T) importance $r"; nice .venv/bin/python src8/importance.py $r > runs/importance_$r.log 2>&1; }
done
(cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 106_laser_freq_importance.ipynb > ../runs/nb106.log 2>&1) && echo "$(date +%T) notebook 106 executed"
run s8-mag-nopose-$TAG     $HYBRID --signal magnitude --modal-order "" $NOPOSE
run s8-mag-poseonly-$TAG   $HYBRID --signal magnitude --modal-order "" $BEST --pose-merge only
run s8-hybrid-nobn-$TAG        $HYBRID --signal log_power --modal-order "" --box-norm 0
run s8-hybrid-nolaser-$TAG     $HYBRID --signal log_power --modal-order "" --modal-laser-layers 0
run s8-hybrid-mag-$TAG         $HYBRID --signal magnitude --modal-order ""
run s8-query-$TAG              $QUERY  --signal log_power
run s8-best-LFLF-$TAG          $HYBRID --signal log_power --modal-order LFLF $BEST
run s8-best-farina-$TAG     $HYBRID --signal log_power --modal-order "" $BEST --window farina --num-workers 2
run s8-best-linskip-$TAG    $HYBRID --signal log_power --modal-order "" $BEST --lin-skip 256
run s8-conv-logpower-z-$TAG    $CONV   --signal log_power --norm z
run s8-conv-mag-$TAG           $CONV   --signal magnitude
