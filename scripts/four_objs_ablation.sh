#!/usr/bin/env bash
# Why did bb-four-objs-v1 score so much higher than the old gastronorm run? Six runs on the four_objs
# capture. Every split has the SAME eval sets as gastronorm_four_objs_2 (see src/model/dataset.py);
# only train changes. "main" = empty box + cube, vase, candle, soap dispenser.
#
#   base         gastronorm_four_objs_2            everything, all speakers                 4761 samples
#   spk-holdout  gastronorm_four_objs_spk_holdout  main, speakers 1,2,3,(5),7 -- no speaker 4 3420
#                                                  + eval/<obj>-spk4-seenpos (new speaker, seen position)
#   spk-control  gastronorm_four_objs_spk_control  main, same positions, 4 random speakers   3422
#                                                  per position (speaker 4 heard) -- same count as holdout
#   scale-s      gastronorm_four_objs_scale_s      main, 53 positions per object            1117
#   scale-m      gastronorm_four_objs_scale_m      main, 125 per object (old gastronorm size) 2556
#   scale-l      gastronorm_four_objs_scale_l      main, every position                     4276
#
# Training length is in BATCHES so every arm gets the same number of gradient steps (an epoch budget
# would give the smaller arms fewer): 9000ba = the base arm's 500 epochs at 18 batches/epoch. Warmup,
# eval, image logging and checkpoints are pinned the same way (100 / 50 / 50 / 250 base epochs).
# One seed (42): small eval sets (cube-vase 10 positions, two-cubes 14) are noisy.
#   TAG=v2 ./scripts/four_objs_ablation.sh 2>&1 | tee runs/four_objs_ablation.log

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.

TAG="${TAG:-v1}"
GROUP="four-objs-ablation-$TAG"

COMMON="--model boombox --d-model 1024 --batch-size 256 \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel \
        --laser-dropout 0 --freq-dropout 0 \
        --max-duration 9000ba --t-warmup 1800ba --eval-interval 900ba --viz-interval 900ba \
        --checkpoint-interval 4500ba --seed 42 \
        --data-dir experiments/2026_09_27_gastronorm_four_objs --wandb-group $GROUP"

for ARM in base spk_holdout spk_control scale_s scale_m scale_l; do
    SPLIT="gastronorm_four_objs_$ARM"
    [ "$ARM" = base ] && SPLIT=gastronorm_four_objs_2
    .venv/bin/python src/run.py $COMMON --split "$SPLIT" --run-name "fo-${ARM//_/-}-$TAG"
done
