#!/usr/bin/env bash
# Cross-object-count generalization: does a model trained on N-object scenes transfer to a
# different number of objects?
#
# Four runs, identical in every hyperparameter -- the only thing that moves is which object counts
# appear in train. All four evaluate on the same held-out eval/1-obj and eval/2-obj samples (5
# positions = 40 samples each), so the numbers line up directly.
#
# The 1-object pool is the binding constraint at 70 usable positions, so the three budget-matched
# runs each train on 65 positions. red-cube is excluded throughout: it is a distinct object, so
# keeping it would confound "how many objects" with "which object".
#
#   1. train on 1-object scenes       65 positions of 1-obj
#   2. train on 2-object scenes       65 positions of 2-obj
#   3. train on both, budget-matched  32 of 1-obj + 33 of 2-obj -- same total data as runs 1 and 2,
#                                     so it asks how a fixed budget is best spent
#   4. train on both, exposure-matched  65 of each -- twice the data, so it asks whether adding the
#                                     other count on top of what you have helps
#
# Runs 3 and 4 bracket the interpretation: run 3 alone cannot separate "mixing helps" from "65 > 32",
# and run 4 alone cannot separate "mixing helps" from "more data helps".

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the sweep

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Only one copy of this sweep at a time: two concurrent launches share the single GPU (and, worse,
# the same --run-name and outputs_history), which OOMs the card instead of just running slower.
exec 200>"/tmp/$(basename "$0").lock"
flock -n 200 || { echo "$(basename "$0") is already running -- refusing to start a second copy" >&2; exit 1; }

TAG=v4
GROUP=object-count-generalization-$TAG

# ***** 1: train on 1-object scenes -- 65 positions of 1-obj *****

python src/run.py --split gastronorm_train1_eval2 --test-size 0.05 \
    --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-pixel --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name objcount-train1-eval2-$TAG

# ***** 2: train on 2-object scenes -- 65 positions of 2-obj *****

python src/run.py --split gastronorm_train2_eval1 --test-size 0.05 \
    --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-pixel --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name objcount-train2-eval1-$TAG

# ***** 3: train on both, budget-matched -- 32 of 1-obj + 33 of 2-obj *****

python src/run.py --split gastronorm_train12_eval12 --test-size 0.05 \
    --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-pixel --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name objcount-train12-budget-$TAG

# ***** 4: train on both, exposure-matched -- 65 of each *****

python src/run.py --split gastronorm_train12_eval12_full --test-size 0.05 \
    --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-pixel --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name objcount-train12-full-$TAG
