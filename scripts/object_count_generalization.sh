#!/usr/bin/env bash
# Cross-object-count generalization: does a model trained on N-object scenes transfer to a
# different number of objects?
#
# Three arms, identical in every hyperparameter -- the only thing that moves is which object
# counts appear in train:
#   1. train on 1-object scenes            -> eval/1-obj (held out), eval/2-obj (OOD)
#   2. train on 2-object scenes            -> eval/2-obj (held out), eval/1-obj (OOD)
#   3. train on both (the control)         -> eval/1-obj, eval/2-obj (both held out, in-distribution)
#
# Arm 3 is the ceiling the two OOD numbers get read against. Do NOT substitute the plain
# `gastronorm` split for it: that split carves its 1-cube eval by sample rather than by position,
# so all 12 of its eval/1-cube positions also appear in train, and it trains on empty-box samples
# the other two arms never see.
#
# Every split holds its eval out by position_id, so no position in an eval set was seen during
# training from any other speaker. Held-out fractions are ~5% of the train-side pool; because whole
# positions move as a unit, the sample-level fraction is approximate.
#
# Object count is read from each row's `n_objects`, not the layout name -- they disagree here
# (x-shift/y-shift are 2-object layouts, lid-purple-cube spans 0 and 1). The 0-object (empty-box)
# and 3-object (purple--green-red-cube) samples fall outside all three arms.
#
# Runs sequentially on the single GPU. Launch under tmux:
#   tmux new -s objcount
#   ./scripts/object_count_generalization.sh 2>&1 | tee runs/object_count_generalization.log
# Detach with ctrl-b d; reattach with `tmux attach -t objcount`.

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the sweep

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Only one copy of this sweep at a time: two concurrent launches share the single GPU (and, worse,
# the same --run-name and outputs_history), which OOMs the card instead of just running slower.
exec 200>"/tmp/$(basename "$0").lock"
flock -n 200 || { echo "$(basename "$0") is already running -- refusing to start a second copy" >&2; exit 1; }

# run.py sets autoresume=True whenever --run-name is passed, so re-running a name that already has
# a checkpoint resumes it instead of starting over, and stale outputs_history from an earlier attempt
# gets interleaved with the new one. Bump TAG to start a clean set of runs.
TAG=v3
GROUP=object-count-generalization-$TAG

# ***** arm 1: train on 1 object, eval on 2 *****

python src/run.py --split gastronorm_train1_eval2 --test-size 0.05 \
    --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-pixel --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name objcount-train1-eval2-$TAG

# ***** arm 2: train on 2 objects, eval on 1 *****

python src/run.py --split gastronorm_train2_eval1 --test-size 0.05 \
    --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-pixel --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name objcount-train2-eval1-$TAG

# ***** arm 3: train on both, eval on both -- the in-distribution control *****

python src/run.py --split gastronorm_train12_eval12 --test-size 0.05 \
    --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
    --loss-fn ce-pixel --max-duration 1000ep \
    --decoder attn --wandb-group "$GROUP" --run-name objcount-train12-eval12-$TAG
