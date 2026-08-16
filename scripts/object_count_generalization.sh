#!/usr/bin/env bash
# Cross-object-count generalization: does a model trained on N-object scenes transfer to a
# different number of objects?
#
# Three arms, identical in every hyperparameter -- the only thing that moves is which object
# counts appear in train:
#   1. train on 1-object scenes            -> eval/1-obj (held out), eval/2-obj (OOD)
#   2. train on 2-object scenes            -> eval/2-obj (held out), eval/1-obj (OOD)
#   3. train on both (the control)         -> eval/1-obj, eval/2-obj (both held out, in-distribution)

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the sweep

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Only one copy of this sweep at a time: two concurrent launches share the single GPU (and, worse,
# the same --run-name and outputs_history), which OOMs the card instead of just running slower.
exec 200>"/tmp/$(basename "$0").lock"
flock -n 200 || { echo "$(basename "$0") is already running -- refusing to start a second copy" >&2; exit 1; }

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
