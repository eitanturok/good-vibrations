#!/usr/bin/env bash
# Train (row, col) heads on a frozen dec-d3-conv-v6. See scripts/com_head.py.
#
# Launch under tmux:
#   tmux new -s com
#   ./scripts/com_head.sh 2>&1 | tee runs/com_head.log
# Detach with ctrl-b d; reattach with `tmux attach -t com`.
#
# Four head types, each on the decoder feature map alone and again with the encoder
# bottleneck concatenated. Every run prints `no-train` -- the centre of mass of the
# decoder's own mask -- which is the bar, since it is what the pipeline does today at
# zero training cost.

set -u  # deliberately NOT -e: one bad arm should not kill the sweep

cd "$(dirname "$0")/.."
export PYTHONPATH=.

PY=.venv/bin/python   # tmux may not have the venv activated

# Shared across every arm below. Epochs and seeds are the two worth changing.
COMMON="--epochs 300 --seeds 3 --hidden 256 --lr 1e-3 --weight-decay 1e-4 --test-frac 0.25 --seed 42"

# Which split. `original` evaluates on dec-d3-conv-v6's own eval/1-cube -- positions the
# BACKBONE never saw, so the frozen features there are not reconstructions of masks it was
# trained to fit. `repartition` re-splits all one-cube data by position, which puts ~47% of
# backbone training data in the eval half and flatters every number (no-train scores 0.0369
# there vs 0.0673 here). Run both to see the gap.
SPLIT="${SPLIT:---split-mode original}"
COMMON="$COMMON $SPLIT"

# ***** 0 cache the frozen features once *****
# Every arm reads this, so each head trains in seconds instead of re-running the backbone.
echo "***** cache *****"
$PY scripts/com_head.py --cache-only $COMMON || { echo "cache failed"; exit 1; }

# ***** 1 heads on the decoder feature map *****
echo
echo "***** softargmax / decoder *****"
$PY scripts/com_head.py --head softargmax $COMMON

echo
echo "***** conv / decoder *****"
$PY scripts/com_head.py --head conv $COMMON

echo
echo "***** mlp / decoder *****"
$PY scripts/com_head.py --head mlp $COMMON

echo
echo "***** linear / decoder *****"
$PY scripts/com_head.py --head linear $COMMON

# ***** 2 same heads, plus the encoder bottleneck *****
# `emb` holds no information the decoder map lacks -- the decoder is a deterministic
# function of it -- so this can only win by offering an easier basis. If it ties, the
# decoder map already carries everything.
echo
echo "***** softargmax / decoder + encoder *****"
$PY scripts/com_head.py --head softargmax --use-emb $COMMON

echo
echo "***** conv / decoder + encoder *****"
$PY scripts/com_head.py --head conv --use-emb $COMMON

echo
echo "***** mlp / decoder + encoder *****"
$PY scripts/com_head.py --head mlp --use-emb $COMMON

echo
echo "***** linear / decoder + encoder *****"
$PY scripts/com_head.py --head linear --use-emb $COMMON

# ***** 3 sharpen the heatmap *****
# Soft-argmax returns the EXPECTED coordinate, so diffuse mass drags the estimate toward
# the image centre. temperature < 1 pulls it back toward the mode.
echo
echo "***** softargmax / decoder, temperature 0.5 *****"
$PY scripts/com_head.py --head softargmax --temperature 0.5 --tag t05 $COMMON

echo
echo "***** softargmax / decoder, temperature 0.2 *****"
$PY scripts/com_head.py --head softargmax --temperature 0.2 --tag t02 $COMMON

echo
echo "***** done *****"
ls -1 runs/dec-d3-conv-v6/com_head/results_*.json 2>/dev/null
