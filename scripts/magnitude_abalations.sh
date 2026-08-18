#!/usr/bin/env bash
# Magnitude normalization sweep: which of the 11 recipes in normalizations.MAG_RECIPES makes
# the best input feature?
#
# Every arm is identical except --mag-recipe. That flag owns the whole magnitude stage: it sets
# --signal-mode (|Z| vs log|Z|) and decides which references get built, and it builds them in
# its OWN domain -- a linear arm divides by a linear reference, a log arm subtracts a log one.
# Mixing those is silently meaningless, which is why the recipe picks all three rather than
# leaving --signal-mode / --subtract-* to be set by hand.
#
# The grid crosses two choices:
#   domain     |Z|  (linear)   vs  log|Z|
#   reference  none / speaker mean (SM) / empty box (EB) / both
#
# and, in the linear domain only, the OPERATION:
#   divide    the matched operation -- the chain Z = C(f) S_s(f) H_l(f) T_n(f) is multiplicative,
#             so a reference comes off by division
#   subtract  the mismatched control -- if the multiplicative story is right these should lose
#
# There is no logmag_div_* arm: dividing by a log is unusable, since log|EB| crosses zero.
# See the comment block in src/model/normalizations.py for why shifting does not rescue it.
#
# rho below is from notebook 70 section 4 (speaker 3, feature cosine similarity vs COM distance;
# more negative = similarity falls off faster with distance = better). It is a LINEAR probe on
# ONE speaker, so treat it as a prior on what to expect, not as the answer -- that is what this
# sweep is for.
#
# NOTE: --mag-recipe is part of the MDS cache key, so each arm builds its own copy of the
# dataset. Check free disk before running the whole sweep.

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the sweep

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Only one copy of this sweep at a time: two concurrent launches share the single GPU (and, worse,
# the same --run-name and outputs_history), which OOMs the card instead of just running slower.
exec 200>"/tmp/$(basename "$0").lock"
flock -n 200 || { echo "$(basename "$0") is already running -- refusing to start a second copy" >&2; exit 1; }

TAG=v1
GROUP=magnitude-ablations-$TAG

# ordered best-first by the rho prior, so the arms most likely to matter run first
RECIPES=(
    logmag_sub_eb     # -0.452  best measured, and the most stable across speakers (sd 0.020)
    mag_div_eb        # -0.438  same correction as above, done in the linear domain
    logmag_sub_both   # -0.420  adding SM on top of EB; rho says this HURTS vs EB alone
    logmag            # -0.417  no reference at all -- most of the gain is just the log
    mag_div_spk       # -0.406
    logmag_sub_spk    # -0.393
    mag_div_both      # -0.370  pre-norm std ~300, EB*SM blows up where both are small
    mag_sub_eb        # -0.368  mismatched op; the gap to mag_div_eb is what that costs
    mag                # -0.308  the floor every other linear arm must beat
    mag_sub_both      # -0.205  control
    mag_sub_spk       # -0.190  control, worst in the grid
)

for RECIPE in "${RECIPES[@]}"; do
    echo "===== $RECIPE ====="
    python src/run.py --mag-recipe "$RECIPE" \
        --split gastronorm --test-size 0.05 \
        --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel --max-duration 1000ep \
        --decoder attn --wandb-group "$GROUP" \
        --run-name "${RECIPE//_/-}-$TAG"
done
