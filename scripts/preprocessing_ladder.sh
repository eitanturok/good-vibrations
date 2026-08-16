#!/usr/bin/env bash
# Preprocessing ladder: how much does the *input representation* matter, holding the model fixed?
#
# Five rungs. Each one adds exactly ONE change to the rung above it, so any difference in the
# metrics is attributable to that change alone:
#
#   R0  magnitude                                          the current default -- the baseline
#   R1  magnitude          + speaker mean subtracted        does removing the speaker help at all?
#   R2  log magnitude                                       is the log alone worth it?
#   R3  log magnitude      + speaker mean subtracted        the physically correct pairing
#   R4  log magnitude      + empty box + speaker mean       also divide out the box's own resonances
#
# Everything else (decoder, loss, resolution, duration, seed) is identical across all five.
#
# ***** why this ordering *****
#
# The speaker chain is MULTIPLICATIVE: Y(f) = S(f) . H_box(f) . (object perturbation). In log space
# multiplication becomes addition, so a subtraction removes it cleanly. That predicts R1 (subtract
# in LINEAR space, a mismatch) should underperform R3 (subtract in LOG space, correct), and R1 may
# even come out below R0. R1 is included precisely so that prediction is tested rather than assumed.
#
# R4 adds a per-speaker empty-box reference. Subtracting a log reference IS dividing by it
# (log A - log B = log(A/B)), done in the domain where it is numerically safe: dividing in linear
# space explodes at anti-resonances where the reference is near zero, and those bins carry more
# position information than the peaks (notebook 57: dips alone rho=-0.523). The empty-box reference
# already contains the speaker chain, so build_dataset computes the speaker mean on already-
# referenced signal -- the two corrections stay complementary instead of removing the gain twice.
#
# ***** predicted ordering *****
#
# From the no-training probes in notebooks/68_preprocessing.ipynb (Spearman rho between signal
# distance and physical distance on the 70-position purple-cube sweep, speaker held fixed):
#
#   magnitude                        +0.352      <- R0
#   log magnitude                    +0.396      <- R2
#   log magnitude - empty box        +0.486      <- R4 (measured at one speaker only)
#
# and across all 8 speakers (notebooks/69, 560 samples), within-speaker rho:
#
#   linear:  raw +0.277 | subtract speaker mean +0.246   <- R1, WORSE than raw, as predicted
#   log:     raw +0.305 | subtract speaker mean +0.364   <- R3, better
#
# Those are rank correlations on a distance metric -- a proxy for learnability, not a trained-model
# result. This ladder is what turns them into one. If the trained ordering disagrees with the
# predicted ordering, the proxy is wrong and that is itself worth knowing.
#
# ***** running it *****
#
# Sequential on the single GPU, ~5 x 1000ep. Launch under tmux:
#   tmux new -s prep
#   ./scripts/preprocessing_ladder.sh 2>&1 | tee runs/preprocessing_ladder.log
# Detach with ctrl-b d; reattach with `tmux attach -t prep`.
#
# NOTE: each rung has a different preprocessing config, so each builds its OWN MDS cache
# (the config is part of the cache key). Expect a one-time ~5-10 min build per rung, and
# ~5 extra copies of the dataset on disk.
#
# NOTE: run.py sets autoresume=True whenever --run-name is passed, so re-running a name RESUMES
# rather than restarting. Bump TAG for a clean set.

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the ladder

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Refuse to run two copies at once: they would share a GPU and, with the same --run-name, the same
# checkpoint and outputs_history directories.
exec 9>/tmp/preprocessing_ladder.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v2
GROUP=preprocessing-ladder-$TAG

# Held fixed across every rung. The split is the REPAIRED gastronorm: eval/1-cube is held out by
# position (it used to leak all 12 of its positions into train), and eval/2-cubes now covers every
# n_objects==2 sample rather than grid4 alone. eval/2-cubes-grid4 is kept as a subset so these
# numbers can still be lined up against the pre-repair runs.
COMMON="--split gastronorm --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel --decoder attn --max-duration 1000ep --wandb-group $GROUP"

# ***** R0: magnitude (the current default) *****
# The baseline. Everything below is measured against this.
python src/run.py $COMMON \
    --signal-mode magnitude \
    --run-name prep-r0-magnitude-$TAG

# ***** R1: magnitude + speaker mean *****
# Subtracting an arithmetic mean of magnitudes. The speaker chain is multiplicative, so this is the
# wrong operation for this domain -- included to test that claim, not because it is expected to win.
python src/run.py $COMMON \
    --signal-mode magnitude --subtract-speaker-mean \
    --run-name prep-r1-magnitude-spk-$TAG

# ***** R2: log magnitude *****
# Compresses the orders-of-magnitude dynamic range of the resonance peaks, which gives the
# anti-resonance dips comparable weight. No speaker handling yet -- isolates the log itself.
python src/run.py $COMMON \
    --signal-mode log_magnitude \
    --run-name prep-r2-logmag-$TAG

# ***** R3: log magnitude + speaker mean *****
# The physically correct pairing: a mean of logs is the log of a geometric mean, so subtracting it
# divides out the speaker's gain. R3 - R2 isolates the speaker correction; R3 - R1 isolates the
# domain it is applied in.
python src/run.py $COMMON \
    --signal-mode log_magnitude --subtract-speaker-mean \
    --run-name prep-r3-logmag-spk-$TAG

# ***** R4: log magnitude + empty box + speaker mean *****
# Adds the per-speaker empty-box reference, removing the box's own transfer function on top of the
# speaker's. Predicted to be the best rung. Requires log_magnitude (build_dataset enforces it).
python src/run.py $COMMON \
    --signal-mode log_magnitude --subtract-empty-box --subtract-speaker-mean \
    --run-name prep-r4-logmag-emptybox-spk-$TAG
