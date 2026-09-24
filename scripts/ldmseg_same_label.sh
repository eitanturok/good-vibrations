#!/usr/bin/env bash
# A/B comparison for LDMSegVibrationModel: does distinguishing between objects (vs. just detecting
# where "something" is) matter for this task? Same architecture/hyperparameters as
# scripts/ldmseg_baseline.sh in both arms -- only the target labels fed to the frozen LDMSeg
# encoder differ:
#
#   1. same-label:  every instance in a sample is bit-encoded with the SAME id (see
#                    src5/precompute_latents.py's build_id_map --same-label and src5/run.py's
#                    --same-label), so z_gt only ever encodes "object present here", never which
#                    object -- the model can't learn to distinguish instances even if it wanted to.
#   2. diff-label:   the existing baseline setup (scripts/ldmseg_baseline.sh) -- every instance
#                    gets its own id, so z_gt (and hence target_class_map) can carry instance
#                    identity. Rerun here (not just pointed at an old ldmseg-baseline-* run) so
#                    both arms share this script's exact TAG/settings for a clean comparison.
#
# --same-label reads a SEPARATE cache file (image/05_ldmseg_latent_same_label.npz) than the
# default (image/05_ldmseg_latent.npz) -- precompute_latents.py never overwrites the other, so
# both arms can coexist against the same --data-dir. Run the precompute step once before this
# script if the same-label cache doesn't exist yet:
#   PYTHONPATH=. python src5/precompute_latents.py --same-label
#
#   TAG=v1 ./scripts/ldmseg_same_label.sh   # bump TAG for a fresh pair of run names

set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.
# see scripts/ldmseg_baseline.sh for why: torch DataLoader workers leak fds and the inherited
# default (1024) silently deadlocks a long run once exhausted.
ulimit -n 65536

TAG="${TAG:-v1}"

COMMON="--lr 1e-4 --max-duration 2000ep --batch-size 8 --eval-batch-size 8 \
        --compile --log-points --augment-fft 0 --augment-mask 0"

# make sure the same-label cache exists for every sample before training touches it -- a no-op if
# already fully cached (precompute_latents.py skips samples that already have the cache file).
python src5/precompute_latents.py --same-label

python src5/run.py $COMMON \
    --same-label \
    --run-name "ldmseg-same-label-$TAG"

python src5/run.py $COMMON \
    --run-name "ldmseg-diff-label-$TAG"
