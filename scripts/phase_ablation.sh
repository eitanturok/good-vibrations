#!/usr/bin/env bash
# Phase ablation: does adding a PHASE gauge to the plain-magnitude input help?
#
# Stage A is held at plain `--signal-mode magnitude` for every arm -- no log, no empty-box,
# no speaker mean. Nothing varies but the phase block, so any difference is attributable to
# phase alone rather than confounded with the preprocessing ladder (scripts/preprocessing_ladder.sh).
#
# ***** the two gauges *****
#
#   relative laser phase   theta_l(f) - theta_ref(f)      difference along LASER
#   group delay            theta_l(f+df) - theta_l(f)     difference along FREQ
#
# They cancel different things, which is why P5/P6 (both at once) is worth running rather than
# assumed redundant. Verified directly against the complex spectrum:
#
#   * rel_laser is EXACTLY invariant to any term constant across lasers -- the chirp's own phase,
#     the speaker transfer function, trigger jitter, the -pi/2 + half-sample ramp from the cumsum
#     in pclk.py:120 -- whatever its shape in f.
#   * group_delay only reduces an f-LINEAR ramp to a constant offset (measured angle spread across
#     f: 1.2e-06, i.e. f-independent, so the model absorbs it). An arbitrary f-shape survives
#     (spread 1.26). This is why the linear-ramp fit in notebooks/57 could not remove speaker
#     group delay.
#
# So rel_laser is the stronger gauge and group_delay keeps information rel_laser throws away
# (rel_laser divides out the reference laser's own f-structure). Complementary, not nested.
#
# ***** predicted ordering *****
#
# From the no-training probes (Spearman rho between signal distance and physical distance,
# purple-cube sweep, mean of 4 speakers):
#
#   rel_laser_w (magnitude-weighted, global mean ref)   +0.428    <- P2, the best probe
#   group_delay                                          nb68's arm
#
# Those are rank correlations on a distance metric -- a proxy for learnability, not a trained
# result. This ladder turns them into one.
#
# ***** running it *****
#
# 8 runs, sequential on the single GPU. Under tmux:
#   tmux new -s phase
#   ./scripts/phase_ablation.sh 2>&1 | tee runs/phase_ablation.log
#
# Then sweep the mix on whichever arm wins:
#   ./scripts/phase_weight_sweep.sh rel_laser_w
#
# NOTE: every arm has its own preprocessing config, so each builds its OWN MDS cache (the phase
# arm is part of the cache key). Expect a one-time ~5-10 min build per arm and ~8 extra copies
# of the dataset on disk.
#
# NOTE: run.py sets autoresume=True whenever --run-name is passed, so re-running a name RESUMES
# rather than restarting. Bump TAG for a clean set.

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the ladder

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Refuse to run two copies at once: they would share a GPU and, with the same --run-name, the
# same checkpoint and outputs_history directories.
exec 9>/tmp/phase_ablation.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v2
GROUP=phase-ablation-$TAG

# BASELINE IS dec-d3-conv-v6 (scripts/decoder_benchmark.sh): the conv model at d_model 512 with
# no dropout, which won that benchmark at 0.272 2-cube soft-iou / 0.206 hard-iou against the mlp
# arm's 0.193. Every flag below is copied from it so the phase block is the only variable, and
# so B1 (no phase) should reproduce dec-d3-conv-v6's numbers -- if it does not, something else
# drifted and nothing here is interpretable.
#
# Pinned explicitly rather than left to defaults:
#   --batch-size 256   what dec-d3-conv-v6 ran. Boombox's conv encoder has no per-laser batch
#                      blowup, so it can afford what the transformer arms could not.
#   --laser-dropout 0  dropout at 0.3 UNDERFITS this model rather than regularizing it:
#   --freq-dropout 0   dec-d3-conv-v5 (drop 0.3) got 0.119 2-cube soft-iou vs 0.286 at drop 0,
#                      with train soft-iou 0.585 vs 0.955. _drop zeros whole lasers and the
#                      10x10 grid carries all the spatial signal.
#   --loss-fn mse      the known-good setting from the decoder benchmark (l1 collapsed to
#                      all-zero; no-sigmoid broke soft-iou by emitting negatives).
# out-h/out-w are already the run.py defaults (21x30), so they are not repeated here.
COMMON="--split gastronorm --augment-mask 0 --augment-fft 0 \
        --loss-fn mse --model boombox --d-model 512 --batch-size 256 \
        --laser-dropout 0 --freq-dropout 0 --max-duration 1000ep \
        --wandb-group $GROUP --signal-mode magnitude"

#***** phase arms *****
# Run FIRST, so the interesting comparisons land before the GPU time is spent on controls.

# P1/P2: laser-referenced. The stronger gauge. _w scales each unit phasor by its bin magnitude,
# so low-energy bins (where phase is near-uniform noise) contribute proportionally less.
python src/run.py $COMMON --phase-arm rel_laser --run-name "phase-p1-rel-laser-$TAG"
python src/run.py $COMMON --phase-arm rel_laser_w --run-name "phase-p2-rel-laser-w-$TAG"

# P3/P4: group delay. Cancels less, but keeps the reference laser's f-structure that P1/P2 divide out.
python src/run.py $COMMON --phase-arm group_delay --run-name "phase-p3-group-delay-$TAG"
python src/run.py $COMMON --phase-arm group_delay_w --run-name "phase-p4-group-delay-w-$TAG"

# P5/P6: both gauges concatenated. Complementary or redundant? If P5 ~ max(P1,P3) they are
# redundant; if P5 > both, the two cancellations are capturing different structure.
python src/run.py $COMMON --phase-arm both --run-name "phase-p5-both-$TAG"
python src/run.py $COMMON --phase-arm both_w --run-name "phase-p6-both-w-$TAG"

#***** baselines, LAST *****

# B1: no phase at all -- plain magnitude, stage A alone. THE reference point. Every arm above
# is measured against this one number, and it is a byte-identical rerun of dec-d3-conv-v6, so it
# doubles as a seed-variance check: it should land at ~0.272 2-cube soft-iou / 0.206 hard-iou.
python src/run.py $COMMON --run-name "phase-b1-none-$TAG"

# B2: ungauged cos/sin of angle(Z). Separates "phase helps" from "the GAUGE FIX helps". Without
# B2 a P1 win is unattributable: it could be the laser referencing, or it could be that any phase
# information at all is useful. B2 has the same channel count as P1-P4 and no gauge, so
# (P1 - B2) isolates the gauge and (B2 - B1) isolates raw phase.
python src/run.py $COMMON --phase-arm raw_phase --run-name "phase-b2-raw-phase-$TAG"
