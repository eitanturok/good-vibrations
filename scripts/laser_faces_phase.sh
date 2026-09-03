#!/usr/bin/env bash
# Does phase help, and does it help the SAME amount from every laser face?
#
# scripts/laser_faces.sh asked which face carries the signal, on magnitude alone. This crosses that
# question with the phase block: if phase pays off on one face and not the other, that is a fact
# about the geometry -- the two faces stand at different angles to the box, so the phase structure
# reaching them is not the same signal viewed twice.
#
# ***** the faces *****
#
#   left     columns 1,2,3,4          32 lasers   8x4
#   right    columns 5,6,7,8          32 lasers   8x4
#   both     columns 1,3,6,8          32 lasers   8x4   -- 2 columns from each face
#   all      columns 1,2,3,4,5,6,7,8  64 lasers   8x8   -- both faces whole
#
# Columns 0 and 9 are dropped everywhere, which is what makes the first three rows COUNT-MATCHED at
# 32 lasers: left vs right vs both is then a clean question about which lasers, with the number
# of them held fixed. Without that, `both` at 4 columns against 5-column faces confounds spread
# with count -- the flaw scripts/laser_faces.sh has and this sweep does not.
#
# `all` is those same 8 columns together, so it is exactly 2x the others rather than an arbitrary
# larger set: the ceiling the three 32-laser subsets are read against, not a peer of them.
#
# ***** the phase arms *****
#
#   baseline      no phase at all -- plain magnitude. THE reference each face is read against.
#   raw_phase     ungauged cos/sin of angle(Z). No gauge fix, so (raw_phase - baseline) is what
#                 raw phase information alone buys.
#   rel_laser     theta_l(f) - theta_ref(f). Differences along LASER, so it is exactly invariant
#                 to anything constant across lasers: the chirp's own phase, the speaker transfer
#                 function, trigger jitter. (rel_laser - raw_phase) isolates the gauge.
#   rel_laser_med same gauge, but theta_ref is the per-(f,c) componentwise MEDIAN phasor over
#                 lasers instead of the magnitude-weighted mean. Robust to a few outlier lasers
#                 -- one on a node, one glitched -- that pull the mean phasor but not the median.
#                 (rel_laser_med - rel_laser) is what swapping the reference statistic buys.
#   group_delay   theta_l(f+df) - theta_l(f). Differences along FREQ, so it cancels an f-linear
#                 ramp and keeps the f-structure rel_laser divides out. Complementary, not nested.
#
# The gauged arms matter more here than in scripts/phase_ablation.sh: rel_laser references lasers
# against each other, so restricting to one face changes what the gauge is relative TO. Whether it
# survives a 40-laser face, or needs both faces to have something to reference against, is exactly
# what the face x arm crossing answers.
#
# ce-pixel throughout, matching every other boombox sweep here.
#
# ***** running it *****
#
# Full sweep is 20 runs, sequential on the single GPU: 4 faces x 5 arms. None of these column
# sets matches an arm of scripts/laser_faces.sh (that sweep used the 5-column faces), so all are
# new work and no baseline here can be reused from it.
#
# RIGHT NOW only the rel_laser_med arm (4 runs) is live -- the baseline / rel_laser / group_delay
# / raw_phase blocks below are wrapped in a `: <<'ALREADY_RAN'` heredoc because they have already
# run under TAG=v1. Delete that wrapper to re-run the whole sweep.
#
#     tmux new -s lasers-phase 'scripts/laser_faces_phase.sh 2>&1 | tee /tmp/laser_faces_phase.log'
#
# NOTE on disk: phase adds channels to the stored X, so the MDS cache is keyed on (columns, arm) --
# a separate build per (face, arm), and the `all` builds are the biggest at 64 lasers. Each is a
# one-off ~1 min for this capture; the 4 rel_laser_med builds are new.

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the sweep

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Only one copy at a time: two concurrent launches share the single GPU and, worse, the same
# --run-name and outputs_history.
exec 9>/tmp/laser_faces_phase.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG=v1
GROUP=laser-faces-phase-$TAG

COMMON="--data-dir experiments/31_08_2026_green_plastic_two_laser_faces \
        --split green_plastic --model boombox --d-model 1024 \
        --out-h 21 --out-w 30 --augment-mask 0 --augment-fft 0 \
        --loss-fn ce-pixel --max-duration 1000ep --batch-size 256 --wandb-group $GROUP"

# ***** rel_laser_med: relative-laser gauge referenced to the MEDIAN laser phasor, not the mean.
# The only live block right now -- see the "running it" note at the top. 4 runs, one per face.

python src/run.py $COMMON --laser-cols 1,2,3,4 --phase-arm rel_laser_med \
    --run-name lasers-left-rel-laser-med-$TAG

python src/run.py $COMMON --laser-cols 5,6,7,8 --phase-arm rel_laser_med \
    --run-name lasers-right-rel-laser-med-$TAG

python src/run.py $COMMON --laser-cols 1,3,6,8 --phase-arm rel_laser_med \
    --run-name lasers-both-rel-laser-med-$TAG

# both faces whole, still without columns 0 and 9: 8x8 = 64
python src/run.py $COMMON --laser-cols 1,2,3,4,5,6,7,8 --phase-arm rel_laser_med \
    --run-name lasers-all-rel-laser-med-$TAG


# ================================================================================================
# Everything below has already run under TAG=v1. It is fed to `:` via a quoted heredoc so nothing
# expands and nothing executes. Delete the `: <<'ALREADY_RAN'` line and the closing `ALREADY_RAN`
# to bring the full 4-faces x 4-arms sweep back.
# ================================================================================================
: <<'ALREADY_RAN'

# ***** baselines: magnitude only, one per face. Run FIRST -- nothing below is readable without them.

python src/run.py $COMMON --laser-cols 1,2,3,4 \
    --run-name lasers-left-baseline-$TAG

python src/run.py $COMMON --laser-cols 5,6,7,8 \
    --run-name lasers-right-baseline-$TAG

python src/run.py $COMMON --laser-cols 1,3,6,8 \
    --run-name lasers-both-baseline-$TAG

# both faces whole, still without columns 0 and 9: 8x8 = 64
python src/run.py $COMMON --laser-cols 1,2,3,4,5,6,7,8 \
    --run-name lasers-all-baseline-$TAG

# ***** rel_laser: the stronger gauge, so it goes before the weaker arms *****

python src/run.py $COMMON --laser-cols 1,2,3,4 --phase-arm rel_laser \
    --run-name lasers-left-rel-laser-$TAG

python src/run.py $COMMON --laser-cols 5,6,7,8 --phase-arm rel_laser \
    --run-name lasers-right-rel-laser-$TAG

python src/run.py $COMMON --laser-cols 1,3,6,8 --phase-arm rel_laser \
    --run-name lasers-both-rel-laser-$TAG

# both faces whole, still without columns 0 and 9: 8x8 = 64
python src/run.py $COMMON --laser-cols 1,2,3,4,5,6,7,8 --phase-arm rel_laser \
    --run-name lasers-all-rel-laser-$TAG

# ***** group_delay: cancels less, keeps the f-structure rel_laser throws away *****

python src/run.py $COMMON --laser-cols 1,2,3,4 --phase-arm group_delay \
    --run-name lasers-left-group-delay-$TAG

python src/run.py $COMMON --laser-cols 5,6,7,8 --phase-arm group_delay \
    --run-name lasers-right-group-delay-$TAG

python src/run.py $COMMON --laser-cols 1,3,6,8 --phase-arm group_delay \
    --run-name lasers-both-group-delay-$TAG

# both faces whole, still without columns 0 and 9: 8x8 = 64
python src/run.py $COMMON --laser-cols 1,2,3,4,5,6,7,8 --phase-arm group_delay \
    --run-name lasers-all-group-delay-$TAG

# ***** raw_phase: ungauged cos/sin. The control that separates "phase helps" from "the GAUGE helps"

python src/run.py $COMMON --laser-cols 1,2,3,4 --phase-arm raw_phase \
    --run-name lasers-left-raw-phase-$TAG

python src/run.py $COMMON --laser-cols 5,6,7,8 --phase-arm raw_phase \
    --run-name lasers-right-raw-phase-$TAG

python src/run.py $COMMON --laser-cols 1,3,6,8 --phase-arm raw_phase \
    --run-name lasers-both-raw-phase-$TAG

# both faces whole, still without columns 0 and 9: 8x8 = 64
python src/run.py $COMMON --laser-cols 1,2,3,4,5,6,7,8 --phase-arm raw_phase \
    --run-name lasers-all-raw-phase-$TAG

ALREADY_RAN
