#!/usr/bin/env bash
# Input-feature ablation on the regular conv-conv boombox model (--model boombox --encoder single,
# ce-pixel loss, full gastronorm split, 32x32 output). The model is held fixed; only the input
# representation changes. Two independent axes:
#
#   magnitude block   magnitude | logmag | logmag-eb (log|Z| - per-speaker empty-box log|Z|) | none
#   phase block       none | raw | src | src-eb        (all [cos, sin], via --phase-arm)
#
#     raw     angle(Z)                                  ungauged
#     src     angle(Z) - angle(S)                       S = DTFT of the played chirp (data/audio/)
#     src-eb  angle(Z) - angle(S) - angle(E_spk)        E_spk = circular mean over the speaker's
#                                                       train empty-box samples of angle(Z.conj(S))
#
#   F1  baseline            magnitude          -- the current conv-conv config (bb-spk-baseline)
#   F2  phase               raw phase only
#   F3  logmag              logmag
#   F4  logmag-eb           logmag-eb
#   F5  logmag+phase        logmag    + raw
#   F6  logmag-eb+phase     logmag-eb + raw
#   F7  phase-src           src phase only
#   F8  phase-src-eb        src-eb phase only
#   F9  logmag+src          logmag    + src
#   F10 logmag+src-eb       logmag    + src-eb
#   F11 logmag-eb+src       logmag-eb + src
#   F12 logmag-eb+src-eb    logmag-eb + src-eb
#
# Pairs to read: F2/F7/F8 = which phase gauge carries signal on its own; F5 vs F9 vs F10 (and F6 vs
# F11 vs F12) = does the gauge matter once magnitude is present; F4-F3 = empty-box magnitude ref.
#
# Every phase block is apply_phase_arm's: [cos, sin], NOT std-normalized, and zeroed outside the
# top 10% of bins by laser-mean magnitude (so phase-only arms see ~124 of 1235 bins).
#
# What to expect (measured on the gastronorm data when these arms were added):
#   * S is fixed across samples, so src is a constant per-bin rotation -- it cannot fix per-sample
#     trigger jitter. What it does do is unwrap the chirp's own phase: mean |dphase/dbin| falls
#     2.03 -> 0.83 rad (1.57 = noise), i.e. the phase becomes smooth along f, which a conv encoder
#     that shares weights across frequency can exploit and raw phase does not offer.
#   * E_spk is averaged over only 3-4 empty-box samples per speaker, with resultant R 0.50-0.79
#     (chance for n=4 is ~0.5). It is a noisy reference; cross-sample R only moves 0.23 -> 0.27.
#
# Each arm has its own preprocessing config -> its own MDS cache (one-time build per arm).
# run.py autoresumes on --run-name, so bump TAG for a clean set.
#
#   tmux new -s feats
#   ./scripts/boombox_input_features.sh 2>&1 | tee runs/boombox_input_features.log

set -u  # deliberately NOT -e: one diverging run should not kill the rest

cd "$(dirname "$0")/.."
export PYTHONPATH=.

exec 9>/tmp/boombox_input_features.sh.lock
flock -n 9 || { echo "another copy of $(basename "$0") is already running; exiting" >&2; exit 1; }

TAG="${TAG:-v1}"
GROUP="boombox-input-features-$TAG"

# Same base config as scripts/boombox_speaker_ablation.sh (bb-spk-baseline).
COMMON="--model boombox --encoder single --d-model 512 --batch-size 1024 \
        --data-dir experiments/31_07_2026_gastronorm_exp1 --split gastronorm \
        --augment-mask 0 --augment-fft 0 --loss-fn ce-pixel --max-duration 1000ep \
        --laser-dropout 0 --freq-dropout 0 --wandb-group $GROUP"

LOGMAG="--signal-mode log_magnitude"
LOGMAG_EB="--signal-mode log_magnitude --subtract-empty-box"
NOMAG="--signal-mode none"

# FROM=5 skips F1-F4 (e.g. after a crash): the run name is the last argument, its fN picks the arm.
FROM="${FROM:-1}"
run() {
    local name="${@: -1}" i
    i=$(sed -E 's/.*-f([0-9]+)-.*/\1/' <<< "$name")
    if [ "$i" -lt "$FROM" ]; then echo "skip $name (FROM=$FROM)"; return; fi
    python src/run.py $COMMON "$@"
}

run --signal-mode magnitude                   --run-name "bb-feat-f1-baseline-$TAG"
run $NOMAG     --phase-arm raw_phase          --run-name "bb-feat-f2-phase-$TAG"
run $LOGMAG                                   --run-name "bb-feat-f3-logmag-$TAG"
run $LOGMAG_EB                                --run-name "bb-feat-f4-logmag-eb-$TAG"
run $LOGMAG    --phase-arm raw_phase          --run-name "bb-feat-f5-logmag-phase-$TAG"
run $LOGMAG_EB --phase-arm raw_phase          --run-name "bb-feat-f6-logmag-eb-phase-$TAG"
run $NOMAG     --phase-arm src                --run-name "bb-feat-f7-phase-src-$TAG"
run $NOMAG     --phase-arm src_eb             --run-name "bb-feat-f8-phase-src-eb-$TAG"
run $LOGMAG    --phase-arm src                --run-name "bb-feat-f9-logmag-src-$TAG"
run $LOGMAG    --phase-arm src_eb             --run-name "bb-feat-f10-logmag-src-eb-$TAG"
run $LOGMAG_EB --phase-arm src                --run-name "bb-feat-f11-logmag-eb-src-$TAG"
run $LOGMAG_EB --phase-arm src_eb             --run-name "bb-feat-f12-logmag-eb-src-eb-$TAG"
