#!/usr/bin/env bash
# Phase v3: do the three fixes recover what v2 lost?
#
# v2 (scripts/phase_ablation.sh, 8 arms at 1000ep) was conclusive that phase HURT, and the
# question was why. 2-cube soft-iou:
#
#   B1 no phase        0.2823   <- winner        P6 both_w        0.2428
#   (seed twin)        0.2716   <- noise = .011  P1 rel_laser     0.2353
#   P2 rel_laser_w     0.2636                    P3 group_delay   0.2307
#   P4 group_delay_w   0.2622                    B2 raw_phase     0.2176  <- worst
#
# Three suspected causes, and what turned out to be true of each:
#
#   1. "we normalized phase to std 1 too"  -- NOT WHAT HAPPENED. normalize_fft runs at
#      dataset.py:349, BEFORE the phase concat at :357, so 'std' divided the MAGNITUDE only
#      and phase was appended raw. All nine v2 runs used plain 'std'; normalize_token is a
#      no-op without a '+token-*' suffix. Measured on synthetic input:
#          std  ->  mag std=1.0000   phase std=0.0734   (phase in [-0.909, 0.992])
#      So phase entered ~14x QUIETER than magnitude. The bug is the opposite of the one
#      suspected: phase was drowned out, not over-normalized. Nothing to fix here, but it
#      reframes the rest -- see the phase-weight note at the bottom.
#
#   2. "we fused mag and phase in the first layer's weights" -- TRUE, and now fixed.
#      conv_block(n_channels, 32, ...) sums across all input channels, so magnitude and
#      phase were linearly mixed at layer 1, before any nonlinearity. --encoder two-stream
#      gives each its own full-width 32->256 frequency stack and fuses after all four conv
#      layers, at 1 freq bin per laser instead of 1248. 21.26M params vs 19.86M (+7%).
#
#   3. "we never tested raw discontinuous phase" -- TRUE. --phase-arm raw_phase is ALREADY
#      cos/sin (torch_raw_phase -> _phasor_cos_sin), so v2's B2 was cos/sin, not raw. The
#      raw angle lives on a different axis: --signal-mode mag_phase = [|Z|, angle(Z)] in
#      radians, with the +pi/-pi wraparound that cos/sin exists to remove. B2 below is the
#      first measurement of it.
#
# SCOPE: the full grid is 9 arms (mag; +raw phase; +cos/sin raw; then rel_laser, group_delay,
# both, each weighted and not). That is too many to read at once, so this ladder runs only
# rel_laser and group_delay -- enough to answer "did the three fixes help?" before spending
# GPU time on `both`, which was the worst gauge family in v2 anyway (P5 0.2269, P6 0.2428).
#
# ***** what each comparison buys *****
#
#   A1 - v2 B1 (0.2823)   two-stream's effect with NO phase present. Should be a wash; if it
#                         is not, the +7% params alone move the number and every other
#                         comparison here needs that offset subtracted.
#   B2 - A1               raw discontinuous phase. The never-tested baseline.
#   B3 - B2               what cos/sin buys over the raw angle (the wraparound fix).
#   G* - B3               what each GAUGE buys over ungauged cos/sin.
#   G1 - v2 P1 (0.2353)   the headline: same gauge, same data, only fusion changed.
#   G2 - v2 P2 (0.2636)   same, for the best v2 arm.
#
# ***** running it *****
#
# 7 runs, sequential on the single GPU, ~35 min each (~4h). Under tmux:
#   tmux new -s phase3
#   ./scripts/phase_v3_fixes.sh 2>&1 | tee runs/phase_v3_fixes.log

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the ladder

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Same lock as the other phase scripts: they share run names, checkpoints and the one GPU.
exec 9>/tmp/phase_ablation.sh.lock
flock -n 9 || { echo "another phase script is already running; exiting" >&2; exit 1; }

TAG=v3
GROUP=phase-v3-fixes-$TAG

# Copied from phase_ablation.sh's COMMON so v2 and v3 numbers are directly comparable, plus:
#   --encoder two-stream  fix #2. Separate frequency stacks for magnitude and phase.
#   --num-workers 2       P6 was OOM-killed by the host at 82.6% RAM with 4 workers.
# --signal-mode is NOT here: B2 varies it to reach the raw-angle path.
COMMON="--split gastronorm --augment-mask 0 --augment-fft 0 \
        --loss-fn mse --model boombox --d-model 512 --batch-size 256 \
        --laser-dropout 0 --freq-dropout 0 --max-duration 1000ep \
        --num-workers 2 --encoder two-stream --wandb-group $GROUP"

#***** baselines *****

# A1: magnitude only. THE reference. With n_channels=2 the two-stream encoder has no phase
# stream and degenerates to exactly the v2 single-stream widths (19.86M, verified), so this
# is also a third independent sample of the no-phase config -- v2 had 0.2823 and 0.2716, and
# a third point is what turns a 2-sample noise guess into a real estimate.
python src/run.py $COMMON --signal-mode magnitude --run-name "v3-a1-mag-$TAG"

# B2: magnitude + RAW ANGLE, [|Z|, angle(Z)] in radians. Fix #3. Discontinuous at +-pi.
# Routed through extract_signal, so unlike --phase-arm this phase IS std-normalized with the
# magnitude -- unavoidable on this axis, and the reason B3 is a separate arm.
python src/run.py $COMMON --signal-mode mag_phase --run-name "v3-b2-mag-rawangle-$TAG"

# B3: magnitude + cos/sin, UNGAUGED. Same extract_signal path as B2, so (B3 - B2) isolates
# the wraparound fix with normalization held constant. This is the honest counterpart to
# v2's B2 (0.2176), which was cos/sin but single-stream.
python src/run.py $COMMON --signal-mode mag_trig_phase --run-name "v3-b3-mag-cossin-$TAG"

#***** gauges: rel_laser and group_delay only *****
# `both` is deliberately excluded: it was the worst family in v2 (P5 0.2269, P6 0.2428) and
# doubles the phase width. Run it only if a gauge below actually clears A1.

python src/run.py $COMMON --signal-mode magnitude --phase-arm rel_laser \
    --run-name "v3-g1-rel-laser-$TAG"
python src/run.py $COMMON --signal-mode magnitude --phase-arm rel_laser_w \
    --run-name "v3-g2-rel-laser-w-$TAG"
python src/run.py $COMMON --signal-mode magnitude --phase-arm group_delay \
    --run-name "v3-g3-group-delay-$TAG"
python src/run.py $COMMON --signal-mode magnitude --phase-arm group_delay_w \
    --run-name "v3-g4-group-delay-w-$TAG"

# NOTE on phase_weight, left at its 1.0 default in every arm above so this ladder varies only
# fusion + encoding: because of finding #1, pw=1.0 is NOT a balanced mix -- phase enters at
# std 0.0734 against magnitude's 1.0000. If the gauged arms here land near A1 rather than
# above it, the next move is to turn phase UP (pw 4.0 puts it near 0.29, pw 14.0 near parity),
# not down. scripts/phase_encoding_ladder.sh has those arms.
