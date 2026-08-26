#!/usr/bin/env bash
# Phase ENCODING + NORMALIZATION ladder (v3).
#
# The v2 ladder (scripts/phase_ablation.sh) asked "which gauge?" and got a clean answer:
# no-phase wins. All 8 arms at 1000ep, 2-cube soft-iou:
#
#   B1 no phase        0.2823   <- winner        P6 both_w        0.2428
#   (seed twin)        0.2716   <- noise = .011  P1 rel_laser     0.2353
#   P2 rel_laser_w     0.2636                    P3 group_delay   0.2307
#   P4 group_delay_w   0.2622                    B2 raw_phase     0.2176  <- worst
#
# Two facts from that ladder set up this one:
#   * The GAUGES WORK. B2 (ungauged) is worst; every gauged arm beats it (P1-B2 = +0.018,
#     P2-B2 = +0.046). The phase content is real.
#   * But phase still costs 0.019-0.065 vs no-phase, and the damage scales with CHANNEL
#     COUNT (2ch arms beat 4ch arms consistently). That is the signature of interference at
#     fusion, not of phase being uninformative.
#
# v2 varied the gauge and held encoding+normalization fixed. This ladder does the opposite:
# it holds the gauge at the v2 winner (rel_laser_w) and varies HOW phase is encoded and
# normalized. Everything else is copied from the v2 COMMON so the two ladders compose.
#
# ***** the encoding axis: why E2 and E3 are different arms *****
#
# --phase-arm raw_phase is ALREADY cos/sin (normalizations.torch_raw_phase -> _phasor_cos_sin).
# The raw ANGLE lives on a different axis entirely, --signal-mode mag_phase, which is
# [|Z|, angle(Z)] in radians (dataset.extract_signal:218).
#
# The difference is the WRAPAROUND. angle(Z) jumps from +pi to -pi at the branch cut, so two
# physically identical phases land at opposite ends of the input range and a conv must learn
# that discontinuity. cos/sin is the standard fix. v2 never tested it: B2 was cos/sin, and
# nothing measured the raw angle. E2 vs E3 measures exactly that, and the prediction is
# E3 > E2 -- if it fails, the wraparound is not what is hurting.
#
# CONFOUND, stated because it cannot be removed on this axis: mag_phase/mag_trig_phase route
# through extract_signal, so normalize_fft std-normalizes the phase alongside the magnitude.
# --phase-arm deliberately does NOT (process_vibration:350, "already bounded in [-1,1] by
# construction, so std-normalizing it would distort the circular geometry"). So E2 vs E3
# differs in BOTH encoding and normalization. E3b is the matched control that separates them:
# it is cos/sin via the same extract_signal path as E2, so (E3b - E2) is encoding alone and
# (E3 - E3b) is the normalization treatment alone.
#
# ***** the normalization axis *****
#
# MEASURED FIRST, so this axis is not built on an assumption. Under plain 'std' -- what ALL
# NINE v2 runs used -- normalize_fft runs at dataset.py:349, BEFORE the phase concat at :357.
# So std divided the MAGNITUDE only; phase was appended raw and never normalized. Confirmed:
#
#   std             mag std=1.0000   phase std=0.0734   (phase values in [-0.909, 0.992])
#   std+token-mean  mag std=1.5952   phase std=0.0968
#
# That kills the premise of the comment at dataset.py:352, which claims normalizing the
# magnitude first puts the blocks "on a common scale, so phase_weight sets a meaningful mix".
# Phase actually enters ~14x QUIETER than magnitude, because apply_phase_arm zeroes 90% of
# the bins (top_frac=0.10) and that crushes the aggregate std even though the surviving
# cos/sin values span +-1. phase_weight=1.0 was never a 50/50 mix; it was closer to 0.07.
#
# Two consequences for this ladder:
#   * N2/N3 sweep phase_weight UP, not down. v2's phase block was already faint, so turning
#     it down would only confirm that faint phase does nothing.
#   * N1 turns on the only mode that touches phase at all. normalize_token pools the channel
#     axis into the denominator on purpose (dataset.py:286) so the blocks share one scale --
#     justified there by the transformer's nn.Linear(patch_size*n_channels, d_model) embed,
#     which does NOT hold for the boombox conv, where channels stay separate to the first
#     conv_block. So it is arguably wrong for the arch every arm here uses.
#
# ***** running it *****
#
# 8 runs, sequential on the single GPU, ~35 min each. Under tmux:
#   tmux new -s phase3
#   ./scripts/phase_encoding_ladder.sh 2>&1 | tee runs/phase_encoding_ladder.log

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the ladder

cd "$(dirname "$0")/.."
export PYTHONPATH=.

# Same lock as the other phase scripts: these share run names, checkpoints and the one GPU.
exec 9>/tmp/phase_ablation.sh.lock
flock -n 9 || { echo "another phase script is already running; exiting" >&2; exit 1; }

TAG=v3
GROUP=phase-encoding-$TAG

# Copied verbatim from phase_ablation.sh's COMMON, plus --num-workers 2: P6 was OOM-killed by
# the host at 82.6% RAM with 4 workers, and the wide arms here are the same shape.
# --signal-mode is NOT in COMMON, because the encoding axis varies it.
COMMON="--split gastronorm --augment-mask 0 --augment-fft 0 \
        --loss-fn mse --model boombox --d-model 512 --batch-size 256 \
        --laser-dropout 0 --freq-dropout 0 --max-duration 1000ep \
        --num-workers 2 --wandb-group $GROUP"

#***** the encoding axis *****

# E1: magnitude only. The reference. A third independent sample of the no-phase config
# (v2 had 0.2823 and 0.2716), so it also tightens the seed-noise estimate from 2 points to 3
# -- which is what decides whether the 0.019 gap to P2 is real or noise.
python src/run.py $COMMON --signal-mode magnitude --run-name "enc-e1-mag-$TAG"

# E2: magnitude + RAW ANGLE, [|Z|, angle(Z)] in radians. Has the +pi/-pi wraparound.
python src/run.py $COMMON --signal-mode mag_phase --run-name "enc-e2-mag-angle-$TAG"

# E3: magnitude + cos/sin, gauged (rel_laser_w, the v2 winner) and NOT std-normalized.
# This is the v2 P2 config, rerun here so the whole ladder shares one set of seeds.
python src/run.py $COMMON --signal-mode magnitude --phase-arm rel_laser_w \
    --run-name "enc-e3-mag-cossin-gauged-$TAG"

# E3b: magnitude + cos/sin, UNGAUGED, via extract_signal so it IS std-normalized.
# The control that splits E2 vs E3 into its two causes. Same path as E2, different encoding.
python src/run.py $COMMON --signal-mode mag_trig_phase --run-name "enc-e3b-mag-cossin-raw-$TAG"

#***** the normalization axis, all on E3's encoding *****

# N1: token-level normalization ON. MEASURED, not assumed: under plain 'std' (what ALL NINE
# v2 runs used) normalize_fft runs at dataset.py:349, BEFORE the phase concat at :357 -- so
# std divided the MAGNITUDE only and phase was appended raw. Verified on synthetic input:
#   std             mag std=1.0000   phase std=0.0734  (phase in [-0.909, 0.992])
#   std+token-mean  mag std=1.5952   phase std=0.0968
# So v2 never std-normalized phase, and this arm turns on the one mode that WOULD touch it.
# normalize_token pools the channel axis into the denominator on purpose (dataset.py:286) so
# magnitude and phase share one scale. Its stated reason is the transformer's
# nn.Linear(patch_size*n_channels, d_model) embed -- which does NOT hold for the boombox conv,
# where channels stay separate. Predicted to hurt; if it does not, the pooling is harmless here.
python src/run.py $COMMON --signal-mode magnitude --phase-arm rel_laser_w \
    --normalize-mode "std+token-mean" --run-name "enc-n1-token-mean-$TAG"

# N2/N3: phase_weight, pinned at 1.0 for all 8 arms of v2 and never swept. The header comment
# at dataset.py:352 claims normalizing the magnitude first puts the two blocks "on a common
# scale, so phase_weight sets a meaningful mix". MEASUREMENT SAYS OTHERWISE: at pw=1.0 the
# phase block enters at std 0.0734 against the magnitude's 1.0000 -- a ~14x deficit, because
# apply_phase_arm zeroes 90% of the bins (top_frac=0.10) and that crushes the aggregate std
# even though the surviving cos/sin values span +-1.
#
# So phase was ALREADY heavily attenuated in v2, and sweeping DOWN would only make a faint
# signal fainter. These arms sweep UP instead: 4.0 puts phase near parity with magnitude
# (0.0734*4 ~ 0.29) and 14.0 puts it at rough std parity. If v2's damage came from phase
# being too loud, both should be worse than E3; if it came from phase being drowned out,
# one of them should beat it.
python src/run.py $COMMON --signal-mode magnitude --phase-arm rel_laser_w --phase-weight 4.0 \
    --run-name "enc-n2-pw4-$TAG"
python src/run.py $COMMON --signal-mode magnitude --phase-arm rel_laser_w --phase-weight 14.0 \
    --run-name "enc-n3-pw14-$TAG"

# N4: log-magnitude instead of magnitude, phase unchanged. From the preprocessing findings,
# log-mag > mag on the magnitude block alone. Worth one arm to check that the phase result
# is not specific to linear magnitude -- if E3-E1 and N4-(logmag baseline) disagree, the
# phase conclusion is entangled with the magnitude domain.
python src/run.py $COMMON --signal-mode log_magnitude --phase-arm rel_laser_w \
    --run-name "enc-n4-logmag-cossin-$TAG"
