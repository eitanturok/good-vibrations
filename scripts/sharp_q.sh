#!/usr/bin/env bash
# sharp masks (per-pixel losses average over position uncertainty -> blur), run one after the other:
# flowmask-v1: a flow-matching UNet mask generator on the frozen base300-v1 encoder's u (+ its sigmoid mask as a prior channel), ~1.2 h
# ganloss-v1: transformer3 defaults (= base300-v1) + a PatchGAN adversarial loss on the mask, x 0.05 ramped in over epochs 50-100, ~1.1 h
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
export PYTHONUTF8=1

$PY src9/flowmask.py --run-name flowmask-v1 2>&1 | tee runs/flowmask-v1.log
$PY src9/run.py --model transformer3 --microbatch 128 --gan-w 0.05 --run-name ganloss-v1 2>&1 | tee runs/ganloss-v1.log
echo "sharp queue done"
