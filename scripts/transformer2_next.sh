#!/usr/bin/env bash
# src9 transformer2 queue (reordered 2026-10-30): Arch A first (per-band W_b + per-laser W_l readout into a 256-d latent u), then the deep4 follow-ups
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
T2="$PY src9/run.py --model transformer2"
BOTH="--freq-cls 1 --band-v 1 --band-proj 1 --laser-cls 1 --laser-v 1 --laser-proj 1"

$T2 --readout archA --latent-dim 256 --run-name archA-v1 2>&1 | tee runs/archA-v1.log                                                              # vs s9-t2-ref-vase-200
$T2 $BOTH --freq-layers 4 --laser-layers 4 --rope 1 --microbatch 128 --run-name deep4-rope-v1 2>&1 | tee runs/deep4-rope-v1.log                    # vs deep4-v1
$T2 $BOTH --freq-layers 4 --laser-layers 4 --heads 8 --microbatch 128 --run-name deep4-h8-v1 2>&1 | tee runs/deep4-h8-v1.log                        # vs deep4-v1
echo "transformer2 next queue done"
