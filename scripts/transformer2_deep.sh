#!/usr/bin/env bash
# src9 transformer2: depth + RoPE on top of bothv-proj (cls + per-position W_V + per-position input projections, both stages). 200 ep each (the transformer2 default), one after the other.
# deep4 = 4 freq + 4 laser blocks (vs 2 + 2) | rope = rotary q/k: 1D over band index (freq stage), 2D (row half / col half) over the laser grid (laser stage), [cls] unrotated
# deep runs at microbatch 128 (4-layer activations ~1.66x of 2-layer at mb32, 256 would not fit next to nothing on 16 GB); grad accumulation only, same results
cd /home/ethantu/workspace/good-vibrations
while pgrep -f "run-name laserv-v1" >/dev/null; do sleep 60; done                                                                                  # wait for the transformer2_v.sh survivor
PY=.venv/bin/python
T2="$PY src9/run.py --model transformer2"
BOTH="--freq-cls 1 --band-v 1 --band-proj 1 --laser-cls 1 --laser-v 1 --laser-proj 1"

$T2 $BOTH --run-name bothv-proj-v1 2>&1 | tee runs/bothv-proj-v1.log                                                                              # vs cls-bothproj-v1
$T2 $BOTH --freq-layers 4 --laser-layers 4 --microbatch 128 --run-name deep4-v1 2>&1 | tee runs/deep4-v1.log                                       # vs bothv-proj-v1
$T2 $BOTH --freq-layers 4 --laser-layers 4 --rope 1 --microbatch 128 --run-name deep4-rope-v1 2>&1 | tee runs/deep4-rope-v1.log                    # vs deep4-v1
echo "transformer2 deep queue done"
