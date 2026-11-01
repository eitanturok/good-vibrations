#!/usr/bin/env bash
# src9 transformer2: per-position value matrices in every attention layer. 200 ep each (the transformer2 default), one after the other, microbatch "auto" (the default).
# band_v = W_V^b, one d x d value matrix per freq band (+ [cls]) | laser_v = W_V^l, one per laser (+ [cls]). the untested side stays at ref (no cls, no proj)
# then the same with the per-position input projections (band_proj / laser_proj), and proj alone without cls
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
T2="$PY src9/run.py --model transformer2"

$T2 --freq-cls 1 --band-v 1 --run-name freqv-v1 2>&1 | tee runs/freqv-v1.log                                                                      # vs s9-t2-freqcls-vase-200
$T2 --laser-cls 1 --laser-v 1 --run-name laserv-v1 2>&1 | tee runs/laserv-v1.log                                                                  # vs s9-t2-lasercls-vase-200
$T2 --freq-cls 1 --band-v 1 --laser-cls 1 --laser-v 1 --run-name bothv-v1 2>&1 | tee runs/bothv-v1.log                                            # vs s9-t2-cls-vase-200
$T2 --freq-cls 1 --band-v 1 --band-proj 1 --run-name freqv-proj-v1 2>&1 | tee runs/freqv-proj-v1.log                                              # vs freqcls-bandproj-v1
$T2 --laser-cls 1 --laser-v 1 --laser-proj 1 --run-name laserv-proj-v1 2>&1 | tee runs/laserv-proj-v1.log                                         # vs lasercls-laserproj-v1
$T2 --freq-cls 1 --band-v 1 --band-proj 1 --laser-cls 1 --laser-v 1 --laser-proj 1 --run-name bothv-proj-v1 2>&1 | tee runs/bothv-proj-v1.log    # vs cls-bothproj-v1
$T2 --band-proj 1 --laser-proj 1 --run-name proj-v1 2>&1 | tee runs/proj-v1.log                                                                   # vs s9-t2-ref-vase-200
echo "transformer2 v queue done"
