#!/usr/bin/env bash
# src9 transformer2: per-position weights before attention, on top of both [cls] tokens. 200 ep each (the transformer2 default), one after the other.
# band_proj = W_x^(b), one embedding matrix per freq band | laser_proj = P_l, one d x d matrix per laser before the laser stage | both
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python

$PY src9/run.py --model transformer2 --freq-cls 1 --laser-cls 1 --band-proj 1 --run-name cls-bandproj-v1
$PY src9/run.py --model transformer2 --freq-cls 1 --laser-cls 1 --laser-proj 1 --run-name cls-laserproj-v1
$PY src9/run.py --model transformer2 --freq-cls 1 --laser-cls 1 --band-proj 1 --laser-proj 1 --run-name cls-bothproj-v1
echo "transformer2 proj queue done"
