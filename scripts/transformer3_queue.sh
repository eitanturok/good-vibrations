#!/usr/bin/env bash
# src9 transformer3 knobs, one at a time on its default (= base-empty5-v1, 200 ep, microbatch 128)
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
T3="$PY src9/run.py --model transformer3 --microbatch 128"

$T3 --laser-drop 0.05 --band-drop 0.05 --run-name drop5-v1 2>&1 | tee runs/drop5-v1.log    # vs base-empty5-v1: 5% laser + band token dropout
$T3 --oversample 3 --run-name oversample3-v1 2>&1 | tee runs/oversample3-v1.log           # vs base-empty5-v1: cylinder + two-object train samples drawn 3x as often
$T3 --edge-w 4 --run-name edge4-v1 2>&1 | tee runs/edge4-v1.log                           # vs base-empty5-v1: bce weight 5x on the true outline pixels
$T3 --pos sincos --run-name sincos-v1 2>&1 | tee runs/sincos-v1.log                       # vs base-empty5-v1: learned fourier pos emb before each encoder, no rope
echo "transformer3 queue done"
$T3 --head-in u --box-head 0 --run-name headu-v1 2>&1 | tee runs/headu-v1.log                     # vs base-empty5-v1: n_objects head reads u (256) not z, no box head / loss
$T3 --head-in u --box-head 0 --up none --run-name noup-v1 2>&1 | tee runs/noup-v1.log             # vs headu-v1: no w_up, one Linear(256 -> 512x4x4) seed
$T3 --head-in u --box-head 0 --up mlp --run-name upmlp-v1 2>&1 | tee runs/upmlp-v1.log           # vs headu-v1: w_up -> GELU -> project
echo "transformer3 queue (handoff) done"
$T3 --run-name std-boxnorm-v1 2>&1 | tee runs/std-boxnorm-v1.log                                   # new standard: per-box empty5 norm, object-class aux (up none, heads on u, 5% laser/band drop)
$T3 --norm sample_z --box-head 1 --run-name std-samplez-v1 2>&1 | tee runs/std-samplez-v1.log      # box-free input: per-sample z, object + box class aux
echo "transformer3 queue (std) done"
