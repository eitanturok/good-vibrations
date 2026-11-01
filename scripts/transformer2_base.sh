#!/usr/bin/env bash
# src9 transformer2 base recipe: archA readout, latent 256, rope only, no box embedding. norm box_z = per-box stats then a per-sample z-score over (L,F,C).
# token dropout (train only, eval on full input): laser_drop 0.75 = p.25 5x5 subgrid else keep U(.25,1) of the lasers; band_drop 0.5 = p.5 keep bands 0..c-1, c ~ U{4..18}
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
T2="$PY src9/run.py --model transformer2"
BASE="--readout archA --latent-dim 256 --pos rope --box-emb 0"

$T2 $BASE --norm box_z --run-name base-boxz-v1 2>&1 | tee runs/base-boxz-v1.log                                                         # main recipe
$T2 $BASE --norm box --run-name base-v1 2>&1 | tee runs/base-v1.log                                                                     # control: no per-sample z
$T2 $BASE --norm box_z --laser-drop 0.75 --run-name base-boxz-droplaser-v1 2>&1 | tee runs/base-boxz-droplaser-v1.log                   # + laser dropout
$T2 $BASE --norm box_z --band-drop 0.5 --run-name base-boxz-dropband-v1 2>&1 | tee runs/base-boxz-dropband-v1.log                       # + band dropout
$T2 $BASE --norm box_z --laser-drop 0.75 --band-drop 0.5 --run-name base-boxz-dropboth-v1 2>&1 | tee runs/base-boxz-dropboth-v1.log     # + both
echo "transformer2 base queue done"
