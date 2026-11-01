#!/usr/bin/env bash
# src9 transformer2 base4: archA latent 256, rope only, no box embedding, box_z norm, 4 freq + 4 laser layers, mask only (no outline), box + n_objects aux,
# bce-dice + imoment. then: no imoment, plain box norm, 10% laser/band token dropout, laser_z norm (fully box-free: no box stats, no box embedding)
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
T2="$PY src9/run.py --model transformer2"
BASE="--readout archA --latent-dim 256 --pos rope --box-emb 0 --norm box_z --freq-layers 4 --laser-layers 4 --outline 0 --aux box_nobj --loss bce_dice_imom --microbatch 128"

$T2 $BASE --run-name base4-v1 2>&1 | tee runs/base4-v1.log
$T2 $BASE --loss bce_dice --run-name base4-noimom-v1 2>&1 | tee runs/base4-noimom-v1.log                                                      # vs base4-v1
$T2 $BASE --norm box --run-name base4-box-v1 2>&1 | tee runs/base4-box-v1.log                                                                  # vs base4-v1
$T2 $BASE --laser-drop 0.1 --band-drop 0.1 --run-name base4-drop10-v1 2>&1 | tee runs/base4-drop10-v1.log                                      # vs base4-v1
$T2 $BASE --norm laser_z --run-name base4-laserz-v1 2>&1 | tee runs/base4-laserz-v1.log                                                        # vs base4-v1, box-free
echo "transformer2 base4 queue done"
