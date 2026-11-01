#!/usr/bin/env bash
# src9 transformer2 loss knobs on base4-noimom-v1 (archA latent 256, rope, no box emb, box_z norm, 4+4 layers, no outline, box + n_objects aux, bce-dice):
# + global log-area loss on the mask, more dice weight, then box-free log_z norm
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
T2="$PY src9/run.py --model transformer2"
BASE="--readout archA --latent-dim 256 --pos rope --box-emb 0 --norm box_z --freq-layers 4 --laser-layers 4 --outline 0 --aux box_nobj --loss bce_dice --microbatch 128"

$T2 $BASE --area-w 0.1 --run-name base-area-v1 2>&1 | tee runs/base-area-v1.log                                                              # vs base4-noimom-v1
$T2 $BASE --dice-w 0.75 --run-name base-dice75-v1 2>&1 | tee runs/base-dice75-v1.log                                                          # vs base4-noimom-v1
$T2 $BASE --norm log_z --run-name base-logz-v1 2>&1 | tee runs/base-logz-v1.log                                                               # vs base4-noimom-v1, box-free
echo "transformer2 loss queue done"
$T2 $BASE --tversky 1 --run-name base-tversky-v1 2>&1 | tee runs/base-tversky-v1.log                                                         # vs base4-noimom-v1
$T2 $BASE --ms-dice 1 --run-name base-msdice-v1 2>&1 | tee runs/base-msdice-v1.log                                                            # vs base4-noimom-v1
$T2 $BASE --tversky 1 --ms-dice 1 --run-name base-tversky-ms-v1 2>&1 | tee runs/base-tversky-ms-v1.log                                        # vs base-tversky-v1, base-msdice-v1
echo "transformer2 loss queue (tversky / ms-dice) done"
$T2 $BASE --norm empty5 --run-name base-empty5-v1 2>&1 | tee runs/base-empty5-v1.log                                                             # vs base4-noimom-v1 (box), 5 empty scans per box
$T2 $BASE --norm log_empty5 --run-name base-logempty5-v1 2>&1 | tee runs/base-logempty5-v1.log                                                    # vs base-empty5-v1, base-logz-v1
$T2 $BASE --norm log_empty5 --tversky 1 --ms-dice 1 --run-name base-combo-v1 2>&1 | tee runs/base-combo-v1.log                                     # best guesses combined: log 5-empty norm + tversky + multi-scale
$T2 $BASE --edge-w 4 --run-name base-edge-v1 2>&1 | tee runs/base-edge-v1.log                                                                    # vs base4-noimom-v1: bce weight 5x on the true outline pixels
