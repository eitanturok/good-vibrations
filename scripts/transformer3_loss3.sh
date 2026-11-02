#!/usr/bin/env bash
# src9 transformer3 vs std-samplez-v1 (ID .656 / OOD .367 at ep200): two loss knobs and a big model with 20% token dropout, 200 ep each
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
T3="$PY src9/run.py --model transformer3 --microbatch 128"

$T3 --objrecall-w 0.5 --run-name objrecall-v1 2>&1 | tee runs/objrecall-v1.log      # + 0.5 x mean over true objects of 1 - (prob mass on it / its area)
$T3 --boundary-w 0.5 --run-name boundary-v1 2>&1 | tee runs/boundary-v1.log          # + 0.5 x Kervadec boundary loss mean(p * signed dist), ramped over ep 0-50
$T3 --d 256 --heads 8 --laser-layers 8 --latent-dim 512 --laser-drop 0.2 --band-drop 0.2 --run-name big-drop20-v1 2>&1 | tee runs/big-drop20-v1.log  # big-2k-v1's model, 20% laser + band dropout
echo "transformer3 queue (loss3) done"
