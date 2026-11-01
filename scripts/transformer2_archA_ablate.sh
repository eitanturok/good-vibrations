#!/usr/bin/env bash
# src9 transformer2 Arch A ablations (latent 256), each vs archA-v1: input norm (per-sample std / z, per-(f,c) z across lasers, no box stats), no box embedding,
# rope only (no additive band/laser embeddings), then a wider d=256 / latent 1024 model. runs after the next queue
while pgrep -f transformer2_next.sh >/dev/null; do sleep 60; done
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
T2="$PY src9/run.py --model transformer2"
A="--readout archA --latent-dim 256"

$T2 $A --norm sample_std --run-name archA-samplestd-v1 2>&1 | tee runs/archA-samplestd-v1.log                                                  # vs archA-v1
$T2 $A --norm sample_z --run-name archA-samplez-v1 2>&1 | tee runs/archA-samplez-v1.log                                                        # vs archA-v1
$T2 $A --norm laser_z --run-name archA-laserz-v1 2>&1 | tee runs/archA-laserz-v1.log                                                           # vs archA-v1
$T2 $A --box-emb 0 --run-name archA-nobox-v1 2>&1 | tee runs/archA-nobox-v1.log                                                                # vs archA-v1
$T2 $A --pos rope --run-name archA-ropeonly-v1 2>&1 | tee runs/archA-ropeonly-v1.log                                                           # vs archA-v1
$T2 $A --d 256 --latent-dim 1024 --microbatch 128 --run-name archA-d256-l1024-v1 2>&1 | tee runs/archA-d256-l1024-v1.log                       # vs archA-v1, ~52M params
echo "transformer2 archA ablate queue done"
