#!/usr/bin/env bash
# src9 transformer3 input normalization (default --norm sample = z over (L,bins,C) per sample, linear |X|):
# z over the lasers per (sample, bin, channel) / over the bins per (sample, laser, channel), each on linear |X| and on log |X| (eps = 1e-3 x the sample's mean)
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
T3="$PY src9/run.py --model transformer3 --microbatch 128"

$T3 --norm laser --run-name zlaser-v1 2>&1 | tee runs/zlaser-v1.log                    # z over lasers
$T3 --norm laser --log-mag 1 --run-name zlaser-log-v1 2>&1 | tee runs/zlaser-log-v1.log  # log |X|, z over lasers
$T3 --norm freq --run-name zfreq-v1 2>&1 | tee runs/zfreq-v1.log                       # z over freq bins
$T3 --norm freq --log-mag 1 --run-name zfreq-log-v1 2>&1 | tee runs/zfreq-log-v1.log     # log |X|, z over freq bins
echo "transformer3 norm queue done"
