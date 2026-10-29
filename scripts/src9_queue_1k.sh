#!/usr/bin/env bash
# src9: the three models on the src9 split (wedge + vase-only held-out region, outliers dropped), 1000 ep each, one after the other
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python

$PY src9/run.py --model hybrid      --max-duration 1000ep --run-name s9-hybrid-vase-1k
$PY src9/run.py --model boombox     --max-duration 1000ep --run-name s9-boombox-vase-1k
$PY src9/run.py --model transformer --max-duration 1000ep --run-name s9-transformer-vase-1k
echo "src9 1k queue done"
