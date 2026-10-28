#!/usr/bin/env bash
# src9: the src9 models on the src9 split (wedge + vase-only held-out region), one after the other
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python

$PY src9/run.py --model hybrid --run-name s9-hybrid-vase-500ep
$PY src9/run.py --model boombox --run-name s9-boombox-vase-3k
echo "src9 queue done"
