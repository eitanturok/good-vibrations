#!/usr/bin/env bash
# src9 transformer2: deep4-v1 with 8 heads x 16 dims (same width 128, same params) -> does the number of heads (distinct weighted averages) matter? runs after the deep queue
cd /home/ethantu/workspace/good-vibrations
while pgrep -f transformer2_deep.sh >/dev/null; do sleep 60; done
PY=.venv/bin/python
T2="$PY src9/run.py --model transformer2"
BOTH="--freq-cls 1 --band-v 1 --band-proj 1 --laser-cls 1 --laser-v 1 --laser-proj 1"

$T2 $BOTH --freq-layers 4 --laser-layers 4 --heads 8 --microbatch 128 --run-name deep4-h8-v1 2>&1 | tee runs/deep4-h8-v1.log                        # vs deep4-v1
