#!/usr/bin/env bash
# t4-v1 = transformer4 (base300-v1's arch, no knobs), every training arg = base300-v1's. Waits for "imomv2 done" in runs/imomv2_q.log AND no src9 run.py.
#   tmux new-window -t train -n t4 "bash scripts/t4_q.sh; exec zsh"     log: runs/t4_q.log (queue), runs/t4-v1.log (run)
cd /home/ethantu/workspace/good-vibrations
export PYTHONUTF8=1
{
until grep -q "imomv2 done" runs/imomv2_q.log 2>/dev/null && ! pgrep -f "python src9/run.py" > /dev/null; do echo "t4: waiting for imomv2 done + no src9/run.py $(date)"; sleep 60; done
echo "t4: starting t4-v1 $(date)"
( until grep -q "\[metric\]\[batch=0\]: loss/train/total" runs/t4-v1/logs-rank0.txt 2>/dev/null; do sleep 20; done
  echo "t4: batch-0 loss t4-v1: $(grep -m1 '\[metric\]\[batch=0\]: loss/train/total' runs/t4-v1/logs-rank0.txt)  base300-v1: $(grep -m1 '\[metric\]\[batch=0\]: loss/train/total' runs/base300-v1/logs-rank0.txt)" ) &
.venv/bin/python src9/run.py --model transformer4 --run-name t4-v1 2>&1 | tee runs/t4-v1.log
echo "t4 done $(date)"
} 2>&1 | tee -a runs/t4_q.log
