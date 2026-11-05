#!/usr/bin/env bash
# transformer4 res decoder + cross-attention onto the 100 laser tokens after the 8x8 and 16x16 doublings (arch/transformer4.py XAttn):
#   decxf-v1 = --decoder res-xattn-freq (FreqEncoder tokens), decxl-v1 = --decoder res-xattn-laser (tokens after the 8 laser blocks). 300 ep, eager, wandb.
# compare: decres-v1 (isolates the cross-attention), t4-nowmask-v1 (ID .678 / OOD .383).
# waits for "dec done" (dec_q.sh) + "decoom done" (dec_oom_redo.sh) + no src9 run.py. microbatch 64 if decres-v1 OOMed (GN/LN only: = mb128), else the default 128.
#   tmux new-window -t train -n xattn "bash scripts/xattn_q.sh; exec zsh"     log: runs/xattn_q.log
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
export PYTHONUTF8=1
{
echo "xattn: waiting $(date)"
until grep -q "dec done" runs/dec_q.log 2>/dev/null && grep -q "decoom done" runs/dec_oom_redo.log 2>/dev/null; do sleep 60; done
for rd in "decxf-v1 res-xattn-freq" "decxl-v1 res-xattn-laser"; do
  set -- $rd
  while pgrep -f "^[^ ]*python[0-9.]* src9/run.py" > /dev/null; do sleep 60; done
  if grep -q "CUDA out of memory\|OutOfMemoryError" runs/decres-v1.log 2>/dev/null; then MB=(--microbatch 64); else MB=(); fi
  echo "xattn: $1 --decoder $2 microbatch ${MB[1]:-128 (default)} $(date)"
  $PY src9/run.py --model transformer4 --decoder $2 "${MB[@]}" --run-name $1 2>&1 | tee runs/$1.log
done
echo "xattn done $(date)"
} 2>&1 | tee -a runs/xattn_q.log
