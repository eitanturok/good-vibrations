#!/usr/bin/env bash
# after dec_q.sh: re-run any res decoder run that died with CUDA OOM at --microbatch 64 (res/res-coord use GroupNorm/LayerNorm only,
# so microbatch 64 x 16 accumulation = the same 1024 batch, mathematically equivalent to mb128). name -> -v2.
#   tmux new-window -t train -n decoom "bash scripts/dec_oom_redo.sh; exec zsh"     log: runs/dec_oom_redo.log
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
export PYTHONUTF8=1
{
until grep -q "dec done" runs/dec_q.log 2>/dev/null; do sleep 60; done
for kind in res:decres res-coord:decres-coord; do
  dec=${kind%%:*}; name=${kind##*:}
  if grep -q "CUDA out of memory\|OutOfMemoryError" runs/$name-v1.log 2>/dev/null; then
    while pgrep -f "^[^ ]*python[0-9.]* src9/run.py" > /dev/null; do sleep 60; done
    echo "decoom: $name-v1 OOMed, rerunning as $name-v2 at mb64 $(date)"
    $PY src9/run.py --model transformer4 --decoder $dec --microbatch 64 --run-name $name-v2 2>&1 | tee runs/$name-v2.log
  else echo "decoom: $name-v1 no OOM, skip $(date)"; fi
done
echo "decoom done $(date)"
} 2>&1 | tee -a runs/dec_oom_redo.log
