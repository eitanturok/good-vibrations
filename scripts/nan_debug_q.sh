#!/usr/bin/env bash
# VERIFICATION of the pool-m-v1 / query-m-v1 NaN fix. root cause (pool-m-v2 / query-m-v2 dumps, 2026-11-03): inductor (torch 2.11) comprehensive padding
# miscompiled the backward for d 400 / 336 (18 x d bf16 per laser not 128-byte aligned): d(embed output) written at stride 7200, read by the embed
# weight / bias grad kernels at the padded 7232 -> garbage / nan / inf embed grads on every step (illegal memory access in isolation).
# fix: src9/run.py turns comprehensive padding off for transformer3 when n_bands * d % 64 != 0 (d 256 = the default: untouched, bit-identical).
# the exact pool-m-v2 / query-m-v2 configs (seed 42 = the v1 init + data order; both dumped at step 0 microbatch 2) + --debug-nan 1, 20 ep, wandb off.
# pass = both reach ep 20 with no "debug-nan: non-finite" / "contains a NaN".     tmux train:verify     log: runs/nan_verify_q.log
cd /home/ethantu/workspace/good-vibrations
PY=.venv/bin/python
export PYTHONUTF8=1
T3="$PY src9/run.py --model transformer3 --laser-drop 0.3 --band-drop 0.3 --debug-nan 1 --seed 42 --max-duration 20ep --wandb 0"
{
for t in "pool-m-v3|--microbatch 64 --d 400 --heads 10 --readout pool" "query-m-v3|--microbatch 64 --d 336 --heads 7 --readout query"; do
    name=${t%%|*}; flags=${t#*|}
    echo "nan_verify: $name $(date)"
    $T3 $flags --run-name $name 2>&1 | tee runs/$name.log
    if grep -q "debug-nan: non-finite\|contains a NaN" runs/$name.log; then echo "nan_verify: $name FAILED (non-finite) $(date)"
    elif grep -q "\[epoch=20\]" runs/$name.log; then echo "nan_verify: $name PASSED (20 ep finite) $(date)"
    else echo "nan_verify: $name did not finish (look at the log) $(date)"; fi
done
echo "nan_verify queue done $(date)"
} 2>&1 | tee -a runs/nan_verify_q.log
