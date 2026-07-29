#!/usr/bin/env bash
# Augmentation ablation: full 3x3 grid over --augment-mask x --augment-fft.
# Everything else is pinned to the "old-config" baseline (lr 1e-4, constant LR,
# batch 128) so the only thing moving between runs is the augmentation strength.
# Runs sequentially on the single GPU. Launch under tmux:
#   tmux new -s aug
#   ./scripts/sweep_augment.sh 2>&1 | tee runs/sweep_augment.log
# Detach with ctrl-b d; reattach with `tmux attach -t aug`.

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the sweep

cd "$(dirname "$0")/.."
export PYTHONPATH=.

MDS_DIR="experiments/experiment-25"
DURATION="2000ep"
GROUP="augment-ablation"
COMMON=(--mds-dir "$MDS_DIR" --max-duration "$DURATION"
        --lr 1e-4 --scheduler constant --batch-size 128
        --wandb-group "$GROUP")

# 0 -> "0", 0.5 -> "05", 1 -> "1", so names stay filesystem/wandb friendly
slug () { echo "${1/./}"; }

run () {  # run <name> <extra args...>
  local name="$1"; shift
  if [ -d "runs/$name" ]; then echo "SKIP $name (runs/$name exists)"; return 0; fi
  echo "=== $(date +%H:%M:%S)  START $name ==="
  python src/run.py "${COMMON[@]}" --run-name "$name" "$@"
  echo "=== $(date +%H:%M:%S)  DONE $name (exit $?) ==="
}

for mask in 0 0.5 1; do
  for fft in 0 0.5 1; do
    run "aug-mask$(slug "$mask")-fft$(slug "$fft")" --augment-mask "$mask" --augment-fft "$fft"
  done
done

echo "ALL DONE $(date +%H:%M:%S)"
