#!/usr/bin/env bash
# Round 1 of the attn-decoder LR sweep: find the LR, then compare arms in round 2.
# Runs sequentially on the single GPU. Launch under tmux:
#   tmux new -s sweep
#   ./scripts/sweep_attn_lr.sh 2>&1 | tee runs/sweep_attn_lr.log
# Detach with ctrl-b d; reattach with `tmux attach -t sweep`.

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the sweep

cd "$(dirname "$0")/.."

MDS_DIR="experiments/experiment-25"
DURATION="500ep"        # long enough to rank LRs; full 2000ep only for the winner
WARMUP="25ep"           # 5% of DURATION
GROUP="attn-lr-sweep"
COMMON=(--mds-dir "$MDS_DIR" --augment-mask 0 --augment-fft 0
        --max-duration "$DURATION" --scheduler cosine-warmup --t-warmup "$WARMUP"
        --wandb-group "$GROUP")

run () {  # run <name> <extra args...>
  local name="$1"; shift
  if [ -d "runs/$name" ]; then echo "SKIP $name (runs/$name exists)"; return 0; fi
  echo "=== $(date +%H:%M:%S)  START $name ==="
  python src/run.py "${COMMON[@]}" --run-name "$name" "$@"
  echo "=== $(date +%H:%M:%S)  DONE $name (exit $?) ==="
}

for lr in 3e-5 1e-4 3e-4 1e-3; do
  run "attn-lr${lr}" --decoder attn --lr "$lr"
done

# same grid on the mlp arm, so the comparison is tuned-vs-tuned
for lr in 3e-5 1e-4 3e-4 1e-3; do
  run "mlp-lr${lr}" --decoder mlp --lr "$lr"
done

echo "ALL DONE $(date +%H:%M:%S)"
