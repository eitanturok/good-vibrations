#!/usr/bin/env bash
# Preprocessing sweep: log signal + the normalization modes added for it.
# Everything else is pinned so the only thing moving between runs is
# --signal-mode / --normalize-mode / --subtract-speaker-mean.
#
# Motivation (notebooks/65_log_signal_ordering.ipynb): speaker identity explains
# 82-84% of the variance under every transform tried, so the modes that attack it
# (2 and 4) are the ones to watch.
#
# Output streams live to the terminal and is also saved per-run under runs/<name>/sweep.log.
#
# EXECUTE it, don't `source` it -- sourcing runs this in your interactive shell, where `set -u`
# below then makes every unset variable your prompt theme touches an error.
#   ./scripts/sweep_normalize.sh
# Runs sequentially on the single GPU, so it's long. Under tmux to survive disconnects:
#   tmux new -s norm
#   ./scripts/sweep_normalize.sh
# Detach with ctrl-b d; reattach with `tmux attach -t norm`.

if [ "${BASH_SOURCE[0]}" != "${0}" ] || [ -n "${ZSH_EVAL_CONTEXT:-}" ]; then
  echo "Run this script, don't source it:  ./scripts/sweep_normalize.sh" >&2
  return 1 2>/dev/null || exit 1
fi

set -u  # deliberately NOT -e: one diverging run should not kill the rest of the sweep

cd "$(dirname "$0")/.."
export PYTHONPATH=src:.  # model/* lives in src/, utils/* at the repo root
export PYTHONUNBUFFERED=1  # stream python's output live instead of in 4KB blocks

DATA_DIR="experiments/experiment-25"
GROUP="normalize-sweep"
COMMON=(--data-dir "$DATA_DIR" --split exp25
        --augment-mask 0 --augment-fft 0 --out-h 20 --out-w 40
        --batch-size 128 --wandb-group "$GROUP")  # --max-duration left at run.py's 2000ep default

run () {  # run <name> <extra args...>
  local name="$1"; shift
  # run.py sets autoresume when --run-name is given, so an existing dir would resume,
  # not restart -- skip it instead and let the user delete it deliberately.
  if [ -d "runs/$name" ]; then echo "SKIP $name (runs/$name exists)"; return 0; fi
  echo "=== $(date +%H:%M:%S)  START $name ($((++i))/$TOTAL) ==="
  mkdir -p "runs/$name"
  # tee streams to the terminal and the log at once; PIPESTATUS[0] is python's exit code, not tee's
  python src/run.py "${COMMON[@]}" --run-name "$name" "$@" 2>&1 | tee "runs/$name/sweep.log"
  echo "=== $(date +%H:%M:%S)  DONE $name (exit ${PIPESTATUS[0]}) ==="
}

i=0
TOTAL=5

# 0: current pipeline, the thing everything else has to beat
run norm-0-baseline-1 --signal-mode magnitude
# 1: does the log help on its own
run norm-1-logz-1     --signal-mode log_magnitude --normalize-mode z
# 2: log makes the speaker mean the right inverse (subtracting logs divides out the gain)
run norm-2-logz-spk-1 --signal-mode log_magnitude --normalize-mode z --subtract-speaker-mean
# 3: kill per-laser sensitivity differences
run norm-3-laserz-1   --signal-mode log_magnitude --normalize-mode per_laser_z
# 4: whiten every bin against train-split stats (most aggressive de-confounder)
run norm-4-binz-1     --signal-mode log_magnitude --normalize-mode per_bin_z

echo "ALL DONE $(date +%H:%M:%S)"
