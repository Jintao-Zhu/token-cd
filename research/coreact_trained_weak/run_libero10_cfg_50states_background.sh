#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/docker/dev_zjt/data/code
ARTIFACT="$ROOT/artifacts/coreact_trained_weak_libero10_cfg_50states_v1_20260812_230341"
PYTHON="$ROOT/task1/.conda-envs/flow-vla/bin/python"
export LIBERO_CONFIG_PATH="$ARTIFACT/libero_config"
export PYTHONPATH="$ROOT/research/worktrees/lerobot_trained_weak_clean_e40b58a/src:$ROOT/LIBERO:$ROOT"
export HF_HUB_OFFLINE=1

cd "$ROOT"
mkdir -p "$ARTIFACT/logs"
printf '{"status":"running_background","episodes_complete":%s,"episodes_planned":2000}\n' \
  "$(find "$ARTIFACT/episodes" -maxdepth 1 -name '*.json' | wc -l)" > "$ARTIFACT/status.json"

init_ids=($(seq 10 49))
pids=()

launch_partition() {
  local label="$1"
  shift
  local tasks=("$@")
  for arm in Strong_15k Weak_10k; do
    "$PYTHON" research/coreact_trained_weak/run_libero10_quality_screen.py \
      --artifact "$ARTIFACT" --arm "$arm" --task-ids "${tasks[@]}" \
      --init-state-ids "${init_ids[@]}" > "$ARTIFACT/logs/${label}_${arm}.log" 2>&1 &
    pids+=("$!")
  done
  for arm in Midpoint CFG; do
    "$PYTHON" research/coreact_trained_weak/run_libero10_exploratory_cfg.py \
      --artifact "$ARTIFACT" --arm "$arm" --task-ids "${tasks[@]}" \
      --init-state-ids "${init_ids[@]}" > "$ARTIFACT/logs/${label}_${arm}.log" 2>&1 &
    pids+=("$!")
  done
}

launch_partition tasks_0_2 0 1 2
launch_partition tasks_3_4 3 4
launch_partition tasks_5_6 5 6
launch_partition tasks_7_9 7 8 9

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

count=$(find "$ARTIFACT/episodes" -maxdepth 1 -name '*.json' | wc -l)
if [[ "$failed" -ne 0 || "$count" -ne 2000 ]]; then
  printf '{"status":"background_runner_failed","episodes_complete":%s,"episodes_planned":2000}\n' "$count" > "$ARTIFACT/status.json"
  exit 1
fi

"$PYTHON" research/coreact_trained_weak/analyze_libero10_cfg_50states.py \
  > "$ARTIFACT/logs/analysis.log" 2>&1
