#!/usr/bin/env bash
set -euo pipefail

artifact="$1"
root=/data/docker/dev_zjt/data/code
python="$root/task1/.conda-envs/flow-vla/bin/python"
export LIBERO_CONFIG_PATH="$artifact/libero_config"
export PYTHONPATH="$root/research/worktrees/lerobot_trained_weak_clean_e40b58a/src:$root/LIBERO:$root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl

while pgrep -f '[r]esearch.coreact_local_success.capture_snapshots' >/dev/null; do sleep 30; done
"$python" -m research.coreact_local_success.finalize_snapshots --artifact "$artifact"
"$python" -m research.coreact_s2_stochastic.run_s0 --workspace "$root" --artifact "$artifact" --task-ids 0 --max-states 1 >"$artifact/logs/s0_dryrun.log" 2>&1
[[ "$(find "$artifact/s0" -name '*.json' | wc -l)" -eq 1 ]]

task_sets=('0 1' '2' '3' '4' '5' '6' '7' '8 9')
pids=()
for index in $(seq 0 7); do
  # shellcheck disable=SC2086
  "$python" -m research.coreact_s2_stochastic.run_s0 --workspace "$root" --artifact "$artifact" --task-ids ${task_sets[$index]} >"$artifact/logs/s0_$index.log" 2>&1 &
  pids+=("$!")
done
bad=0
for pid in "${pids[@]}"; do wait "$pid" || bad=1; done
[[ "$bad" -eq 0 ]]
"$python" -m research.coreact_s2_stochastic.analyze_s0 --artifact "$artifact" >"$artifact/logs/s0_analysis.log" 2>&1
