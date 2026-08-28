#!/usr/bin/env bash
set -euo pipefail

root=/data/docker/dev_zjt/data/code
artifact="${1:-$root/artifacts/coreact_local_success_u0_v1_20260817_152043}"
python="$root/task1/.conda-envs/flow-vla/bin/python"

export LIBERO_CONFIG_PATH="$artifact/libero_config"
export PYTHONPATH="$root/research/worktrees/lerobot_trained_weak_clean_e40b58a/src:$root/LIBERO:$root"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl

while pgrep -f '[r]esearch.coreact_local_success.capture_snapshots' >/dev/null; do
  sleep 30
done

"$python" -m research.coreact_local_success.finalize_snapshots --artifact "$artifact"
find "$artifact/neighbors" -type f -delete
find "$artifact/neighbor_parts" -type f -delete
rm -f "$artifact/u0_manifest.jsonl" "$artifact/u0_manifest.sha256"

task_sets=('0 1' '2' '3' '4' '5' '6' '7' '8 9')
pids=()
for index in $(seq 0 7); do
  # shellcheck disable=SC2086
  "$python" -m research.coreact_local_success.build_neighbors \
    --workspace "$root" --artifact "$artifact" \
    --task-ids ${task_sets[$index]} \
    >"$artifact/logs/neighbors_vector_$index.log" 2>&1 &
  pids+=("$!")
done
bad=0
for pid in "${pids[@]}"; do
  wait "$pid" || bad=1
done
[[ "$bad" -eq 0 ]]
[[ "$(find "$artifact/neighbors" -name '*.npz' | wc -l)" -eq 100 ]]

"$python" -m research.coreact_local_success.prepare_u0 --artifact "$artifact"
"$python" -m research.coreact_local_success.run_u0 \
  --workspace "$root" --artifact "$artifact" --max-units 1 \
  >"$artifact/logs/u0_dryrun.log" 2>&1
[[ "$(find "$artifact/u0" -name '*.json' | wc -l)" -eq 5 ]]

pids=()
for index in $(seq 0 7); do
  "$python" -m research.coreact_local_success.run_u0 \
    --workspace "$root" --artifact "$artifact" \
    --shard-count 8 --shard-index "$index" \
    >"$artifact/logs/u0_$index.log" 2>&1 &
  pids+=("$!")
done
bad=0
for pid in "${pids[@]}"; do
  wait "$pid" || bad=1
done
[[ "$bad" -eq 0 ]]

"$python" -m research.coreact_local_success.analyze_u0 \
  --artifact "$artifact" >"$artifact/logs/u0_analysis.log" 2>&1
