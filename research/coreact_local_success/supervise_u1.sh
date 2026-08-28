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

while pgrep -f '[r]esearch.coreact_local_success.capture_snapshots' >/dev/null || pgrep -f '[r]esearch.coreact_local_success.build_pca' >/dev/null; do
  sleep 30
done

"$python" -m research.coreact_local_success.finalize_snapshots --artifact "$artifact"
[[ -f "$artifact/pca_basis.npz" ]]
"$python" -m research.coreact_local_success.prepare_u1_manifest --artifact "$artifact"

"$python" -m research.coreact_local_success.run_u1 \
  --workspace "$root" --artifact "$artifact" --max-units 1 \
  >"$artifact/logs/u1_dryrun.log" 2>&1
[[ "$(find "$artifact/u1" -name '*.json' | wc -l)" -eq 9 ]]

pids=()
for index in $(seq 0 7); do
  "$python" -m research.coreact_local_success.run_u1 \
    --workspace "$root" --artifact "$artifact" \
    --shard-count 8 --shard-index "$index" \
    >"$artifact/logs/u1_$index.log" 2>&1 &
  pids+=("$!")
done
bad=0
for pid in "${pids[@]}"; do
  wait "$pid" || bad=1
done
[[ "$bad" -eq 0 ]]

"$python" -m research.coreact_local_success.analyze_u1 \
  --artifact "$artifact" >"$artifact/logs/u1_analysis.log" 2>&1
