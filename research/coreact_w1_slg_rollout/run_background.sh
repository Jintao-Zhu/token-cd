#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/docker/dev_zjt/data/code
ARTIFACT="$1"
SHARDS="${2:-4}"
PYTHON="$ROOT/task1/.conda-envs/flow-vla/bin/python"
export LIBERO_CONFIG_PATH="$ARTIFACT/libero_config"
export PYTHONPATH="$ROOT/research/worktrees/lerobot_trained_weak_clean_e40b58a/src:$ROOT/LIBERO:$ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl
cd "$ROOT"
mkdir -p "$ARTIFACT/logs" "$ARTIFACT/libero_config"
count=$(find "$ARTIFACT/episodes" -maxdepth 1 -name '*.json' | wc -l)
printf '{"status":"running_background","episodes_complete":%s,"episodes_planned":2000,"shards":%s}\n' "$count" "$SHARDS" > "$ARTIFACT/status.json"
pids=()
for ((shard=0; shard<SHARDS; shard++)); do
  "$PYTHON" -m research.coreact_w1_slg_rollout.run \
    --workspace "$ROOT" --artifact "$ARTIFACT" --shard-count "$SHARDS" --shard-index "$shard" \
    > "$ARTIFACT/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
count=$(find "$ARTIFACT/episodes" -maxdepth 1 -name '*.json' | wc -l)
invalid=$(find "$ARTIFACT/invalid_pairs" -maxdepth 1 -name '*.json' | wc -l)
if [[ "$failed" -ne 0 || "$count" -ne 2000 || "$invalid" -ne 0 ]]; then
  printf '{"status":"rollout_failed","episodes_complete":%s,"episodes_planned":2000,"invalid_pairs":%s}\n' "$count" "$invalid" > "$ARTIFACT/status.json"
  exit 1
fi
"$PYTHON" -m research.coreact_w1_slg_rollout.analyze --artifact "$ARTIFACT" > "$ARTIFACT/logs/analysis.log" 2>&1
