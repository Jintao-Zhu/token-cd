#!/usr/bin/env bash
set -euo pipefail

workspace="$1"
root="$2"
artifact="$root/rollout"
reference="$root/reference_gate"
means="$workspace/artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"
python_bin="$workspace/task1/.conda-envs/flow-vla/bin/python"
workers=8

export PYTHONPATH="$workspace/lerobot/src:$workspace/LIBERO:$workspace"
export HF_HOME="$workspace/task1/.hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl

mkdir -p "$artifact/logs"
printf '{"status":"running","episodes_complete":0,"episodes_planned":1680,"aggregate_read_forbidden":true}\n' > "$artifact/status.json"
pids=()
for worker in $(seq 0 $((workers-1))); do
  "$python_bin" -u "$workspace/research/coreact_signed_utility/run_oracle_u0.py" \
    --workspace "$workspace" --artifact "$artifact" --reference "$reference" --means "$means" \
    --unit-start "$worker" --unit-stride "$workers" > "$artifact/logs/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$artifact/worker.pids"
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
count=$(find "$artifact/episodes" -maxdepth 1 -name '*.json' | wc -l)
invalid=$(find "$artifact/invalid_units" -maxdepth 1 -name '*.json' | wc -l)
if [[ "$failed" -ne 0 || "$count" -ne 1680 || "$invalid" -ne 0 ]]; then
  printf '{"status":"rollout_failed","episodes_complete":%s,"episodes_planned":1680,"invalid_units":%s}\n' "$count" "$invalid" > "$artifact/status.json"
  exit 1
fi
"$python_bin" "$workspace/research/coreact_signed_utility/analyze_oracle_u0.py" --artifact "$artifact" > "$artifact/logs/analysis.log" 2>&1
