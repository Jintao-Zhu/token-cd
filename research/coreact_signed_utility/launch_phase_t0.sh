#!/usr/bin/env bash
set -euo pipefail

workspace="$1"
artifact="$2"
reference="$3"
means="$4"
python_bin="$workspace/task1/.conda-envs/flow-vla/bin/python"
launcher="${artifact}_launcher"
mkdir -p "$launcher"

export PYTHONPATH="$workspace/lerobot/src:$workspace/LIBERO:$workspace"
export HF_HOME="$workspace/task1/.hf-cache"
export TRANSFORMERS_CACHE="$workspace/task1/.hf-cache/hub"
export MUJOCO_GL=egl

"$python_bin" -u "$workspace/research/coreact_signed_utility/prepare_phase_t0.py" \
  --workspace "$workspace" \
  --reference-artifact "$reference" \
  --means "$means" \
  --output "$artifact" > "$launcher/prepare.log" 2>&1

decision="$($python_bin -c "import json; print(json.load(open('$artifact/decision.json'))['decision'])")"
if [[ "$decision" != "PHASE_T0_CANDIDATES_LOCKED_READY_FOR_CAUSAL_DRYRUN" ]]; then
  printf '%s\n' "$decision" > "$launcher/stopped_at_candidate_gate"
  exit 1
fi

"$python_bin" -u "$workspace/research/coreact_signed_utility/run_phase_t0.py" \
  --workspace "$workspace" \
  --artifact "$artifact" \
  --reference-artifact "$reference" \
  --means "$means" \
  --max-units 1 \
  --dry-run > "$launcher/dryrun.log" 2>&1

decision="$($python_bin -c "import json; print(json.load(open('$artifact/decision.json'))['decision'])")"
if [[ "$decision" != "PHASE_T0_CAUSAL_DRYRUN_PASS_READY_FOR_ROLLOUT" ]]; then
  printf '%s\n' "$decision" > "$launcher/stopped_at_causal_gate"
  exit 1
fi

pids=()
for worker in 0 1; do
  "$python_bin" -u "$workspace/research/coreact_signed_utility/run_phase_t0.py" \
    --workspace "$workspace" \
    --artifact "$artifact" \
    --reference-artifact "$reference" \
    --means "$means" \
    --unit-start "$worker" \
    --unit-stride 2 > "$launcher/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$launcher/worker.pids"

failure=0
for pid in "${pids[@]}"; do
  wait "$pid" || failure=1
done
if [[ "$failure" -ne 0 ]]; then
  printf 'worker failure\n' > "$launcher/rollout.failed"
  exit 1
fi
printf 'workers complete\n' > "$launcher/rollout.complete"
