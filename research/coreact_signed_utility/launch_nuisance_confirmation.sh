#!/usr/bin/env bash
set -euo pipefail

workspace="$1"
root="$2"
python_bin="$workspace/task1/.conda-envs/flow-vla/bin/python"
reference="$root/reference_gate"
artifact="$root/rollout"
launcher="$root/launcher"
means="$workspace/artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt"
parent="$workspace/artifacts/coreact_state_level_matched_causal_calibration_v1_20260809_225300"
mkdir -p "$root" "$launcher"

export PYTHONPATH="$workspace/lerobot/src:$workspace/LIBERO:$workspace"
export HF_HOME="$workspace/task1/.hf-cache"
export TRANSFORMERS_CACHE="$workspace/task1/.hf-cache/hub"
export MUJOCO_GL=egl

cat > "$root/confirmation_protocol.lock.yaml" <<'EOF'
experiment_name: coreact_independent_nuisance_confirmation_v1
stage: independent_confirmation
frozen_development_artifact: coreact_signed_token_utility_phase_t0_v1_20260812_152500
stage_entry_decision: ISS_IMPORTANCE_ESTABLISHED_SIGNED_UTILITY_NOT_ESTABLISHED
fresh_data:
  tasks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  init_state_ids: [5, 6, 7, 8, 9]
  progress: [0.25, 0.65]
  snapshots_per_task: 10
  total_snapshots: 100
  matched_noise_seeds_per_snapshot: 5
arms:
  - full
  - iss_max
  - attention_high_relevance_low
  - random_control
planned_causal_units: 500
planned_episodes: 2000
frozen_components:
  region: local 2x2 visual-token neighborhood
  replacement: camera-and-position-conditioned mean embedding
  intervention_timing: branch-point replan only; all later replans vanilla
  attention: late-half action-to-context attention
  ISS: full 10-step flow action-chunk RMS after replacement
  relevance: frozen visual-region embedding vs instruction-token embedding cosine
  nuisance_selector: rank(attention) * (1-rank(relevance))
  progress_rule: max(1, floor(actual_vanilla_trajectory_length * target_progress))
primary_hypothesis:
  estimand: P(success|attention_high_relevance_low perturb) - P(success|full)
  direction: positive
  go_requires_all:
    - snapshot_cluster_bootstrap_95pct_lower_gt_0
    - at_least_7_of_10_tasks_perturb_ge_full
    - no_task_harm_gt_10pp
    - pooled_nuisance_perturb_gt_random_control
mechanistic_control:
  prediction: P(success|iss_max perturb) < P(success|full)
analysis:
  independent_unit: snapshot
  noise_seeds_are_repeated_measurements: true
  progress_is_stratification_only: true
  report_taskwise_effects: true
forbidden:
  - new_selector
  - threshold_or_weight_tuning
  - signed_classifier
  - guidance_or_CFG
  - partial_outcome_analysis
  - dropping_failed_snapshots_or_causal_units
EOF

"$python_bin" -u "$workspace/research/coreact_self_guidance/reference_snapshot_gate.py" \
  --workspace "$workspace" \
  --parent-artifact "$parent" \
  --output "$reference" \
  --tasks 0,1,2,3,4,5,6,7,8,9 \
  --inits 5,6,7,8,9 \
  --progress 0.25,0.65 > "$launcher/reference_gate.log" 2>&1

decision="$($python_bin -c "import json; print(json.load(open('$reference/decision.json'))['decision'])")"
if [[ "$decision" != "REFERENCE_SNAPSHOT_GATE_PASS_READY_FOR_MATCHED_CAUSAL_ROLLOUT" ]]; then
  printf '%s\n' "$decision" > "$launcher/stopped_at_reference_gate"
  exit 1
fi

"$python_bin" -u "$workspace/research/coreact_signed_utility/prepare_phase_t0.py" \
  --workspace "$workspace" \
  --reference-artifact "$reference" \
  --means "$means" \
  --output "$artifact" \
  --init-ids 5,6,7,8,9 \
  --rollout-arms full,iss_max,attention_high_relevance_low,random_control \
  > "$launcher/candidate_gate.log" 2>&1

decision="$($python_bin -c "import json; print(json.load(open('$artifact/decision.json'))['decision'])")"
if [[ "$decision" != "PHASE_T0_CANDIDATES_LOCKED_READY_FOR_CAUSAL_DRYRUN" ]]; then
  printf '%s\n' "$decision" > "$launcher/stopped_at_candidate_gate"
  exit 1
fi

"$python_bin" -u "$workspace/research/coreact_signed_utility/run_phase_t0.py" \
  --workspace "$workspace" --artifact "$artifact" --reference-artifact "$reference" \
  --means "$means" --max-units 1 --dry-run > "$launcher/dryrun.log" 2>&1

decision="$($python_bin -c "import json; print(json.load(open('$artifact/decision.json'))['decision'])")"
if [[ "$decision" != "PHASE_T0_CAUSAL_DRYRUN_PASS_READY_FOR_ROLLOUT" ]]; then
  printf '%s\n' "$decision" > "$launcher/stopped_at_causal_gate"
  exit 1
fi

pids=()
for worker in 0 1; do
  "$python_bin" -u "$workspace/research/coreact_signed_utility/run_phase_t0.py" \
    --workspace "$workspace" --artifact "$artifact" --reference-artifact "$reference" \
    --means "$means" --unit-start "$worker" --unit-stride 2 \
    > "$launcher/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$launcher/worker.pids"
failure=0
for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
if [[ "$failure" -ne 0 ]]; then
  printf 'worker failure\n' > "$launcher/rollout.failed"
  exit 1
fi
printf 'workers complete\n' > "$launcher/rollout.complete"
