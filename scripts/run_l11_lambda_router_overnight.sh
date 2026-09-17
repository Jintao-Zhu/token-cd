#!/usr/bin/env bash
set -uo pipefail

cd /home/leju-suzhou/zjt_ws/token-cd
export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
PYTHON=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
DISCOVERY=artifacts/prompt_attn_l11_lambda_heterogeneity_discovery_v1
CONFIRM=artifacts/prompt_attn_l11_lambda_router_confirmation_v1
CLOSED=artifacts/prompt_attn_l11_lambda_router_closed_loop_v1
CANONICAL=artifacts/vanilla_recon_shr_canonical_0_299_v2

mkdir -p "$CONFIRM/logs" "$CLOSED/logs"

fail() {
  local stage=$1
  local message=$2
  "$PYTHON" - "$stage" "$message" "$CONFIRM" <<'PY'
import json, os, sys
from pathlib import Path
stage, message, root = sys.argv[1:]
path = Path(root) / "FAILED.json"
temporary = path.with_suffix(f".{os.getpid()}.tmp")
temporary.write_text(json.dumps({"stage": stage, "message": message}, indent=2) + "\n")
os.replace(temporary, path)
PY
  exit 1
}

while [[ ! -f "$DISCOVERY/COMPLETE.json" && ! -f "$DISCOVERY/FAILED.json" ]]; do
  sleep 60
done
[[ -f "$DISCOVERY/COMPLETE.json" ]] || fail discovery "discovery rollout or analysis failed"

discovery_go=$($PYTHON - "$DISCOVERY" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads((Path(sys.argv[1]) / "analysis" / "DISCOVERY_RESULTS.json").read_text())
print("1" if payload["go_no_go"]["passed"] else "0")
PY
)
if [[ "$discovery_go" != 1 ]]; then
  printf '{"stage":"discovery","decision":"STOP","reason":"locked Go criteria failed"}\n' > "$CONFIRM/STOPPED.json"
  exit 0
fi

$PYTHON - "$DISCOVERY" "$CONFIRM" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
discovery, confirm = map(Path, sys.argv[1:])
results = json.loads((discovery / "analysis" / "DISCOVERY_RESULTS.json").read_text())
payload = {
    "protocol_id": "PROMPT_ATTN_L11_LAMBDA_ROUTER_CONFIRM_V1",
    "created_date": "2026-09-14",
    "purpose": "frozen held-out confirmation of episode-start lambda selection",
    "tasks": ["google_robot_pick_coke_can", "google_robot_move_near"],
    "seeds": [100, 199],
    "lambdas": [0.0, 0.25, 0.5],
    "new_rollouts": {"l11_positive_only": 200, "l11_fixed_025": 200},
    "reused_lambda_050": "prompt_attn_l11_matched_full_9task_0_299_v1",
    "frozen_model": results["frozen_model"],
    "no_test_retraining_or_calibration": True,
    "go_criteria": {
        "heldout_oracle_gap_at_least_0_05": True,
        "pooled_net_at_least_8_of_200": True,
        "pooled_rescue_gt_harm": True,
        "each_task_net_at_least_minus_2": True,
        "at_least_one_task_net_positive": True,
    },
    "gpu_policy": "only GPUs 2 and 3; at most two worker processes per GPU",
}
path = confirm / "CONFIG_LOCK.json"
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists() and json.loads(path.read_text()) != payload:
    raise RuntimeError("confirmation CONFIG_LOCK differs")
temporary = path.with_suffix(f".{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY

run_confirmation_slot() {
  local gpu=$1 task=$2 seeds=$3 slot=$4
  local log="$CONFIRM/logs/${task}_${seeds}_gpu${gpu}_${slot}.log"
  {
    "$PYTHON" research/semantic_token_cd/collect_l11_lambda_confirmation_features.py \
      --artifact "$CONFIRM" --canonical "$CANONICAL" --task "$task" --seeds "$seeds" --gpu "$gpu" --worker-id "$slot" &&
    "$PYTHON" research/semantic_token_cd/prompt_attn_l11_lambda_confirmation_rollout.py \
      --artifact "$CONFIRM" --canonical "$CANONICAL" --task "$task" --seeds "$seeds" --gpu "$gpu" --worker-id "$slot"
  } > "$log" 2>&1
}

pids=()
run_confirmation_slot 2 google_robot_pick_coke_can 100-149 gpu2_slot0 & pids+=("$!")
run_confirmation_slot 2 google_robot_pick_coke_can 150-199 gpu2_slot1 & pids+=("$!")
run_confirmation_slot 3 google_robot_move_near 100-149 gpu3_slot0 & pids+=("$!")
run_confirmation_slot 3 google_robot_move_near 150-199 gpu3_slot1 & pids+=("$!")
statuses=()
for pid in "${pids[@]}"; do
  if wait "$pid"; then statuses+=(0); else statuses+=("$?"); fi
done
[[ "${statuses[*]}" == "0 0 0 0" ]] || fail confirmation_rollout "worker statuses: ${statuses[*]}"

zero_count=$(find "$CONFIRM/episodes" -path '*/l11_positive_only/episode_1??_summary.json' | wc -l)
quarter_count=$(find "$CONFIRM/episodes" -path '*/l11_fixed_025/episode_1??_summary.json' | wc -l)
feature_count=$(find "$CONFIRM/features" -name 'seed_1??.npz' | wc -l)
[[ "$zero_count" -eq 200 && "$quarter_count" -eq 200 && "$feature_count" -eq 200 ]] || \
  fail confirmation_counts "zero=$zero_count quarter=$quarter_count features=$feature_count"

"$PYTHON" research/semantic_token_cd/analyze_l11_lambda_router_confirmation.py \
  --artifact "$CONFIRM" --discovery "$DISCOVERY" > "$CONFIRM/logs/analysis.log" 2>&1 || fail confirmation_analysis "analysis failed"

confirmation_go=$($PYTHON - "$CONFIRM" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads((Path(sys.argv[1]) / "analysis" / "CONFIRMATION_RESULTS.json").read_text())
print("1" if payload["go_no_go"]["passed"] else "0")
PY
)
if [[ "$confirmation_go" != 1 ]]; then
  printf '{"stage":"confirmation","decision":"STOP","reason":"locked frozen-test Go criteria failed"}\n' > "$CONFIRM/STOPPED.json"
  exit 0
fi
printf '{"complete":true,"decision":"GO","analysis":"analysis/CONFIRMATION_RESULTS.json"}\n' > "$CONFIRM/COMPLETE.json"

run_router_slot() {
  local gpu=$1 task=$2 seeds=$3 slot=$4
  "$PYTHON" research/semantic_token_cd/prompt_attn_l11_lambda_router_rollout.py \
    --artifact "$CLOSED" --confirmation "$CONFIRM" --discovery "$DISCOVERY" --canonical "$CANONICAL" \
    --task "$task" --seeds "$seeds" --gpu "$gpu" --worker-id "$slot" \
    > "$CLOSED/logs/${task}_${seeds}_gpu${gpu}_${slot}.log" 2>&1
}

pids=()
run_router_slot 2 google_robot_pick_coke_can 100-149 gpu2_slot0 & pids+=("$!")
run_router_slot 2 google_robot_pick_coke_can 150-199 gpu2_slot1 & pids+=("$!")
run_router_slot 3 google_robot_move_near 100-149 gpu3_slot0 & pids+=("$!")
run_router_slot 3 google_robot_move_near 150-199 gpu3_slot1 & pids+=("$!")
statuses=()
for pid in "${pids[@]}"; do
  if wait "$pid"; then statuses+=(0); else statuses+=("$?"); fi
done
[[ "${statuses[*]}" == "0 0 0 0" ]] || fail closed_loop_rollout "worker statuses: ${statuses[*]}"

router_count=$(find "$CLOSED/episodes" -path '*/router/episode_1??_summary.json' | wc -l)
[[ "$router_count" -eq 200 ]] || fail closed_loop_counts "router=$router_count"
"$PYTHON" research/semantic_token_cd/analyze_l11_lambda_router_closed_loop.py \
  --artifact "$CLOSED" --confirmation "$CONFIRM" > "$CLOSED/logs/analysis.log" 2>&1 || fail closed_loop_analysis "analysis failed"
printf '{"complete":true,"episodes":200,"analysis":"analysis/CLOSED_LOOP_RESULTS.json"}\n' > "$CLOSED/COMPLETE.json"
