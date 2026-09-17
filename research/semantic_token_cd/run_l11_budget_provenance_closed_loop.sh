#!/usr/bin/env bash
set -euo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_matched_budget_provenance_closed_loop_100_199_v1"
CANONICAL="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
MATCHED="$WS/artifacts/prompt_attn_l11_token_count_v1"
SCHEDULE="$WS/artifacts/l11_matched_budget_causal_pilot_v1/BUDGET_SCHEDULES.json"
ARMS=wrong_entity,random_cluster,within_task_shuffle
TASKS=(google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near)

export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"
cd "$WS"

status_json() {
  local stage="$1" message="$2"
  "$PY" - "$ROOT" "$stage" "$message" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
root, stage, message = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
payload = {"stage": stage, "message": message, "updated_at": datetime.now(timezone.utc).astimezone().isoformat()}
temporary = root / f"QUEUE_STATUS.{os.getpid()}.tmp"
temporary.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(temporary, root / "QUEUE_STATUS.json")
PY
}

run_rollout() {
  local gpu="$1" task="$2" seeds="$3" worker="$4"
  "$PY" research/semantic_token_cd/prompt_attn_l11_budget_provenance_rollout.py \
    --task "$task" --seeds "$seeds" --gpu "$gpu" --worker-id "$worker" \
    --artifact "$ROOT" --snapshot-artifact "$CANONICAL" --schedule-file "$SCHEDULE" \
    --arms "$ARMS"
}

run_worker() {
  local gpu="$1" seeds="$2" worker="$3"
  for task in "${TASKS[@]}"; do
    local attempt=1
    while true; do
      if run_rollout "$gpu" "$task" "$seeds" "$worker"; then
        break
      fi
      if [[ "$attempt" -ge 3 ]]; then
        return 1
      fi
      attempt=$((attempt + 1))
      sleep 60
    done
  done
}

cat > "$ROOT/CONFIG_LOCK.json" <<EOF
{
  "protocol_id": "PROMPT_ATTN_L11_BUDGET_PROVENANCE_CLOSED_LOOP_V1",
  "created_date": "2026-09-17",
  "tasks": ["google_robot_open_drawer", "google_robot_close_drawer", "google_robot_pick_coke_can", "google_robot_move_near"],
  "seeds": [100, 199],
  "reused_arm": "true_matched",
  "new_arms": ["wrong_entity", "random_cluster", "within_task_shuffle"],
  "new_episode_count": 1200,
  "attention_layers": [11],
  "lambda": 0.5,
  "ranking": "Prompt-L11 descending score with stable token-id tie break",
  "reconstruction": "harmonic beta=0",
  "gpu_policy": "only GPUs 2 and 3; exactly two worker processes per GPU after preflight"
}
EOF

status_json preflight "running three-arm seed-100 preflight on GPU 2"
run_rollout 2 google_robot_open_drawer 100 preflight > "$ROOT/logs/preflight.log" 2>&1
"$PY" - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
paths = sorted((root / "episodes/google_robot_open_drawer").glob("*/episode_100_summary.json"))
if len(paths) != 3:
    raise RuntimeError(f"preflight expected 3 summaries, got {len(paths)}")
for path in paths:
    data = json.loads(path.read_text())
    if not data.get("technical_pass"):
        raise RuntimeError(f"preflight technical failure: {path}")
(root / "PREFLIGHT_PASS.json").write_text(json.dumps({"passed": True, "episodes": 3}, indent=2) + "\n")
PY

status_json rollout "preflight passed; running four workers on GPUs 2 and 3"
pids=()
run_worker 2 100-124 gpu2_slot0 > "$ROOT/logs/gpu2_slot0.log" 2>&1 & pids+=("$!")
sleep 12
run_worker 2 125-149 gpu2_slot1 > "$ROOT/logs/gpu2_slot1.log" 2>&1 & pids+=("$!")
sleep 12
run_worker 3 150-174 gpu3_slot0 > "$ROOT/logs/gpu3_slot0.log" 2>&1 & pids+=("$!")
sleep 12
run_worker 3 175-199 gpu3_slot1 > "$ROOT/logs/gpu3_slot1.log" 2>&1 & pids+=("$!")

statuses=()
for pid in "${pids[@]}"; do
  if wait "$pid"; then statuses+=(0); else statuses+=("$?"); fi
done
if [[ "${statuses[*]}" != "0 0 0 0" ]]; then
  status_json failed "worker statuses: ${statuses[*]}"
  exit 1
fi

count=$(find "$ROOT/episodes" -type f -name 'episode_*_summary.json' | wc -l)
if [[ "$count" -ne 1200 ]]; then
  status_json failed "expected 1200 summaries, got $count"
  exit 1
fi

status_json analysis "1200 candidate episodes complete; running paired analysis"
"$PY" research/semantic_token_cd/analyze_l11_budget_provenance_closed_loop.py \
  --artifact "$ROOT" --matched-artifact "$MATCHED" --seed-start 100 --seed-end 199 \
  > "$ROOT/logs/analysis.log" 2>&1
status_json complete "closed-loop causal test and paired analysis complete"
