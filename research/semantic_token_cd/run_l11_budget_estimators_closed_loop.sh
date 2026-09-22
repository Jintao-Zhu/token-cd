#!/usr/bin/env bash
set -uo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_budget_estimators_closed_loop_100_199_v1"
CANONICAL="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
MATCHED="$WS/artifacts/prompt_attn_l11_token_count_v1"
OFFLINE="$WS/artifacts/l11_budget_estimators_v1/OFFLINE_RESULTS.json"
TASKS=(google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near)

export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"
cd "$WS"

if [[ ! -f "$OFFLINE" ]]; then
  echo "missing offline calibration result: $OFFLINE" >&2
  exit 2
fi

read -r GAMMA TAU < <("$PY" - "$OFFLINE" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(d['gamma_star'], d['tau_star'])
PY
)
if [[ -z "${GAMMA:-}" || -z "${TAU:-}" ]]; then
  echo "failed to read gamma_star/tau_star" >&2
  exit 2
fi

status_json() {
  local stage="$1" message="$2"
  "$PY" - "$ROOT" "$stage" "$message" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
root, stage, message = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
payload = {
    "stage": stage,
    "message": message,
    "updated_at": datetime.now(timezone.utc).astimezone().isoformat(),
    "gamma_star": None,
    "tau_star": None,
}
try:
    import json as _json
    off = _json.loads((root.parent / "l11_budget_estimators_v1" / "OFFLINE_RESULTS.json").read_text())
    payload["gamma_star"] = off.get("gamma_star")
    payload["tau_star"] = off.get("tau_star")
except Exception:
    pass
tmp = root / f"QUEUE_STATUS.{os.getpid()}.tmp"
tmp.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(tmp, root / "QUEUE_STATUS.json")
PY
}

run_rollout() {
  local gpu="$1" task="$2" seeds="$3" worker="$4"
  "$PY" research/semantic_token_cd/prompt_attn_l11_budget_estimators_rollout.py \
    --task "$task" --seeds "$seeds" --gpu "$gpu" --worker-id "$worker" \
    --artifact "$ROOT" --snapshot-artifact "$CANONICAL" \
    --gamma "$GAMMA" --tau "$TAU" --arms relative,spectral
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
  "protocol_id": "PROMPT_ATTN_L11_BUDGET_ESTIMATOR_CLOSED_LOOP_V1",
  "created_date": "2026-09-19",
  "tasks": ["google_robot_open_drawer", "google_robot_close_drawer", "google_robot_pick_coke_can", "google_robot_move_near"],
  "seeds": [100, 199],
  "arms": ["relative", "spectral"],
  "matched_reused": true,
  "new_episode_count": 800,
  "attention_layers": [11],
  "lambda": 0.5,
  "ranking": "Prompt-L11 descending score with stable token-id tie break",
  "reconstruction": "harmonic beta=0",
  "gamma_star": $GAMMA,
  "tau_star": $TAU,
  "gpu_policy": "only GPUs 2 and 3; two worker processes per GPU",
  "calibration": "gamma_star/tau_star fixed from OFFLINE_RESULTS.json; no success labels"
}
EOF

status_json preflight "running Relative/Spectral preflight on seed 100, open_drawer, GPU 2"
if [[ ! -f "$ROOT/PREFLIGHT_PASS.json" ]]; then
  run_rollout 2 google_robot_open_drawer 100 preflight > "$ROOT/logs/preflight.log" 2>&1
  "$PY" - "$ROOT" "$MATCHED" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); matched_root = Path(sys.argv[2])
task = "google_robot_open_drawer"
matched_path = matched_root / "episodes" / task / "l11_matched" / "episode_100_summary.json"
if not matched_path.exists():
    raise RuntimeError(f"missing matched preflight: {matched_path}")
matched = json.loads(matched_path.read_text())
for arm in ("relative", "spectral"):
    path = root / "episodes" / task / arm / "episode_100_summary.json"
    data = json.loads(path.read_text())
    if not data.get("technical_pass"):
        raise RuntimeError(f"technical failure: {path}")
    if data.get("budget_source") != arm:
        raise RuntimeError(f"bad budget source: {path}")
    if not data.get("kmeans_bypassed"):
        raise RuntimeError(f"KMeans was not bypassed: {path}")
    for key in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
        if data.get(key) != matched.get(key):
            raise RuntimeError(f"paired {key} mismatch: {path}")
    m0 = matched["selector_trace"][0]
    a0 = data["selector_trace"][0]
    if a0.get("attention_sha256") != m0.get("attention_sha256"):
        raise RuntimeError(f"L11 attention mismatch: {path}")
    if a0.get("positive_token_ids") != m0.get("positive_token_ids"):
        raise RuntimeError(f"positive branch mismatch: {path}")
(root / "PREFLIGHT_PASS.json").write_text(json.dumps({"passed": True, "seed": 100}, indent=2) + "\n")
print(json.dumps({"preflight": "passed", "gamma": data.get("relative_gamma"), "tau": data.get("spectral_tau")}))
PY
fi
if [[ ! -f "$ROOT/PREFLIGHT_PASS.json" ]]; then
  status_json failed "preflight did not pass"
  exit 1
fi

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

count=$(find "$ROOT/episodes" -type f \( -path '*/relative/episode_*_summary.json' -o -path '*/spectral/episode_*_summary.json' \) | wc -l)
if [[ "$count" -ne 800 ]]; then
  status_json failed "expected 800 summaries, got $count"
  exit 1
fi

status_json analysis "800 estimator episodes complete; running paired analysis"
"$PY" research/semantic_token_cd/analyze_l11_budget_estimators_closed_loop.py \
  --artifact "$ROOT" --matched-artifact "$MATCHED" --seed-start 100 --seed-end 199 \
  > "$ROOT/logs/analysis.log" 2>&1
status_json complete "Relative/Spectral closed-loop test and paired analysis complete"
