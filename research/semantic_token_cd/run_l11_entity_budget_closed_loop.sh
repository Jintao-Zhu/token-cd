#!/usr/bin/env bash
set -uo pipefail

WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_entity_budget_closed_loop_100_199_v1"
CANONICAL="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
MATCHED="$WS/artifacts/prompt_attn_l11_token_count_v1"
CAL="$WS/artifacts/l11_entity_budget_calibration_v1/CALIBRATION_RESULTS.json"
TASKS=(google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near)

export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"; cd "$WS"

read -r ET GT FT < <("$PY" - "$CAL" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(d['tau']['entity'], d['tau']['generic'], d['tau']['full'])
PY
)
if [[ -z "${ET:-}" || -z "${GT:-}" || -z "${FT:-}" ]]; then echo "failed to read calibration tau" >&2; exit 2; fi

status_json() {
  "$PY" - "$ROOT" "$1" "$2" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
root, stage, message = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
payload={"stage":stage,"message":message,"updated_at":datetime.now(timezone.utc).astimezone().isoformat()}
tmp=root/f"QUEUE_STATUS.{os.getpid()}.tmp"; tmp.write_text(json.dumps(payload,indent=2)+"\n"); os.replace(tmp,root/"QUEUE_STATUS.json")
PY
}

run_rollout(){ local gpu="$1" task="$2" seeds="$3" worker="$4" arm="$5"; "$PY" research/semantic_token_cd/prompt_attn_l11_entity_budget_rollout.py --task "$task" --seeds "$seeds" --gpu "$gpu" --worker-id "$worker" --artifact "$ROOT" --snapshot-artifact "$CANONICAL" --entity-tau "$ET" --generic-tau "$GT" --full-tau "$FT" --arms "$arm"; }
run_worker(){ local gpu="$1" seeds="$2" worker="$3" arm="$4"; for task in "${TASKS[@]}"; do local attempt=1; while true; do if run_rollout "$gpu" "$task" "$seeds" "$worker" "$arm"; then break; fi; if [[ "$attempt" -ge 3 ]]; then return 1; fi; attempt=$((attempt+1)); sleep 60; done; done; }

cat > "$ROOT/CONFIG_LOCK.json" <<EOF
{
  "protocol_id": "PROMPT_ATTN_L11_ENTITY_BUDGET_CLOSED_LOOP_V1",
  "created_date": "2026-09-20",
  "tasks": ["google_robot_open_drawer", "google_robot_close_drawer", "google_robot_pick_coke_can", "google_robot_move_near"],
  "seeds": [100, 199],
  "arm": "entity_top_p",
  "new_episode_count": 400,
  "entity_tau": $ET,
  "generic_tau_for_bookkeeping": $GT,
  "full_tau_for_bookkeeping": $FT,
  "entity_template": "the {entity}",
  "token_ranking": "Top-m of full-prompt L11 attention",
  "only_variable": "m source"
}
EOF

status_json preflight "running entity_top_p preflight on seed 100 open_drawer"
if [[ ! -f "$ROOT/PREFLIGHT_PASS.json" ]]; then
  run_rollout 2 google_robot_open_drawer 100 preflight entity_top_p > "$ROOT/logs/preflight.log" 2>&1
  "$PY" - "$ROOT" "$MATCHED" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); matched_root=Path(sys.argv[2]); task='google_robot_open_drawer'
a=json.loads((root/'episodes'/task/'entity_top_p'/'episode_100_summary.json').read_text())
m=json.loads((matched_root/'episodes'/task/'l11_matched'/'episode_100_summary.json').read_text())
if not a.get('technical_pass'): raise RuntimeError('entity preflight technical failure')
if a.get('budget_source')!='entity_top_p': raise RuntimeError('wrong budget source')
if not a.get('kmeans_bypassed'): raise RuntimeError('KMeans was not bypassed')
for k in ('canonical_snapshot_sha256','initial_state_sha256','initial_rgb_sha256'):
    if a.get(k)!=m.get(k): raise RuntimeError(f'paired {k} mismatch')
a0=a['selector_trace'][0]; m0=m['selector_trace'][0]
if a0.get('attention_sha256')!=m0.get('attention_sha256'): raise RuntimeError('full L11 attention mismatch')
if a0.get('positive_token_ids')!=m0.get('positive_token_ids'): raise RuntimeError('positive branch mismatch')
(root/'PREFLIGHT_PASS.json').write_text(json.dumps({'passed':True,'seed':100},indent=2)+'\n')
print(json.dumps({'preflight':'passed'}))
PY
fi
[[ -f "$ROOT/PREFLIGHT_PASS.json" ]] || { status_json failed "preflight failed"; exit 1; }

status_json rollout "preflight passed; running 4 workers on GPUs 2 and 3"
pids=()
run_worker 2 100-124 gpu2_slot0 entity_top_p > "$ROOT/logs/gpu2_slot0.log" 2>&1 & pids+=("$!")
sleep 12
run_worker 2 125-149 gpu2_slot1 entity_top_p > "$ROOT/logs/gpu2_slot1.log" 2>&1 & pids+=("$!")
sleep 12
run_worker 3 150-174 gpu3_slot0 entity_top_p > "$ROOT/logs/gpu3_slot0.log" 2>&1 & pids+=("$!")
sleep 12
run_worker 3 175-199 gpu3_slot1 entity_top_p > "$ROOT/logs/gpu3_slot1.log" 2>&1 & pids+=("$!")
st=(); for p in "${pids[@]}"; do if wait "$p"; then st+=(0); else st+=("$?"); fi; done
if [[ "${st[*]}" != "0 0 0 0" ]]; then status_json failed "worker statuses: ${st[*]}"; exit 1; fi

count=$(find "$ROOT/episodes" -type f -path '*/entity_top_p/episode_*_summary.json' | wc -l)
if [[ "$count" -ne 400 ]]; then status_json failed "expected 400 summaries, got $count"; exit 1; fi

status_json analysis "400 entity episodes complete; running paired analysis"
"$PY" research/semantic_token_cd/analyze_l11_entity_budget_closed_loop.py --artifact "$ROOT" --matched-artifact "$MATCHED" --seed-start 100 --seed-end 199 --arms entity_top_p > "$ROOT/logs/analysis.log" 2>&1
status_json complete "Entity-TopP closed-loop test and paired analysis complete"
