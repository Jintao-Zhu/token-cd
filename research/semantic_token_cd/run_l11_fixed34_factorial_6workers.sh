#!/usr/bin/env bash
set -uo pipefail
WS=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
ROOT="$WS/artifacts/l11_fixed34_factorial_100_199_v1"
CANON="$WS/artifacts/vanilla_recon_shr_canonical_0_299_v2"
MATCHED="$WS/artifacts/prompt_attn_l11_token_count_v1"
ARMS=l11_fixed34,random_fixed34
export PYTHONPATH="$WS:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/logs"; cd "$WS"
status_json(){ "$PY" - "$ROOT" "$1" "$2" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path
root,stage,msg=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
p={"stage":stage,"message":msg,"updated_at":datetime.now(timezone.utc).astimezone().isoformat()}
t=root/f"QUEUE_STATUS.{os.getpid()}.tmp"; t.write_text(json.dumps(p,indent=2)+"\n"); os.replace(t,root/"QUEUE_STATUS.json")
PY
}
run_rollout(){ "$PY" research/semantic_token_cd/prompt_attn_l11_fixed34_factorial_rollout.py --task "$2" --seeds "$3" --gpu "$1" --worker-id "$4" --artifact "$ROOT" --snapshot-artifact "$CANON" --arms "$ARMS"; }
run_worker(){ local gpu="$1" task="$2" seeds="$3" worker="$4"; local attempt=1; while true; do if run_rollout "$gpu" "$task" "$seeds" "$worker"; then return 0; fi; if [[ "$attempt" -ge 3 ]]; then return 1; fi; attempt=$((attempt+1)); sleep 60; done; }
status_json rollout "resuming with 3 workers per GPU on disjoint task/seed shards"
pids=()
run_worker 2 google_robot_close_drawer 100-149 gpu2_close > "$ROOT/logs/gpu2_close.log" 2>&1 & pids+=("$!")
run_worker 2 google_robot_pick_coke_can 100-149 gpu2_pick > "$ROOT/logs/gpu2_pick.log" 2>&1 & pids+=("$!")
run_worker 2 google_robot_move_near 100-149 gpu2_move > "$ROOT/logs/gpu2_move.log" 2>&1 & pids+=("$!")
run_worker 3 google_robot_close_drawer 150-199 gpu3_close > "$ROOT/logs/gpu3_close.log" 2>&1 & pids+=("$!")
run_worker 3 google_robot_pick_coke_can 150-199 gpu3_pick > "$ROOT/logs/gpu3_pick.log" 2>&1 & pids+=("$!")
run_worker 3 google_robot_move_near 150-199 gpu3_move > "$ROOT/logs/gpu3_move.log" 2>&1 & pids+=("$!")
st=(); for p in "${pids[@]}"; do if wait "$p"; then st+=(0); else st+=($?); fi; done
if [[ "${st[*]}" != "0 0 0 0 0 0" ]]; then status_json failed "worker statuses: ${st[*]}"; exit 1; fi
count=$(find "$ROOT/episodes" -type f \( -path '*/l11_fixed34/episode_*_summary.json' -o -path '*/random_fixed34/episode_*_summary.json' \) | wc -l)
if [[ "$count" -ne 800 ]]; then status_json failed "expected 800 summaries, got $count"; exit 1; fi
status_json analysis "800 fixed34 episodes complete; running paired analysis"
"$PY" research/semantic_token_cd/analyze_l11_fixed34_factorial_closed_loop.py --artifact "$ROOT" --matched-artifact "$MATCHED" --seed-start 100 --seed-end 199 --arms l11_fixed34,random_fixed34 > "$ROOT/logs/analysis.log" 2>&1
status_json complete "Fixed34 factorial closed loop complete"
