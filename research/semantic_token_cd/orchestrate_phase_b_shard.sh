#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd/artifacts/l11_budget_response_curve_v1
RUN=/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/run_phase_b_shard.sh
LOG="$ROOT/logs/PHASEB_SHARD_ORCH.log"; mkdir -p "$ROOT/logs"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== Phase B sharded start (4 workers, 2 shards x 2 tasks) ==="
bash "$RUN" 2 google_robot_open_drawer  2 0 pb2_od0 > "$ROOT/logs/pb2_od0.log" 2>&1 & P1=$!
sleep 15
bash "$RUN" 3 google_robot_open_drawer  2 1 pb2_od1 > "$ROOT/logs/pb2_od1.log" 2>&1 & P2=$!
sleep 15
bash "$RUN" 2 google_robot_close_drawer 2 0 pb2_cd0 > "$ROOT/logs/pb2_cd0.log" 2>&1 & P3=$!
sleep 15
bash "$RUN" 3 google_robot_close_drawer 2 1 pb2_cd1 > "$ROOT/logs/pb2_cd1.log" 2>&1 & P4=$!
say "workers: [$P1 $P2 $P3 $P4]"
st=(); for p in $P1 $P2 $P3 $P4; do wait $p && st+=(0) || st+=($?); done
N=$(find "$ROOT/phase_b" -name "seed_*_step_*.json" | wc -l)
say "exit=${st[*]}  state files=$N (expected 160)"
