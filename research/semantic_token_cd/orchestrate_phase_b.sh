#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd/artifacts/l11_budget_response_curve_v1
RUN=/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/run_phase_b.sh
LOG="$ROOT/logs/PHASEB_ORCH.log"; mkdir -p "$ROOT/logs"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== Phase B start (4 workers, 2/GPU) ==="
bash "$RUN" w 2 google_robot_open_drawer pb_g2a > "$ROOT/logs/pb_g2a.log" 2>&1 & P1=$!
sleep 15
bash "$RUN" w 2 google_robot_close_drawer pb_g2b > "$ROOT/logs/pb_g2b.log" 2>&1 & P2=$!
sleep 15
bash "$RUN" w 3 google_robot_pick_coke_can pb_g3a > "$ROOT/logs/pb_g3a.log" 2>&1 & P3=$!
sleep 15
bash "$RUN" w 3 google_robot_move_near pb_g3b > "$ROOT/logs/pb_g3b.log" 2>&1 & P4=$!
say "workers: gpu2[$P1 $P2] gpu3[$P3 $P4]"
st=(); for p in $P1 $P2 $P3 $P4; do wait $p && st+=(0) || st+=($?); done
N=$(find "$ROOT/phase_b" -name "seed_*_step_*.json" 2>/dev/null | wc -l)
say "exit=${st[*]}  state files=$N (expected 160)"
