#!/usr/bin/env bash
# Runs inside tmux: FOUR workers, two per GPU on GPUs 2 and 3.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd/artifacts/l11_matched_plus_uniform16_v1
RUN=/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/run_l11_matched_plus_uniform.sh
LOG="$ROOT/logs/ORCHESTRATOR.log"; mkdir -p "$ROOT/logs"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== uniform16 start (4 workers); GPU snapshot ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$LOG" 2>&1
bash "$RUN" 2 100-124 u_gpu2_a > "$ROOT/logs/gpu2_slotA.log" 2>&1 & P1=$!
sleep 20
bash "$RUN" 2 125-149 u_gpu2_b > "$ROOT/logs/gpu2_slotB.log" 2>&1 & P2=$!
sleep 20
bash "$RUN" 3 150-174 u_gpu3_a > "$ROOT/logs/gpu3_slotA.log" 2>&1 & P3=$!
sleep 20
bash "$RUN" 3 175-199 u_gpu3_b > "$ROOT/logs/gpu3_slotB.log" 2>&1 & P4=$!
say "workers: gpu2[$P1 $P2] gpu3[$P3 $P4]"
st=()
for p in $P1 $P2 $P3 $P4; do wait $p && st+=(0) || st+=($?); done
N=$(find "$ROOT/episodes" -name 'episode_*_summary.json' | wc -l)
say "exit=${st[*]}  summaries=$N (expected 900)"
