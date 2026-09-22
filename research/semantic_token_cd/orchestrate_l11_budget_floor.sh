#!/usr/bin/env bash
# Runs inside tmux: FOUR workers, two per GPU on GPUs 2 and 3.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd/artifacts/l11_matched_budget_floor_v1
RUN=/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/run_l11_budget_floor.sh
LOG="$ROOT/logs/ORCHESTRATOR.log"
mkdir -p "$ROOT/logs"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== start; GPU snapshot ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$LOG" 2>&1

# exactly two new workers per GPU, staggered 20s to avoid a load spike
bash "$RUN" worker 2 100-124 floor_gpu2_slotA > "$ROOT/logs/gpu2_slotA.log" 2>&1 & P1=$!
sleep 20
bash "$RUN" worker 2 125-149 floor_gpu2_slotB > "$ROOT/logs/gpu2_slotB.log" 2>&1 & P2=$!
sleep 20
bash "$RUN" worker 3 150-174 floor_gpu3_slotA > "$ROOT/logs/gpu3_slotA.log" 2>&1 & P3=$!
sleep 20
bash "$RUN" worker 3 175-199 floor_gpu3_slotB > "$ROOT/logs/gpu3_slotB.log" 2>&1 & P4=$!

say "workers: gpu2[A=$P1 B=$P2] gpu3[A=$P3 B=$P4]"

st=()
for p in $P1 $P2 $P3 $P4; do wait $p && st+=(0) || st+=($?); done
say "worker exit statuses: ${st[*]}"

N=$(find "$ROOT/episodes" -name 'episode_*_summary.json' | wc -l)
if [[ "${st[*]}" == "0 0 0 0" ]]; then
  say "ALL DONE - $N summaries (expected 1200)"
else
  say "FAILED - statuses ${st[*]}, summaries so far: $N"
fi
