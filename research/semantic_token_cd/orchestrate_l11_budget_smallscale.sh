#!/usr/bin/env bash
# Runs inside tmux: SIX workers, three per GPU on GPUs 2 and 3.
# Seeds 100-199 split into six contiguous ranges so workers never overlap.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd/artifacts/l11_matched_budget_smallscale_v1
RUN=/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/run_l11_budget_smallscale.sh
LOG="$ROOT/logs/ORCHESTRATOR.log"
mkdir -p "$ROOT/logs"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== start (6 workers); GPU snapshot ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$LOG" 2>&1

bash "$RUN" worker 2 100-116 gpu2_slot1 > "$ROOT/logs/gpu2_slot1.log" 2>&1 & P1=$!
sleep 20
bash "$RUN" worker 2 117-132 gpu2_slot2 > "$ROOT/logs/gpu2_slot2.log" 2>&1 & P2=$!
sleep 20
bash "$RUN" worker 2 133-149 gpu2_slot3 > "$ROOT/logs/gpu2_slot3.log" 2>&1 & P3=$!
sleep 20
bash "$RUN" worker 3 150-166 gpu3_slot1 > "$ROOT/logs/gpu3_slot1.log" 2>&1 & P4=$!
sleep 20
bash "$RUN" worker 3 167-182 gpu3_slot2 > "$ROOT/logs/gpu3_slot2.log" 2>&1 & P5=$!
sleep 20
bash "$RUN" worker 3 183-199 gpu3_slot3 > "$ROOT/logs/gpu3_slot3.log" 2>&1 & P6=$!

say "workers: gpu2[$P1 $P2 $P3] gpu3[$P4 $P5 $P6]"

st=()
for p in $P1 $P2 $P3 $P4 $P5 $P6; do wait $p && st+=(0) || st+=($?); done
say "worker exit statuses: ${st[*]}"

N=$(find "$ROOT/episodes" -name 'episode_*_summary.json' | wc -l)
if [[ "${st[*]}" == "0 0 0 0 0 0" ]]; then
  say "ALL DONE - $N summaries (expected 1200)"
else
  say "FAILED - statuses ${st[*]}, summaries so far: $N"
fi
