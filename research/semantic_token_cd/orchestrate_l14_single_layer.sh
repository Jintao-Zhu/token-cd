#!/usr/bin/env bash
# Runs inside tmux: TWO workers, one per GPU on GPUs 2 and 3.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd/artifacts/l14_single_layer_v1/closed_loop
RUN=/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/run_l14_single_layer.sh
LOG="$ROOT/logs/ORCHESTRATOR.log"
mkdir -p "$ROOT/logs"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== L14 start (2 workers); GPU snapshot ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$LOG" 2>&1
bash "$RUN" 2 0-49 l14_gpu2 > "$ROOT/logs/gpu2.log" 2>&1 & P1=$!
sleep 20
bash "$RUN" 3 50-99 l14_gpu3 > "$ROOT/logs/gpu3.log" 2>&1 & P2=$!
say "workers: gpu2=$P1 seeds=0-49 | gpu3=$P2 seeds=50-99"
st=()
for p in $P1 $P2; do wait $p && st+=(0) || st+=($?); done
N=$(find "$ROOT/episodes" -name 'episode_*_summary.json' | wc -l)
say "exit=${st[*]}  summaries=$N (expected 400)"
