#!/usr/bin/env bash
# Runs inside tmux: TWO Phase-A workers, one per GPU on GPUs 2 and 3.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd/artifacts/l11_budget_response_curve_v1
RUN=/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/run_phase_a_budget_response.sh
LOG="$ROOT/logs/ORCHESTRATOR.log"; mkdir -p "$ROOT/logs"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== Phase A start (2 workers); GPU snapshot ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$LOG" 2>&1
bash "$RUN" A > "$ROOT/logs/workerA.log" 2>&1 & P1=$!
sleep 20
bash "$RUN" B > "$ROOT/logs/workerB.log" 2>&1 & P2=$!
say "workers: A(gpu2: open_drawer,close_drawer)=$P1  B(gpu3: pick_coke_can,move_near)=$P2"
st=(); for p in $P1 $P2; do wait $p && st+=(0) || st+=($?); done
N=$(find "$ROOT/phase_a" -name "seed_*.json" 2>/dev/null | wc -l)
say "exit=${st[*]}  seed files=$N (expected 200)"
