#!/usr/bin/env bash
# Runs inside tmux: SIX workers, three per GPU on GPUs 2 and 3.
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd/artifacts/l11_matched_patch_ablation_v1
RUN=/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/run_l11_matched_patch_ablation.sh
LOG="$ROOT/logs/ORCHESTRATOR.log"; mkdir -p "$ROOT/logs"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
say "=== patch-ablation start (6 workers); GPU snapshot ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$LOG" 2>&1
bash "$RUN" 2 100-116 pa_g2a > "$ROOT/logs/gpu2_slotA.log" 2>&1 & P1=$!; sleep 18
bash "$RUN" 2 117-132 pa_g2b > "$ROOT/logs/gpu2_slotB.log" 2>&1 & P2=$!; sleep 18
bash "$RUN" 2 133-149 pa_g2c > "$ROOT/logs/gpu2_slotC.log" 2>&1 & P3=$!; sleep 18
bash "$RUN" 3 150-166 pa_g3a > "$ROOT/logs/gpu3_slotA.log" 2>&1 & P4=$!; sleep 18
bash "$RUN" 3 167-182 pa_g3b > "$ROOT/logs/gpu3_slotB.log" 2>&1 & P5=$!; sleep 18
bash "$RUN" 3 183-199 pa_g3c > "$ROOT/logs/gpu3_slotC.log" 2>&1 & P6=$!
say "workers: gpu2[$P1 $P2 $P3] gpu3[$P4 $P5 $P6]"
st=(); for p in $P1 $P2 $P3 $P4 $P5 $P6; do wait $p && st+=(0) || st+=($?); done
N=$(find "$ROOT/episodes" -name 'episode_*_summary.json' | wc -l)
say "exit=${st[*]}  summaries=$N (expected 1200)"
