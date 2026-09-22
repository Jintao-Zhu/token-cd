#!/usr/bin/env bash
# Runs inside tmux: preflight on GPU 2, then one worker per GPU (2 and 3).
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd/artifacts/l11_top_p85_unclipped_100_199_v1
RUN=/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/run_l11_top_p_noclip.sh
LOG="$ROOT/logs/ORCHESTRATOR.log"
mkdir -p "$ROOT/logs"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== start; GPU free-mem snapshot ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$LOG" 2>&1

say "preflight: 1 episode on GPU 2 (open_drawer seed 100)"
if ! bash "$RUN" preflight > "$ROOT/logs/preflight.log" 2>&1; then
  say "PREFLIGHT FAILED - not launching workers. tail:"
  tail -30 "$ROOT/logs/preflight.log" | tee -a "$LOG"
  exit 1
fi
if ! ls "$ROOT"/episodes/google_robot_open_drawer/l11_top_p85_noclip/episode_100_summary.json >/dev/null 2>&1; then
  say "PREFLIGHT produced no summary - aborting"
  exit 1
fi
say "PREFLIGHT OK - fanning out 2 workers"

bash "$RUN" worker 2 100-149 gpu2_noclip > "$ROOT/logs/gpu2_noclip.log" 2>&1 &
P2=$!
sleep 15
bash "$RUN" worker 3 150-199 gpu3_noclip > "$ROOT/logs/gpu3_noclip.log" 2>&1 &
P3=$!
say "workers started: gpu2 pid=$P2 seeds=100-149 | gpu3 pid=$P3 seeds=150-199"

wait $P2; S2=$?
wait $P3; S3=$?
say "workers finished: gpu2=$S2 gpu3=$S3"
if [[ "$S2" == "0" && "$S3" == "0" ]]; then
  N=$(find "$ROOT/episodes" -name 'episode_*_summary.json' | wc -l)
  say "ALL DONE - $N summaries (expected 400)"
else
  say "FAILED - gpu2=$S2 gpu3=$S3"
fi
