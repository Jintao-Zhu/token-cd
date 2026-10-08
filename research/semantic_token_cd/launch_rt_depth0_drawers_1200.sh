#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
OUT="$ROOT/artifacts/simpler_rt_depth0_drawers_0_299_v1"
RUN="$ROOT/research/semantic_token_cd/run_rt_depth0_drawer_paired.py"
LOG="$OUT/logs"
mkdir -p "$LOG"

echo "[$(date -Is)] PRE-FLIGHT: drawer RT-depth0 paired smoke on GPUs 1-3" | tee -a "$LOG/launcher.log"
"$PY" "$RUN" --gpu 1 --task google_robot_open_drawer --seeds 0 --worker-id preflight_gpu1_open_s000 --output "$OUT" >"$LOG/preflight_gpu1_open.log" 2>&1 & p1=$!
"$PY" "$RUN" --gpu 2 --task google_robot_close_drawer --seeds 0 --worker-id preflight_gpu2_close_s000 --output "$OUT" >"$LOG/preflight_gpu2_close.log" 2>&1 & p2=$!
"$PY" "$RUN" --gpu 3 --task google_robot_close_drawer --seeds 1 --worker-id preflight_gpu3_close_s001 --output "$OUT" >"$LOG/preflight_gpu3_close.log" 2>&1 & p3=$!
status=0
for pid in "$p1" "$p2" "$p3"; do wait "$pid" || status=1; done
if [[ $status -ne 0 ]]; then
  echo "[$(date -Is)] PRE-FLIGHT FAILED; formal queue not launched" | tee -a "$LOG/launcher.log"
  exit 20
fi
"$PY" "$ROOT/research/semantic_token_cd/finalize_rt_depth0_drawers.py" --output "$OUT" --preflight-only | tee -a "$LOG/launcher.log"
echo "[$(date -Is)] PRE-FLIGHT PASS; launch 9 workers, 3 per GPU" | tee -a "$LOG/launcher.log"

# Exactly three workers per physical GPU; all shards are disjoint.
"$PY" "$RUN" --gpu 1 --task google_robot_open_drawer --seeds 0-49 --worker-id gpu1_open_000_049 --output "$OUT" >"$LOG/gpu1_open_000_049.log" 2>&1 & w1=$!
"$PY" "$RUN" --gpu 1 --task google_robot_open_drawer --seeds 50-99 --worker-id gpu1_open_050_099 --output "$OUT" >"$LOG/gpu1_open_050_099.log" 2>&1 & w2=$!
"$PY" "$RUN" --gpu 1 --task google_robot_close_drawer --seeds 0-99 --worker-id gpu1_close_000_099 --output "$OUT" >"$LOG/gpu1_close_000_099.log" 2>&1 & w3=$!

"$PY" "$RUN" --gpu 2 --task google_robot_open_drawer --seeds 100-149 --worker-id gpu2_open_100_149 --output "$OUT" >"$LOG/gpu2_open_100_149.log" 2>&1 & w4=$!
"$PY" "$RUN" --gpu 2 --task google_robot_open_drawer --seeds 150-199 --worker-id gpu2_open_150_199 --output "$OUT" >"$LOG/gpu2_open_150_199.log" 2>&1 & w5=$!
"$PY" "$RUN" --gpu 2 --task google_robot_close_drawer --seeds 100-199 --worker-id gpu2_close_100_199 --output "$OUT" >"$LOG/gpu2_close_100_199.log" 2>&1 & w6=$!

"$PY" "$RUN" --gpu 3 --task google_robot_open_drawer --seeds 200-249 --worker-id gpu3_open_200_249 --output "$OUT" >"$LOG/gpu3_open_200_249.log" 2>&1 & w7=$!
"$PY" "$RUN" --gpu 3 --task google_robot_open_drawer --seeds 250-299 --worker-id gpu3_open_250_299 --output "$OUT" >"$LOG/gpu3_open_250_299.log" 2>&1 & w8=$!
"$PY" "$RUN" --gpu 3 --task google_robot_close_drawer --seeds 200-299 --worker-id gpu3_close_200_299 --output "$OUT" >"$LOG/gpu3_close_200_299.log" 2>&1 & w9=$!
formal_status=0
for pid in "$w1" "$w2" "$w3" "$w4" "$w5" "$w6" "$w7" "$w8" "$w9"; do wait "$pid" || formal_status=1; done
if [[ $formal_status -ne 0 ]]; then
  echo "[$(date -Is)] FORMAL WORKER FAILURE; inspect failure_*.json and worker logs" | tee -a "$LOG/launcher.log"
  exit 21
fi
"$PY" "$ROOT/research/semantic_token_cd/finalize_rt_depth0_drawers.py" --output "$OUT" --finalize | tee -a "$LOG/launcher.log"
echo "[$(date -Is)] COMPLETE" | tee -a "$LOG/launcher.log"
