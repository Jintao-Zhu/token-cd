#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/leju-suzhou/zjt_ws/token-cd
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
OUT="$ROOT/artifacts/simpler_rt_depth0_7task_4200_v1"
RUN="$ROOT/research/semantic_token_cd/run_rt_depth0_seven_task_paired.py"
LOG="$OUT/logs"
mkdir -p "$LOG"

echo "[$(date -Is)] PRE-FLIGHT: 7 task x seed 0, paired Vanilla/L11-Matched" | tee -a "$LOG/launcher.log"

# Parallel smoke cases across only GPUs 1-3. Their seed-0 outputs are part of
# the locked 0-299 cohort and will be skipped by the formal shards below.
"$PY" "$RUN" --gpu 1 --task google_robot_place_apple_in_closed_top_drawer --seeds 0 --worker-id preflight_gpu1_apple --output "$OUT" >"$LOG/preflight_gpu1_apple.log" 2>&1 & p1=$!
"$PY" "$RUN" --gpu 1 --task widowx_carrot_on_plate --seeds 0 --worker-id preflight_gpu1_carrot --output "$OUT" >"$LOG/preflight_gpu1_carrot.log" 2>&1 & p2=$!
"$PY" "$RUN" --gpu 2 --task google_robot_pick_coke_can --seeds 0 --worker-id preflight_gpu2_pick --output "$OUT" >"$LOG/preflight_gpu2_pick.log" 2>&1 & p3=$!
"$PY" "$RUN" --gpu 2 --task widowx_put_eggplant_in_basket --seeds 0 --worker-id preflight_gpu2_eggplant --output "$OUT" >"$LOG/preflight_gpu2_eggplant.log" 2>&1 & p4=$!
"$PY" "$RUN" --gpu 3 --task google_robot_move_near --seeds 0 --worker-id preflight_gpu3_move --output "$OUT" >"$LOG/preflight_gpu3_move.log" 2>&1 & p5=$!
"$PY" "$RUN" --gpu 3 --task widowx_spoon_on_towel --seeds 0 --worker-id preflight_gpu3_spoon --output "$OUT" >"$LOG/preflight_gpu3_spoon.log" 2>&1 & p6=$!
"$PY" "$RUN" --gpu 3 --task widowx_stack_cube --seeds 0 --worker-id preflight_gpu3_stack --output "$OUT" >"$LOG/preflight_gpu3_stack.log" 2>&1 & p7=$!

status=0
for pid in "$p1" "$p2" "$p3" "$p4" "$p5" "$p6" "$p7"; do
  wait "$pid" || status=1
done
if [[ $status -ne 0 ]]; then
  echo "[$(date -Is)] PRE-FLIGHT FAILED; formal jobs NOT started" | tee -a "$LOG/launcher.log"
  exit 20
fi

"$PY" "$ROOT/research/semantic_token_cd/validate_rt_depth0_seven_task.py" --output "$OUT" --preflight-only | tee -a "$LOG/launcher.log"
echo "[$(date -Is)] PRE-FLIGHT PASS; launching 9 formal workers (3/GPU)" | tee -a "$LOG/launcher.log"

"$PY" "$RUN" --gpu 1 --task google_robot_place_apple_in_closed_top_drawer --seeds 0-149 --worker-id gpu1_apple_000_149 --output "$OUT" >"$LOG/gpu1_apple_000_149.log" 2>&1 & w1=$!
"$PY" "$RUN" --gpu 1 --task widowx_carrot_on_plate --seeds 0-299 --worker-id gpu1_carrot_000_299 --output "$OUT" >"$LOG/gpu1_carrot_000_299.log" 2>&1 & w2=$!
"$PY" "$RUN" --gpu 1 --task widowx_stack_cube --seeds 0-299 --worker-id gpu1_stack_000_299 --output "$OUT" >"$LOG/gpu1_stack_000_299.log" 2>&1 & w3=$!

"$PY" "$RUN" --gpu 2 --task google_robot_place_apple_in_closed_top_drawer --seeds 150-299 --worker-id gpu2_apple_150_299 --output "$OUT" >"$LOG/gpu2_apple_150_299.log" 2>&1 & w4=$!
"$PY" "$RUN" --gpu 2 --task widowx_put_eggplant_in_basket --seeds 0-149 --worker-id gpu2_eggplant_000_149 --output "$OUT" >"$LOG/gpu2_eggplant_000_149.log" 2>&1 & w5=$!
"$PY" "$RUN" --gpu 2 --task widowx_spoon_on_towel --seeds 0-299 --worker-id gpu2_spoon_000_299 --output "$OUT" >"$LOG/gpu2_spoon_000_299.log" 2>&1 & w6=$!

"$PY" "$RUN" --gpu 3 --task google_robot_pick_coke_can --seeds 0-299 --worker-id gpu3_pick_000_299 --output "$OUT" >"$LOG/gpu3_pick_000_299.log" 2>&1 & w7=$!
"$PY" "$RUN" --gpu 3 --task google_robot_move_near --seeds 0-299 --worker-id gpu3_move_000_299 --output "$OUT" >"$LOG/gpu3_move_000_299.log" 2>&1 & w8=$!
"$PY" "$RUN" --gpu 3 --task widowx_put_eggplant_in_basket --seeds 150-299 --worker-id gpu3_eggplant_150_299 --output "$OUT" >"$LOG/gpu3_eggplant_150_299.log" 2>&1 & w9=$!

formal_status=0
for pid in "$w1" "$w2" "$w3" "$w4" "$w5" "$w6" "$w7" "$w8" "$w9"; do
  wait "$pid" || formal_status=1
done
if [[ $formal_status -ne 0 ]]; then
  echo "[$(date -Is)] FORMAL WORKER FAILURE; inspect failure JSON and logs" | tee -a "$LOG/launcher.log"
  exit 21
fi

"$PY" "$ROOT/research/semantic_token_cd/validate_rt_depth0_seven_task.py" --output "$OUT" --finalize | tee -a "$LOG/launcher.log"
echo "[$(date -Is)] COMPLETE" | tee -a "$LOG/launcher.log"
