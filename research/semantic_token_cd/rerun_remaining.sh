#!/usr/bin/env bash
# Serial re-run of remaining work, GPU-0-aware (GPU 0 is DEAD, Xid 45).
# Lambda sweep is ALREADY running on GPU 6,7 (launched manually); this script
# waits for it alongside the surviving Phase-1 shards, then runs stats and the
# selection + extension phases serially. Uses GPUs 1,4,5,6,7 only.
set -u
cd ~/zjt_ws/token-cd

PY=~/zjt_ws/openvla-ar-h100/bin/python
SCD=research/semantic_token_cd
WS=artifacts/attn_semantic_layer_window_sweep_v1
LS=artifacts/attn_semantic_lambda_sweep_v1
SS=artifacts/attn_semantic_selection_sweep_v1
PP=~/zjt_ws/token-cd:~/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PP"

SHARDS=("100-112" "113-124" "125-136" "137-148" "149-160" "161-172" "173-184" "185-199")

wait_done() {
  local files=$1 expected=$2
  while true; do
    local got crashed alive
    got=$(grep -ls '"DONE": true' $files 2>/dev/null | wc -l)
    crashed=$(grep -ls 'Traceback\|ErrorInitializationFailed\|OutOfMemoryError\|launch failure\|DeviceLost' $files 2>/dev/null | wc -l)
    alive=$(pgrep -f 'attn_semantic_.*sweep.py' 2>/dev/null | wc -l)
    if [ "$got" -ge "$expected" ]; then
      echo "[$(date +%T)] DONE $got/$expected"
      return 0
    fi
    echo "[$(date +%T)] wait $got/$expected DONE (crashed=$crashed alive=$alive)"
    sleep 60
  done
}

echo "[$(date +%T)] === Phase 1c: wait for move_near re-run (4) + close_drawer (8) + lambda (8) ==="
wait_done "$WS/logs/move_near_0.log $WS/logs/move_near_1.log $WS/logs/move_near_2.log $WS/logs/move_near_3.log $WS/logs/close_drawer_[0-7].log $LS/logs/lambda_move_near_[0-7].log" 20

echo "[$(date +%T)] === Phase 1 + lambda stats ==="
"$PY" "$SCD/analyze_layer_window_sweep.py" --artifact "$WS" | tee "$WS/RESULTS_$(date +%F_%H%M%S).txt"
"$PY" "$SCD/analyze_lambda_sweep.py" --artifact "$LS" | tee "$LS/RESULTS_$(date +%F_%H%M%S).txt"

echo "[$(date +%T)] === Phase 2: selection sweep (16 shards, GPU 4,5,6,7) ==="
for i in 0 1 2 3; do
  nohup env CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_selection_sweep.py" --artifact "$SS" --task google_robot_move_near --gpu 4 --seeds "${SHARDS[$i]}" \
    > "$SS/logs/sel_move_near_${i}.log" 2>&1 &
done
for i in 4 5 6 7; do
  nohup env CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_selection_sweep.py" --artifact "$SS" --task google_robot_move_near --gpu 5 --seeds "${SHARDS[$i]}" \
    > "$SS/logs/sel_move_near_${i}.log" 2>&1 &
done
for i in 0 1 2 3; do
  nohup env CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_selection_sweep.py" --artifact "$SS" --task google_robot_close_drawer --gpu 6 --seeds "${SHARDS[$i]}" \
    > "$SS/logs/sel_close_drawer_${i}.log" 2>&1 &
done
for i in 4 5 6 7; do
  nohup env CUDA_VISIBLE_DEVICES=7 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_selection_sweep.py" --artifact "$SS" --task google_robot_close_drawer --gpu 7 --seeds "${SHARDS[$i]}" \
    > "$SS/logs/sel_close_drawer_${i}.log" 2>&1 &
done
wait_done "$SS/logs/sel_*.log" 16
"$PY" "$SCD/analyze_selection_sweep.py" --artifact "$SS" | tee "$SS/RESULTS_$(date +%F_%H%M%S).txt"

echo "[$(date +%T)] === Phase 3: window sweep extension, open_drawer + stack_cube (16 shards, GPU 1,4,5,6) ==="
for i in 0 1 2 3; do
  nohup env CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_layer_window_sweep.py" --artifact "$WS" --task google_robot_open_drawer --gpu 1 --seeds "${SHARDS[$i]}" \
    > "$WS/logs/ext_open_drawer_${i}.log" 2>&1 &
done
for i in 4 5 6 7; do
  nohup env CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_layer_window_sweep.py" --artifact "$WS" --task google_robot_open_drawer --gpu 4 --seeds "${SHARDS[$i]}" \
    > "$WS/logs/ext_open_drawer_${i}.log" 2>&1 &
done
for i in 0 1 2 3; do
  nohup env CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_layer_window_sweep.py" --artifact "$WS" --task widowx_stack_cube --gpu 5 --seeds "${SHARDS[$i]}" \
    > "$WS/logs/ext_stack_cube_${i}.log" 2>&1 &
done
for i in 4 5 6 7; do
  nohup env CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_layer_window_sweep.py" --artifact "$WS" --task widowx_stack_cube --gpu 6 --seeds "${SHARDS[$i]}" \
    > "$WS/logs/ext_stack_cube_${i}.log" 2>&1 &
done
wait_done "$WS/logs/ext_*.log" 16
"$PY" "$SCD/analyze_layer_window_sweep.py" --artifact "$WS" | tee "$WS/RESULTS_FULL_$(date +%F_%H%M%S).txt"

echo "[$(date +%T)] ALL PHASES COMPLETE"
