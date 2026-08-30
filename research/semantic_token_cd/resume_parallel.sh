#!/usr/bin/env bash
# Parallel resume of selection + extension phases, using ALL live GPUs (1,4,5,6,7).
# GPU 0 is DEAD (Xid 45). Replaces rerun_remaining.sh, which was stuck waiting
# serially for close_drawer phase-1 while GPUs 1/6/7 sat idle.
#
# Immediately dispatches 12 selection shards to the currently-idle GPUs 1/6/7,
# then fills the remaining 4 selection close_drawer shards onto GPUs 4/5 as soon
# as the phase-1 close_drawer shards finish, then runs selection stats and the
# Phase-3 extension (open_drawer + stack_cube) on all 5 GPUs.
set -u
cd ~/zjt_ws/token-cd

PY=~/zjt_ws/openvla-ar-h100/bin/python
SCD=research/semantic_token_cd
SS=artifacts/attn_semantic_selection_sweep_v1
WS=artifacts/attn_semantic_layer_window_sweep_v1
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

mkdir -p "$SS/logs" "$WS/logs"

echo "[$(date +%T)] === dispatch selection move_near 0-3 -> GPU1 ==="
for i in 0 1 2 3; do
  nohup env CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_selection_sweep.py" --artifact "$SS" --task google_robot_move_near --gpu 1 --seeds "${SHARDS[$i]}" \
    > "$SS/logs/sel_move_near_${i}.log" 2>&1 &
done

echo "[$(date +%T)] === dispatch selection move_near 4-7 -> GPU6 ==="
for i in 4 5 6 7; do
  nohup env CUDA_VISIBLE_DEVICES=6 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_selection_sweep.py" --artifact "$SS" --task google_robot_move_near --gpu 6 --seeds "${SHARDS[$i]}" \
    > "$SS/logs/sel_move_near_${i}.log" 2>&1 &
done

echo "[$(date +%T)] === dispatch selection close_drawer 0-3 -> GPU7 ==="
for i in 0 1 2 3; do
  nohup env CUDA_VISIBLE_DEVICES=7 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_selection_sweep.py" --artifact "$SS" --task google_robot_close_drawer --gpu 7 --seeds "${SHARDS[$i]}" \
    > "$SS/logs/sel_close_drawer_${i}.log" 2>&1 &
done

echo "[$(date +%T)] === wait phase-1 close_drawer (8) to free GPU 4,5 ==="
wait_done "$WS/logs/close_drawer_[0-7].log" 8

echo "[$(date +%T)] === dispatch selection close_drawer 4-7 -> GPU4/5 ==="
for i in 4 5; do
  nohup env CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_selection_sweep.py" --artifact "$SS" --task google_robot_close_drawer --gpu 4 --seeds "${SHARDS[$i]}" \
    > "$SS/logs/sel_close_drawer_${i}.log" 2>&1 &
done
for i in 6 7; do
  nohup env CUDA_VISIBLE_DEVICES=5 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_selection_sweep.py" --artifact "$SS" --task google_robot_close_drawer --gpu 5 --seeds "${SHARDS[$i]}" \
    > "$SS/logs/sel_close_drawer_${i}.log" 2>&1 &
done

echo "[$(date +%T)] === wait all 16 selection DONE ==="
wait_done "$SS/logs/sel_*.log" 16

echo "[$(date +%T)] === selection stats ==="
"$PY" "$SCD/analyze_selection_sweep.py" --artifact "$SS" | tee "$SS/RESULTS_$(date +%F_%H%M%S).txt"

echo "[$(date +%T)] === Phase 3 extension: open_drawer + stack_cube (16 shards, GPU 1/4/5/6/7) ==="
for i in 0 1 2; do
  nohup env CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_layer_window_sweep.py" --artifact "$WS" --task google_robot_open_drawer --gpu 1 --seeds "${SHARDS[$i]}" \
    > "$WS/logs/ext_open_drawer_${i}.log" 2>&1 &
done
for i in 3 4 5; do
  nohup env CUDA_VISIBLE_DEVICES=4 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_layer_window_sweep.py" --artifact "$WS" --task google_robot_open_drawer --gpu 4 --seeds "${SHARDS[$i]}" \
    > "$WS/logs/ext_open_drawer_${i}.log" 2>&1 &
done
for i in 6 7; do
  nohup env CUDA_VISIBLE_DEVICES=7 PYTHONPATH="$PP" "$PY" \
    "$SCD/attn_semantic_layer_window_sweep.py" --artifact "$WS" --task google_robot_open_drawer --gpu 7 --seeds "${SHARDS[$i]}" \
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
echo "[$(date +%T)] === full stats ==="
"$PY" "$SCD/analyze_layer_window_sweep.py" --artifact "$WS" | tee "$WS/RESULTS_FULL_$(date +%F_%H%M%S).txt"
echo "[$(date +%T)] ALL PHASES COMPLETE"
