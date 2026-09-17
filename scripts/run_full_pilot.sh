#!/usr/bin/env bash
# Pilot inside tmux: wait for capacity test + smoke to finish, then run the
# full XSWAP-V1 coordinator on GPUs 2,3 with the chosen workers-per-gpu.
set -u
cd /home/leju-suzhou/zjt_ws/token-cd
export PYTHONPATH=/home/leju-suzhou/zjt_ws/token-cd:/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=/home/leju-suzhou/zjt_ws/openvla-ar-h100/bin/python
LOG=logs/xswap

echo "[pilot] waiting for capacity test..."
for i in $(seq 1 120); do
  grep -q CAPACITY_TEST_ALL_DONE "$LOG/capacity_staged.log" 2>/dev/null && break
  sleep 10
done
echo "[pilot] waiting for open_drawer smoke..."
for i in $(seq 1 90); do
  grep -q SMOKE_V3_DONE "$LOG/smoke_opendrawer_v3.log" 2>/dev/null && break
  sleep 10
done
sleep 20   # let any stragglers flush output

if [ -f "$LOG/workers_per_gpu.env" ]; then
  WP=$(cat "$LOG/workers_per_gpu.env")
else
  WP=${XSWAP_WORKERS_PER_GPU:-2}
fi
echo "[pilot] starting full coordinator with workers_per_gpu=$WP at $(date '+%F %T')"
setsid "$PY" research/semantic_token_cd/xswap_coordinator.py --workers-per-gpu "$WP" \
  >> "$LOG/coordinator_main.log" 2>&1 &
COORD_PID=$!
echo "[pilot] coordinator pid=$COORD_PID"
# Keep the pilot alive so tmux shows the coordinator session; if the
# coordinator exits unexpectedly, log and exit.
wait $COORD_PID
echo "[pilot] coordinator exited rc=$? at $(date '+%F %T')"
tail -3 "$LOG/coordinator_main.log"
