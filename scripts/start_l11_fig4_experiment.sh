#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd
BASE="$ROOT/artifacts/l11_matched_dose_response_fig4_v1"
mkdir -p "$BASE/logs"
if tmux has-session -t fig4_coordinator 2>/dev/null; then
  echo "fig4_coordinator already exists" >&2; exit 1
fi
for i in $(seq 1 15); do
  if tmux has-session -t "fig4_pre_$i" 2>/dev/null; then
    echo "fig4_pre_$i already exists" >&2; exit 1
  fi
done
launch() {
  local id="$1" gpu="$2"; shift 2
  local name="fig4_pre_${id}"
  local args=("$ROOT/scripts/run_l11_fig4_worker.sh" preflight "$gpu" "$name" "$@")
  local command="bash"
  for arg in "${args[@]}"; do command+=" '$arg'"; done
  tmux new-session -d -s "$name" "$command > '$BASE/logs/$name.log' 2>&1"
  echo "STARTED $name on GPU$gpu: $*"
}
# 16 arm-task smoke shards on 15 workers; the final worker handles both short tasks.
launch 1 1 dose_0p50 google_robot_open_drawer 100-104
launch 2 1 dose_0p50 google_robot_pick_coke_can 100-104
launch 3 1 dose_0p75 google_robot_close_drawer 100-104
launch 4 2 dose_0p50 google_robot_close_drawer 100-104
launch 5 2 dose_0p75 google_robot_open_drawer 100-104
launch 6 2 dose_0p75 google_robot_move_near 100-104
launch 7 3 dose_0p50 google_robot_move_near 100-104
launch 8 3 dose_0p75 google_robot_pick_coke_can 100-104
launch 9 3 dose_1p25 google_robot_open_drawer 100-104
launch 10 4 dose_1p25 google_robot_close_drawer 100-104
launch 11 4 dose_1p25 google_robot_pick_coke_can 100-104
launch 12 4 dose_1p25 google_robot_move_near 100-104
launch 13 5 dose_1p50 google_robot_open_drawer 100-104
launch 14 5 dose_1p50 google_robot_close_drawer 100-104
launch 15 5 dose_1p50 google_robot_pick_coke_can 100-104 dose_1p50 google_robot_move_near 100-104
tmux new-session -d -s fig4_coordinator "bash '$ROOT/scripts/l11_fig4_coordinator.sh' > '$BASE/logs/coordinator.log' 2>&1"
echo "Started 15 preflight workers (3 per GPU 1-5) and coordinator."
