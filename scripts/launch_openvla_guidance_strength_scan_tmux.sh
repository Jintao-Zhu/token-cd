#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
ARTIFACT="$REPO/artifacts/openvla_guidance_strength_scan_100_199_v1"

if tmux ls 2>/dev/null | grep -q '^openvla_gs_[g45]_'; then
  echo "OpenVLA guidance strength scan tmux sessions already exist" >&2
  exit 1
fi
mkdir -p "$ARTIFACT/logs"

start() {
  local gpu="$1" slot="$2"
  shift 2
  local name="openvla_gs_g${gpu}_s${slot}"
  local cmd=("$REPO/scripts/run_openvla_guidance_strength_worker.sh" "$gpu" "$name" "$@")
  printf -v quoted '%q ' "${cmd[@]}"
  tmux new-session -d -s "$name" "cd '$REPO' && $quoted 2>&1 | tee '$ARTIFACT/logs/${name}.log'"
  echo "started $name"
}

# Three independent workers per card. Existing exact C6 rows are linked into
# the scan artifact, so the rollout automatically skips completed seed/arm rows.
start 4 1 google_robot_open_drawer 100-199 widowx_carrot_on_plate 100-199
start 4 2 google_robot_close_drawer 100-199 widowx_put_eggplant_in_basket 100-199
start 4 3 google_robot_pick_coke_can 100-199 widowx_spoon_on_towel 100-199
start 5 1 google_robot_move_near 100-199 widowx_stack_cube 100-199
start 5 2 google_robot_place_apple_in_closed_top_drawer 100-149
start 5 3 google_robot_place_apple_in_closed_top_drawer 150-199

echo "Launched six workers on GPUs 4 and 5 for the OpenVLA lambda scan."
