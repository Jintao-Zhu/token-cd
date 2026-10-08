#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
ARTIFACT="$REPO/artifacts/univla_nine_task_hidden_cd_rtdepth0_extension_v1"
PROTO="UNIVLA_NINE_TASK_HIDDEN_CD_RT_DEPTH0_EXTENSION_V1"
mkdir -p "$ARTIFACT/logs" "$ARTIFACT/worker_manifests" "$ARTIFACT/locks"

if tmux ls 2>/dev/null | grep -q '^univla_n9d0_'; then
  echo "A worker session with prefix univla_n9d0_ already exists" >&2
  exit 1
fi

launch_google() {
  local gpu="$1" task="$2" seeds="$3" wid="$4"
  tmux new-session -d -s "$wid" \
    "cd '$REPO' && env UNIVLA_ARTIFACT='$ARTIFACT' UNIVLA_PROTOCOL_ID='$PROTO' bash '$REPO/scripts/run_univla_ft30k_depth0_paired_fill_worker.sh' '$gpu' '$task' '$seeds' '$wid'"
  echo "started $wid gpu=$gpu task=$task seeds=$seeds arms=vanilla,hidden_cd_025,hidden_cd_05 checkpoint=Fractal-30k renderer=RT-depth0"
}

launch_widowx() {
  local gpu="$1" task="$2" seeds="$3" arms="$4" wid="$5"
  tmux new-session -d -s "$wid" \
    "cd '$REPO' && env UNIVLA_ARTIFACT='$ARTIFACT' UNIVLA_PROTOCOL_ID='$PROTO' bash '$REPO/scripts/run_univla_hidden_cd_depth0_three_arm_worker.sh' '$gpu' '$task' '$seeds' '$arms' '$wid'"
  echo "started $wid gpu=$gpu task=$task seeds=$seeds arms=$arms checkpoint=Bridge renderer=RT-depth0"
}

# The fixed allocation is 3 sessions on each physical GPU 1-7; GPU 0 is excluded.
launch_google 1 google_robot_pick_coke_can 100-199 univla_n9d0_g1_pick_100_199
launch_google 1 google_robot_close_drawer 200-299 univla_n9d0_g1_close_200_299
launch_widowx 1 widowx_carrot_on_plate 0-99 hidden_cd_025 univla_n9d0_g1_carrot_h25_000_099
sleep 4

launch_google 2 google_robot_pick_coke_can 200-299 univla_n9d0_g2_pick_200_299
launch_google 2 google_robot_place_apple_in_closed_top_drawer 100-166 univla_n9d0_g2_apple_100_166
launch_widowx 2 widowx_put_eggplant_in_basket 0-99 hidden_cd_025 univla_n9d0_g2_eggplant_h25_000_099
sleep 4

launch_google 3 google_robot_move_near 100-199 univla_n9d0_g3_move_100_199
launch_google 3 google_robot_place_apple_in_closed_top_drawer 167-233 univla_n9d0_g3_apple_167_233
launch_widowx 3 widowx_spoon_on_towel 0-99 hidden_cd_025 univla_n9d0_g3_spoon_h25_000_099
sleep 4

launch_google 4 google_robot_move_near 200-299 univla_n9d0_g4_move_200_299
launch_google 4 google_robot_place_apple_in_closed_top_drawer 234-299 univla_n9d0_g4_apple_234_299
launch_widowx 4 widowx_stack_cube 0-99 hidden_cd_025 univla_n9d0_g4_stack_h25_000_099
sleep 4

launch_google 5 google_robot_open_drawer 100-199 univla_n9d0_g5_open_100_199
launch_widowx 5 widowx_carrot_on_plate 100-199 vanilla,hidden_cd_025,hidden_cd_05 univla_n9d0_g5_carrot_100_199
launch_widowx 5 widowx_carrot_on_plate 200-299 vanilla,hidden_cd_025,hidden_cd_05 univla_n9d0_g5_carrot_200_299
sleep 4

launch_google 6 google_robot_open_drawer 200-299 univla_n9d0_g6_open_200_299
launch_widowx 6 widowx_put_eggplant_in_basket 100-199 vanilla,hidden_cd_025,hidden_cd_05 univla_n9d0_g6_eggplant_100_199
launch_widowx 6 widowx_put_eggplant_in_basket 200-299 vanilla,hidden_cd_025,hidden_cd_05 univla_n9d0_g6_eggplant_200_299
sleep 4

launch_google 7 google_robot_close_drawer 100-199 univla_n9d0_g7_close_100_199
launch_widowx 7 widowx_spoon_on_towel 100-299 vanilla,hidden_cd_025,hidden_cd_05 univla_n9d0_g7_spoon_100_299
launch_widowx 7 widowx_stack_cube 100-299 vanilla,hidden_cd_025,hidden_cd_05 univla_n9d0_g7_stack_100_299

echo "All 21 worker sessions launched. GPU 0 was not used. Artifact: $ARTIFACT"
