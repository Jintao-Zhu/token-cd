#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
ARTIFACT="$REPO/artifacts/univla_hidden_cd_lambda_010_015_rtdepth0_0_99_v1"
PROTOCOL="UNIVLA_NINE_TASK_HIDDEN_CD_LAMBDA_010_015_RT_DEPTH0_V1"
mkdir -p "$ARTIFACT/logs" "$ARTIFACT/worker_manifests" "$ARTIFACT/locks" "$ARTIFACT/pairs" "$ARTIFACT/snapshots"

if tmux ls 2>/dev/null | grep -q '^univla_hcdlo_'; then
  echo "A worker session with prefix univla_hcdlo_ already exists" >&2
  exit 1
fi

launch_google() {
  local gpu="$1" task="$2" worker="$3"
  tmux new-session -d -s "$worker" \
    "cd '$REPO' && env UNIVLA_ARTIFACT='$ARTIFACT' UNIVLA_PROTOCOL_ID='$PROTOCOL' UNIVLA_ARMS='hidden_cd_010,hidden_cd_015' bash '$REPO/scripts/run_univla_ft30k_depth0_paired_fill_worker.sh' '$gpu' '$task' '0-99' '$worker'"
  echo "started $worker gpu=$gpu task=$task seeds=0-99 arms=hidden_cd_010,hidden_cd_015 checkpoint=Fractal-30k renderer=RT-depth0"
}

launch_widowx() {
  local gpu="$1" task="$2" worker="$3"
  tmux new-session -d -s "$worker" \
    "cd '$REPO' && env UNIVLA_ARTIFACT='$ARTIFACT' UNIVLA_PROTOCOL_ID='$PROTOCOL' bash '$REPO/scripts/run_univla_hidden_cd_depth0_three_arm_worker.sh' '$gpu' '$task' '0-99' 'hidden_cd_010,hidden_cd_015' '$worker'"
  echo "started $worker gpu=$gpu task=$task seeds=0-99 arms=hidden_cd_010,hidden_cd_015 checkpoint=Bridge renderer=RT-depth0"
}

# Exactly three independent model/render workers on each physical GPU 1-3.
launch_google 1 google_robot_pick_coke_can univla_hcdlo_g1_pick
launch_google 1 google_robot_open_drawer univla_hcdlo_g1_open
launch_widowx 1 widowx_carrot_on_plate univla_hcdlo_g1_carrot

launch_google 2 google_robot_move_near univla_hcdlo_g2_move
launch_google 2 google_robot_close_drawer univla_hcdlo_g2_close
launch_widowx 2 widowx_put_eggplant_in_basket univla_hcdlo_g2_eggplant

launch_google 3 google_robot_place_apple_in_closed_top_drawer univla_hcdlo_g3_apple
launch_widowx 3 widowx_spoon_on_towel univla_hcdlo_g3_spoon
launch_widowx 3 widowx_stack_cube univla_hcdlo_g3_stack

echo "All 9 workers launched on GPUs 1-3 only. Target: 9 tasks x 100 seeds x 2 arms = 1800 episode-arm evaluations. Artifact: $ARTIFACT"
