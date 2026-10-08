#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd
ARTIFACT=$ROOT/artifacts/openvla_global_lambda_grid_225_250_0_199_v1
WORKER=$ROOT/scripts/run_openvla_lambda225_250_worker.sh
mkdir -p "$ARTIFACT/logs"
for s in lambda225_g4_w1 lambda225_g4_w2 lambda225_g4_w3 lambda225_g5_w1 lambda225_g5_w2 lambda225_g5_w3; do
  if tmux has-session -t "$s" 2>/dev/null; then echo "session already exists: $s" >&2; exit 1; fi
done
launch() {
  local gpu=$1 name=$2; shift 2
  local cmd=(bash "$WORKER" "$gpu" "$name" preflight "$@")
  printf -v quoted '%q ' "${cmd[@]}"
  tmux new-session -d -s "$name" "cd '$ROOT' && $quoted 2>&1 | tee '$ARTIFACT/logs/${name}.log'"
  echo "started $name on GPU $gpu"
}
launch 4 lambda225_g4_w1 google_robot_open_drawer 0-4 google_robot_move_near 0-4
launch 4 lambda225_g4_w2 google_robot_close_drawer 0-4 google_robot_pick_coke_can 0-4
launch 4 lambda225_g4_w3 google_robot_place_apple_in_closed_top_drawer 0-4
launch 5 lambda225_g5_w1 widowx_carrot_on_plate 0-4
launch 5 lambda225_g5_w2 widowx_put_eggplant_in_basket 0-4
launch 5 lambda225_g5_w3 widowx_spoon_on_towel 0-4 widowx_stack_cube 0-4
echo "Started 6 preflight workers (3 each on GPUs 4 and 5); 9 tasks x 5 seeds x 2 lambdas = 90 evaluations."
