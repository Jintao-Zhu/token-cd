#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
ARTIFACT="$REPO/artifacts/univla_hidden_cd_lambda_035_rtdepth0_0_99_v1"
PROTOCOL="UNIVLA_NINE_TASK_HIDDEN_CD_LAMBDA_035_RT_DEPTH0_V1"
mkdir -p "$ARTIFACT/logs" "$ARTIFACT/worker_manifests" "$ARTIFACT/locks" "$ARTIFACT/pairs" "$ARTIFACT/episodes"

if tmux ls 2>/dev/null | grep -q '^univla_hcd035_'; then
  echo "A lambda-0.35 worker session with prefix univla_hcd035_ already exists" >&2
  exit 1
fi

for spec in \
  "1|google_robot_pick_coke_can google_robot_open_drawer widowx_carrot_on_plate" \
  "2|google_robot_move_near google_robot_close_drawer widowx_put_eggplant_in_basket" \
  "3|google_robot_place_apple_in_closed_top_drawer widowx_spoon_on_towel widowx_stack_cube"; do
  gpu="${spec%%|*}"
  tasks="${spec#*|}"
  # shellcheck disable=SC2086
  tmux new-session -d -s "univla_hcd035_g${gpu}" \
    "cd '$REPO' && bash '$REPO/scripts/run_univla_hidden_cd_lambda_035_queue_worker.sh' '$gpu' $tasks"
  echo "started queued lambda-0.35 worker on GPU $gpu for: $tasks"
done

cat > "$ARTIFACT/RUN_CONFIG.json" <<EOF
{
  "protocol_id": "$PROTOCOL",
  "status": "QUEUED_THEN_RUNNING",
  "tasks": {
    "google_robot_pick_coke_can": "Fractal Google FT-30k",
    "google_robot_move_near": "Fractal Google FT-30k",
    "google_robot_open_drawer": "Fractal Google FT-30k",
    "google_robot_close_drawer": "Fractal Google FT-30k",
    "google_robot_place_apple_in_closed_top_drawer": "Fractal Google FT-30k",
    "widowx_carrot_on_plate": "Bridge checkpoint",
    "widowx_put_eggplant_in_basket": "Bridge checkpoint",
    "widowx_spoon_on_towel": "Bridge checkpoint",
    "widowx_stack_cube": "Bridge checkpoint"
  },
  "seeds": "0-99 per task",
  "arm": {"hidden_cd_035": 0.35},
  "guidance": "H*=H+ + 0.35*(H+ - H-); L14 selector, dynamic Matched budget, harmonic negative, ACT1-4 ActionDecoder hidden guidance",
  "renderer": {"shader_dir": "rt", "path_depth": 0, "spp": 32, "denoiser": true, "offscreen_only": true},
  "episode_arm_evaluations": 900,
  "reference_artifact": "artifacts/univla_hidden_cd_lambda_010_015_rtdepth0_0_99_v1",
  "pairing": "Same task and seed are the comparison unit. Initial state/RGB hashes are recorded diagnostically but do not block rollout; WidowX reuses the source artifact's persisted canonical simulator snapshots.",
  "gpu_allocation": {"GPU1": "one queued worker", "GPU2": "one queued worker", "GPU3": "one queued worker", "GPU0": "untouched", "GPU4-7": "untouched"}
}
EOF

echo "Launched 3 queued workers; 900 lambda-0.35 episode evaluations. They wait for exact reference pairs and only use GPUs 1-3. Artifact: $ARTIFACT"
