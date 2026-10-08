#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "preflight" && "$MODE" != "formal" ]]; then
  echo "usage: $0 preflight|formal" >&2
  exit 2
fi

REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$MODE" == "preflight" ]]; then
  ARTIFACT="$REPO/artifacts/openvla_ood_robustness_v3_preflight"
  TASKS=(
    google_robot_open_drawer google_robot_open_drawer google_robot_open_drawer google_robot_open_drawer
    google_robot_close_drawer google_robot_close_drawer google_robot_close_drawer google_robot_close_drawer
    google_robot_pick_coke_can google_robot_pick_coke_can google_robot_pick_coke_can google_robot_pick_coke_can
    google_robot_move_near google_robot_move_near google_robot_move_near
  )
  SEEDS=(200 201 202 203-204 200 201 202 203-204 200 201 202 203-204 200-201 202-203 204)
  EXTRA=(--smoke-steps 1 --save-audit-images)
else
  ARTIFACT="$REPO/artifacts/openvla_ood_robustness_v3"
  TASKS=(
    google_robot_open_drawer google_robot_open_drawer google_robot_open_drawer google_robot_open_drawer
    google_robot_close_drawer google_robot_close_drawer google_robot_close_drawer google_robot_close_drawer
    google_robot_pick_coke_can google_robot_pick_coke_can google_robot_pick_coke_can google_robot_pick_coke_can
    google_robot_move_near google_robot_move_near google_robot_move_near
  )
  SEEDS=(200-224 225-249 250-274 275-299 200-224 225-249 250-274 275-299 200-224 225-249 250-274 275-299 200-233 234-266 267-299)
  EXTRA=()
fi

mkdir -p "$ARTIFACT/logs"
source "$REPO/scripts/activate_simpler_local.sh"
"$SIMPLER_ENV/bin/python" - "$ARTIFACT/assets/arena_ood_texture.png" <<'PY'
import sys
from research.semantic_token_cd.ood_visual_shift_utils import create_arena_texture
create_arena_texture(sys.argv[1])
PY
GPUS=(1 2 3 4 5)
for i in "${!TASKS[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  worker="${MODE}_$(printf '%02d' "$i")_gpu${gpu}"
  session="ood_${MODE}_$(printf '%02d' "$i")"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "already running: $session"
    continue
  fi
  printf -v cmd 'source scripts/activate_simpler_local.sh && export CUDA_VISIBLE_DEVICES=%q && exec "$SIMPLER_ENV/bin/python" research/semantic_token_cd/openvla_ood_robustness_rollout.py --artifact %q --task %q --seeds %q --gpu %q --worker-id %q' \
    "$gpu" "$ARTIFACT" "${TASKS[$i]}" "${SEEDS[$i]}" "$gpu" "$worker"
  if [[ "$MODE" == "preflight" ]]; then
    cmd+=" --smoke-steps 1 --save-audit-images"
  fi
  cmd+=" > $(printf '%q' "$ARTIFACT/logs/${worker}.log") 2>&1"
  tmux new-session -d -s "$session" -c "$REPO" "bash -lc $(printf '%q' "$cmd")"
  printf '%-24s gpu=%s task=%s seeds=%s\n' "$session" "$gpu" "${TASKS[$i]}" "${SEEDS[$i]}"
done

echo "Started ${#TASKS[@]} $MODE workers. GPU 0 is not assigned."
