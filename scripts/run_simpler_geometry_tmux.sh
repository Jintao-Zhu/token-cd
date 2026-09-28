#!/usr/bin/env bash
# Launch one GPU worker per task in persistent tmux sessions inside Docker.
set -euo pipefail

MODE="${1:?usage: run_simpler_geometry_tmux.sh preflight|full}"
case "$MODE" in
  preflight) SEEDS="100-104" ;;
  full) SEEDS="100-199" ;;
  *) echo "mode must be preflight or full" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DOCKER_REPO="${SIMPLER_DOCKER_REPO_MIRROR:-/data/docker/dev_zjt/data/code/token-cd}"
CONTAINER="${LIBERO_DOCKER_CONTAINER:-dev_zjt_container}"
ARTIFACT="${SIMPLER_GEOMETRY_ARTIFACT:-/root/code/token-cd/artifacts/simpler_geometry_selector_4arm_v2}"
SNAPSHOTS="${SIMPLER_CANONICAL_ARTIFACT:-/root/code/token-cd/github_results/vanilla_recon_shr_canonical_0_299_v2}"
TASKS=(google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near)

"$SCRIPT_DIR/prepare_simpler_docker.sh"
mkdir -p "$DOCKER_REPO"
(cd "$REPO_ROOT" && tar --exclude='./.git' --exclude='*/__pycache__' --exclude='*.pyc' -cf - .) | \
  (cd "$DOCKER_REPO" && tar -xf -)

if [[ "$MODE" == "full" ]]; then
  docker exec "$CONTAINER" python3 -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["pass"] and p["structural_pass"] and p["visual_maps_reviewed"]' \
    "$ARTIFACT/PREFLIGHT_GATE.json"
fi

for task in "${TASKS[@]}"; do
  session="simpler_geo_${MODE}_${task##*google_robot_}"
  if docker exec "$CONTAINER" tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
  log="$ARTIFACT/logs/${MODE}_${task}.log"
  done_file="$ARTIFACT/logs/${MODE}_${task}.exit"
  worker="set +e; cd /root/code/token-cd; source scripts/activate_simpler_docker.sh; mkdir -p '$ARTIFACT/logs'; /root/code/task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/simpler_geometry_mask_rollout.py --artifact '$ARTIFACT' --snapshot-artifact '$SNAPSHOTS' --task '$task' --seeds '$SEEDS' --gpu 0 --worker-id '$session' --phase '$MODE' > '$log' 2>&1; code=\$?; echo \$code > '$done_file'; exit \$code"
  docker exec "$CONTAINER" tmux new-session -d -s "$session" "bash -lc '$worker'"
  echo "started tmux=$session task=$task seeds=$SEEDS log=$log"
done

echo "Attached worker list: docker exec $CONTAINER tmux list-sessions"
