#!/usr/bin/env bash
# Launch one GPU worker per task in persistent local tmux sessions.
set -euo pipefail

MODE="${1:?usage: run_simpler_geometry_tmux.sh preflight|full}"
case "$MODE" in
  preflight) SEEDS="100-104" ;;
  full) SEEDS="100-199" ;;
  *) echo "mode must be preflight or full" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ARTIFACT="${SIMPLER_GEOMETRY_ARTIFACT:-$REPO_ROOT/artifacts/simpler_geometry_selector_4arm_v2}"
SNAPSHOTS="${SIMPLER_CANONICAL_ARTIFACT:-$REPO_ROOT/github_results/vanilla_recon_shr_canonical_0_299_v2}"
PYTHON="${SIMPLER_PYTHON:-$REPO_ROOT/../openvla-ar-h100/bin/python}"
GPUS_CSV="${SIMPLER_GPUS:-1,2,3}"
TASKS=(google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near)

source "$SCRIPT_DIR/activate_simpler_local.sh"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "SIMPLER_GPUS must contain at least one physical GPU id" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if [[ "$gpu" != "1" && "$gpu" != "2" && "$gpu" != "3" ]]; then
    echo "This launcher is restricted to the previously authorized GPUs 1,2,3; invalid id: $gpu" >&2
    exit 2
  fi
done
if [[ ! -x "$PYTHON" ]]; then
  echo "SIMPLER Python environment is missing: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$SNAPSHOTS/SNAPSHOT_MANIFEST.jsonl" ]]; then
  echo "Canonical snapshot manifest is missing: $SNAPSHOTS/SNAPSHOT_MANIFEST.jsonl" >&2
  exit 1
fi
mkdir -p "$ARTIFACT/logs"

if [[ "$MODE" == "full" ]]; then
  "$PYTHON" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["pass"] and p["structural_pass"] and p["visual_maps_reviewed"]' \
    "$ARTIFACT/PREFLIGHT_GATE.json"
fi

for task_index in "${!TASKS[@]}"; do
  task="${TASKS[$task_index]}"
  gpu="${GPUS[$((task_index % ${#GPUS[@]}))]}"
  session="simpler_geo_${MODE}_${task##*google_robot_}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
  log="$ARTIFACT/logs/${MODE}_${task}.log"
  done_file="$ARTIFACT/logs/${MODE}_${task}.exit"
  worker="set +e; cd '$REPO_ROOT'; source '$SCRIPT_DIR/activate_simpler_local.sh'; mkdir -p '$ARTIFACT/logs'; '$PYTHON' research/semantic_token_cd/simpler_geometry_mask_rollout.py --artifact '$ARTIFACT' --snapshot-artifact '$SNAPSHOTS' --task '$task' --seeds '$SEEDS' --gpu '$gpu' --worker-id '$session' --phase '$MODE' > '$log' 2>&1; code=\$?; echo \$code > '$done_file'; exit \$code"
  printf -v worker_quoted '%q' "$worker"
  tmux new-session -d -s "$session" "bash -lc $worker_quoted"
  echo "started tmux=$session task=$task gpu=$gpu seeds=$SEEDS log=$log"
done

echo "Attached worker list: tmux list-sessions"
