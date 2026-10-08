#!/usr/bin/env bash
# Resume the locked SIMPLER geometry study with nine non-overlapping shards.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ARTIFACT="${SIMPLER_GEOMETRY_ARTIFACT:-$REPO_ROOT/artifacts/simpler_geometry_selector_4arm_v2_deterministic}"
SNAPSHOTS="${SIMPLER_CANONICAL_ARTIFACT:-$REPO_ROOT/github_results/vanilla_recon_shr_canonical_0_299_v2}"
PYTHON="${SIMPLER_PYTHON:-$REPO_ROOT/../openvla-ar-h100/bin/python}"

source "$SCRIPT_DIR/activate_simpler_local.sh"
if [[ ! -x "$PYTHON" ]]; then
  echo "SIMPLER Python environment is missing: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$SNAPSHOTS/SNAPSHOT_MANIFEST.jsonl" ]]; then
  echo "Canonical snapshot manifest is missing: $SNAPSHOTS/SNAPSHOT_MANIFEST.jsonl" >&2
  exit 1
fi
"$PYTHON" -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["pass"] and p["structural_pass"] and p["visual_maps_reviewed"]' \
  "$ARTIFACT/PREFLIGHT_GATE.json"
mkdir -p "$ARTIFACT/logs"

# Every seed-task pair belongs to exactly one worker. Existing complete pairs
# are skipped, and any partially saved pair is resumed by its unique shard.
declare -a SHARDS=(
  "1|google_robot_open_drawer|105-152|open_a"
  "2|google_robot_open_drawer|153-199|open_b"
  "2|google_robot_close_drawer|105-152|close_a"
  "3|google_robot_close_drawer|153-199|close_b"
  "1|google_robot_pick_coke_can|105-152|pick_a"
  "3|google_robot_pick_coke_can|153-199|pick_b"
  "1|google_robot_move_near|105-136|move_a"
  "2|google_robot_move_near|137-167|move_b"
  "3|google_robot_move_near|168-199|move_c"
)

for shard in "${SHARDS[@]}"; do
  IFS='|' read -r gpu task seeds suffix <<< "$shard"
  session="simpler_geo_full9_${suffix}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $session" >&2
    exit 1
  fi
done

for shard in "${SHARDS[@]}"; do
  IFS='|' read -r gpu task seeds suffix <<< "$shard"
  session="simpler_geo_full9_${suffix}"
  log="$ARTIFACT/logs/full9_${suffix}.log"
  done_file="$ARTIFACT/logs/full9_${suffix}.exit"
  worker="set +e; cd '$REPO_ROOT'; source '$SCRIPT_DIR/activate_simpler_local.sh'; '$PYTHON' research/semantic_token_cd/simpler_geometry_mask_rollout.py --artifact '$ARTIFACT' --snapshot-artifact '$SNAPSHOTS' --task '$task' --seeds '$seeds' --gpu '$gpu' --worker-id '$session' --phase full > '$log' 2>&1; code=\$?; echo \$code > '$done_file'; exit \$code"
  printf -v worker_quoted '%q' "$worker"
  tmux new-session -d -s "$session" "bash -lc $worker_quoted"
  echo "started tmux=$session task=$task gpu=$gpu seeds=$seeds log=$log"
done

echo "Nine workers started: three each on physical GPUs 1, 2, and 3."
