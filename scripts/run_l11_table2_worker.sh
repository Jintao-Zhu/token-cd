#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU CARD ARM WORKER_ID" >&2
  exit 2
fi
GPU=$1
CARD=$2
ARM=$3
WORKER=$4
ROOT=/home/leju-suzhou/zjt_ws/token-cd
SNAPSHOT="$ROOT/artifacts/vanilla_recon_shr_canonical_0_299_v2"
BASE="$ROOT/artifacts/l11_table2_main_ablation_v1"
source "$ROOT/scripts/activate_simpler_local.sh"
cd "$ROOT"
TASKS=(
  google_robot_open_drawer
  google_robot_close_drawer
  google_robot_pick_coke_can
  google_robot_move_near
)
run_tasks() {
  local seeds=$1 artifact=$2
  for task in "${TASKS[@]}"; do
    echo "[$(date -Is)] worker=$WORKER gpu=$GPU arm=$ARM task=$task seeds=$seeds artifact=$artifact"
    python research/semantic_token_cd/l11_table2_openvla_rollout.py \
      --card "$CARD" --artifact "$artifact" --snapshot-artifact "$SNAPSHOT" \
      --task "$task" --seeds "$seeds" --gpu "$GPU" --worker-id "$WORKER" \
      --arms "$ARM"
  done
}

if [[ "$ARM" == random_ranking || "$ARM" == geometry_ranking || "$CARD" == 6 ]]; then
  preflight="$BASE/preflight/$ARM"
  run_tasks 100-104 "$preflight"
  python3 - "$preflight" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
paths = list(root.glob("episodes/*/*/episode_*_summary.json"))
if len(paths) != 20:
    raise SystemExit(f"preflight expected 20 summaries (4 tasks x 5 seeds), found {len(paths)}")
bad = [str(p) for p in paths if json.loads(p.read_text()).get("technical_pass") is not True]
if bad:
    raise SystemExit("preflight technical audit failed: " + ", ".join(bad[:5]))
(root / "PREFLIGHT_PASS.json").write_text(json.dumps({
    "status": "PASS", "episodes": 20, "arm": root.name,
    "tasks": 4, "seeds": [100, 101, 102, 103, 104]
}, indent=2) + "\n")
print(f"PREFLIGHT PASS: {root.name} (20/20)")
PY
fi

run_tasks 100-199 "$BASE/formal/$ARM"
echo "[$(date -Is)] COMPLETE worker=$WORKER arm=$ARM"
