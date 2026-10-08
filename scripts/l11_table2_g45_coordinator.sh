#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd
BASE="$ROOT/artifacts/l11_table2_main_ablation_v1"
while tmux has-session -t l11t2_pre_full_open 2>/dev/null || \
      tmux has-session -t l11t2_pre_full_pick 2>/dev/null || \
      tmux has-session -t l11t2_pre_kmeans_open 2>/dev/null || \
      tmux has-session -t l11t2_pre_kmeans_pick 2>/dev/null || \
      tmux has-session -t l11t2_pre_fixed 2>/dev/null || \
      tmux has-session -t l11t2_pre_shuffle 2>/dev/null; do
  sleep 15
done
python3 - "$BASE" <<'PY'
import json, pathlib, sys
base = pathlib.Path(sys.argv[1])
for arm in ("full", "kmeans_ranking", "fixed32", "shuffle_m"):
    root = base / "preflight" / arm
    paths = list(root.glob("episodes/*/*/episode_*_summary.json"))
    if len(paths) != 20:
        raise SystemExit(f"{arm}: expected 20 preflight summaries, found {len(paths)}")
    bad = [str(p) for p in paths if json.loads(p.read_text()).get("technical_pass") is not True]
    if bad:
        raise SystemExit(f"{arm}: preflight failed: {bad[:5]}")
    (root / "PREFLIGHT_PASS.json").write_text(json.dumps({
        "status": "PASS", "arm": arm, "episodes": 20,
        "tasks": 4, "seeds": [100, 101, 102, 103, 104]
    }, indent=2) + "\n")
    print(f"PREFLIGHT PASS {arm}: 20/20", flush=True)
PY
launch() {
  local gpu="$1" arm="$2" suffix="$3"; shift 3
  local log="$BASE/logs/g45_${suffix}.log"
  local task_args=""
  for task in "$@"; do task_args+=" '$task'"; done
  tmux new-session -d -s "l11t2_${suffix}" \
    "bash '$ROOT/scripts/run_l11_table2_formal_shard.sh' '$gpu' '$arm' 'l11t2_${suffix}'${task_args} > '$log' 2>&1"
  echo "STARTED formal shard l11t2_${suffix} gpu=$gpu arm=$arm tasks=$*" | tee -a "$BASE/logs/g45_coordinator.log"
}
# Balance each GPU at 400 drawer + 400 pick/move episodes (3 workers/GPU).
launch 4 full full_open_pick google_robot_open_drawer google_robot_pick_coke_can
launch 5 full full_close_move google_robot_close_drawer google_robot_move_near
launch 5 kmeans_ranking kmeans_open_pick google_robot_open_drawer google_robot_pick_coke_can
launch 4 kmeans_ranking kmeans_close_move google_robot_close_drawer google_robot_move_near
launch 4 fixed32 fixed_all google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near
launch 5 shuffle_m shuffle_all google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can google_robot_move_near
echo "ALL SIX FORMAL SHARDS STARTED" | tee -a "$BASE/logs/g45_coordinator.log"
