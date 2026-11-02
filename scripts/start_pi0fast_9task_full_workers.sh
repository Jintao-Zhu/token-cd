#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD="$ROOT/token-cd"
OUT="$TOKEN_CD/runs/pi0fast_9task_vanilla_vs_l6jit_lambda050_rtfull_100seed_v1"
WORKER_LAUNCHER="$TOKEN_CD/scripts/run_pi0fast_9task_jit_compare_worker.sh"

python3 - "$OUT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
tasks = [
    "google_robot_open_drawer", "google_robot_close_drawer", "google_robot_pick_coke_can",
    "google_robot_move_near", "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel", "widowx_stack_cube",
]
for task in tasks:
    for seed in (0,):
        pair = root / "pairs" / task / f"seed{seed:03d}.json"
        if not pair.is_file():
            raise SystemExit(f"missing preflight pair: {pair}")
        row = json.loads(pair.read_text())
        if row.get("initial_rgb_coarse_equal") is not True or row.get("initial_state_equal") is not True:
            raise SystemExit(f"pair hash mismatch: {pair}")
        if "initial_rgb_lowres_mae" not in row or "initial_rgb_lowres_max_abs_diff" not in row:
            raise SystemExit(f"pair is from an older preflight format: {pair}")
        for arm in ("vanilla", "l6_jit_lambda050"):
            ep = root / "episodes" / task / f"{arm}_seed{seed:03d}.json"
            if not ep.is_file() or json.loads(ep.read_text()).get("status") != "complete":
                raise SystemExit(f"missing/incomplete preflight episode: {ep}")
        guided = json.loads((root / "episodes" / task / f"l6_jit_lambda050_seed{seed:03d}.json").read_text())
        if guided.get("guided_fallbacks", 0) > 0:
            print(f"NOTICE: {task}/{seed} used {guided['guided_fallbacks']} decoder fallback(s)")
print("Preflight passed: each of the 9 tasks has a complete, hash-matched seed-0 pair; full workers can resume all seeds 0-99.")
PY

for session in pi0f9_v05_pre_g4 pi0f9_v05_pre_g5 pi0f9_v05_pre_w6 pi0f9_v05_pre_w7; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Preflight worker $session is still active; wait for all preflight workers to exit before starting full shards." >&2
    exit 2
  fi
done

start_worker() {
  local gpu="$1" port="$2" family="$3" tag="$4"
  shift 4
  local session="pi0f9_v05_full_${tag}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing to replace existing tmux session $session." >&2
    exit 2
  fi
  local args=""
  for job in "$@"; do args+=" $job"; done
  tmux new-session -d -s "$session" "bash -lc '$WORKER_LAUNCHER $gpu $port $family $OUT${args}'"
}

start_worker 4 18841 google g4 \
  google_robot_open_drawer:0-49 google_robot_close_drawer:0-49 google_robot_pick_coke_can:0-49 \
  google_robot_move_near:0-49 google_robot_place_apple_in_closed_top_drawer:0-49
start_worker 5 18842 google g5 \
  google_robot_open_drawer:50-99 google_robot_close_drawer:50-99 google_robot_pick_coke_can:50-99 \
  google_robot_move_near:50-99 google_robot_place_apple_in_closed_top_drawer:50-99
start_worker 6 18843 widowx w6 \
  widowx_carrot_on_plate:0-99 widowx_put_eggplant_in_basket:0-49 \
  widowx_spoon_on_towel:0-25 widowx_stack_cube:0-24
start_worker 7 18844 widowx w7 \
  widowx_put_eggplant_in_basket:50-99 widowx_spoon_on_towel:26-99 widowx_stack_cube:25-99

echo "Full 9-task workers started. Each (task, seed, arm) appears once; preflight rows 0-1 are reused."
