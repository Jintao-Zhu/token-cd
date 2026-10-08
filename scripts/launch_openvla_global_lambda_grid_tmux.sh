#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
ARTIFACT="$REPO/artifacts/openvla_global_lambda_grid_0_199_v1"
PREFLIGHT="$ARTIFACT/PREFLIGHT_45_SCENES.json"
WORKER="$REPO/scripts/run_openvla_global_lambda_grid_worker.sh"

if [[ ! -f "$PREFLIGHT" ]] || ! python3 - "$PREFLIGHT" <<'PY'
import json, sys
obj = json.load(open(sys.argv[1]))
raise SystemExit(0 if obj.get("all_checks_passed") is True and obj.get("n_scenes") == 45 else 1)
PY
then
  echo "45-scene preflight must pass before formal rollout" >&2
  exit 1
fi

if tmux ls 2>/dev/null | grep -q '^openvla_lgs_g[1-5]_'; then
  echo "Global lambda-grid tmux sessions already exist" >&2
  exit 1
fi

mkdir -p "$ARTIFACT/logs"
PLAN="$ARTIFACT/WORK_PLAN.tsv"
python3 - "$PLAN" <<'PY'
import json, sys
from pathlib import Path

# Historical mean rollout seconds per episode; split each task into four paired
# 50-seed chunks and greedily balance total worker-seconds across 15 workers.
means = {
    "google_robot_open_drawer": 161.7,
    "google_robot_close_drawer": 152.5,
    "google_robot_pick_coke_can": 37.5,
    "google_robot_move_near": 37.8,
    "google_robot_place_apple_in_closed_top_drawer": 82.0,
    "widowx_carrot_on_plate": 24.8,
    "widowx_put_eggplant_in_basket": 50.9,
    "widowx_spoon_on_towel": 24.3,
    "widowx_stack_cube": 24.7,
}
lambdas = 9
jobs = []
for task, seconds in means.items():
    for lo in range(0, 200, 50):
        hi = lo + 49
        weight = 50 * lambdas * seconds
        jobs.append((weight, task, f"{lo}-{hi}"))
jobs.sort(reverse=True)
bins = [[] for _ in range(15)]
loads = [0.0] * 15
for weight, task, seeds in jobs:
    i = min(range(15), key=lambda j: loads[j])
    bins[i].append((task, seeds))
    loads[i] += weight
path = Path(sys.argv[1])
with path.open("w") as f:
    for i, chunks in enumerate(bins):
        worker = f"openvla_lgs_g{i // 3 + 1}_w{i + 1:02d}"
        gpu = i // 3 + 1
        fields = [f"{task}={seeds}" for task, seeds in chunks]
        f.write("\t".join([worker, str(gpu), *fields]) + "\n")
manifest = {
    "protocol_id": "OPENVLA_GLOBAL_LAMBDA_GRID_0_199_V1",
    "artifact": "/home/leju-suzhou/zjt_ws/token-cd/artifacts/openvla_global_lambda_grid_0_199_v1",
    "gpus": [1, 2, 3, 4, 5], "workers_per_gpu": 3, "workers": 15,
    "tasks": list(means), "seeds": list(range(200)),
    "lambdas": [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2],
    "episode_arm_evaluations": 16200,
    "estimated_worker_hours_from_historical_task_means": [round(x / 3600, 2) for x in loads],
    "assignment": [{"worker": f"openvla_lgs_g{i // 3 + 1}_w{i + 1:02d}",
                    "gpu": i // 3 + 1,
                    "estimated_hours": round(loads[i] / 3600, 2),
                    "chunks": [{"task": t, "seeds": s} for t, s in bins[i]]}
                   for i in range(15)],
}
Path(str(path).replace(".tsv", ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
PY

started=0
while IFS=$'\t' read -r name gpu rest; do
  [[ -n "$name" ]] || continue
  args=("$gpu" "$name")
  for item in $rest; do
    args+=("${item%%=*}" "${item#*=}")
  done
  cmd=(bash "$WORKER" "${args[@]}")
  printf -v quoted '%q ' "${cmd[@]}"
  tmux new-session -d -s "$name" "cd '$REPO' && $quoted 2>&1 | tee '$ARTIFACT/logs/${name}.log'"
  echo "started $name on GPU $gpu: ${#args[@]} worker arguments"
  started=$((started + 1))
done < "$PLAN"

if [[ "$started" -ne 15 ]]; then
  echo "Expected 15 workers; started $started" >&2
  exit 1
fi
echo "Launched 15 workers (3 each on GPUs 1-5); stage 1 workload = 16,200 episode-arm evaluations."
