#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/leju-suzhou/zjt_ws"
REPO="$ROOT/token-cd"
ARTIFACT="$REPO/artifacts/openvla_lambda1_extension_200_299_v1"
WORKER="$REPO/scripts/run_openvla_lambda1_extension_worker.sh"
mkdir -p "$ARTIFACT/logs"

if tmux ls 2>/dev/null | grep -q '^openvla_l1ext_'; then
  echo "OpenVLA lambda=1 extension tmux sessions already exist" >&2
  exit 1
fi

# One real seed from each task acts as preflight and remains part of the 900.
tasks=(google_robot_open_drawer google_robot_close_drawer google_robot_pick_coke_can
       google_robot_move_near google_robot_place_apple_in_closed_top_drawer
       widowx_carrot_on_plate widowx_put_eggplant_in_basket widowx_spoon_on_towel widowx_stack_cube)
for idx in 0 1 2; do
  gpu=$((idx + 1))
  name="openvla_l1ext_preflight_g${gpu}"
  args=("$gpu" "$name")
  for task_idx in $(seq "$idx" 3 8); do args+=("${tasks[$task_idx]}" "200"); done
  printf -v quoted '%q ' "$WORKER" "${args[@]}"
  tmux new-session -d -s "$name" "cd '$REPO' && bash $quoted 2>&1 | tee '$ARTIFACT/logs/${name}.log'"
  echo "started $name on GPU $gpu"
done

while tmux ls 2>/dev/null | grep -q '^openvla_l1ext_preflight_g'; do sleep 5; done
source "$REPO/scripts/activate_simpler_local.sh"
python "$REPO/research/semantic_token_cd/validate_openvla_lambda1_extension_preflight.py"

# Remaining 891 task-seed pairs, split into 10-seed chunks and balanced across
# 15 workers (three each on GPUs 1-5) using historical task runtimes.
python3 - "$ARTIFACT/WORK_PLAN_200_299.tsv" "$ARTIFACT/WORK_PLAN_200_299.json" <<'PY'
import json, sys
from pathlib import Path

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
jobs = []
for task, seconds in means.items():
    seeds = list(range(201, 300))
    for start in range(0, len(seeds), 10):
        chunk = seeds[start:start + 10]
        jobs.append((seconds * len(chunk), task, f"{chunk[0]}-{chunk[-1]}"))
jobs.sort(reverse=True)
bins = [[] for _ in range(15)]
loads = [0.0] * 15
for weight, task, seed_spec in jobs:
    i = min(range(15), key=lambda j: loads[j])
    bins[i].append((task, seed_spec))
    loads[i] += weight

tsv = Path(sys.argv[1])
with tsv.open("w") as f:
    for i, chunks in enumerate(bins):
        worker = f"openvla_l1ext_g{i // 3 + 1}_w{i % 3 + 1}"
        fields = [f"{task}={seed_spec}" for task, seed_spec in chunks]
        f.write("\t".join([worker, str(i // 3 + 1), *fields]) + "\n")
manifest = {
    "protocol_id": "OPENVLA_LAMBDA1_EXTENSION_200_299_V1",
    "artifact": "/home/leju-suzhou/zjt_ws/token-cd/artifacts/openvla_lambda1_extension_200_299_v1",
    "gpus": [1, 2, 3, 4, 5], "workers_per_gpu": 3, "workers": 15,
    "tasks": list(means), "seeds": list(range(200, 300)), "arm": "lambda_100",
    "total_episode_evaluations": 900, "preflight_included": 9,
    "remaining_episode_evaluations": 891,
    "estimated_worker_hours": [round(x / 3600, 2) for x in loads],
    "assignment": [{"worker": f"openvla_l1ext_g{i // 3 + 1}_w{i % 3 + 1}",
                    "gpu": i // 3 + 1, "estimated_hours": round(loads[i] / 3600, 2),
                    "chunks": [{"task": task, "seeds": seed_spec} for task, seed_spec in bins[i]]}
                   for i in range(15)],
}
Path(sys.argv[2]).write_text(json.dumps(manifest, indent=2) + "\n")
PY

started=0
while IFS=$'\t' read -r name gpu rest; do
  [[ -n "$name" ]] || continue
  args=("$gpu" "$name")
  for item in $rest; do args+=("${item%%=*}" "${item#*=}"); done
  printf -v quoted '%q ' "$WORKER" "${args[@]}"
  tmux new-session -d -s "$name" "cd '$REPO' && bash $quoted 2>&1 | tee '$ARTIFACT/logs/${name}.log'"
  echo "started $name on GPU $gpu"
  started=$((started + 1))
done < "$ARTIFACT/WORK_PLAN_200_299.tsv"
[[ "$started" -eq 15 ]]
echo "Launched lambda=1 extension: 9 tasks x seeds 200-299, 900 total episodes, GPUs 1-5 (3 workers/GPU)."
