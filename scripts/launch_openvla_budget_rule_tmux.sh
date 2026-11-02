#!/usr/bin/env bash
# Launch the OpenVLA budget-rule study: 12 tmux workers (4 per GPU on GPU 1/2/3) driven by
# artifacts/openvla_budget_rule_9task_200_299_v1/WORK_PLAN.json.
#
# GPU policy: GPU 1/2/3 only -- never 0, never 4-7 (zxr's jobs live there).
#
# Usage: bash launch_openvla_budget_rule_tmux.sh [--dry-run]
set -uo pipefail

ROOT=/home/leju-suzhou/zjt_ws
REPO=$ROOT/token-cd
ARTIFACT="$REPO/artifacts/openvla_budget_rule_9task_200_299_v1"
PLAN="$ARTIFACT/WORK_PLAN.json"
WORKER="$REPO/scripts/run_openvla_budget_rule_worker.sh"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

[[ -f "$PLAN" ]] || { echo "missing $PLAN (run scripts/plan_openvla_budget_rule_run.py --write)" >&2; exit 2; }
mkdir -p "$ARTIFACT/logs"

python3 - "$PLAN" <<'PY' > /tmp/br_launch_cmds.txt
import json, sys, shlex
plan = json.load(open(sys.argv[1]))
for w in plan["assignment"]:
    specs = [f"{c['task']}:{c['arm']}:{c['seeds']}" for c in w["chunks"]]
    print(shlex.join([w["worker"], str(w["gpu"])] + specs))
PY

n=0
while read -r line; do
  [[ -z "$line" ]] && continue
  n=$((n + 1))
  worker=$(echo "$line" | cut -d' ' -f1)
  gpu=$(echo "$line" | cut -d' ' -f2)
  sess="br_${worker}"
  if [[ "$DRY" == 1 ]]; then
    echo "[dry-run] $sess gpu$gpu chunks=$(echo "$line" | wc -w | awk '{print $1-2}')"
    continue
  fi
  if tmux has-session -t "$sess" 2>/dev/null; then echo "$sess already up"; continue; fi
  tmux new-session -d -s "$sess" "bash -lc 'bash $WORKER $gpu $worker $(echo "$line" | cut -d' ' -f3-) >> $ARTIFACT/logs/${worker}.log 2>&1; echo \"[exit \$?] \$(date -Is)\" >> $ARTIFACT/logs/${worker}.log'"
  echo "started $sess (gpu$gpu)"
  sleep 1
done < /tmp/br_launch_cmds.txt

echo "--- $n workers ---"
tmux ls 2>/dev/null | cut -d: -f1 | grep '^br_' | sort | tr '\n' ' '; echo
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed -n '2,4p'
