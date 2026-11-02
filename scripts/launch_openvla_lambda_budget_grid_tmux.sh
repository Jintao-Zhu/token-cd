#!/usr/bin/env bash
# Launch the lambda x budget-rule grid: 12 tmux workers (4 per GPU on GPU 1/2/3) driven by
# artifacts/openvla_lambda_budget_grid_200_299_v1/WORK_PLAN.json.
#
# GPU policy: GPU 1/2/3 only -- never 0, never 4-7.
# Usage: bash launch_openvla_lambda_budget_grid_tmux.sh [--dry-run]
set -uo pipefail

ROOT=/home/leju-suzhou/zjt_ws
REPO=$ROOT/token-cd
ARTIFACT="${ARTIFACT:-$REPO/artifacts/openvla_lambda_budget_grid_200_299_v1}"
PLAN="${PLAN:-$ARTIFACT/WORK_PLAN.json}"
SESSPREFIX="${SESSPREFIX:-ovgrid}"
WORKER="$REPO/scripts/run_openvla_lambda_budget_grid_worker.sh"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

[[ -f "$PLAN" ]] || { echo "missing $PLAN" >&2; exit 2; }
mkdir -p "$ARTIFACT/logs"

python3 - "$PLAN" <<'PY' > /tmp/ovgrid_cmds.txt
import json, sys, shlex
plan = json.load(open(sys.argv[1]))
for w in plan["assignment"]:
    specs = [f"{c['task']}:{c['arm']}:{c['seeds']}" for c in w["chunks"]]
    print(shlex.join([w["worker"], str(w["gpu"])] + specs))
PY

n=0
want="${GPUS:-1,2,3}"
while read -r line; do
  [[ -z "$line" ]] && continue
  worker=$(echo "$line" | cut -d' ' -f1)
  gpu=$(echo "$line" | cut -d' ' -f2)
  case ",$want," in *",$gpu,"*) ;; *) continue ;; esac
  n=$((n + 1))
  sess="${SESSPREFIX}_${worker}"
  chunks=$(echo "$line" | wc -w | awk '{print $1-2}')
  if [[ "$DRY" == 1 ]]; then echo "[dry-run] $sess gpu$gpu chunks=$chunks"; continue; fi
  if tmux has-session -t "$sess" 2>/dev/null; then echo "$sess already up"; continue; fi
  tmux new-session -d -s "$sess" "bash -lc 'bash $WORKER $gpu $worker $(echo "$line" | cut -d' ' -f3-) >> $ARTIFACT/logs/${worker}.log 2>&1; echo \"[exit \$?] \$(date -Is)\" >> $ARTIFACT/logs/${worker}.log'"
  echo "started $sess (gpu$gpu chunks=$chunks)"
  sleep 1
done < /tmp/ovgrid_cmds.txt

echo "--- $n workers ---"
tmux ls 2>/dev/null | cut -d: -f1 | grep "^${SESSPREFIX}_" | sort | tr '\n' ' '; echo
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed -n '2,4p'
