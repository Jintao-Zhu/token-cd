#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd
ARTIFACT=$ROOT/artifacts/openvla_global_lambda_grid_225_250_0_199_v1
WORKER=$ROOT/scripts/run_openvla_lambda225_250_worker.sh
source "$ROOT/scripts/activate_simpler_local.sh"
cd "$ROOT"
# This reallocation assumes GPUs 1-5 are free of other model jobs.
python research/semantic_token_cd/plan_openvla_lambda225_250_rebalance.py | tee "$ARTIFACT/logs/rebalance_plan.log"
PLAN=$ARTIFACT/WORK_PLAN_REBALANCED_G12345.tsv
while IFS=$'\t' read -r name gpu rest; do
  [[ -n "$name" ]] || continue
  if tmux has-session -t "$name" 2>/dev/null; then echo "session already exists: $name" >&2; exit 1; fi
  args=("$gpu" "$name" rebalance)
  for item in $rest; do args+=("${item%%=*}" "${item#*=}"); done
  cmd=(bash "$WORKER" "${args[@]}")
  printf -v quoted '%q ' "${cmd[@]}"
  tmux new-session -d -s "$name" "cd '$ROOT' && $quoted 2>&1 | tee '$ARTIFACT/logs/${name}.log'"
  echo "started $name on GPU $gpu with ${#args[@]} worker arguments"
done < "$PLAN"
echo "Rebalanced remaining lambda=2.25/2.50 evaluations across 15 workers (3 each on GPUs 1-5)."
