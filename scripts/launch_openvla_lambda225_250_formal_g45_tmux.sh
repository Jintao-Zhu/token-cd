#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/leju-suzhou/zjt_ws/token-cd
ARTIFACT=$ROOT/artifacts/openvla_global_lambda_grid_225_250_0_199_v1
WORKER=$ROOT/scripts/run_openvla_lambda225_250_worker.sh
PASS=$ARTIFACT/PREFLIGHT_PASS_225_250.json
if [[ ! -f "$PASS" ]] || ! python3 - "$PASS" <<'PY'
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get('status')=='PASS' else 1)
PY
then echo "2.25/2.50 preflight has not passed" >&2; exit 1; fi
mkdir -p "$ARTIFACT/logs"
PLAN=$ARTIFACT/WORK_PLAN_225_250.tsv
python3 - "$PLAN" <<'PY'
import json,sys
from pathlib import Path
means={
 'google_robot_open_drawer':161.7,'google_robot_close_drawer':152.5,
 'google_robot_pick_coke_can':37.5,'google_robot_move_near':37.8,
 'google_robot_place_apple_in_closed_top_drawer':82.0,
 'widowx_carrot_on_plate':24.8,'widowx_put_eggplant_in_basket':50.9,
 'widowx_spoon_on_towel':24.3,'widowx_stack_cube':24.7,
}
jobs=[]
for task,sec in means.items():
 for lo in (0,100): jobs.append((100*2*sec,task,f'{lo}-{lo+99}'))
jobs.sort(reverse=True)
bins=[[] for _ in range(6)]; loads=[0.0]*6
for weight,task,seeds in jobs:
 i=min(range(6),key=lambda j:loads[j]); bins[i].append((task,seeds));loads[i]+=weight
with Path(sys.argv[1]).open('w') as f:
 for i,chunks in enumerate(bins):
  worker=f'lambda225_g{4+i//3}_w{i%3+1}_formal'; gpu=4+i//3
  f.write('\t'.join([worker,str(gpu),*[f'{t}={s}' for t,s in chunks]])+'\n')
manifest={
 'protocol_id':'OPENVLA_GLOBAL_LAMBDA_GRID_0_199_V1',
 'artifact':str(Path(sys.argv[1]).parent),'gpus':[4,5],'workers_per_gpu':3,'workers':6,
 'tasks':list(means),'seeds':list(range(200)),'lambdas':[2.25,2.5],
 'episode_arm_evaluations':9*200*2,
 'estimated_worker_hours':[round(x/3600,2) for x in loads],
 'assignment':[{'worker':f'lambda225_g{4+i//3}_w{i%3+1}_formal','gpu':4+i//3,
  'estimated_hours':round(loads[i]/3600,2),
  'chunks':[{'task':t,'seeds':s} for t,s in bins[i]]} for i in range(6)]}
Path(str(sys.argv[1]).replace('.tsv','.json')).write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
PY
while IFS=$'\t' read -r name gpu rest; do
  [[ -n "$name" ]] || continue
  if tmux has-session -t "$name" 2>/dev/null; then echo "session already exists: $name" >&2; exit 1; fi
  args=("$gpu" "$name" formal)
  for item in $rest; do args+=("${item%%=*}" "${item#*=}"); done
  cmd=(bash "$WORKER" "${args[@]}")
  printf -v quoted '%q ' "${cmd[@]}"
  tmux new-session -d -s "$name" "cd '$ROOT' && $quoted 2>&1 | tee '$ARTIFACT/logs/${name}.log'"
  echo "started $name on GPU $gpu with ${#args[@]} worker arguments"
done < "$PLAN"
echo "Launched 6 formal workers (3 each on GPUs 4 and 5); 3,600 episode-arm evaluations."
