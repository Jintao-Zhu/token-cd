#!/usr/bin/env bash
# Progress monitor for the OpenVLA budget-rule study: appends one line per $INTERVAL s to
# artifacts/openvla_budget_rule_9task_200_299_v1/logs/monitor.log.
#   nohup env INTERVAL=600 bash scripts/monitor_openvla_budget_rule.sh >/dev/null 2>&1 &
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws
REPO=$ROOT/token-cd
ART=$REPO/artifacts/openvla_budget_rule_9task_200_299_v1
INTERVAL=${INTERVAL:-600}
mkdir -p "$ART/logs"
while true; do
  python3 - "$ART" >> "$ART/logs/monitor.log" <<'PY'
import glob, json, os, sys, time, collections
art = sys.argv[1]
now = time.time()
per = collections.Counter(); succ = collections.Counter(); mts = collections.defaultdict(list)
for f in glob.glob(art + '/episodes/*/*/episode_*_summary.json'):
    d = json.load(open(f)); a = d['arm']
    per[a] += 1; succ[a] += 1 if d.get('success') else 0
    mts[a].append(d.get('mean_selected_token_count') or 0)
done = sum(per.values())
recent = sum(1 for f in glob.glob(art + '/episodes/*/*/episode_*_summary.json')
             if now - os.path.getmtime(f) <= 600)
parts = ' '.join(f"{a.replace('_l100','')}:{succ[a]}/{per[a]}" for a in sorted(per))
print(f"{time.strftime('%H:%M:%S')} done={done}/9900 rate10m={recent*6}/h | {parts}", flush=True)
PY
  sleep "$INTERVAL"
done
