#!/usr/bin/env bash
# Append a one-line progress/rate record for the pi0-FAST WidowX 2800 run every
# $INTERVAL seconds (default 600) to logs/w18/monitor.log.  Cheap: it only reads
# episode JSONs.  Start with:
#   nohup bash scripts/monitor_pi0fast_widowx_2800.sh >/dev/null 2>&1 &
set -uo pipefail
ROOT=/home/leju-suzhou/zjt_ws
TOKEN_CD=$ROOT/token-cd
RUN=$TOKEN_CD/runs/pi0fast_bridge_widowx_4task_100seed
LOG=$RUN/logs/w18/monitor.log
INTERVAL=${INTERVAL:-600}
mkdir -p "$(dirname "$LOG")"
while true; do
  python3 - "$RUN" >> "$LOG" <<'PY'
import glob, json, os, sys, time, collections
run = sys.argv[1]
now = time.time()
tot = comp = err = 0
by_arm = collections.Counter(); succ = collections.Counter()
rt = []
for f in glob.glob(os.path.join(run, 'episodes', '*', '*.json')):
    tot += 1
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get('status') == 'complete':
        comp += 1
        by_arm[d.get('arm')] += 1
        succ[d.get('arm')] += 1 if d.get('success') else 0
        rt.append(d.get('runtime_seconds') or 0)
    elif d.get('status') == 'error':
        err += 1
recent = [m for m in (os.path.getmtime(f) for f in glob.glob(os.path.join(run, 'episodes', '*', '*.json')))
          if now - m <= 3600]
r10 = sum(1 for m in recent if now - m <= 600)
arms = ['vanilla', 'l6_matched_025', 'l6_matched_050', 'l6_matched_075',
        'l6_matched_100', 'l6_matched_125', 'l6_matched_150']
cells = ' '.join(f"{a.replace('l6_matched_','L')}:{succ[a]}/{by_arm[a]}" for a in arms)
mean_rt = sum(rt) / len(rt) if rt else 0
rate = r10 * 6
print(f"{time.strftime('%H:%M:%S')} done={comp}/{tot} err={err} rate10m={rate}/h "
      f"mean_rt={mean_rt:.0f}s | {cells}", flush=True)
PY
  sleep "$INTERVAL"
done
