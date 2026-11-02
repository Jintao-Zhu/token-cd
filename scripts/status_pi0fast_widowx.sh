#!/usr/bin/env bash
# Compact status for the pi0-FAST WidowX 2800-episode run.
# Usage: bash status_pi0fast_widowx.sh
RUN=/home/leju-suzhou/zjt_ws/token-cd/runs/pi0fast_bridge_widowx_4task_100seed
PY=/home/leju-suzhou/miniconda3/envs/pi05/bin/python
echo "== $(date '+%F %T') =="
echo "tmux: workers=$(tmux ls 2>/dev/null | grep -c pf_wx_w) servers=$(tmux ls 2>/dev/null | grep -c pf_wx_srv)"
echo "procs: workers=$(pgrep -cf '^python .*pi0_fast_simpler_l6_matched_rollout') servers=$(pgrep -cf '^python .*pi0_fast_l6_simpler_server')"
"$PY" - "$RUN" <<'PY'
import glob, json, os, sys, time, collections
from pathlib import Path
run = Path(sys.argv[1])
files = sorted((run / 'episodes').glob('*/*.json'))
n = len(files)
target = json.loads((run / 'protocol.json').read_text())['episodes']
now = time.time()
rate = {}
for w in (15, 60):
    k = sum(1 for f in files if now - os.path.getmtime(f) < w * 60)
    rate[w] = k / w * 60
done = collections.Counter()
succ = collections.Counter()
for f in files:
    try: d = json.loads(f.read_text())
    except Exception: continue
    done[d.get('arm')] += 1
    if d.get('success'): succ[d.get('arm')] += 1
print(f"progress: {n}/{target} ({n/target*100:.1f}%)  remaining {target-n}")
print(f"rate: last15min {rate[15]:.0f}/h  last60min {rate[60]:.0f}/h" +
      (f"  ETA {(target-n)/(rate[15] or 1)*60/3600:.1f}h" if rate[15] else ""))
for a in sorted(done):
    print(f"   {a:16s} {done[a]:4d} eps  success {succ[a]:3d} ({succ[a]/done[a]*100:.1f}%)")
errs = list((run / 'logs').glob('worker_*.log'))
bad = [f.name for f in errs if 'Traceback' in f.read_text(errors='replace')]
print("worker logs with Traceback:", bad[:5] if bad else "none")
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i 1,2,3
