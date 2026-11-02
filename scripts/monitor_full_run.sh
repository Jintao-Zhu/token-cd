#!/usr/bin/env bash
# Long-run monitor for the 4500-episode pi0-FAST experiment.
# Appends a progress line every 5 minutes and writes a machine-readable status file, so the
# run can be checked without re-deriving state each time.
R=/home/leju-suzhou/zjt_ws/token-cd/runs/pi0fast_l6_matched_9task_100seed
LOG=$R/logs/monitor.log
TARGET=4500
mkdir -p "$R/logs"
echo "monitor started $(date '+%F %T')" >> "$LOG"
while true; do
  n=$(find "$R/episodes" -name '*.json' 2>/dev/null | wc -l)
  w=$(pgrep -cf "[p]i0_fast_simpler_l6_matched_rollout" || true)
  s=$(pgrep -cf "[p]i0_fast_l6_simpler_server" || true)
  line="$(date '+%F %T') episodes=$n/$TARGET workers=$w servers=$s"
  echo "$line" >> "$LOG"
  python3 - "$R" "$n" "$w" "$s" <<'PY' 2>/dev/null
import json, sys, glob, collections, statistics, time
R, n, w, s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = []
for f in glob.glob(f"{R}/episodes/*/*.json"):
    try:
        d = json.load(open(f)); rows.append(d)
    except Exception:
        pass
by = collections.Counter(d.get("arm") for d in rows)
status = {
    "updated": time.strftime("%F %T"), "episodes": n, "target": 4500,
    "workers": w, "servers": s,
    "per_arm": {a: sum(1 for d in rows if d.get("arm") == a) for a in sorted(by)},
    "success": sum(1 for d in rows if d.get("success")),
    "errors": sum(1 for d in rows if d.get("status") != "complete"),
    "tasks_covered": len({d.get("task") for d in rows}),
    "pairs_complete": sum(
        1 for k, v in collections.defaultdict(set, {
            (d.get("task"), d.get("seed")): set() for d in rows
        }).items() if len(v) == 5
    ),
}
# proper 5-arm pairing count
pairs = collections.defaultdict(set)
for d in rows:
    pairs[(d.get("task"), d.get("seed"))].add(d.get("arm"))
status["pairs_complete"] = sum(1 for v in pairs.values() if len(v) == 5)
json.dump(status, open(f"{R}/status.json", "w"), indent=2)
PY
  if [ "$n" -ge "$TARGET" ]; then echo "$(date '+%F %T') ALL_COMPLETE" >> "$LOG"; fi
  if [ "$w" -eq 0 ] && [ "$n" -gt 24 ]; then echo "$(date '+%F %T') WORKERS_EXITED episodes=$n" >> "$LOG"; fi
  sleep 300
done
