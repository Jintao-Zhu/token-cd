#!/usr/bin/env bash
# Wait for the smoke run to finish (45 episodes or all workers gone), then summarise.
R=/home/leju-suzhou/zjt_ws/token-cd/runs/pi0fast_l6_matched_9task_100seed
TARGET=45
for i in $(seq 1 400); do
  n=$(find "$R/episodes" -name '*.json' 2>/dev/null | wc -l)
  w=$(pgrep -cf "[p]i0_fast_simpler_l6_matched_rollout" || true)
  echo "$(date '+%H:%M:%S') episodes=$n/$TARGET workers=$w"
  if [ "$n" -ge "$TARGET" ]; then echo "SMOKE_COMPLETE"; break; fi
  if [ "$w" -eq 0 ]; then echo "ALL_WORKERS_EXITED at $n/$TARGET"; break; fi
  sleep 20
done
echo "=== final summary ==="
python3 - <<'PY'
import json, glob, collections, statistics
R = "/home/leju-suzhou/zjt_ws/token-cd/runs/pi0fast_l6_matched_9task_100seed/episodes"
rows = []
for f in glob.glob(f"{R}/*/*.json"):
    d = json.load(open(f))
    rows.append({"task": d.get("task"), "arm": d.get("arm"), "seed": d.get("seed"),
                 "success": bool(d.get("success")), "steps": d.get("steps") or 0,
                 "rt": d.get("runtime_seconds") or 0, "status": d.get("status")})
print(f"total={len(rows)}")
by = collections.Counter(r["arm"] for r in rows)
for arm in sorted(by):
    sub = [r for r in rows if r["arm"] == arm]
    ok = sum(1 for r in sub if r["success"])
    print(f"  {arm:16s} n={len(sub):3d}  success={ok:3d} ({100*ok/len(sub):5.1f}%)  "
          f"steps={statistics.fmean(r['steps'] for r in sub):6.1f}  "
          f"runtime={statistics.fmean(r['rt'] for r in sub):7.1f}s")
# pairing: every task/seed should have all five arms
pairs = collections.defaultdict(set)
for r in rows:
    pairs[(r["task"], r["seed"])].add(r["arm"])
full = sum(1 for v in pairs.values() if len(v) == 5)
print(f"complete 5-arm pairs: {full}/{len(pairs)}")
bad = [r for r in rows if r["status"] != "complete"]
print(f"non-complete statuses: {len(bad)}")
PY
