#!/usr/bin/env python3
"""Self-check for the pi0-FAST run at 3500/3500 episodes.

Exit 0 only when every gate passes; exit 1 otherwise. Prints a JSON report to stdout and
writes it to <run>/logs/handoff_selfcheck.json.

Gates (from the experiment goal):
  1. episode count == 3500 (5 tasks x 7 arms x 100 seeds)
  2. per arm exactly 500 episodes, per task exactly 700
  3. every record has status == "complete", no duplicate (task, seed, arm)
  4. status.json reports errors == 0
  5. worker logs contain no error / traceback
"""
from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path

RUN = Path("/home/leju-suzhou/zjt_ws/token-cd/runs/pi0fast_l6_matched_9task_100seed")
ARMS = ["vanilla", "l6_matched_025", "l6_matched_050", "l6_matched_075",
        "l6_matched_100", "l6_matched_125", "l6_matched_150"]
TASKS = ["google_robot_close_drawer", "google_robot_move_near", "google_robot_open_drawer",
         "google_robot_pick_coke_can", "google_robot_place_apple_in_closed_top_drawer"]
TARGET = 3500


def main() -> int:
    report: dict = {"checked": __import__("time").strftime("%F %T"), "gates": {}}
    gates = report["gates"]

    files = sorted(glob.glob(str(RUN / "episodes" / "*" / "*.json")))
    report["episode_files"] = len(files)
    gates["count_3500"] = len(files) == TARGET

    per_arm = collections.Counter()
    per_task = collections.Counter()
    seen: dict[tuple, int] = collections.Counter()
    bad_status = []
    parse_errors = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception as exc:                       # noqa: BLE001
            parse_errors.append([f, repr(exc)])
            continue
        key = (d.get("task"), d.get("seed"), d.get("arm"))
        seen[key] += 1
        per_arm[d.get("arm")] += 1
        per_task[d.get("task")] += 1
        if d.get("status") != "complete":
            bad_status.append([f, d.get("status")])

    report["per_arm"] = dict(per_arm)
    report["per_task"] = dict(per_task)
    gates["per_arm_500"] = all(per_arm.get(a, 0) == 500 for a in ARMS)
    gates["per_task_700"] = all(per_task.get(t, 0) == 700 for t in TASKS)
    gates["all_complete_status"] = not bad_status and not parse_errors
    dupes = {str(k): v for k, v in seen.items() if v > 1}
    report["duplicates"] = dupes
    gates["no_duplicates"] = not dupes
    report["bad_status"] = bad_status[:10]
    report["parse_errors"] = parse_errors[:10]

    try:
        st = json.loads((RUN / "status.json").read_text())
        report["status_json"] = {k: st.get(k) for k in
                                 ("episodes", "success", "errors", "updated", "workers", "servers")}
        gates["status_errors_zero"] = st.get("errors") == 0
    except Exception as exc:                           # noqa: BLE001
        report["status_json_error"] = repr(exc)
        gates["status_errors_zero"] = False

    dirty = []
    for log in sorted(glob.glob(str(RUN / "logs" / "final_worker_*.log"))):
        try:
            text = Path(log).read_text(errors="replace")
        except Exception:                              # noqa: BLE001
            continue
        if re.search(r"error|traceback", text, re.I):
            dirty.append(Path(log).name)
    report["worker_logs_with_errors"] = dirty
    gates["worker_logs_clean"] = not dirty

    report["passed"] = all(gates.values())
    out = RUN / "logs" / "handoff_selfcheck.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
