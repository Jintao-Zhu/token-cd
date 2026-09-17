"""Aggregate physical phase traces from replay_prompt_action_key_pairs.py."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts/prompt_action_complement_v1/mechanism_analysis")
SRC = ROOT / "trajectory_replay"


def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return None if not vals else float(np.mean(vals))


def main():
    flat, summary = [], {}
    for task_dir in sorted(SRC.iterdir()):
        task = task_dir.name
        grouped = defaultdict(list)
        for p in sorted(task_dir.glob("seed_*.json")):
            x = json.loads(p.read_text())
            for arm, a in x["arms"].items():
                row = {"task": task, "seed": x["seed"], "category": x["category"], "arm": arm,
                       "success": a["success"], "n_steps": a["n_steps"]}
                if task.endswith("move_near"):
                    for k in ("first_moved", "first_near", "min_src_tgt_xy", "min_tcp_src", "final_src_xy_move", "final_src_tgt_xy"):
                        row[k] = a[k]
                else:
                    row.update(qpos_initial=a["qpos_initial"], qpos_final=a["qpos_final"], qpos_min=a["qpos_min"])
                    row["first_progress_001"] = a["first_progress"]["0.01"]
                    row["first_progress_005"] = a["first_progress"]["0.05"]
                    row["first_progress_010"] = a["first_progress"]["0.1"]
                    row["ever_closed"] = a["qpos_min"] <= 0.05
                    row["reopened_after_closed"] = a["qpos_min"] <= 0.05 and a["qpos_final"] > 0.05
                flat.append(row)
                grouped[(x["category"], arm)].append(row)
        t = {}
        for (cat, arm), rows in grouped.items():
            rec = {"n": len(rows)}
            keys = ("first_moved", "first_near", "min_src_tgt_xy", "min_tcp_src", "final_src_xy_move", "final_src_tgt_xy") if task.endswith("move_near") else ("qpos_initial", "qpos_final", "qpos_min", "first_progress_001", "first_progress_005", "first_progress_010")
            rec.update({k: mean(rows, k) for k in keys})
            if not task.endswith("move_near"):
                rec["ever_closed_rate"] = float(np.mean([r["ever_closed"] for r in rows]))
                rec["reopened_after_closed_rate"] = float(np.mean([r["reopened_after_closed"] for r in rows]))
            t.setdefault(cat, {})[arm] = rec
        summary[task] = t
    fields = sorted({k for r in flat for k in r})
    with (ROOT / "trajectory_phase_per_episode.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(flat)
    (ROOT / "trajectory_phase_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
