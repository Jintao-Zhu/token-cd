"""Analyze code-L15 versus paper-L16:31 P75 and verified formal Vanilla."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
ART = ROOT / "artifacts/vla_pruner_openvla_reproduction/paper_v5_p75/formal"
VAN = ROOT / "artifacts/vla_pruner_openvla_reproduction/formal_1200/episodes"
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)
ARMS = ("vanilla", "vla_pruner_prune75_code_l15", "vla_pruner_prune75_paper_l16_31")


def load_summary(path: Path):
    return json.loads(path.read_text())


def paired(rows, left, right):
    rescue = sum((not r[left]) and r[right] for r in rows)
    harm = sum(r[left] and (not r[right]) for r in rows)
    p = binomtest(rescue, rescue + harm, 0.5).pvalue if rescue + harm else 1.0
    return {"rescue": rescue, "harm": harm, "net": rescue - harm, "mcnemar_exact_p": p}


def main():
    rows, mismatches, technical_failures = [], [], []
    for task in TASKS:
        for seed in range(100):
            vanilla_path = VAN / task / "vanilla" / f"episode_{seed:03d}_summary.json"
            if not vanilla_path.exists():
                continue
            v = load_summary(vanilla_path)
            row = {"task": task, "seed": seed, "vanilla": bool(v["success"])}
            base_hash = (v["initial_state_sha256"], v["initial_rgb_sha256"])
            complete = True
            for arm in ARMS[1:]:
                path = ART / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                if not path.exists():
                    complete = False
                    break
                s = load_summary(path)
                row[arm] = bool(s["success"])
                if not bool(s.get("technical_pass", False)):
                    technical_failures.append((task, seed, arm))
                if (s.get("initial_state_sha256"), s.get("initial_rgb_sha256")) != base_hash:
                    mismatches.append((task, seed, arm))
            if complete:
                rows.append(row)

    result = {
        "complete_arm_episodes": len(rows) * 2,
        "expected_arm_episodes": 800,
        "complete_scenes": len(rows),
        "hash_mismatches": mismatches,
        "technical_failures": technical_failures,
        "by_task": {},
    }
    for task in TASKS + ("OVERALL",):
        sub = rows if task == "OVERALL" else [r for r in rows if r["task"] == task]
        result["by_task"][task] = {
            "n": len(sub),
            "success": {arm: sum(r[arm] for r in sub) for arm in ARMS},
            "paper_vs_vanilla": paired(sub, "vanilla", ARMS[2]),
            "code_vs_vanilla": paired(sub, "vanilla", ARMS[1]),
            "paper_vs_code": paired(sub, ARMS[1], ARMS[2]),
        }

    ART.mkdir(parents=True, exist_ok=True)
    (ART / "FINAL_RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# VLA-Pruner P75: official-code L15 vs paper-v5 L16-31", "",
        f"Completed: {result['complete_arm_episodes']}/800 arm-episodes; "
        f"hash mismatches={len(mismatches)}; technical failures={len(technical_failures)}", "",
        "| Task | N | Vanilla | Code-L15 P75 | Paper-L16-31 P75 |",
        "|---|---:|---:|---:|---:|",
    ]
    for task, data in result["by_task"].items():
        n = data["n"]
        vals = [data["success"][a] for a in ARMS]
        fmt = [f"{v}/{n} ({100*v/n:.1f}%)" if n else "—" for v in vals]
        lines.append(f"| {task} | {n} | {fmt[0]} | {fmt[1]} | {fmt[2]} |")
    lines += ["", "## Paired comparisons", ""]
    for task, data in result["by_task"].items():
        p = data["paper_vs_code"]
        lines.append(f"- {task}: Paper vs Code rescue={p['rescue']}, harm={p['harm']}, "
                     f"net={p['net']}, exact p={p['mcnemar_exact_p']:.4g}")
    (ART / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result["by_task"], indent=2))


if __name__ == "__main__":
    main()
