"""Paired analysis for the Entity/Generic Top-P budget closed loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import stats

TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
ARMS = ("taskmean",)


def load_summary(root: Path, task: str, arm: str, seed: int) -> dict:
    path = root / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def mcnemar(matched: list[bool], other: list[bool]) -> dict:
    rescue = sum((not a) and b for a, b in zip(matched, other))
    harm = sum(a and (not b) for a, b in zip(matched, other))
    discordant = rescue + harm
    if discordant == 0:
        p = 1.0
    else:
        p = float(stats.binomtest(min(rescue, harm), discordant, 0.5).pvalue)
    return {"rescue": int(rescue), "harm": int(harm), "net": int(rescue - harm), "p": p}


def arm_stats(root: Path, matched_root: Path, task: str, arm: str, seeds: list[int]) -> dict:
    rows = []
    for seed in seeds:
        m = load_summary(matched_root, task, "l11_matched", seed)
        a = load_summary(root, task, arm, seed)
        rows.append({
            "seed": seed,
            "matched_success": bool(m["success"]),
            "arm_success": bool(a["success"]),
            "matched_m_mean": float(m.get("mean_selected_token_count", 0.0)),
            "arm_m_mean": float(a.get("mean_selected_token_count", 0.0)),
            "steps": int(a.get("control_steps", 0)),
        })
    matched = [r["matched_success"] for r in rows]
    other = [r["arm_success"] for r in rows]
    test = mcnemar(matched, other)
    return {
        "task": task,
        "arm": arm,
        "n": len(rows),
        "matched_success": sum(matched),
        "arm_success": sum(other),
        "matched_rate": sum(matched) / len(rows) if rows else float("nan"),
        "arm_rate": sum(other) / len(rows) if rows else float("nan"),
        "matched_m_mean": sum(r["matched_m_mean"] for r in rows) / len(rows) if rows else float("nan"),
        "arm_m_mean": sum(r["arm_m_mean"] for r in rows) / len(rows) if rows else float("nan"),
        **test,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--matched-artifact", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-end", type=int, default=199)
    parser.add_argument("--arms", default="entity_top_p")
    args = parser.parse_args()

    root = args.artifact.resolve()
    matched_root = args.matched_artifact.resolve()
    seeds = list(range(args.seed_start, args.seed_end + 1))
    arms = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError(f"invalid arms: {arms}")

    overall = []
    per_task = []
    for arm in arms:
        for task in TASKS:
            st = arm_stats(root, matched_root, task, arm, seeds)
            per_task.append(st)
        # Overall arm is pooled across tasks.
        rows = []
        for task in TASKS:
            for seed in seeds:
                m = load_summary(matched_root, task, "l11_matched", seed)
                a = load_summary(root, task, arm, seed)
                rows.append((bool(m["success"]), bool(a["success"]),
                             float(m.get("mean_selected_token_count", 0.0)),
                             float(a.get("mean_selected_token_count", 0.0))))
        matched = [r[0] for r in rows]
        other = [r[1] for r in rows]
        test = mcnemar(matched, other)
        overall.append({
            "arm": arm,
            "n": len(rows),
            "matched_success": sum(matched),
            "arm_success": sum(other),
            "matched_rate": sum(matched) / len(rows),
            "arm_rate": sum(other) / len(rows),
            "matched_m_mean": sum(r[2] for r in rows) / len(rows),
            "arm_m_mean": sum(r[3] for r in rows) / len(rows),
            **test,
        })

    L = ["# TaskMean budget closed loop", "",
         "Matched is the shared paired baseline. Lower p values are exact McNemar tests.",
         ""]
    for arm in arms:
        L += [f"## {arm.capitalize()}", "",
              "| Task | Matched | Arm | Rescue | Harm | Net | McNemar p | Matched m | Arm m |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for st in overall:
            if st["arm"] != arm:
                continue
            L.append(f"| **OVERALL** | {st['matched_success']}/{st['n']} | {st['arm_success']}/{st['n']} | "
                     f"{st['rescue']} | {st['harm']} | {st['net']:+d} | {st['p']:.4f} | "
                     f"{st['matched_m_mean']:.2f} | {st['arm_m_mean']:.2f} |")
        for st in per_task:
            if st["arm"] != arm:
                continue
            L.append(f"| {st['task'].replace('google_robot_','')} | {st['matched_success']}/{st['n']} | "
                     f"{st['arm_success']}/{st['n']} | {st['rescue']} | {st['harm']} | {st['net']:+d} | "
                     f"{st['p']:.4f} | {st['matched_m_mean']:.2f} | {st['arm_m_mean']:.2f} |")
        L.append("")

    output = {
        "protocol_id": "PROMPT_ATTN_L11_TASKMEAN_BUDGET_V1",
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "n_per_arm": len(TASKS) * len(seeds),
        "overall": overall,
        "per_task": per_task,
    }
    (root / "CLOSED_LOOP_RESULTS.json").write_text(json.dumps(output, indent=2) + "\n")
    (root / "CLOSED_LOOP_REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
