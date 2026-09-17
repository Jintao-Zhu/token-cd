"""Aggregate DTP closed-loop calibration episodes and freeze one unified config.

Decision rule is pre-registered: choose the candidate arm with the larger total
success count over the overlap-free calibration scenes; ties go to L11.  The
resulting choice is written to CONFIG_LOCK.json under the calibration artifact.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from research.semantic_token_cd.dtp_closed_loop_protocol import (
    ARM_CONFIG,
    ARMS,
    ARTIFACT,
    PROTOCOL,
    TASKS,
    calibration_seeds,
)


def load_episodes(root: Path) -> list[dict]:
    episodes = []
    for arm in ARMS:
        for summary_path in sorted((root / "episodes").glob(f"*/{arm}/episode_*_summary.json")):
            episodes.append(json.loads(summary_path.read_text()))
    return episodes


def summarize(episodes: list[dict], arms: tuple[str, ...] = ARMS) -> dict:
    out = {}
    for arm in arms:
        rows = [e for e in episodes if e["arm"] == arm]
        out[arm] = {
            "n": len(rows),
            "success": sum(bool(e["success"]) for e in rows),
            "prune_activation_steps_total": sum(int(e["prune_activation_steps"]) for e in rows),
            "control_steps_total": sum(int(e["control_steps"]) for e in rows),
        }
    return out


def main() -> None:
    root = ARTIFACT / "closed_loop"
    episodes = load_episodes(root)
    if len(episodes) != len(ARMS) * sum(len(calibration_seeds(t)) for t in TASKS):
        raise RuntimeError(
            f"expected {len(ARMS) * sum(len(calibration_seeds(t)) for t in TASKS)} "
            f"calibration episodes, found {len(episodes)}"
        )
    total = summarize(episodes)
    per_task = {task: summarize([e for e in episodes if e["task"] == task]) for task in TASKS}

    pairs = {}
    for task in TASKS:
        for seed in calibration_seeds(task):
            control = next(e for e in episodes if e["task"] == task and e["seed"] == seed and e["arm"] == "control")
            pairs[(task, seed)] = control
    table = []
    for arm in ARMS:
        for task in TASKS:
            rows = [e for e in episodes if e["arm"] == arm and e["task"] == task]
            table.append({"arm": arm, "task": task, "n": len(rows),
                          "success": sum(bool(e["success"]) for e in rows)})
    csv_path = root / "calibration_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    decisions = {}
    for candidate in ("l11_k64_t05", "l7_k64_t05"):
        rescue = harm = unchanged = 0
        for task in TASKS:
            for seed in calibration_seeds(task):
                control = next(e for e in episodes
                               if e["task"] == task and e["seed"] == seed and e["arm"] == "control")
                cand = next(e for e in episodes
                            if e["task"] == task and e["seed"] == seed and e["arm"] == candidate)
                if cand["success"] and not control["success"]:
                    rescue += 1
                elif not cand["success"] and control["success"]:
                    harm += 1
                else:
                    unchanged += 1
        decisions[candidate] = {"rescue": rescue, "harm": harm, "unchanged": unchanged,
                                "net": rescue - harm, "success": total[candidate]["success"]}

    winner = max(("l11_k64_t05", "l7_k64_t05"),
                 key=lambda arm: (total[arm]["success"], -int(arm == "l7_k64_t05")))
    if total[winner]["success"] == total["control"]["success"]:
        chosen = "l11_k64_t05"
    elif total[winner]["success"] > total["control"]["success"]:
        chosen = winner
    else:
        chosen = "l11_k64_t05"

    lock = {
        "protocol_id": PROTOCOL,
        "stage": "calibration",
        "calibration_scenes": {task: calibration_seeds(task) for task in TASKS},
        "arms": list(ARMS),
        "totals": total,
        "per_task": per_task,
        "paired_rescue_harm": decisions,
        "decision_rule": "choose candidate with larger total calibration success; "
                         "tie -> L11; if neither exceeds control, freeze L11 as offline default",
        "chosen_config": ARM_CONFIG[chosen],
        "chosen_arm": chosen,
        "summary_csv": str(csv_path.relative_to(root)),
    }
    lock_path = root / "CONFIG_LOCK.json"
    tmp = lock_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    tmp.replace(lock_path)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
