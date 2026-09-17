"""Analyze the closed-loop causal test of L11 budget provenance."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


TASKS = (
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
TASK_LABELS = {
    "google_robot_open_drawer": "open_drawer",
    "google_robot_close_drawer": "close_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
}
ARMS = ("wrong_entity", "random_cluster", "within_task_shuffle")


def exact_mcnemar(rescue: int, harm: int) -> float:
    discordant = rescue + harm
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(rescue, harm) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--matched-artifact", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-end", type=int, default=199)
    args = parser.parse_args()
    root = args.artifact.resolve()
    matched_root = args.matched_artifact.resolve()
    seeds = range(args.seed_start, args.seed_end + 1)
    rows = []
    episode_rows = []

    for task in TASKS:
        matched_values = []
        arm_values = {arm: [] for arm in ARMS}
        arm_counts = {arm: [] for arm in ARMS}
        for seed in seeds:
            matched_path = matched_root / "episodes" / task / "l11_matched" / f"episode_{seed:03d}_summary.json"
            matched = json.loads(matched_path.read_text())
            if not matched.get("technical_pass"):
                raise RuntimeError(f"matched technical failure: {matched_path}")
            matched_values.append(bool(matched["success"]))
            for arm in ARMS:
                path = root / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                summary = json.loads(path.read_text())
                if not summary.get("technical_pass"):
                    raise RuntimeError(f"candidate technical failure: {path}")
                for key in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
                    if summary[key] != matched[key]:
                        raise RuntimeError(f"paired hash mismatch for {task}/{seed}/{arm}/{key}")
                arm_values[arm].append(bool(summary["success"]))
                arm_counts[arm].append(float(summary["mean_selected_token_count"]))
                episode_rows.append({
                    "task": TASK_LABELS[task],
                    "seed": seed,
                    "arm": arm,
                    "matched_success": int(matched["success"]),
                    "arm_success": int(summary["success"]),
                    "mean_selected_token_count": summary["mean_selected_token_count"],
                    "mean_feature_perturbation_relative": summary["mean_feature_perturbation_relative"],
                    "mean_centered_logit_residual_norm": summary["mean_centered_logit_residual_norm"],
                    "runtime_seconds": summary["runtime_seconds"],
                })
        for arm in ARMS:
            candidate = arm_values[arm]
            rescue = sum(not base and value for base, value in zip(matched_values, candidate))
            harm = sum(base and not value for base, value in zip(matched_values, candidate))
            rows.append({
                "task": TASK_LABELS[task],
                "arm": arm,
                "episodes": len(matched_values),
                "matched_success": sum(matched_values),
                "arm_success": sum(candidate),
                "rescue": rescue,
                "harm": harm,
                "net_vs_matched": rescue - harm,
                "exact_p": exact_mcnemar(rescue, harm),
                "mean_selected_token_count": float(np.mean(arm_counts[arm])),
            })

    for arm in ARMS:
        chosen = [row for row in episode_rows if row["arm"] == arm]
        rescue = sum(not row["matched_success"] and row["arm_success"] for row in chosen)
        harm = sum(row["matched_success"] and not row["arm_success"] for row in chosen)
        rows.append({
            "task": "Overall",
            "arm": arm,
            "episodes": len(chosen),
            "matched_success": sum(row["matched_success"] for row in chosen),
            "arm_success": sum(row["arm_success"] for row in chosen),
            "rescue": rescue,
            "harm": harm,
            "net_vs_matched": rescue - harm,
            "exact_p": exact_mcnemar(rescue, harm),
            "mean_selected_token_count": float(np.mean([row["mean_selected_token_count"] for row in chosen])),
        })

    write_csv(root / "PAIRED_RESULTS.csv", rows)
    write_csv(root / "EPISODE_RESULTS.csv", episode_rows)
    overall = {row["arm"]: row for row in rows if row["task"] == "Overall"}
    payload = {
        "protocol_id": "PROMPT_ATTN_L11_BUDGET_PROVENANCE_CLOSED_LOOP_V1",
        "seeds": [args.seed_start, args.seed_end],
        "tasks": list(TASK_LABELS.values()),
        "matched_source": str(matched_root),
        "overall": overall,
        "task_results": [row for row in rows if row["task"] != "Overall"],
        "technical_audit_pass": True,
    }
    (root / "FINAL_RESULTS.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# L11 matched-budget provenance closed-loop test",
        "",
        f"Four tasks, seeds {args.seed_start}–{args.seed_end}. True Matched is reused from the paired canonical run.",
        "All candidate arms preserve L11 ranking, harmonic reconstruction, and λ=0.5; only the budget source changes.",
        "",
        "## Overall",
        "",
        "| Budget source | True Matched | Candidate | Rescue | Harm | Net | Exact p | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = overall[arm]
        lines.append(
            f"| {arm} | {row['matched_success']}/{row['episodes']} | {row['arm_success']}/{row['episodes']} | "
            f"{row['rescue']} | {row['harm']} | {row['net_vs_matched']:+d} | {row['exact_p']:.6g} | "
            f"{row['mean_selected_token_count']:.2f} |"
        )
    lines += ["", "## Per task", "", "| Task | Arm | Matched | Candidate | Rescue | Harm | Net | Exact p |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["task"] == "Overall":
            continue
        lines.append(
            f"| {row['task']} | {row['arm']} | {row['matched_success']}/{row['episodes']} | "
            f"{row['arm_success']}/{row['episodes']} | {row['rescue']} | {row['harm']} | "
            f"{row['net_vs_matched']:+d} | {row['exact_p']:.6g} |"
        )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "- Matched > Wrong-Entity: the budget must be conditioned on the real task entities.",
        "- Matched > Random-Cluster: the result is not explained by an arbitrary KMeans cluster size.",
        "- Matched > Within-task Shuffle: the budget must remain paired with its own state, beyond task-level count distribution.",
    ]
    (root / "FULL_REPORT.md").write_text("\n".join(lines) + "\n")
    (root / "COMPLETE.json").write_text(json.dumps({"complete": True, "results": "FINAL_RESULTS.json"}, indent=2) + "\n")
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
