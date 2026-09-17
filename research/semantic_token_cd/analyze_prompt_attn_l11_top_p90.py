"""Combined paired analysis for Matched and L11 visual Top-p 0.75--0.90."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from research.semantic_token_cd.prompt_attn_l11_count_rollout import TASKS


SHORT = {
    "google_robot_open_drawer": "open_drawer",
    "google_robot_close_drawer": "close_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
}
LABELS = {
    "l11_matched": "Matched",
    "l11_top_p75": "TopP75",
    "l11_top_p80": "TopP80",
    "l11_top_p85": "TopP85",
    "l11_top_p90": "TopP90",
}
TOP_P_ARMS = tuple(arm for arm in LABELS if arm != "l11_matched")


def exact_mcnemar(rescue: int, harm: int) -> float:
    total = rescue + harm
    if not total:
        return 1.0
    tail = min(rescue, harm)
    return min(1.0, 2.0 * sum(math.comb(total, value) for value in range(tail + 1)) / 2 ** total)


def holm(values: dict[str, float]) -> dict[str, float]:
    keys = sorted(values, key=values.get)
    result = {}
    running = 0.0
    for rank, key in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - rank) * values[key]))
        result[key] = running
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_row(task: str, candidate: str, baseline: str, values: dict) -> dict:
    seeds = range(100, 200) if task != "Overall" else (
        (task_name, seed) for task_name in TASKS for seed in range(100, 200)
    )
    if task == "Overall":
        pairs = [(values[(task_name, candidate)][seed], values[(task_name, baseline)][seed])
                 for task_name, seed in seeds]
    else:
        pairs = [(values[(task, candidate)][seed], values[(task, baseline)][seed]) for seed in seeds]
    rescue = sum(candidate_value and not baseline_value for candidate_value, baseline_value in pairs)
    harm = sum(not candidate_value and baseline_value for candidate_value, baseline_value in pairs)
    return {
        "task": task if task == "Overall" else SHORT[task],
        "candidate": LABELS[candidate],
        "baseline": LABELS[baseline],
        "n": len(pairs),
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "exact_mcnemar_p": exact_mcnemar(rescue, harm),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--matched-artifact", type=Path, required=True)
    parser.add_argument("--base-top-p-artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve()
    matched = args.matched_artifact.resolve()
    base_top_p = args.base_top_p_artifact.resolve()
    results = defaultdict(dict)
    summaries = {}

    for task in TASKS:
        for arm in LABELS:
            if arm == "l11_matched":
                source = matched
            elif arm == "l11_top_p90":
                source = root
            else:
                source = base_top_p
            for seed in range(100, 200):
                path = source / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                if not path.exists():
                    raise RuntimeError(f"missing result: {path}")
                payload = json.loads(path.read_text())
                if payload.get("technical_pass") is not True:
                    raise RuntimeError(f"technical audit failed: {path}")
                results[(task, arm)][seed] = bool(payload["success"])
                summaries[(task, arm, seed)] = payload

    hash_checks = []
    for task in TASKS:
        for seed in range(100, 200):
            group = [summaries[(task, arm, seed)] for arm in LABELS]
            hash_checks.append(all(
                len({payload[key] for payload in group}) == 1
                for key in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256")
            ))
    if not all(hash_checks):
        raise RuntimeError("paired snapshot/state/RGB hash mismatch")

    success_rows = []
    for task in [*TASKS, "Overall"]:
        row = {"task": task if task == "Overall" else SHORT[task], "n": 400 if task == "Overall" else 100}
        for arm, label in LABELS.items():
            if task == "Overall":
                success = sum(results[(task_name, arm)][seed] for task_name in TASKS for seed in range(100, 200))
            else:
                success = sum(results[(task, arm)].values())
            row[f"{label}_success"] = success
            row[f"{label}_rate"] = success / row["n"]
        success_rows.append(row)

    paired_rows = []
    for task in [*TASKS, "Overall"]:
        paired_rows.extend(paired_row(task, arm, "l11_matched", results) for arm in TOP_P_ARMS)
        paired_rows.append(paired_row(task, "l11_top_p90", "l11_top_p85", results))
    overall_vs_matched = {
        row["candidate"]: row["exact_mcnemar_p"]
        for row in paired_rows if row["task"] == "Overall" and row["baseline"] == "Matched"
    }
    adjusted = holm(overall_vs_matched)
    for row in paired_rows:
        row["holm_adjusted_p_over_4_overall_vs_matched"] = (
            adjusted[row["candidate"]]
            if row["task"] == "Overall" and row["baseline"] == "Matched" else ""
        )

    distribution_rows = []
    for task in [*TASKS, "Overall"]:
        for arm, label in LABELS.items():
            payloads = (
                [summaries[(task_name, arm, seed)] for task_name in TASKS for seed in range(100, 200)]
                if task == "Overall" else [summaries[(task, arm, seed)] for seed in range(100, 200)]
            )
            counts = np.asarray([
                step["actual_selected_count"] for payload in payloads for step in payload["selector_trace"]
            ], dtype=float)
            masses = [
                step.get("selected_attention_mass") for payload in payloads for step in payload["selector_trace"]
                if step.get("selected_attention_mass") is not None
            ]
            distribution_rows.append({
                "task": task if task == "Overall" else SHORT[task],
                "arm": label,
                "control_steps": len(counts),
                "mean_m_t": float(counts.mean()),
                "std_m_t": float(counts.std()),
                "median_m_t": float(np.median(counts)),
                "p90_m_t": float(np.percentile(counts, 90)),
                "max_m_t": int(counts.max()),
                "mean_attention_mass": float(np.mean(masses)) if masses else "",
            })

    statistics = root / "statistics"
    paired = root / "paired_results"
    write_csv(statistics / "combined_success_rates.csv", success_rows)
    write_csv(statistics / "combined_count_distribution.csv", distribution_rows)
    write_csv(paired / "combined_paired_results.csv", paired_rows)
    (statistics / "combined_success_rates.json").write_text(json.dumps(success_rows, indent=2) + "\n")
    (paired / "combined_paired_results.json").write_text(json.dumps(paired_rows, indent=2) + "\n")

    thresholds = [75, 80, 85, 90]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for row in success_rows:
        axes[0].plot(thresholds, [100 * row[f"TopP{value}_rate"] for value in thresholds],
                     marker="o", label=row["task"])
    axes[0].axhline(100 * success_rows[-1]["Matched_rate"], color="black", ls="--",
                    label="Matched overall")
    axes[0].set(xlabel="Cumulative visual attention threshold (%)", ylabel="Success rate (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    overall_distribution = {row["arm"]: row for row in distribution_rows if row["task"] == "Overall"}
    axes[1].errorbar(
        thresholds,
        [overall_distribution[f"TopP{value}"]["mean_m_t"] for value in thresholds],
        yerr=[overall_distribution[f"TopP{value}"]["std_m_t"] for value in thresholds],
        marker="o",
    )
    axes[1].set(xlabel="Cumulative visual attention threshold (%)", ylabel="Selected tokens (mean +/- SD)")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(statistics / "threshold_75_90_success_and_count.png", dpi=180)
    plt.close(figure)

    def cell(row: dict, label: str) -> str:
        return f'{row[f"{label}_success"]}/{row["n"]} ({100 * row[f"{label}_rate"]:.1f}%)'

    lines = [
        "# L11 Visual Top-p 0.90 Extension — Final Report",
        "",
        "TopP90 adds 400 new episodes on the same four tasks and seeds 100--199. Matched and TopP75/80/85 are reused; all paired snapshot/state/RGB hashes and technical audits passed.",
        "",
        "| Task | Matched | TopP75 | TopP80 | TopP85 | TopP90 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in success_rows:
        lines.append("| " + row["task"] + " | " + " | ".join(cell(row, label) for label in LABELS.values()) + " |")
    lines.extend([
        "",
        "| Candidate | Baseline | Rescue | Harm | Net | Exact p | Holm-adjusted p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in paired_rows:
        if row["task"] == "Overall":
            adjusted_value = row["holm_adjusted_p_over_4_overall_vs_matched"]
            adjusted_text = f"{adjusted_value:.6g}" if adjusted_value != "" else "—"
            lines.append(
                f'| {row["candidate"]} | {row["baseline"]} | {row["rescue"]} | {row["harm"]} | '
                f'{row["net"]:+d} | {row["exact_mcnemar_p"]:.6g} | {adjusted_text} |'
            )
    lines.extend([
        "",
        "## TopP90 Per-task Paired Results",
        "",
        "| Task | Baseline | Rescue | Harm | Net | Exact p |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in paired_rows:
        if row["task"] != "Overall" and row["candidate"] == "TopP90":
            lines.append(
                f'| {row["task"]} | {row["baseline"]} | {row["rescue"]} | {row["harm"]} | '
                f'{row["net"]:+d} | {row["exact_mcnemar_p"]:.6g} |'
            )
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "complete": True,
        "new_top_p90_episodes": 400,
        "reused_matched_episodes": 400,
        "reused_top_p75_80_85_episodes": 1200,
        "success": success_rows,
        "overall_paired": [row for row in paired_rows if row["task"] == "Overall"],
    }, indent=2))


if __name__ == "__main__":
    main()
