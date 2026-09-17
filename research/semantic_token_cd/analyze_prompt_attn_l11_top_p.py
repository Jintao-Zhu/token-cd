"""Final paired and mechanism analysis for L11 clipped visual Top-p."""
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
from research.semantic_token_cd.prompt_attn_l11_top_p_rollout import ARM_THRESHOLDS


SHORT = {
    "google_robot_open_drawer": "open_drawer",
    "google_robot_close_drawer": "close_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
}
LABELS = {"l11_matched": "Matched", "l11_top_p75": "TopP75",
          "l11_top_p80": "TopP80", "l11_top_p85": "TopP85"}
METRICS = ("actual_selected_count", "m_raw", "selected_attention_mass",
           "feature_perturbation_relative", "centered_logit_residual_norm",
           "guided_changed_dims", "guided_change_ratio", "guided_clean_action_l2")


def exact_mcnemar(rescue: int, harm: int) -> float:
    total = rescue + harm
    if not total:
        return 1.0
    tail = min(rescue, harm)
    return min(1.0, 2.0 * sum(math.comb(total, x) for x in range(tail + 1)) / 2 ** total)


def holm(values: dict[str, float]) -> dict[str, float]:
    keys = sorted(values, key=values.get)
    result = {}; running = 0.0
    for rank, key in enumerate(keys):
        running = max(running, min(1.0, (len(keys) - rank) * values[key]))
        result[key] = running
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--matched-artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact.resolve(); matched = args.matched_artifact.resolve()
    arms = tuple(LABELS)
    results = defaultdict(dict); summaries = {}; step_rows = []

    for task in TASKS:
        for arm in arms:
            base = matched if arm == "l11_matched" else root
            for seed in range(100, 200):
                path = base / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                if not path.exists():
                    raise RuntimeError(f"missing result: {path}")
                payload = json.loads(path.read_text())
                if payload.get("technical_pass") is not True:
                    raise RuntimeError(f"technical audit failed: {path}")
                results[(task, arm)][seed] = bool(payload["success"])
                summaries[(task, arm, seed)] = payload
                for index, step in enumerate(payload["selector_trace"]):
                    row = {"task": SHORT[task], "seed": seed, "arm": LABELS[arm], "step": index}
                    for metric in METRICS:
                        row[metric] = step.get(metric)
                    row["lower_bound_triggered"] = bool(step.get("lower_bound_triggered", False))
                    row["upper_bound_triggered"] = bool(step.get("upper_bound_triggered", False))
                    step_rows.append(row)

    for task in TASKS:
        for seed in range(100, 200):
            group = [summaries[(task, arm, seed)] for arm in arms]
            for key in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
                if len({item[key] for item in group}) != 1:
                    raise RuntimeError(f"paired hash mismatch: {task}/{seed}/{key}")

    success_rows = []
    for task_key in [*TASKS, "Overall"]:
        chosen = TASKS if task_key == "Overall" else (task_key,)
        n = 100 * len(chosen)
        row = {"task": "Overall" if task_key == "Overall" else SHORT[task_key], "n": n}
        for arm in arms:
            count = sum(results[(task, arm)][seed] for task in chosen for seed in range(100, 200))
            row[f"{LABELS[arm]}_success"] = count; row[f"{LABELS[arm]}_rate"] = count / n
        success_rows.append(row)

    paired_rows = []; overall_p = {}
    for arm in ARM_THRESHOLDS:
        for task_key in [*TASKS, "Overall"]:
            chosen = TASKS if task_key == "Overall" else (task_key,)
            pairs = [(results[(task, "l11_matched")][seed], results[(task, arm)][seed])
                     for task in chosen for seed in range(100, 200)]
            rescue = sum(not old and new for old, new in pairs)
            harm = sum(old and not new for old, new in pairs)
            p = exact_mcnemar(rescue, harm)
            paired_rows.append({"task": "Overall" if task_key == "Overall" else SHORT[task_key],
                                "arm": LABELS[arm], "n": len(pairs), "rescue": rescue,
                                "harm": harm, "net": rescue - harm, "exact_mcnemar_p": p})
            if task_key == "Overall": overall_p[arm] = p
    adjusted = holm(overall_p)
    for row in paired_rows:
        row["holm_adjusted_p_over_3_overall_tests"] = (
            adjusted[next(arm for arm in ARM_THRESHOLDS if LABELS[arm] == row["arm"])]
            if row["task"] == "Overall" else ""
        )

    distribution_rows = []
    for task_key in [*[SHORT[x] for x in TASKS], "Overall"]:
        for arm in LABELS.values():
            selected = [row for row in step_rows if row["arm"] == arm and
                        (task_key == "Overall" or row["task"] == task_key)]
            counts = np.asarray([row["actual_selected_count"] for row in selected], dtype=float)
            raw = np.asarray([row["m_raw"] for row in selected if row["m_raw"] is not None], dtype=float)
            distribution_rows.append({
                "task": task_key, "arm": arm, "control_steps": len(selected),
                "mean_m_t": float(counts.mean()), "std_m_t": float(counts.std()),
                "min_m_t": int(counts.min()), "p10_m_t": float(np.percentile(counts, 10)),
                "median_m_t": float(np.median(counts)), "p90_m_t": float(np.percentile(counts, 90)),
                "max_m_t": int(counts.max()), "mean_m_raw": float(raw.mean()) if len(raw) else "",
                "lower_trigger_rate": float(np.mean([row["lower_bound_triggered"] for row in selected])),
                "upper_trigger_rate": float(np.mean([row["upper_bound_triggered"] for row in selected])),
                "mean_attention_mass": float(np.mean([row["selected_attention_mass"] for row in selected
                                                        if row["selected_attention_mass"] is not None]))
                if arm != "Matched" else "",
            })

    mechanism_rows = []
    for task_key in [*[SHORT[x] for x in TASKS], "Overall"]:
        for arm in LABELS.values():
            selected = [row for row in step_rows if row["arm"] == arm and
                        (task_key == "Overall" or row["task"] == task_key)]
            item = {"task": task_key, "arm": arm, "control_steps": len(selected)}
            for metric in METRICS[3:]:
                item[f"mean_{metric}"] = float(np.mean([row[metric] for row in selected]))
            mechanism_rows.append(item)

    statistics = root / "statistics"; paired = root / "paired_results"
    statistics.mkdir(exist_ok=True); paired.mkdir(exist_ok=True)
    write_csv(statistics / "success_rates.csv", success_rows)
    write_csv(statistics / "adaptive_count_distribution.csv", distribution_rows)
    write_csv(statistics / "intervention_effects.csv", mechanism_rows)
    write_csv(statistics / "per_step_adaptive_counts.csv", step_rows)
    write_csv(paired / "paired_vs_matched.csv", paired_rows)
    (statistics / "success_rates.json").write_text(json.dumps(success_rows, indent=2) + "\n")
    (paired / "paired_vs_matched.json").write_text(json.dumps(paired_rows, indent=2) + "\n")

    x = [75, 80, 85]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for row in success_rows:
        axes[0].plot(x, [100 * row[f"TopP{value}_rate"] for value in x], marker="o", label=row["task"])
    axes[0].axhline(100 * success_rows[-1]["Matched_rate"], color="black", ls="--", label="Matched overall")
    axes[0].set(xlabel="Cumulative visual attention threshold (%)", ylabel="Success rate (%)")
    axes[0].grid(alpha=.25); axes[0].legend(fontsize=7)
    overall_dist = {row["arm"]: row for row in distribution_rows if row["task"] == "Overall"}
    axes[1].errorbar(x, [overall_dist[f"TopP{value}"]["mean_m_t"] for value in x],
                     yerr=[overall_dist[f"TopP{value}"]["std_m_t"] for value in x], marker="o")
    axes[1].set(xlabel="Cumulative visual attention threshold (%)", ylabel="Selected tokens (mean +/- SD)")
    axes[1].grid(alpha=.25); fig.tight_layout(); fig.savefig(statistics / "threshold_success_and_count.png", dpi=180)
    plt.close(fig)

    def cell(row: dict, label: str) -> str:
        return f'{row[f"{label}_success"]}/{row["n"]} ({100*row[f"{label}_rate"]:.1f}%)'
    lines = ["# L11 Visual Top-p Adaptive Count — Final Report", "",
             "All 1,200 new adaptive episodes completed; 400 paired Matched episodes were reused. Technical audits and initial snapshot/state/RGB hashes passed.", "",
             "| Task | Matched | TopP75 | TopP80 | TopP85 |", "|---|---:|---:|---:|---:|"]
    for row in success_rows:
        lines.append("| " + row["task"] + " | " + " | ".join(cell(row, label) for label in LABELS.values()) + " |")
    lines += ["", "| Arm | Rescue | Harm | Net | Exact p | Holm-adjusted p |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in paired_rows:
        if row["task"] == "Overall":
            lines.append(f'| {row["arm"]} | {row["rescue"]} | {row["harm"]} | {row["net"]} | {row["exact_mcnemar_p"]:.6g} | {row["holm_adjusted_p_over_3_overall_tests"]:.6g} |')
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": True, "new_episodes": 1200, "reused_matched": 400,
                      "success": success_rows,
                      "overall_paired": [row for row in paired_rows if row["task"] == "Overall"]}, indent=2))


if __name__ == "__main__":
    main()
