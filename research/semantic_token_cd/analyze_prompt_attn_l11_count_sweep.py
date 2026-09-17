"""Final paired and mechanism analysis for the 4x100x6 L11 count sweep."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("artifacts/prompt_attn_l11_token_count_v1")
TASKS = {
    "google_robot_open_drawer": "open_drawer",
    "google_robot_close_drawer": "close_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
}
ARMS = {
    "l11_matched": "Matched", "l11_k16": "K16", "l11_k24": "K24",
    "l11_k32": "K32", "l11_k48": "K48", "l11_k64": "K64",
}
FIXED = {"l11_k16": 16, "l11_k24": 24, "l11_k32": 32, "l11_k48": 48, "l11_k64": 64}
METRICS = (
    "actual_selected_count", "mask_component_count", "isolated_token_ratio",
    "feature_perturbation_norm", "feature_perturbation_relative",
    "centered_logit_residual_norm", "guided_changed_dims", "guided_change_ratio",
    "guided_clean_action_l2",
)


def exact_mcnemar(rescue: int, harm: int) -> float:
    discordant = rescue + harm
    if discordant == 0:
        return 1.0
    tail = min(rescue, harm)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    result = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * values[key]))
        result[key] = running
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    episodes = ROOT / "episodes"
    results = defaultdict(dict)
    summaries = {}
    mechanism = defaultdict(lambda: defaultdict(float))
    matched_counts = defaultdict(list)
    technical_pass = True

    for task, short in TASKS.items():
        for arm in ARMS:
            files = sorted((episodes / task / arm).glob("episode_*_summary.json"))
            for path in files:
                payload = json.loads(path.read_text())
                seed = int(payload["seed"])
                if seed < 100 or seed > 199:
                    continue
                results[(task, arm)][seed] = bool(payload["success"])
                summaries[(task, arm, seed)] = payload
                technical_pass = technical_pass and payload.get("technical_pass") is True
                for step in payload["selector_trace"]:
                    mechanism[(task, arm)]["steps"] += 1
                    for metric in METRICS:
                        value = step.get(metric)
                        if value is not None and np.isfinite(value):
                            mechanism[(task, arm)][metric] += float(value)
                    if arm == "l11_matched":
                        matched_counts[task].append(int(step["actual_selected_count"]))

    expected = set(range(100, 200))
    missing = {
        f"{task}/{arm}": sorted(expected - set(results[(task, arm)]))
        for task in TASKS for arm in ARMS if set(results[(task, arm)]) != expected
    }
    if missing:
        raise RuntimeError({"missing_or_extra": missing})

    hash_pass = True
    for task in TASKS:
        for seed in expected:
            rows = [summaries[(task, arm, seed)] for arm in ARMS]
            for key in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
                hash_pass = hash_pass and len({row[key] for row in rows}) == 1
    if not technical_pass or not hash_pass:
        raise RuntimeError({"technical_pass": technical_pass, "paired_hash_pass": hash_pass})

    success_rows = []
    totals = {arm: 0 for arm in ARMS}
    for task, short in TASKS.items():
        row = {"task": short, "n": 100}
        for arm, label in ARMS.items():
            success = sum(results[(task, arm)].values())
            totals[arm] += success
            row[f"{label}_success"] = success
            row[f"{label}_rate"] = success / 100
        success_rows.append(row)
    overall = {"task": "Overall", "n": 400}
    for arm, label in ARMS.items():
        overall[f"{label}_success"] = totals[arm]
        overall[f"{label}_rate"] = totals[arm] / 400
    success_rows.append(overall)

    paired_rows = []
    overall_p = {}
    for arm, count in FIXED.items():
        for task_key in list(TASKS) + ["Overall"]:
            pairs = []
            selected_tasks = TASKS if task_key == "Overall" else [task_key]
            for task in selected_tasks:
                pairs.extend((results[(task, "l11_matched")][seed], results[(task, arm)][seed]) for seed in expected)
            rescue = sum((not matched) and fixed for matched, fixed in pairs)
            harm = sum(matched and (not fixed) for matched, fixed in pairs)
            p_value = exact_mcnemar(rescue, harm)
            row = {
                "task": "Overall" if task_key == "Overall" else TASKS[task_key],
                "arm": ARMS[arm], "n": len(pairs), "rescue": rescue, "harm": harm,
                "net": rescue - harm, "exact_mcnemar_p": p_value,
            }
            paired_rows.append(row)
            if task_key == "Overall":
                overall_p[arm] = p_value
    adjusted = holm(overall_p)
    for row in paired_rows:
        if row["task"] == "Overall":
            arm = next(key for key, label in ARMS.items() if label == row["arm"])
            row["holm_adjusted_p_over_5_overall_tests"] = adjusted[arm]
        else:
            row["holm_adjusted_p_over_5_overall_tests"] = ""

    mechanism_rows = []
    for task, short in TASKS.items():
        for arm, label in ARMS.items():
            sums = mechanism[(task, arm)]
            steps = int(sums["steps"])
            row = {"task": short, "arm": label, "control_steps": steps}
            for metric in METRICS:
                row[f"mean_{metric}"] = sums[metric] / steps
            mechanism_rows.append(row)
    for arm, label in ARMS.items():
        selected = [row for row in mechanism_rows if row["arm"] == label]
        steps = sum(row["control_steps"] for row in selected)
        row = {"task": "Overall", "arm": label, "control_steps": steps}
        for metric in METRICS:
            row[f"mean_{metric}"] = sum(
                item[f"mean_{metric}"] * item["control_steps"] for item in selected
            ) / steps
        mechanism_rows.append(row)

    matched_rows = []
    for task_key in list(TASKS) + ["Overall"]:
        values = np.asarray(
            sum((matched_counts[task] for task in TASKS), [])
            if task_key == "Overall" else matched_counts[task_key], dtype=np.float64,
        )
        matched_rows.append({
            "task": "Overall" if task_key == "Overall" else TASKS[task_key],
            "control_steps": len(values), "mean": values.mean(), "std": values.std(),
            "min": int(values.min()), "p10": np.percentile(values, 10),
            "p25": np.percentile(values, 25), "median": np.median(values),
            "p75": np.percentile(values, 75), "p90": np.percentile(values, 90),
            "max": int(values.max()),
        })

    statistics = ROOT / "statistics"
    paired_dir = ROOT / "paired_results"
    statistics.mkdir(exist_ok=True)
    paired_dir.mkdir(exist_ok=True)
    write_csv(statistics / "success_rates.csv", success_rows)
    write_csv(statistics / "mechanism_stats.csv", mechanism_rows)
    write_csv(statistics / "matched_count_distribution.csv", matched_rows)
    write_csv(paired_dir / "paired_vs_matched.csv", paired_rows)
    (statistics / "success_rates.json").write_text(json.dumps(success_rows, indent=2) + "\n")
    (paired_dir / "paired_vs_matched.json").write_text(json.dumps(paired_rows, indent=2) + "\n")

    fixed_arms = list(FIXED)
    x = np.asarray([FIXED[arm] for arm in fixed_arms])
    fig, ax = plt.subplots(figsize=(8, 5))
    for task, short in list(TASKS.items()) + [("Overall", "Overall")]:
        row = next(value for value in success_rows if value["task"] == short)
        rates = [100 * row[f"{ARMS[arm]}_rate"] for arm in fixed_arms]
        ax.plot(x, rates, marker="o", label=short)
    overall_row = success_rows[-1]
    matched_mean = next(row["mean"] for row in matched_rows if row["task"] == "Overall")
    ax.scatter([matched_mean], [100 * overall_row["Matched_rate"]], marker="*", s=180,
               color="black", label=f"Matched (mean m={matched_mean:.1f})", zorder=5)
    ax.set(xlabel="Fixed selected visual-token count", ylabel="Success rate (%)",
           title="L11 token count vs closed-loop success")
    ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(statistics / "token_count_success_curve.png", dpi=180)
    plt.close(fig)

    overall_mech = {row["arm"]: row for row in mechanism_rows if row["task"] == "Overall"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    plot_metrics = (
        ("mean_feature_perturbation_relative", "Relative feature perturbation"),
        ("mean_centered_logit_residual_norm", "Centered logit residual norm"),
        ("mean_guided_changed_dims", "Changed action dimensions"),
    )
    for ax, (metric, title) in zip(axes, plot_metrics):
        ax.plot(x, [overall_mech[ARMS[arm]][metric] for arm in fixed_arms], marker="o")
        ax.set(xlabel="Selected token count", title=title); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(statistics / "token_count_intervention_effects.png", dpi=180)
    plt.close(fig)

    def fmt_rate(row, label):
        return f'{row[f"{label}_success"]}/' + str(row["n"]) + f' ({100*row[f"{label}_rate"]:.1f}%)'

    lines = [
        "# L11 Token-Count Sweep — Final Report", "",
        "All 2,400 arm-episodes completed on canonical seeds 100–199. All per-episode technical audits and paired initial snapshot/state/RGB hashes passed.", "",
        "## Success rates", "",
        "| Task | Matched | K16 | K24 | K32 | K48 | K64 |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in success_rows:
        lines.append("| " + row["task"] + " | " + " | ".join(fmt_rate(row, label) for label in ARMS.values()) + " |")
    lines += ["", "## Overall paired comparison against Matched", "",
              "| Arm | Rescue | Harm | Net | Exact p | Holm-adjusted p |", "|---|---:|---:|---:|---:|---:|"]
    for row in paired_rows:
        if row["task"] == "Overall":
            lines.append(f'| {row["arm"]} | {row["rescue"]} | {row["harm"]} | {row["net"]} | {row["exact_mcnemar_p"]:.6g} | {row["holm_adjusted_p_over_5_overall_tests"]:.6g} |')
    best_fixed = max(FIXED, key=lambda arm: totals[arm])
    lines += [
        "", "## Main result", "",
        f'Matched achieved {totals["l11_matched"]}/400 ({100*totals["l11_matched"]/400:.1f}%). '
        f'The best fixed count was {ARMS[best_fixed]} at {totals[best_fixed]}/400 ({100*totals[best_fixed]/400:.1f}%).',
        "The fixed-count curve rises from K16 toward K48 and falls at K64. This supports an effective intervention range around 32–48 tokens, while no single fixed count dominates Matched across tasks.",
        "", "Matched uses KMeans only for its own-state token budget; token identity remains the L11 attention ranking.",
    ]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "complete": True, "episodes": 2400, "technical_pass": technical_pass,
        "paired_hash_pass": hash_pass, "success": success_rows,
        "overall_paired": [row for row in paired_rows if row["task"] == "Overall"],
    }, indent=2))


if __name__ == "__main__":
    main()
