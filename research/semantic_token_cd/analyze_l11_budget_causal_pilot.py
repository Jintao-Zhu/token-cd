"""Analyze the 25-seed four-task L11 matched-budget causal pilot."""
from __future__ import annotations

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
TASK_LABEL = {task: task.removeprefix("google_robot_") for task in TASKS}
ARMS = (
    "global_shuffle", "within_task_shuffle", "episode_fixed",
    "matched_scale_050", "matched_scale_075", "matched_scale_125", "matched_scale_150",
)


def exact_mcnemar(rescue: int, harm: int) -> float:
    discordant = rescue + harm
    if discordant == 0:
        return 1.0
    low = min(rescue, harm)
    tail = sum(math.comb(discordant, index) for index in range(low + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def trace_values(summary: dict, key: str) -> np.ndarray:
    values = [item.get(key) for item in summary.get("selector_trace", [])]
    return np.asarray([
        float(value) for value in values
        if value is not None and np.isfinite(float(value))
    ], dtype=np.float64)


def finite_stat(values: np.ndarray, reducer, default: float = float("nan")) -> float:
    return float(reducer(values)) if values.size else default


def phase_name(index: int, length: int) -> str:
    fraction = (index + .5) / max(length, 1)
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def centered_correlation(rows: list[dict], x_key: str, y_key: str) -> float:
    centered_x = []
    centered_y = []
    for task in TASKS:
        selected = [row for row in rows if row["task_id"] == task]
        x = np.asarray([row[x_key] for row in selected], dtype=np.float64)
        y = np.asarray([row[y_key] for row in selected], dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            continue
        centered_x.extend((x[valid] - x[valid].mean()).tolist())
        centered_y.extend((y[valid] - y[valid].mean()).tolist())
    x = np.asarray(centered_x, dtype=np.float64)
    y = np.asarray(centered_y, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--matched-artifact", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-end", type=int, default=124)
    args = parser.parse_args()
    root = args.artifact.resolve()
    matched_root = args.matched_artifact.resolve()
    seeds = range(args.seed_start, args.seed_end + 1)
    outcomes: dict[tuple[str, int, str], bool] = {}
    summaries: dict[tuple[str, int, str], dict] = {}
    for task in TASKS:
        for seed in seeds:
            matched_path = matched_root / "episodes" / task / "l11_matched" / f"episode_{seed:03d}_summary.json"
            matched = json.loads(matched_path.read_text())
            outcomes[(task, seed, "matched")] = bool(matched["success"])
            summaries[(task, seed, "matched")] = matched
            for arm in ARMS:
                path = root / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
                summary = json.loads(path.read_text())
                if not summary.get("technical_pass"):
                    raise RuntimeError(f"technical audit failed: {path}")
                if (
                    summary["canonical_snapshot_sha256"] != matched["canonical_snapshot_sha256"]
                    or summary["initial_state_sha256"] != matched["initial_state_sha256"]
                    or summary["initial_rgb_sha256"] != matched["initial_rgb_sha256"]
                ):
                    raise RuntimeError(f"pairing mismatch: {path}")
                outcomes[(task, seed, arm)] = bool(summary["success"])
                summaries[(task, seed, arm)] = summary

    rows = []
    for task_key in (*TASKS, "Overall"):
        selected_tasks = TASKS if task_key == "Overall" else (task_key,)
        for arm in ARMS:
            paired = [
                (outcomes[(task, seed, "matched")], outcomes[(task, seed, arm)])
                for task in selected_tasks for seed in seeds
            ]
            matched_success = sum(base for base, _ in paired)
            arm_success = sum(candidate for _, candidate in paired)
            rescue = sum((not base) and candidate for base, candidate in paired)
            harm = sum(base and (not candidate) for base, candidate in paired)
            arm_summaries = [summaries[(task, seed, arm)] for task in selected_tasks for seed in seeds]
            rows.append({
                "task": "Overall" if task_key == "Overall" else TASK_LABEL[task_key],
                "arm": arm,
                "episodes": len(paired),
                "matched_success": matched_success,
                "arm_success": arm_success,
                "success_delta": arm_success - matched_success,
                "rescue": rescue,
                "harm": harm,
                "net": rescue - harm,
                "exact_p": exact_mcnemar(rescue, harm),
                "mean_selected_count": float(np.mean([item["mean_selected_token_count"] for item in arm_summaries])),
                "mean_feature_perturbation_relative": float(np.mean([item["mean_feature_perturbation_relative"] for item in arm_summaries])),
                "mean_logit_residual": float(np.mean([item["mean_centered_logit_residual_norm"] for item in arm_summaries])),
                "technical_pass": 1,
            })
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_csv(analysis / "PAIRED_RESULTS.csv", rows)

    episode_rows = []
    phase_accumulator: dict[tuple[str, str, str], list[dict]] = {}
    for task in TASKS:
        for seed in seeds:
            matched = summaries[(task, seed, "matched")]
            matched_counts = trace_values(matched, "actual_selected_count")
            matched_perturbation = trace_values(matched, "feature_perturbation_relative")
            matched_residual = trace_values(matched, "centered_logit_residual_norm")
            for arm in ARMS:
                summary = summaries[(task, seed, arm)]
                counts = trace_values(summary, "actual_selected_count")
                perturbation = trace_values(summary, "feature_perturbation_relative")
                residual = trace_values(summary, "centered_logit_residual_norm")
                action_l2 = trace_values(summary, "guided_clean_action_l2")
                changed_dims = trace_values(summary, "guided_changed_dims")
                row = {
                    "task_id": task,
                    "task": TASK_LABEL[task],
                    "seed": seed,
                    "arm": arm,
                    "matched_success": int(matched["success"]),
                    "arm_success": int(summary["success"]),
                    "outcome_delta": int(summary["success"]) - int(matched["success"]),
                    "control_steps": int(summary["control_steps"]),
                    "mean_count": finite_stat(counts, np.mean),
                    "std_count": finite_stat(counts, np.std),
                    "count_range": finite_stat(counts, np.ptp),
                    "mean_perturbation": finite_stat(perturbation, np.mean),
                    "mean_residual": finite_stat(residual, np.mean),
                    "mean_action_l2": finite_stat(action_l2, np.mean),
                    "mean_changed_dims": finite_stat(changed_dims, np.mean),
                    "action_jitter": float(summary["action_jitter_index"]),
                    "matched_mean_count": finite_stat(matched_counts, np.mean),
                    "matched_std_count": finite_stat(matched_counts, np.std),
                    "matched_mean_perturbation": finite_stat(matched_perturbation, np.mean),
                    "matched_mean_residual": finite_stat(matched_residual, np.mean),
                }
                row["mean_count_delta"] = row["mean_count"] - row["matched_mean_count"]
                row["std_count_delta"] = row["std_count"] - row["matched_std_count"]
                row["mean_perturbation_delta"] = row["mean_perturbation"] - row["matched_mean_perturbation"]
                row["mean_residual_delta"] = row["mean_residual"] - row["matched_mean_residual"]
                episode_rows.append(row)
                trace = summary.get("selector_trace", [])
                for index, item in enumerate(trace):
                    phase = phase_name(index, len(trace))
                    phase_accumulator.setdefault((task, arm, phase), []).append(item)
    write_csv(analysis / "EPISODE_MECHANISMS.csv", episode_rows)

    phase_rows = []
    for (task, arm, phase), items in sorted(phase_accumulator.items()):
        def item_mean(key: str) -> float:
            values = np.asarray([
                float(item[key]) for item in items
                if item.get(key) is not None and np.isfinite(float(item[key]))
            ], dtype=np.float64)
            return finite_stat(values, np.mean)
        phase_rows.append({
            "task": TASK_LABEL[task],
            "arm": arm,
            "phase": phase,
            "steps": len(items),
            "mean_count": item_mean("actual_selected_count"),
            "mean_perturbation": item_mean("feature_perturbation_relative"),
            "mean_residual": item_mean("centered_logit_residual_norm"),
            "mean_action_l2": item_mean("guided_clean_action_l2"),
            "mean_changed_dims": item_mean("guided_changed_dims"),
        })
    write_csv(analysis / "PHASE_MECHANISMS.csv", phase_rows)
    overall = {row["arm"]: row for row in rows if row["task"] == "Overall"}
    primary_net = -int(overall["within_task_shuffle"]["net"])
    temporal_net = -int(overall["episode_fixed"]["net"])
    global_net = -int(overall["global_shuffle"]["net"])
    scale_deltas = {arm: int(row["net"]) for arm, row in overall.items() if arm.startswith("matched_scale")}
    extend = (
        primary_net >= 5 or temporal_net >= 5 or global_net >= 5
        or max((abs(value) for value in scale_deltas.values()), default=0) >= 5
    )
    decision = {
        "pilot_complete": True,
        "episodes_new": len(TASKS) * len(list(seeds)) * len(ARMS),
        "matched_reused": len(TASKS) * len(list(seeds)),
        "screening_rule_frozen_before_rollout": True,
        "matched_advantage_vs_within_task_shuffle": primary_net,
        "matched_advantage_vs_global_shuffle": global_net,
        "matched_advantage_vs_episode_fixed": temporal_net,
        "scale_arm_net_vs_matched": scale_deltas,
        "extend_to_100_seeds": extend,
        "extension_rule": "Extend if any primary/shuffle/temporal contrast or any scale contrast has absolute paired net >=5/100 pilot episodes.",
        "mechanism_correlations_task_centered": {
            "count_vs_feature_perturbation": centered_correlation(
                episode_rows, "mean_count", "mean_perturbation"
            ),
            "count_vs_logit_residual": centered_correlation(
                episode_rows, "mean_count", "mean_residual"
            ),
            "feature_perturbation_vs_logit_residual": centered_correlation(
                episode_rows, "mean_perturbation", "mean_residual"
            ),
        },
    }
    (analysis / "PILOT_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")
    lines = [
        "# L11 matched-budget causal pilot", "",
        f"Seeds {args.seed_start}–{args.seed_end}, four tasks. Matched is reused; {decision['episodes_new']} candidate episodes are newly run.", "",
        "| Arm | Matched | Candidate | Rescue | Harm | Net | Exact p | Mean count |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = overall[arm]
        lines.append(
            f"| {arm} | {row['matched_success']}/{row['episodes']} | {row['arm_success']}/{row['episodes']} | "
            f"{row['rescue']} | {row['harm']} | {row['net']:+d} | {row['exact_p']:.6g} | {row['mean_selected_count']:.2f} |"
        )
    lines += ["", "## Frozen screening decision", "", f"**{'EXTEND' if extend else 'STOP'}** to the full 100-seed study.", ""]
    correlations = decision["mechanism_correlations_task_centered"]
    lines += [
        "## Mechanism decomposition", "",
        "The contrasts have distinct causal meanings:", "",
        "- `matched > global_shuffle`: the budget is not explained by its marginal distribution alone.",
        "- `matched > within_task_shuffle`: the budget must remain paired with its own state, beyond task identity.",
        "- `matched > episode_fixed`: within-episode budget changes matter.",
        "- A peak at scale 1.0: the absolute KMeans-derived intervention dose is locally calibrated.", "",
        "Task-centered episode-level correlations trace how count changes propagate through the negative branch:", "",
        f"- count vs feature perturbation: {correlations['count_vs_feature_perturbation']:.4f}",
        f"- count vs centered logit residual: {correlations['count_vs_logit_residual']:.4f}",
        f"- feature perturbation vs centered logit residual: {correlations['feature_perturbation_vs_logit_residual']:.4f}", "",
        "These correlations establish the intervention-strength pathway; paired success contrasts determine whether that pathway is beneficial.", "",
        "Detailed evidence is saved in `EPISODE_MECHANISMS.csv` and `PHASE_MECHANISMS.csv`.", "",
    ]
    (analysis / "REPORT.md").write_text("\n".join(lines))
    (root / "COMPLETE.json").write_text(json.dumps({"complete": True, "analysis": "analysis/PILOT_DECISION.json"}, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
