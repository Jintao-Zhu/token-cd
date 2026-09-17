"""Aggregate Target Positive-Boost token-swap causal audit."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from research.semantic_token_cd.target_boost_token_causal_worker import ARTIFACT


def mean(rows: list[dict], key: str):
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    files = sorted((ARTIFACT / "results").rglob("seed_*.json"))
    if len(files) != 400:
        raise RuntimeError(f"incomplete audit: {len(files)}/400 episodes")
    states, swaps = [], []
    for path in files:
        data = json.loads(path.read_text())
        if not data.get("technical_pass"):
            raise RuntimeError(f"technical failure: {path}")
        task = data["task"].removeprefix("google_robot_")
        for state in data["states"]:
            c, b = state["correct"], state["boost"]
            row = {
                "task": task, "seed": data["seed"], "stage": state["stage"],
                "category": data["outcome_category"], "step": state["step"], "m": state["m"],
                "swap_count": state["swap_count"],
                "historical_mask_jaccard": state["historical_correct_mask_jaccard"],
                "residual_cosine": state["boost_residual_cosine_vs_correct"],
                "action_l2": state["boost_action_l2_vs_correct"],
                "action_changed_dims": state["boost_final_changed_dims_vs_correct"],
                "feature_norm_delta": b["feature_perturbation_norm"] - c["feature_perturbation_norm"],
                "residual_norm_delta": b["centered_residual_norm"] - c["centered_residual_norm"],
                "component_delta": b["mask_component_count"] - c["mask_component_count"],
                "isolated_ratio_delta": b["isolated_token_ratio"] - c["isolated_token_ratio"],
                "max_single_swap_action_l2": max(
                    [float(x["only_action_l2_vs_correct"]) for x in state["swaps"]], default=0.0
                ),
                "sum_single_swap_action_l2": sum(
                    float(x["only_action_l2_vs_correct"]) for x in state["swaps"]
                ),
            }
            row["interaction_only_action_change"] = int(
                row["action_l2"] > 1e-9 and row["max_single_swap_action_l2"] <= 1e-9
            )
            states.append(row)
            for swap in state["swaps"]:
                swaps.append({"task": task, "seed": data["seed"], "stage": state["stage"],
                    "category": data["outcome_category"], "step": state["step"], **swap})
    write_csv(ARTIFACT / "state_rows.csv", states)
    write_csv(ARTIFACT / "atomic_swap_rows.csv", swaps)

    summaries = []
    for keys in ((), ("category",), ("task",), ("stage",), ("task", "category"),
                 ("stage", "category")):
        grouped = defaultdict(list)
        for row in states:
            grouped[tuple(row[key] for key in keys)].append(row)
        for values, subset in sorted(grouped.items()):
            record = {key: value for key, value in zip(keys, values)}
            record.update({"n_states": len(subset), "episodes": len({(x["task"], x["seed"]) for x in subset}),
                "mean_m": mean(subset, "m"), "mean_swap_count": mean(subset, "swap_count"),
                "zero_swap_fraction": float(np.mean([x["swap_count"] == 0 for x in subset])),
                "mean_residual_cosine": mean(subset, "residual_cosine"),
                "mean_action_l2": mean(subset, "action_l2"),
                "action_flip_state_fraction": float(np.mean([x["action_changed_dims"] > 0 for x in subset])),
                "mean_residual_norm_delta": mean(subset, "residual_norm_delta"),
                "mean_feature_norm_delta": mean(subset, "feature_norm_delta"),
                "interaction_only_action_fraction": float(np.mean(
                    [x["interaction_only_action_change"] for x in subset]
                )),
                "mean_historical_mask_jaccard": mean(subset, "historical_mask_jaccard")})
            summaries.append(record)
    (ARTIFACT / "GROUP_SUMMARIES.json").write_text(json.dumps(summaries, indent=2) + "\n")

    atomic_summary = []
    for keys in ((), ("category",), ("task",), ("stage",), ("task", "category")):
        grouped = defaultdict(list)
        for row in swaps:
            grouped[tuple(row[key] for key in keys)].append(row)
        for values, subset in sorted(grouped.items()):
            record = {key: value for key, value in zip(keys, values)}
            record.update({"n_swaps": len(subset),
                "mean_entered_p_rank": mean(subset, "entered_p_rank"),
                "mean_exited_p_rank": mean(subset, "exited_p_rank"),
                "mean_entered_difference": mean(subset, "entered_difference"),
                "mean_exited_difference": mean(subset, "exited_difference"),
                "mean_only_action_l2": mean(subset, "only_action_l2_vs_correct"),
                "mean_undo_action_l2": mean(subset, "undo_action_l2_vs_boost"),
                "mean_only_residual_cosine": mean(subset, "only_residual_cosine_vs_correct"),
                "mean_undo_residual_cosine": mean(subset, "undo_residual_cosine_vs_boost")})
            atomic_summary.append(record)
    (ARTIFACT / "ATOMIC_SUMMARIES.json").write_text(json.dumps(atomic_summary, indent=2) + "\n")

    tests = {}
    for metric in ("swap_count", "residual_cosine", "action_l2", "residual_norm_delta", "feature_norm_delta"):
        harm = [x[metric] for x in states if x["category"] == "harm"]
        rescue = [x[metric] for x in states if x["category"] == "rescue"]
        result = mannwhitneyu(harm, rescue, alternative="two-sided")
        tests[metric] = {"harm_mean": float(np.mean(harm)), "rescue_mean": float(np.mean(rescue)),
                         "statistic": float(result.statistic), "p": float(result.pvalue)}
    atomic_tests = {}
    for metric in ("entered_p_rank", "entered_difference", "only_action_l2_vs_correct",
                   "only_residual_cosine_vs_correct"):
        harm = [x[metric] for x in swaps if x["category"] == "harm"]
        rescue = [x[metric] for x in swaps if x["category"] == "rescue"]
        result = mannwhitneyu(harm, rescue, alternative="two-sided")
        atomic_tests[metric] = {"harm_mean": float(np.mean(harm)),
                                "rescue_mean": float(np.mean(rescue)),
                                "statistic": float(result.statistic), "p": float(result.pvalue)}
    # Exploratory only: these thresholds are inspected after outcomes and must
    # be validated on held-out scenes before becoming a selector rule.
    gate_diagnostics = []
    for threshold in (0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010):
        record = {"entered_difference_min": threshold, "entered_p_rank_max": 40}
        for category in ("rescue", "harm"):
            subset = [x for x in swaps if x["category"] == category]
            record[f"{category}_swap_retention"] = float(np.mean([
                x["entered_p_rank"] <= 40 and x["entered_difference"] >= threshold
                for x in subset
            ]))
        gate_diagnostics.append(record)
    overall = next(x for x in summaries if not any(k in x for k in ("task", "category", "stage")))
    outcome = {x["category"]: x for x in summaries if "category" in x and
               not any(k in x for k in ("task", "stage"))}
    final = {"protocol_id": "TARGET_BOOST_TOKEN_CAUSAL_AUDIT_V1", "complete": True,
             "episodes": 400, "states": len(states), "atomic_swaps": len(swaps),
             "overall": overall, "by_outcome": outcome, "harm_vs_rescue_tests": tests,
             "atomic_harm_vs_rescue_tests": atomic_tests,
             "posthoc_confidence_gate_diagnostics": gate_diagnostics,
             "interpretation_boundary": "same-state mask/logit/action causality; no new closed-loop outcomes"}
    (ARTIFACT / "FINAL_RESULTS.json").write_text(json.dumps(final, indent=2) + "\n")
    lines = ["# Target Positive-Boost Token Causal Audit", "", "## Scale", "",
             f"- 400 episodes, {len(states)} early/middle/late same states, {len(swaps)} budget-preserving atomic swaps.",
             "- Every state compares Correct and Positive-Boost-0.5 on the same image and action prefix.",
             "", "## Overall", "",
             "| Group | States | Mean swaps | Zero-swap | Action-flip states | Interaction-only | Residual cosine | Action L2 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("rescue", "harm", "both_success", "both_fail"):
        x = outcome.get(name)
        if x:
            lines.append(f"| {name} | {x['n_states']} | {x['mean_swap_count']:.3f} | "
                         f"{100*x['zero_swap_fraction']:.1f}% | {100*x['action_flip_state_fraction']:.1f}% | "
                         f"{100*x['interaction_only_action_fraction']:.1f}% | "
                         f"{x['mean_residual_cosine']:.4f} | {x['mean_action_l2']:.4f} |")
    lines += ["", "## Harm versus Rescue", ""]
    for metric, value in tests.items():
        lines.append(f"- {metric}: Harm {value['harm_mean']:.5f}, Rescue {value['rescue_mean']:.5f}, p={value['p']:.4g}.")
    lines += ["", "## Atomic swap confidence", ""]
    for metric, value in atomic_tests.items():
        lines.append(f"- {metric}: Harm {value['harm_mean']:.5f}, Rescue {value['rescue_mean']:.5f}, p={value['p']:.4g}.")
    lines += ["", "The outcome-associated pattern is low-confidence rather than high-magnitude Harm: "
              "Harm swaps enter from deeper Correct-P ranks and carry smaller P-Q evidence. "
              "The exploratory gate `P rank <= 40 and P-Q >= 0.004` retains "
              f"{100*next(x for x in gate_diagnostics if x['entered_difference_min'] == .004)['rescue_swap_retention']:.1f}% "
              "of Rescue-associated swaps versus "
              f"{100*next(x for x in gate_diagnostics if x['entered_difference_min'] == .004)['harm_swap_retention']:.1f}% "
              "of Harm-associated swaps. This gate is post-hoc and requires held-out validation.",
              "", "## Reproducibility boundary", "",
              f"Mean recomputed-vs-historical Correct-mask Jaccard is {overall['mean_historical_mask_jaccard']:.3f}; "
              "it decreases under long replay because historical executed actions were stored at finite precision. "
              "Correct-vs-Boost comparisons within every audited state remain exact same-state comparisons, but "
              "outcome-conditioned middle/late associations are diagnostic rather than definitive causal attribution."]
    lines += ["", "## Files", "", "- `state_rows.csv`: one row per same state.",
              "- `atomic_swap_rows.csv`: one row per budget-preserving token swap.",
              "- `GROUP_SUMMARIES.json` and `ATOMIC_SUMMARIES.json`: task/stage/outcome aggregates.",
              "- Per-state masks, images and centered residuals are retained under `results/`."]
    (ARTIFACT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"episodes": 400, "states": len(states), "swaps": len(swaps)}))


if __name__ == "__main__":
    main()
