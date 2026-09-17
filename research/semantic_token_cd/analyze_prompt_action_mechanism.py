"""Mechanism analysis from existing Stage A/B and closed-loop artifacts."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.semantic_token_cd.prompt_action_complement_protocol import (
    ARMS, ARTIFACT, ORIGINAL_ROOT, TASKS, atomic_json,
)

OUT = ARTIFACT / "mechanism_analysis"
SHORT = {task: task.removeprefix("google_robot_") for task in TASKS}


def mean(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else None


def quantiles(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not values: return None
    return {key: float(value) for key, value in zip(
        ("q10", "median", "q90"), np.quantile(values, (0.1, 0.5, 0.9))
    )}


def jaccard(a, b):
    a, b = set(a), set(b); return len(a & b) / max(1, len(a | b))


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def analyze_stage_a():
    rows = []
    for path in sorted((ARTIFACT / "stage_a/states").glob("**/step_*.json")):
        data = json.loads(path.read_text()); original = data["arms"]["original"]["selected"]
        for arm in ARMS:
            spec = data["arms"][arm]
            rows.append({
                "task": data["task"], "seed": data["seed"], "control_step": data["control_step"],
                "arm": arm, "m": data["m"], "r": data["r"],
                "actual_replacements": spec["actual_replacements_vs_original"],
                "mask_jaccard_vs_original": jaccard(spec["selected"], original),
                "supplement_prompt_rank_mean": mean(spec["supplement_prompt_global_ranks"]),
                "supplement_action_rank_mean": mean(spec["supplement_action_global_ranks"]),
                "supplement_prompt_score_mean": mean(spec["supplement_prompt_scores"]),
                "supplement_action_score_mean": mean(spec["supplement_action_scores"]),
            })
    write_csv(OUT / "stage_a_selection_per_state.csv", rows)
    summary = {}
    for task in (*TASKS, "overall"):
        subset_task = rows if task == "overall" else [row for row in rows if row["task"] == task]
        summary[SHORT.get(task, task)] = {}
        for arm in ARMS:
            subset = [row for row in subset_task if row["arm"] == arm]
            summary[SHORT.get(task, task)][arm] = {
                "n_states": len(subset), "mean_m": mean(row["m"] for row in subset),
                "mean_r": mean(row["r"] for row in subset),
                "mean_actual_replacements": mean(row["actual_replacements"] for row in subset),
                "mean_jaccard_vs_original": mean(row["mask_jaccard_vs_original"] for row in subset),
                "mean_supplement_prompt_rank": mean(row["supplement_prompt_rank_mean"] for row in subset),
                "mean_supplement_action_rank": mean(row["supplement_action_rank_mean"] for row in subset),
                "supplement_prompt_rank_quantiles": quantiles(row["supplement_prompt_rank_mean"] for row in subset),
                "supplement_action_rank_quantiles": quantiles(row["supplement_action_rank_mean"] for row in subset),
            }
    atomic_json(OUT / "stage_a_selection_summary.json", summary)
    return summary


def analyze_stage_b():
    rows = []
    for path in sorted((ARTIFACT / "stage_b/states").glob("**/step_*.json")):
        data = json.loads(path.read_text()); original = data["metrics"]["original"]
        clean_margin = np.asarray(original["clean_top2_margin"], dtype=float)
        original_ids = np.asarray(original["final_token_ids"][:6])
        for arm in ARMS:
            item = data["metrics"][arm]
            final_ids = np.asarray(item["final_token_ids"][:6])
            flips = final_ids != original_ids
            action_delta = np.abs(np.asarray(item["guided_action"][:6]) - np.asarray(original["guided_action"][:6]))
            row = {
                "task": data["task"], "seed": data["seed"], "control_step": data["control_step"],
                "arm": arm, "m": data["m"], "feature_perturbation_norm": item["feature_perturbation_norm"],
                "feature_perturbation_relative": item["feature_perturbation_relative"],
                "centered_residual_norm": item["centered_residual_norm"],
                "residual_cosine_vs_original": item["residual_cosine_vs_original"],
                "residual_norm_ratio_vs_original": item["centered_residual_norm"] / max(1e-12, original["centered_residual_norm"]),
                "action_exact_vs_original": bool(np.array_equal(final_ids, original_ids)),
                "changed_dimensions_vs_original": int(flips.sum()),
                "mean_abs_action_delta_vs_original": float(action_delta.mean()),
                "clean_margin_on_flipped_dims": float(clean_margin[flips].mean()) if flips.any() else None,
                "clean_margin_on_unflipped_dims": float(clean_margin[~flips].mean()) if (~flips).any() else None,
            }
            for dimension in range(6):
                row[f"dim{dimension}_flip"] = bool(flips[dimension])
                row[f"dim{dimension}_abs_action_delta"] = float(action_delta[dimension])
            rows.append(row)
    write_csv(OUT / "stage_b_action_per_state.csv", rows)
    summary = {}
    for task in (*TASKS, "overall"):
        subset_task = rows if task == "overall" else [row for row in rows if row["task"] == task]
        summary[SHORT.get(task, task)] = {}
        for arm in ARMS:
            subset = [row for row in subset_task if row["arm"] == arm]
            summary[SHORT.get(task, task)][arm] = {
                "n_states": len(subset),
                "mean_feature_perturbation_norm": mean(row["feature_perturbation_norm"] for row in subset),
                "mean_residual_norm": mean(row["centered_residual_norm"] for row in subset),
                "mean_residual_norm_ratio_vs_original": mean(row["residual_norm_ratio_vs_original"] for row in subset),
                "mean_residual_cosine_vs_original": mean(row["residual_cosine_vs_original"] for row in subset),
                "action_changed_fraction_vs_original": mean(not row["action_exact_vs_original"] for row in subset),
                "mean_changed_dimensions_vs_original": mean(row["changed_dimensions_vs_original"] for row in subset),
                "mean_abs_action_delta_vs_original": mean(row["mean_abs_action_delta_vs_original"] for row in subset),
                "clean_margin_flipped": mean(row["clean_margin_on_flipped_dims"] for row in subset),
                "clean_margin_unflipped": mean(row["clean_margin_on_unflipped_dims"] for row in subset),
                "dimension_flip_rates": [mean(row[f"dim{dimension}_flip"] for row in subset) for dimension in range(6)],
            }
    atomic_json(OUT / "stage_b_action_summary.json", summary)
    return summary


def load_episode(task, arm, seed):
    path = (ORIGINAL_ROOT[task] / f"episode_{seed:03d}_summary.json" if arm == "original" else
            ARTIFACT / "closed_loop/episodes" / task / arm / f"episode_{seed:03d}_summary.json")
    return json.loads(path.read_text())


def outcome_category(a, b):
    if a and not b: return "rescue"
    if not a and b: return "harm"
    return "both_success" if a else "both_failure"


def episode_trace_summary(row):
    trace = row.get("selector_trace", [])
    thirds = {"early": trace[:len(trace)//3], "middle": trace[len(trace)//3:2*len(trace)//3],
              "late": trace[2*len(trace)//3:]}
    result = {"n_steps": len(trace), "mean_m": mean(x.get("m_t") for x in trace),
              "mean_replacements": mean(x.get("actual_replacements_vs_original") for x in trace),
              "mean_feature_perturbation_norm": mean(x.get("feature_perturbation_norm") for x in trace),
              "mean_residual_norm": mean(x.get("centered_logit_residual_norm") for x in trace),
              "mean_guided_changed_dims": mean(x.get("guided_changed_dims") for x in trace)}
    for phase, items in thirds.items():
        result[f"{phase}_residual_norm"] = mean(x.get("centered_logit_residual_norm") for x in items)
        result[f"{phase}_guided_changed_dims"] = mean(x.get("guided_changed_dims") for x in items)
    return result


def analyze_closed():
    comparisons = (("google_robot_move_near", "prompt_high_action_high", "original"),
                   ("google_robot_close_drawer", "prompt_low_action_high", "prompt_high_action_high"))
    output = {}
    flat = []
    for task, candidate, baseline in comparisons:
        key = f"{SHORT[task]}__{candidate}_vs_{baseline}"; output[key] = {}
        buckets = defaultdict(list)
        for seed in range(100):
            a = load_episode(task, candidate, seed); b = load_episode(task, baseline, seed)
            category = outcome_category(bool(a["success"]), bool(b["success"]))
            buckets[category].append((seed, a, b))
            flat.append({"task": task, "candidate": candidate, "baseline": baseline, "seed": seed,
                         "category": category, "candidate_success": a["success"], "baseline_success": b["success"]})
        for category in ("rescue", "harm", "both_success", "both_failure"):
            items = buckets[category]
            record = {"n": len(items), "seeds": [seed for seed, _, _ in items]}
            if items:
                for label, position in (("candidate", 1), ("baseline", 2)):
                    traces = [episode_trace_summary(item[position]) for item in items]
                    record[label] = {field: mean(trace[field] for trace in traces)
                                     for field in traces[0]}
                    results = [item[position].get("result", {}) for item in items]
                    flags = sorted(set().union(*(result.keys() for result in results)) - {"failure_reason"})
                    record[label]["final_task_metrics"] = {
                        flag: mean(result.get(flag) for result in results if isinstance(result.get(flag), (bool, int, float)))
                        for flag in flags
                    }
            output[key][category] = record
    write_csv(OUT / "closed_loop_key_pair_categories.csv", flat)
    atomic_json(OUT / "closed_loop_key_pair_summary.json", output)
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"stage_a": analyze_stage_a(), "stage_b": analyze_stage_b(), "closed": analyze_closed()}
    atomic_json(OUT / "EXISTING_DATA_MECHANISM_RESULTS.json", result)
    print(json.dumps({"status": "complete", "output": str(OUT)}, indent=2))


if __name__ == "__main__": main()
