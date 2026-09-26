#!/usr/bin/env python3
"""Summarize the paired two-bowl attention localization audit."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
ART = ROOT / "artifacts/libero_two_bowl_attention_localization_v1_20260926"
TASKS = {
    2: "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    8: "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
}
MODES = ("full_instruction", "source_clause", "object_words", "relation_words", "reference_words")


def mean(xs):
    return float(statistics.mean(xs)) if xs else None


def median(xs):
    return float(statistics.median(xs)) if xs else None


def main():
    rows = []
    for path in sorted((ART / "workers").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    keys = [(r["task_id"], r["init_state"]) for r in rows]
    if len(rows) != 100 or len(set(keys)) != 100:
        raise RuntimeError(f"expected 100 unique records, got {len(rows)} / {len(set(keys))}")
    if not all(r["state_hash_match"] for r in rows):
        raise RuntimeError("at least one state hash differs from the archived paired rollout")
    if not all(abs(r["recomputed_l11_topm_jaccard_to_historical"] - 1.0) < 1e-12 for r in rows):
        raise RuntimeError("at least one recomputed L11 top-m differs from archived selected token IDs")

    layer_csv = []
    task_summary = {}
    for tid, task_name in TASKS.items():
        group = sorted((r for r in rows if r["task_id"] == tid), key=lambda x: x["init_state"])
        if len(group) != 50 or any(r["task"] != task_name for r in group):
            raise RuntimeError(f"task {tid} is not the expected 50-case group")
        counts = Counter(r["outcome"]["label"] for r in group)
        if tid == 2 and (counts["rescue"], counts["harm"]) != (7, 2):
            raise RuntimeError(f"unexpected official paired outcomes for task 2: {counts}")
        if tid == 8 and (counts["rescue"], counts["harm"]) != (0, 12):
            raise RuntimeError(f"unexpected official paired outcomes for task 8: {counts}")

        historical_target = [r["historical_selected_target_coverage"] for r in group]
        historical_distractor = [r["historical_selected_distractor_coverage"] for r in group]
        modes = {}
        for mode in MODES:
            mm = [r["layer_metrics"][11]["modes"][mode] for r in group]
            modes[mode] = {
                "density_ratio_mean": mean([x["target_vs_distractor_density_ratio"] for x in mm]),
                "density_ratio_median": median([x["target_vs_distractor_density_ratio"] for x in mm]),
                "target_topm_coverage_mean": mean([x["topk_target_coverage"] for x in mm]),
                "distractor_topm_coverage_mean": mean([x["topk_distractor_coverage"] for x in mm]),
                "density_favors_target_n": sum(x["target_vs_distractor_density_ratio"] > 1 for x in mm),
                "topm_coverage_favors_target_n": sum(x["topk_target_coverage"] > x["topk_distractor_coverage"] for x in mm),
            }

        per_layer = []
        for layer in range(32):
            m = [r["layer_metrics"][layer]["modes"]["full_instruction"] for r in group]
            ratios = [x["target_vs_distractor_density_ratio"] for x in m]
            diffs = [x["topk_target_coverage"] - x["topk_distractor_coverage"] for x in m]
            record = {
                "task_id": tid, "task": task_name, "layer": layer,
                "mean_density_ratio_target_over_distractor": mean(ratios),
                "median_density_ratio_target_over_distractor": median(ratios),
                "fraction_density_favors_target": sum(x > 1 for x in ratios) / len(ratios),
                "mean_topm_coverage_difference_target_minus_distractor": mean(diffs),
                "fraction_topm_coverage_favors_target": sum(x > 0 for x in diffs) / len(diffs),
            }
            layer_csv.append(record)
            per_layer.append(record)

        label_stats = {}
        for label in ("rescue", "harm", "both_success", "both_fail"):
            subset = [r for r in group if r["outcome"]["label"] == label]
            if not subset:
                continue
            label_stats[label] = {
                "n": len(subset),
                "historical_selected_target_coverage_mean": mean([r["historical_selected_target_coverage"] for r in subset]),
                "historical_selected_distractor_coverage_mean": mean([r["historical_selected_distractor_coverage"] for r in subset]),
                "l11_full_density_ratio_mean": mean([r["layer_metrics"][11]["modes"]["full_instruction"]["target_vs_distractor_density_ratio"] for r in subset]),
                "l11_full_topm_target_minus_distractor_mean": mean([
                    r["layer_metrics"][11]["modes"]["full_instruction"]["topk_target_coverage"]
                    - r["layer_metrics"][11]["modes"]["full_instruction"]["topk_distractor_coverage"]
                    for r in subset
                ]),
            }

        switch = {}
        for layer in range(32):
            b = [r["layer_metrics"][layer]["modes"]["full_instruction"]["target_vs_distractor_density_ratio"] for r in group]
            a = [r["alternate_prompt_metrics"][layer]["metrics"]["target_vs_distractor_density_ratio"] for r in group]
            switch[layer] = {
                "mean_log_ratio_change_base_minus_alternate": mean([math.log(max(x, 1e-30)) - math.log(max(y, 1e-30)) for x, y in zip(b, a)]),
                "base_prefers_designated_bowl_and_alternate_prefers_other_n": sum(x > 1 and y < 1 for x, y in zip(b, a)),
            }

        task_summary[str(tid)] = {
            "task": task_name,
            "n": len(group),
            "paired_outcome_counts": dict(counts),
            "mean_matched_budget_m": mean([r["matched_budget_m"] for r in group]),
            "historical_l11_selected_target_coverage_mean": mean(historical_target),
            "historical_l11_selected_distractor_coverage_mean": mean(historical_distractor),
            "l11_query_modes": modes,
            "per_layer_full_instruction": per_layer,
            "outcome_strata": label_stats,
            "relation_prompt_switch_by_layer": switch,
        }

    with (ART / "LAYER_METRICS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(layer_csv[0]))
        w.writeheader()
        w.writerows(layer_csv)
    payload = {
        "protocol": "LIBERO_SPATIAL_TWO_BOWL_ATTENTION_LOCALIZATION_AUDIT_V1",
        "checkpoint": str(Path("/home/leju-suzhou/zjt_ws/checkpoints/libero/openvla-7b-finetuned-libero-spatial")),
        "renderer": "MuJoCo EGL physical GPU 7 (resolved EGL ordinal 4)",
        "inference_gpus": [1, 2, 3, 6],
        "worker_processes_per_inference_gpu": 3,
        "records": len(rows), "unique_task_initial_state_pairs": len(set(keys)),
        "state_hash_matches": sum(r["state_hash_match"] for r in rows),
        "recomputed_l11_topm_exact_matches": sum(abs(r["recomputed_l11_topm_jaccard_to_historical"] - 1.0) < 1e-12 for r in rows),
        "closed_loop_episodes_added": 0,
        "tasks": task_summary,
    }
    (ART / "SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n")

    def percent(x): return f"{100*x:.1f}%"
    lines = [
        "# Two-bowl prompt-attention localization audit",
        "",
        "## Audit validity",
        "",
        f"- Reanalyzed {len(rows)}/100 paired initial states; all 100 state hashes match the official archived rollouts.",
        f"- Recomputed L11 Top-m exactly matches the archived first-step token IDs in {payload['recomputed_l11_topm_exact_matches']}/100 cases (Jaccard = 1.0).",
        "- No closed-loop episodes were added. The primary query is the original full task instruction; other query modes are diagnostic splits.",
        "- Segmentation is used only as a measurement reference to label the designated and distractor bowl image patches.",
        "",
        "## Main comparison",
        "",
        "| Task | Rescue / Harm | Mean historical L11-mask coverage: designated bowl | Distractor bowl | L11 full-prompt density ratio (designated / distractor) | Cases with ratio > 1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tid, task in task_summary.items():
        full = task["l11_query_modes"]["full_instruction"]
        lines.append(
            f"| {task['task']} | {task['paired_outcome_counts'].get('rescue',0)} / {task['paired_outcome_counts'].get('harm',0)} "
            f"| {percent(task['historical_l11_selected_target_coverage_mean'])} "
            f"| {percent(task['historical_l11_selected_distractor_coverage_mean'])} "
            f"| {full['density_ratio_mean']:.2f}× (median {full['density_ratio_median']:.2f}×) "
            f"| {full['density_favors_target_n']}/50 |"
        )
    lines += [
        "",
        "## Layer and relation checks",
        "",
        "For each task, L11 full-prompt attention was also split into source clause, object words, relation words, and reference words. The complete per-layer table is in `LAYER_METRICS.csv`.",
        "",
    ]
    for tid, task in task_summary.items():
        m = task["l11_query_modes"]
        l11 = task["per_layer_full_instruction"][11]
        density_candidates = sorted(
            task["per_layer_full_instruction"],
            key=lambda x: (x["fraction_density_favors_target"], x["median_density_ratio_target_over_distractor"]),
            reverse=True,
        )[:1]
        coverage_candidate = max(
            task["per_layer_full_instruction"],
            key=lambda x: x["mean_topm_coverage_difference_target_minus_distractor"],
        )
        sw = task["relation_prompt_switch_by_layer"][11]
        lines += [
            f"### Task {tid}: `{task['task']}`",
            "",
            f"- L11 designated/distractor density ratio by query: full {m['full_instruction']['density_ratio_mean']:.2f}×, source clause {m['source_clause']['density_ratio_mean']:.2f}×, object words {m['object_words']['density_ratio_mean']:.2f}×, relation words {m['relation_words']['density_ratio_mean']:.2f}×, reference words {m['reference_words']['density_ratio_mean']:.2f}×.",
            f"- L11 full-prompt Top-m coverage difference (designated minus distractor): {l11['mean_topm_coverage_difference_target_minus_distractor']:+.3f}; designated bowl wins by density in {l11['fraction_density_favors_target']*100:.0f}% of states.",
            f"- Strongest density-favoring layer by fraction then median ratio: L{density_candidates[0]['layer']} "
            f"({density_candidates[0]['fraction_density_favors_target']*100:.0f}% of states, median ratio "
            f"{density_candidates[0]['median_density_ratio_target_over_distractor']:.2f}×). "
            f"Largest Top-m coverage separation is L{coverage_candidate['layer']} "
            f"({coverage_candidate['mean_topm_coverage_difference_target_minus_distractor']:+.3f}). "
            "This is descriptive layer scanning, not a held-out layer selection result.",
            f"- Counterfactual relation-prompt check at L11: {sw['base_prefers_designated_bowl_and_alternate_prefers_other_n']}/50 states flip the bowl-density preference in the expected direction; mean log-ratio change (base minus alternate) = {sw['mean_log_ratio_change_base_minus_alternate']:+.3f}.",
            "- Rescue/Harm strata are exploratory and small; see `SUMMARY.json` for their counts and localization means.",
            "",
        ]
    lines += [
        "## Interpretation and limits",
        "",
        "The first-step localization pattern differs sharply across the two tasks: in the table-center task, canonical L11 generally places more attention and selected-mask coverage on the designated bowl; in the ramekin task, it generally favors the distractor bowl. This makes prompt-attention localization a plausible contributor to the task-level Rescue/Harm contrast.",
        "",
        "This does not establish that localization causes the closed-loop outcome difference. Within the ramekin task, L11 also favors the distractor in many both-success cases, and the Rescue/Harm subgroups are small. The audit only examines the initial control state; it does not measure how localization changes later in the trajectory. A causal conclusion would require a separate controlled closed-loop intervention after reviewing this audit.",
        "",
    ]
    (ART / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"summary": str(ART / "SUMMARY.json"), "report": str(ART / "REPORT.md"),
                      "records": len(rows), "hash_matches": payload["state_hash_matches"],
                      "selector_exact_matches": payload["recomputed_l11_topm_exact_matches"]}, sort_keys=True))


if __name__ == "__main__":
    main()
