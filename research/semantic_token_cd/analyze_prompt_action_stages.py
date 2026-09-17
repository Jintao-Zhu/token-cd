"""Validate and summarize Prompt/Action complement stages A and B."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from research.semantic_token_cd.prompt_action_complement_protocol import ARMS, ARTIFACT, TASKS, atomic_json


def mean(values): return float(np.mean(list(values)))


def stage_a():
    files = sorted((ARTIFACT / "stage_a/states").glob("**/step_*.json"))
    if len(files) != 918: raise RuntimeError(f"Stage A incomplete: {len(files)}/918")
    rows = [json.loads(path.read_text()) for path in files]
    if not all(row["original_mask_exact"] for row in rows): raise RuntimeError("Original reproduction failed")
    summary = {"n_states": len(rows), "audits_passed": True, "m_min": min(r["m"] for r in rows),
               "m_max": max(r["m"] for r in rows), "arms": {}, "action_layer_rank_overlap": {}}
    for arm in ARMS:
        spec = [row["arms"][arm] for row in rows]
        summary["arms"][arm] = {
            "mean_replacements_vs_original": mean(x["actual_replacements_vs_original"] for x in spec),
            "mean_supplement_prompt_global_rank": mean(
                rank for x in spec for rank in x["supplement_prompt_global_ranks"]
            ),
            "mean_supplement_action_global_rank": mean(
                rank for x in spec for rank in x["supplement_action_global_ranks"]
            ),
        }
    for tag in ("action_attention_l11", "action_attention_full_layers"):
        overlaps = []
        for path, row in zip(files, rows):
            arrays = np.load(path.with_suffix(".npz")); m = row["m"]
            late = set(np.lexsort((np.arange(256), -arrays["action_attention"]))[:m])
            other = set(np.lexsort((np.arange(256), -arrays[tag]))[:m])
            overlaps.append(len(late & other) / max(1, len(late | other)))
        summary["action_layer_rank_overlap"][f"late16_31_vs_{tag}"] = mean(overlaps)
    atomic_json(ARTIFACT / "stage_a/statistics/summary.json", summary)
    (ARTIFACT / "stage_a/COMPLETE").write_text("918/918 states complete and audited\n")
    print(json.dumps(summary, indent=2))


def stage_b():
    files = sorted((ARTIFACT / "stage_b/states").glob("**/step_*.json"))
    if len(files) != 240: raise RuntimeError(f"Stage B incomplete: {len(files)}/240")
    rows = [json.loads(path.read_text()) for path in files]
    if not all(row["audit"]["masks_exact"] for row in rows): raise RuntimeError("Stage B audit failed")
    summary = {"n_states": len(rows), "audits_passed": True, "arms": {}}
    for arm in ARMS:
        metrics = [row["metrics"][arm] for row in rows]
        summary["arms"][arm] = {
            "mean_feature_perturbation_norm": mean(x["feature_perturbation_norm"] for x in metrics),
            "mean_centered_residual_norm": mean(x["centered_residual_norm"] for x in metrics),
            "mean_residual_cosine_vs_original": mean(x["residual_cosine_vs_original"] for x in metrics),
            "action_exact_fraction_vs_original": mean(
                x["final_token_ids"] == row["metrics"]["original"]["final_token_ids"]
                for x, row in zip(metrics, rows)
            ),
            "mean_guided_changed_dims": mean(x["guided_changed_dims"] for x in metrics),
            "mean_guided_clean_action_l2": mean(x["guided_clean_action_l2"] for x in metrics),
        }
    atomic_json(ARTIFACT / "stage_b/statistics/summary.json", summary)
    (ARTIFACT / "stage_b/COMPLETE").write_text("240/240 states complete and audited\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=("a", "b")); args = parser.parse_args()
    stage_a() if args.stage == "a" else stage_b()


if __name__ == "__main__": main()

