"""Validate the 12 adaptive episodes before launching the full Top-p sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.semantic_token_cd.prompt_attn_l11_count_rollout import TASKS
from research.semantic_token_cd.prompt_attn_l11_top_p_rollout import ARM_THRESHOLDS
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--matched-artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    matched_artifact = args.matched_artifact.resolve()
    report = {"passed": True, "seed": args.seed, "tasks": {}}
    arms = tuple(ARM_THRESHOLDS)

    for task in TASKS:
        summaries: dict[str, dict] = {}
        arrays: dict[str, np.lib.npyio.NpzFile] = {}
        for arm in arms:
            root = artifact / "episodes" / task / arm
            summary_path = root / f"episode_{args.seed:03d}_summary.json"
            arrays_path = root / f"episode_{args.seed:03d}_arrays.npz"
            if not summary_path.exists() or not arrays_path.exists():
                raise RuntimeError(f"missing preflight output: {task}/{arm}")
            summaries[arm] = json.loads(summary_path.read_text())
            arrays[arm] = np.load(arrays_path)

        matched_root = matched_artifact / "episodes" / task / "l11_matched"
        matched_summary = json.loads((matched_root / f"episode_{args.seed:03d}_summary.json").read_text())
        matched_arrays = np.load(matched_root / f"episode_{args.seed:03d}_arrays.npz")

        all_summaries = [matched_summary, *summaries.values()]
        hashes_equal = {
            key: len({value[key] for value in all_summaries}) == 1
            for key in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256")
        }
        initial_positive = [matched_arrays["positive"][0], *[arrays[arm]["positive"][0] for arm in arms]]
        clean_logits_equal = all(np.array_equal(initial_positive[0], value) for value in initial_positive[1:])
        initial_tokens = [matched_summary["selector_trace"][0]["positive_token_ids"], *[
            summaries[arm]["selector_trace"][0]["positive_token_ids"] for arm in arms
        ]]
        clean_tokens_equal = all(value == initial_tokens[0] for value in initial_tokens[1:])
        masks = {arm: set(np.flatnonzero(arrays[arm]["selected_mask"][0]).tolist()) for arm in arms}
        nested = masks["l11_top_p75"] <= masks["l11_top_p80"] <= masks["l11_top_p85"]

        formula_checks = []
        for arm, threshold in ARM_THRESHOLDS.items():
            for step in summaries[arm]["selector_trace"]:
                raw = int(step["m_raw"])
                count = int(step["actual_selected_count"])
                formula_checks.append(
                    abs(float(step["top_p_threshold"]) - threshold) < 1e-12
                    and count == min(64, max(16, raw))
                    and bool(step["lower_bound_triggered"]) == (raw < 16)
                    and bool(step["upper_bound_triggered"]) == (raw > 64)
                    and 0.0 < float(step["selected_attention_mass"]) <= 1.0
                )

        checks = {
            "three_adaptive_outputs_present": True,
            "snapshot_hash_equal_to_reused_matched": hashes_equal["canonical_snapshot_sha256"],
            "initial_state_hash_equal_to_reused_matched": hashes_equal["initial_state_sha256"],
            "initial_rgb_hash_equal_to_reused_matched": hashes_equal["initial_rgb_sha256"],
            "initial_clean_logits_bit_equal_to_matched": clean_logits_equal,
            "initial_clean_action_tokens_equal_to_matched": clean_tokens_equal,
            "initial_masks_monotonic": nested,
            "top_p_formula_and_bounds_exact": bool(formula_checks) and all(formula_checks),
            "all_episode_technical_audits_pass": all(value.get("technical_pass") is True for value in summaries.values()),
            "all_adaptive_arms_bypass_kmeans": all(value.get("kmeans_used") is False for value in summaries.values()),
        }
        checks["passed"] = all(checks.values())
        report["tasks"][task] = checks
        report["passed"] = report["passed"] and checks["passed"]

    atomic_json(artifact / "preflight" / "PREFLIGHT_REPORT.json", report)
    if not report["passed"]:
        raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
