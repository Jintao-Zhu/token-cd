"""Validate TopP90 smoke episodes against Matched and the frozen TopP85 arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.semantic_token_cd.prompt_attn_l11_count_rollout import TASKS
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json


ARM = "l11_top_p90"
THRESHOLD = 0.90


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--matched-artifact", type=Path, required=True)
    parser.add_argument("--base-top-p-artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    matched_artifact = args.matched_artifact.resolve()
    base_top_p_artifact = args.base_top_p_artifact.resolve()
    report = {"passed": True, "seed": args.seed, "tasks": {}}

    for task in TASKS:
        top90_root = artifact / "episodes" / task / ARM
        top85_root = base_top_p_artifact / "episodes" / task / "l11_top_p85"
        matched_root = matched_artifact / "episodes" / task / "l11_matched"
        top90_summary = json.loads((top90_root / f"episode_{args.seed:03d}_summary.json").read_text())
        top85_summary = json.loads((top85_root / f"episode_{args.seed:03d}_summary.json").read_text())
        matched_summary = json.loads((matched_root / f"episode_{args.seed:03d}_summary.json").read_text())
        top90_arrays = np.load(top90_root / f"episode_{args.seed:03d}_arrays.npz")
        top85_arrays = np.load(top85_root / f"episode_{args.seed:03d}_arrays.npz")
        matched_arrays = np.load(matched_root / f"episode_{args.seed:03d}_arrays.npz")

        summaries = (matched_summary, top85_summary, top90_summary)
        hashes_equal = {
            key: len({value[key] for value in summaries}) == 1
            for key in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256")
        }
        clean_logits_equal = (
            np.array_equal(matched_arrays["positive"][0], top85_arrays["positive"][0])
            and np.array_equal(matched_arrays["positive"][0], top90_arrays["positive"][0])
        )
        clean_tokens_equal = (
            matched_summary["selector_trace"][0]["positive_token_ids"]
            == top85_summary["selector_trace"][0]["positive_token_ids"]
            == top90_summary["selector_trace"][0]["positive_token_ids"]
        )
        top85_mask = set(np.flatnonzero(top85_arrays["selected_mask"][0]).tolist())
        top90_mask = set(np.flatnonzero(top90_arrays["selected_mask"][0]).tolist())
        formula_checks = []
        for step in top90_summary["selector_trace"]:
            raw = int(step["m_raw"])
            count = int(step["actual_selected_count"])
            formula_checks.append(
                abs(float(step["top_p_threshold"]) - THRESHOLD) < 1e-12
                and count == min(64, max(16, raw))
                and bool(step["lower_bound_triggered"]) == (raw < 16)
                and bool(step["upper_bound_triggered"]) == (raw > 64)
                and 0.0 < float(step["selected_attention_mass"]) <= 1.0
            )

        checks = {
            "top_p90_output_present": True,
            "snapshot_hash_equal_to_references": hashes_equal["canonical_snapshot_sha256"],
            "initial_state_hash_equal_to_references": hashes_equal["initial_state_sha256"],
            "initial_rgb_hash_equal_to_references": hashes_equal["initial_rgb_sha256"],
            "initial_clean_logits_bit_equal_to_references": clean_logits_equal,
            "initial_clean_action_tokens_equal_to_references": clean_tokens_equal,
            "initial_top_p85_mask_subset_of_top_p90": top85_mask <= top90_mask,
            "top_p90_formula_and_bounds_exact": bool(formula_checks) and all(formula_checks),
            "episode_technical_audit_pass": top90_summary.get("technical_pass") is True,
            "kmeans_bypassed": top90_summary.get("kmeans_used") is False,
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
