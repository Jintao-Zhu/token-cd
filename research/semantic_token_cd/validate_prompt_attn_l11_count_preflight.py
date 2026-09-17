"""Validate the 24-episode L11 count-sweep preflight before full launch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.semantic_token_cd.prompt_attn_l11_count_rollout import ARM_COUNTS, TASKS
from research.semantic_token_cd.prompt_attn_shr_rollout import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    report = {"passed": True, "seed": args.seed, "tasks": {}}
    fixed = ["l11_k16", "l11_k24", "l11_k32", "l11_k48", "l11_k64"]

    for task in TASKS:
        summaries = {}
        arrays = {}
        for arm in ARM_COUNTS:
            root = artifact / "episodes" / task / arm
            summary_path = root / f"episode_{args.seed:03d}_summary.json"
            arrays_path = root / f"episode_{args.seed:03d}_arrays.npz"
            if not summary_path.exists() or not arrays_path.exists():
                raise RuntimeError(f"missing preflight output: {task}/{arm}")
            summaries[arm] = json.loads(summary_path.read_text())
            arrays[arm] = np.load(arrays_path)

        snapshot_hashes = {value["canonical_snapshot_sha256"] for value in summaries.values()}
        state_hashes = {value["initial_state_sha256"] for value in summaries.values()}
        rgb_hashes = {value["initial_rgb_sha256"] for value in summaries.values()}
        initial_positive = [arrays[arm]["positive"][0] for arm in ARM_COUNTS]
        initial_clean_logits_equal = all(
            np.array_equal(initial_positive[0], value) for value in initial_positive[1:]
        )
        initial_clean_tokens = [
            summaries[arm]["selector_trace"][0]["positive_token_ids"] for arm in ARM_COUNTS
        ]
        initial_clean_actions_equal = all(value == initial_clean_tokens[0] for value in initial_clean_tokens[1:])
        masks = {arm: set(np.flatnonzero(arrays[arm]["selected_mask"][0]).tolist()) for arm in fixed}
        fixed_counts_exact = all(len(masks[arm]) == ARM_COUNTS[arm] for arm in fixed)
        nested = all(masks[left] <= masks[right] for left, right in zip(fixed, fixed[1:]))
        all_technical = all(value.get("technical_pass") is True for value in summaries.values())
        matched_path_exact = (
            summaries["l11_matched"]["selected_token_count_config"] is None
            and summaries["l11_matched"]["selector_trace"][0]["coverage_mode"]
            == "own_state_standard_shr_matched"
        )
        checks = {
            "six_episode_outputs_present": True,
            "snapshot_hash_equal": len(snapshot_hashes) == 1,
            "initial_state_hash_equal": len(state_hashes) == 1,
            "initial_rgb_hash_equal": len(rgb_hashes) == 1,
            "initial_clean_logits_bit_equal": initial_clean_logits_equal,
            "initial_clean_action_tokens_equal": initial_clean_actions_equal,
            "fixed_counts_exact": fixed_counts_exact,
            "fixed_masks_nested": nested,
            "matched_uses_original_l11_path": matched_path_exact,
            "all_episode_technical_audits_pass": all_technical,
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
