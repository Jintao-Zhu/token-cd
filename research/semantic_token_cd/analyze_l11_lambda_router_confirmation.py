"""Evaluate a frozen lambda-conditioned predictor on held-out seeds."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from research.semantic_token_cd.l11_lambda_router_protocol import (
    ARM_NAMES,
    LAMBDAS,
    PROTOCOL,
    SEEDS,
    SHORT,
    TASKS,
    arm_summary_path,
    atomic_json,
    choose_lambda,
    feature_metadata_path,
    feature_path,
    load_frozen_model,
)


HALF_ROOT = Path("artifacts/prompt_attn_l11_matched_full_9task_0_299_v1/run/episodes")


def half_summary(task: str, seed: int) -> Path:
    return HALF_ROOT / task / "prompt_single" / f"episode_{seed:03d}_summary.json"


def paired(selected: np.ndarray, baseline: np.ndarray) -> dict:
    rescue = int(np.sum((selected == 1) & (baseline == 0)))
    harm = int(np.sum((selected == 0) & (baseline == 1)))
    return {
        "rescue": rescue,
        "harm": harm,
        "net": rescue - harm,
        "exact_p": float(binomtest(min(rescue, harm), rescue + harm, 0.5).pvalue) if rescue + harm else 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    model, model_path = load_frozen_model(args.discovery.resolve())

    rows = []
    all_selected = []
    all_fixed = []
    all_oracle = []
    pattern_counts = Counter()
    by_task = {}
    for task in TASKS:
        task_selected = []
        task_fixed = []
        task_oracle = []
        choice_counts = Counter()
        for seed in SEEDS:
            metadata = json.loads(feature_metadata_path(artifact, task, seed).read_text())
            zero = json.loads(arm_summary_path(artifact, task, "l11_positive_only", seed).read_text())
            quarter = json.loads(arm_summary_path(artifact, task, "l11_fixed_025", seed).read_text())
            half = json.loads(half_summary(task, seed).read_text())
            expected_hashes = (
                metadata["canonical_snapshot_sha256"], metadata["initial_state_sha256"], metadata["initial_rgb_sha256"]
            )
            for summary in (zero, quarter, half):
                hashes = (
                    summary["canonical_snapshot_sha256"], summary["initial_state_sha256"], summary["initial_rgb_sha256"]
                )
                if hashes != expected_hashes:
                    raise RuntimeError(f"held-out pairing mismatch: {task} seed={seed}")
            if metadata["protocol_id"] != PROTOCOL or not zero["technical_pass"] or not quarter["technical_pass"]:
                raise RuntimeError(f"held-out technical audit failure: {task} seed={seed}")
            outcomes = np.asarray([zero["success"], quarter["success"], half["success"]], dtype=np.int64)
            chosen_lambda, scores = choose_lambda(model, feature_path(artifact, task, seed), task)
            choice = int(np.flatnonzero(np.isclose(LAMBDAS, chosen_lambda))[0])
            selected = int(outcomes[choice])
            fixed = int(outcomes[2])
            oracle = int(outcomes.max())
            pattern = "".join(str(int(value)) for value in outcomes)
            pattern_counts[pattern] += 1
            choice_counts[str(chosen_lambda)] += 1
            task_selected.append(selected)
            task_fixed.append(fixed)
            task_oracle.append(oracle)
            rows.append({
                "task": SHORT[task], "seed": seed,
                "q_0": scores[0], "q_025": scores[1], "q_050": scores[2],
                "chosen_lambda": chosen_lambda, "pattern": pattern,
                "selected_success": selected, "fixed_050_success": fixed, "oracle_success": oracle,
            })
        selected_array = np.asarray(task_selected)
        fixed_array = np.asarray(task_fixed)
        pair = paired(selected_array, fixed_array)
        pair.update({
            "router_successes": int(selected_array.sum()),
            "fixed_050_successes": int(fixed_array.sum()),
            "oracle_successes": int(np.sum(task_oracle)),
            "choice_counts": dict(choice_counts),
        })
        by_task[SHORT[task]] = pair
        all_selected.extend(task_selected)
        all_fixed.extend(task_fixed)
        all_oracle.extend(task_oracle)

    selected = np.asarray(all_selected)
    fixed = np.asarray(all_fixed)
    oracle = np.asarray(all_oracle)
    pair = paired(selected, fixed)
    go = {
        "heldout_oracle_gap_at_least_0_05": float(oracle.mean() - fixed.mean()) >= 0.05,
        "pooled_net_at_least_8_of_200": pair["net"] >= 8,
        "pooled_rescue_gt_harm": pair["rescue"] > pair["harm"],
        "each_task_net_at_least_minus_2": all(result["net"] >= -2 for result in by_task.values()),
        "at_least_one_task_net_positive": any(result["net"] > 0 for result in by_task.values()),
    }
    go["passed"] = all(go.values())
    payload = {
        "protocol_id": PROTOCOL,
        "test_seeds": [100, 199],
        "tasks": list(TASKS),
        "frozen_model_path": str(model_path),
        "pattern_counts": {pattern: int(pattern_counts.get(pattern, 0)) for pattern in ("000", "100", "010", "001", "110", "101", "011", "111")},
        "router_successes": int(selected.sum()),
        "fixed_050_successes": int(fixed.sum()),
        "oracle_successes": int(oracle.sum()),
        "oracle_gap_vs_050": float(oracle.mean() - fixed.mean()),
        "paired_vs_fixed_050": pair,
        "by_task": by_task,
        "go_no_go": go,
        "retraining_or_calibration_on_test": False,
    }
    output = artifact / "analysis"
    atomic_json(output / "CONFIRMATION_RESULTS.json", payload)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Frozen L11 Lambda Router Confirmation", "",
        "| Task | Router | Fixed .5 | Oracle | Rescue | Harm | Net | p |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        result = by_task[SHORT[task]]
        lines.append(f"| {SHORT[task]} | {result['router_successes']}/100 | {result['fixed_050_successes']}/100 | {result['oracle_successes']}/100 | {result['rescue']} | {result['harm']} | {result['net']:+d} | {result['exact_p']:.4g} |")
    lines.extend([
        "", f"- Pooled Router: **{int(selected.sum())}/200**",
        f"- Fixed λ=.5: **{int(fixed.sum())}/200**",
        f"- Oracle: **{int(oracle.sum())}/200**",
        f"- Paired Rescue/Harm/Net: **{pair['rescue']}/{pair['harm']}/{pair['net']:+d}**",
        f"- Decision: **{'GO' if go['passed'] else 'STOP'}**",
    ])
    (output / "CONFIRMATION_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
