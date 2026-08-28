from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ARMS = ("A_vanilla", "B_toward_top8", "C_dual_seed_chunk_average", "D_random8_toward")
COMPARISONS = (
    ("C_dual_seed_chunk_average", "A_vanilla"),
    ("D_random8_toward", "A_vanilla"),
    ("B_toward_top8", "A_vanilla"),
    ("B_toward_top8", "D_random8_toward"),
    ("C_dual_seed_chunk_average", "B_toward_top8"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval_contains_zero(interval: list[float]) -> bool:
    return interval[0] <= 0 <= interval[1]


def compare(rows: list[dict], first: str, second: str, seed: int) -> dict:
    first_values = np.asarray([int(row[first]["success"]) for row in rows])
    second_values = np.asarray([int(row[second]["success"]) for row in rows])
    delta = first_values - second_values
    rng = np.random.default_rng(seed)
    boot = [float(np.mean(delta[rng.integers(0, len(delta), len(delta))])) for _ in range(2000)]
    first_only = int(np.sum((first_values == 1) & (second_values == 0)))
    second_only = int(np.sum((first_values == 0) & (second_values == 1)))
    discordant = first_only + second_only
    return {
        "comparison": f"{first}_minus_{second}",
        "point_estimate": float(np.mean(delta)),
        "paired_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "first_only_success": first_only,
        "second_only_success": second_only,
        "discordant": discordant,
        "exact_mcnemar_p": float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0,
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_values[key]))
        adjusted[key] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    specs = [json.loads(line) for line in (artifact / "episode_manifest.jsonl").read_text().splitlines()]
    expected = {row["episode_id"] for row in specs}
    files = list((artifact / "episodes").glob("*.json"))
    actual = {path.stem for path in files}
    if len(specs) != 400 or len(files) != 400 or actual != expected:
        raise RuntimeError(f"analysis forbidden before exact 400/400 completion: specs={len(specs)} files={len(files)} missing={len(expected-actual)} extra={len(actual-expected)}")
    records = [json.loads(path.read_text()) for path in files]
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    failures = []
    for record in records:
        grouped[record["pair_id"]][record["arm"]] = record
        if record["status"] != "complete" or not record["all_actions_finite"]:
            failures.append({"episode_id": record["episode_id"], "reason": "status or finite failure"})
    paired_rows = []
    for pair_id, arms in sorted(grouped.items()):
        if set(arms) != set(ARMS):
            failures.append({"episode_id": pair_id, "reason": "incomplete arm set"})
            continue
        reference = arms["A_vanilla"]
        identity_keys = ("suite", "task_id", "init_state_id", "reset_seed", "action_noise_seed", "selection_seed", "second_action_noise_seed", "language")
        if any(arms[arm][key] != reference[key] for arm in ARMS for key in identity_keys):
            failures.append({"episode_id": pair_id, "reason": "paired identity mismatch"})
            continue
        if len({arms[arm]["initial_sim_state_sha256"] for arm in ARMS}) != 1 or len({arms[arm]["initial_prepared_input_sha256"] for arm in ARMS}) != 1:
            failures.append({"episode_id": pair_id, "reason": "initial state or preprocessing mismatch"})
            continue
        common_replans = min(arms[arm]["replans"] for arm in ARMS)
        native_noise = reference["first_noise_sha256_by_replan"][:common_replans]
        if any(arms[arm]["first_noise_sha256_by_replan"][:common_replans] != native_noise for arm in ARMS):
            failures.append({"episode_id": pair_id, "reason": "shared first noise mismatch"})
            continue
        ensemble = arms["C_dual_seed_chunk_average"]
        if len(ensemble["second_noise_sha256_by_replan"]) != ensemble["replans"] or any(trace["masked_token_count"] != 0 or trace["selected_token_indices"] or trace["changed_token_indices"] for trace in ensemble["replan_traces"]):
            failures.append({"episode_id": pair_id, "reason": "arm C mask/second-noise failure"})
            continue
        trace_failure = False
        for arm in ("B_toward_top8", "D_random8_toward"):
            for trace in arms[arm]["replan_traces"]:
                selected = trace["selected_token_indices"]
                changed = trace["changed_token_indices"]
                if len(selected) != 8 or sorted(selected) != sorted(changed) or not trace["protected_tokens_untouched"] or not trace["all_output_finite"]:
                    trace_failure = True
                if arm == "D_random8_toward" and trace["overlap_with_attention_top8"] != len(set(selected) & set(trace["attention_top8_indices"])):
                    trace_failure = True
        if trace_failure:
            failures.append({"episode_id": pair_id, "reason": "B/D intervention trace failure"})
            continue
        paired_rows.append({"pair_id": pair_id, "task_id": reference["task_id"], "init_state_id": reference["init_state_id"], **arms})
    if len(grouped) != 100 or len(paired_rows) != 100 or failures:
        (artifact / "analysis_integrity_failure.json").write_text(json.dumps({"failures": failures, "pairs": len(paired_rows)}, indent=2) + "\n")
        raise RuntimeError("analysis integrity failure")

    task_results = {}
    flat_csv = []
    for task_id in (4, 7):
        task_rows = [row for row in paired_rows if row["task_id"] == task_id]
        rates = {arm: float(np.mean([row[arm]["success"] for row in task_rows])) for arm in ARMS}
        comparisons = {}
        for index, (first, second) in enumerate(COMPARISONS):
            result = compare(task_rows, first, second, 20260809 + task_id * 100 + index)
            comparisons[result["comparison"]] = result
        task_results[str(task_id)] = {"states": len(task_rows), "success_rates": rates, "comparisons": comparisons}
        for row in task_rows:
            output = {"pair_id": row["pair_id"], "task_id": task_id, "init_state_id": row["init_state_id"]}
            for arm in ARMS:
                record = row[arm]
                output.update({
                    f"{arm}_success": int(record["success"]),
                    f"{arm}_control_steps": record["control_steps"],
                    f"{arm}_action_total_variation": record["action_total_variation"],
                    f"{arm}_action_total_variation_first_30_steps": record["action_total_variation_first_30_steps"],
                    f"{arm}_chunk_discontinuity": record["chunk_discontinuity"],
                    f"{arm}_median_replan_latency_seconds": record["median_replan_latency_seconds"],
                })
            flat_csv.append(output)

    family_p = {}
    for task_id, result in task_results.items():
        for name in ("C_dual_seed_chunk_average_minus_A_vanilla", "D_random8_toward_minus_A_vanilla"):
            family_p[f"task{task_id}:{name}"] = result["comparisons"][name]["exact_mcnemar_p"]
    adjusted = holm(family_p)
    for key, value in adjusted.items():
        task_label, comparison = key.split(":", 1)
        task_results[task_label.removeprefix("task")]["comparisons"][comparison]["holm_adjusted_p"] = value

    def comparison(task_id: int, name: str) -> dict:
        return task_results[str(task_id)]["comparisons"][name]

    b_a = [comparison(task, "B_toward_top8_minus_A_vanilla") for task in (4, 7)]
    c_a = [comparison(task, "C_dual_seed_chunk_average_minus_A_vanilla") for task in (4, 7)]
    d_a = [comparison(task, "D_random8_toward_minus_A_vanilla") for task in (4, 7)]
    b_d = [comparison(task, "B_toward_top8_minus_D_random8_toward") for task in (4, 7)]
    c_b = [comparison(task, "C_dual_seed_chunk_average_minus_B_toward_top8") for task in (4, 7)]
    if all(interval_contains_zero(result["paired_bootstrap_95_ci"]) for result in b_a):
        decision = "TOWARD_EFFECT_NOT_REPLICATED"
    elif all(result["point_estimate"] > 0 and result["paired_bootstrap_95_ci"][0] > 0 for result in c_a) and all(interval_contains_zero(result["paired_bootstrap_95_ci"]) for result in c_b):
        decision = "ENSEMBLE_EXPLAINS_THE_GAIN"
    elif all(interval_contains_zero(result["paired_bootstrap_95_ci"]) for result in c_a) and all(result["point_estimate"] > 0 and result["paired_bootstrap_95_ci"][0] > 0 for result in d_a) and all(interval_contains_zero(result["paired_bootstrap_95_ci"]) for result in b_d):
        decision = "OPERATOR_NOT_TOKEN"
    elif all(interval_contains_zero(result["paired_bootstrap_95_ci"]) for result in c_a + d_a) and all(result["point_estimate"] > 0 and result["paired_bootstrap_95_ci"][0] > 0 for result in b_a):
        decision = "TOKEN_IDENTITY_MATTERS"
    else:
        decision = "INCONCLUSIVE"

    secondary = {}
    for task_id in (4, 7):
        rows = [row for row in paired_rows if row["task_id"] == task_id]
        secondary[str(task_id)] = {}
        for arm in ARMS:
            records_for_arm = [row[arm] for row in rows]
            secondary[str(task_id)][arm] = {
                metric: float(np.mean([record[metric] for record in records_for_arm]))
                for metric in ("control_steps", "action_total_variation", "action_total_variation_first_30_steps", "chunk_discontinuity", "median_replan_latency_seconds")
            }
            traces = [trace for record in records_for_arm for trace in record["replan_traces"]]
            if arm in ("B_toward_top8", "D_random8_toward"):
                secondary[str(task_id)][arm]["clip_active_replan_fraction"] = float(np.mean([trace["trust_region_clipping_active_bool"] for trace in traces]))
                secondary[str(task_id)][arm]["mean_attention_top8_overlap"] = float(np.mean([trace["overlap_with_attention_top8"] for trace in traces]))
            if arm == "C_dual_seed_chunk_average":
                secondary[str(task_id)][arm]["clip_active_replan_fraction"] = float(np.mean([trace["trust_region_clipping_active"] for trace in traces]))

    pooled_rates = {arm: float(np.mean([row[arm]["success"] for row in paired_rows])) for arm in ARMS}
    summary = {
        "experiment_name": "coreact_ensemble_vs_contrast_control_v1",
        "stage": "mechanism_decision_experiment_not_confirmation",
        "completion": {"episodes": 400, "paired_units": 100, "task4_pairs": 50, "task7_pairs": 50, "missing": 0, "duplicates": 0},
        "integrity_pass": True,
        "task_results": task_results,
        "pooled_success_rates_descriptive_only": pooled_rates,
        "holm_family": {"raw_p": family_p, "adjusted_p": adjusted},
        "secondary_metrics": secondary,
        "decision": decision,
        "confirmation_claim_allowed": False,
    }
    (artifact / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (artifact / "decision.json").write_text(json.dumps({"decision": decision, "integrity_pass": True, "confirmation_claim_allowed": False}, indent=2, sort_keys=True) + "\n")
    with (artifact / "paired_results.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_csv[0])); writer.writeheader(); writer.writerows(flat_csv)
    with (artifact / "missingness.csv").open("x", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=("episode_id", "reason")).writeheader()
    report = f"""# CoreAct Ensemble vs Contrast Control

This is a mechanism decision experiment, not confirmation. Task 4 outcomes were known. Task 7 was selected using known vanilla success only.

## Integrity

- Episodes: 400/400
- Complete paired units: 100/100
- Missing/duplicate/nonfinite: 0
- Mandatory gate: PASS

## Results

```json
{json.dumps(task_results, indent=2, sort_keys=True)}
```

Pooled rates are descriptive only: `{json.dumps(pooled_rates, sort_keys=True)}`.

## Decision

`{decision}`

The preregistered rules were evaluated in order. No confirmation claim is allowed.
"""
    (artifact / "report.md").write_text(report, encoding="utf-8")
    reproduction = f"""#!/usr/bin/env bash
set -euo pipefail
WORKSPACE={artifact.parent.parent}
ARTIFACT={artifact}
PYTHON=$WORKSPACE/task1/.conda-envs/flow-vla/bin/python
export HF_HOME=$WORKSPACE/task1/.hf-cache TRANSFORMERS_CACHE=$WORKSPACE/task1/.hf-cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYTHONPATH=$WORKSPACE:$WORKSPACE/lerobot/src:$WORKSPACE/LIBERO
$PYTHON -m research.coreact_ensemble_control.integrity --workspace $WORKSPACE --artifact $ARTIFACT
for shard in 0 1 2 3; do $PYTHON -m research.coreact_ensemble_control.run --workspace $WORKSPACE --artifact $ARTIFACT --shard-index $shard --shard-count 4; done
$PYTHON -m research.coreact_ensemble_control.analyze --artifact $ARTIFACT
"""
    (artifact / "reproduction_commands.sh").write_text(reproduction, encoding="utf-8")
    audit = {str(path.relative_to(artifact)): sha256(path) for path in sorted(artifact.rglob("*")) if path.is_file() and path.name != "sha256_audit.json"}
    (artifact / "sha256_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (artifact / "status/analysis.complete").touch(exist_ok=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
