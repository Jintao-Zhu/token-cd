#!/usr/bin/env python3
"""Audit and decide the pre-encoder image-space Effect-Existence Gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from research.coreact_closed_loop.audit_qualification import video_frames
from research.coreact_closed_loop.run_pilot import read_jsonl, write_json
from research.coreact_region.effect_existence import HORIZONS


COMPONENTS = ("composite_progress", "reach_progress", "transport_progress", "grasp_ever", "predicate_ever")


def summarize(rows: list[dict], condition: str, horizon: int) -> dict:
    selected = [row for row in rows if row["condition"] == condition and row["horizon"] == horizon]
    values = np.asarray([row["D_composite_progress"] for row in selected], dtype=float)
    return {
        "condition": condition, "horizon": horizon, "n": len(selected),
        "mean_D": float(np.mean(values)) if len(values) else None,
        "median_D": float(np.median(values)) if len(values) else None,
        "fraction_D_gt_0_05": float(np.mean(values > 0.05)) if len(values) else None,
        "median_D_by_suite": {
            suite: float(np.median([
                row["D_composite_progress"] for row in selected if row["suite"] == suite
            ])) for suite in sorted({row["suite"] for row in selected})
        },
        "median_D_by_phase": {
            phase: float(np.median([
                row["D_composite_progress"] for row in selected if row["phase"] == phase
            ])) for phase in sorted({row["phase"] for row in selected})
        },
        "median_first_chunk_action_rmse": float(np.median([
            row["first_chunk_action_rmse"] for row in selected
        ])) if selected else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    specs = read_jsonl(artifact / "rollout_manifest.jsonl")
    records, failures = {}, []
    for spec in specs:
        path = artifact / "episodes" / spec["episode_id"] / "episode.json"
        if not path.exists():
            failures.append({"episode_id": spec["episode_id"], "reason": "missing episode"})
            continue
        record = json.loads(path.read_text())
        expected_replans = [] if spec["condition"] == "vanilla" else [0]
        checks = (
            record["control_steps"] == 60,
            record["all_outputs_finite"],
            record["initial_sim_state_sha256"] == spec["sim_state_sha256"],
            record["mask_bundle_sha256_runtime"] == spec["mask_bundle_sha256"],
            record["intervention_replans"] == expected_replans,
            video_frames(artifact / record["video"]) == 60,
            len(read_jsonl(artifact / record["step_log"])) == 60,
        )
        if not all(checks):
            failures.append({"episode_id": spec["episode_id"], "reason": "record integrity mismatch"})
        records[spec["episode_id"]] = record

    grouped = defaultdict(dict)
    for record in records.values():
        grouped[record["pair_id"]][record["condition"]] = record
    effects = []
    for pair_id, conditions in sorted(grouped.items()):
        if "vanilla" not in conditions:
            failures.append({"episode_id": pair_id, "reason": "missing vanilla"})
            continue
        vanilla = conditions["vanilla"]
        vanilla_actions = np.asarray([
            row["model_action"] for row in read_jsonl(artifact / vanilla["step_log"])
        ])
        for condition, masked in sorted(conditions.items()):
            if condition == "vanilla":
                continue
            if (
                masked["initial_sim_state_sha256"] != vanilla["initial_sim_state_sha256"]
                or masked["noise_sha256_by_replan"] != vanilla["noise_sha256_by_replan"]
                or masked["mask_bundle_sha256_runtime"] != vanilla["mask_bundle_sha256_runtime"]
            ):
                failures.append({"episode_id": masked["episode_id"], "reason": "paired state/noise/mask mismatch"})
                continue
            masked_actions = np.asarray([
                row["model_action"] for row in read_jsonl(artifact / masked["step_log"])
            ])
            delta = vanilla_actions[:10] - masked_actions[:10]
            semantic, replacement = condition.rsplit("_", 1)
            for horizon in HORIZONS:
                clean_progress = vanilla["progress_by_horizon"][str(horizon)]
                mask_progress = masked["progress_by_horizon"][str(horizon)]
                row = {
                    "pair_id": pair_id, "suite": vanilla["suite"], "phase": vanilla["phase"],
                    "condition": condition, "semantic": semantic, "replacement": replacement,
                    "horizon": horizon,
                    "first_chunk_action_rmse": float(np.sqrt(np.mean(delta**2))),
                    "first_chunk_action_max_abs": float(np.max(np.abs(delta))),
                }
                for component in COMPONENTS:
                    row[f"vanilla_{component}"] = float(clean_progress[component])
                    row[f"mask_{component}"] = float(mask_progress[component])
                    row[f"D_{component}"] = float(clean_progress[component]) - float(mask_progress[component])
                effects.append(row)

    primary = [row for row in effects if row["condition"] == "target_inpaint" and row["horizon"] == 60]
    primary_summary = summarize(effects, "target_inpaint", 60)
    blur_summary = summarize(effects, "target_blur", 60)
    black_summary = summarize(effects, "target_black", 60)
    background_summary = summarize(effects, "background_match_target_inpaint", 60)
    component_variation = sum(
        abs(row["D_reach_progress"]) > 0.05
        or abs(row["D_transport_progress"]) > 0.05
        or row["D_grasp_ever"] != 0
        or row["D_predicate_ever"] != 0
        for row in primary
    )
    suite_positive = (
        len(primary_summary["median_D_by_suite"]) == 2
        and all(value > 0 for value in primary_summary["median_D_by_suite"].values())
    )
    complete_rate = len(records) / len(specs) if specs else 0.0
    passed = (
        not failures and complete_rate == 1.0 and len(primary) >= 6
        and primary_summary["median_D"] is not None and primary_summary["median_D"] > 0.05
        and primary_summary["fraction_D_gt_0_05"] >= 0.5 and suite_positive
        and component_variation >= 4 and blur_summary["median_D"] > 0
    )
    decision = "IMAGE_OUTCOME_DYNAMIC_RANGE_ESTABLISHED" if passed else "IMAGE_OUTCOME_DYNAMIC_RANGE_NOT_ESTABLISHED"
    summaries = [
        summarize(effects, condition, horizon)
        for condition in sorted({row["condition"] for row in effects})
        for horizon in HORIZONS
    ]
    result = {
        "stage": "development_image_space_positive_control",
        "expected_episodes": len(specs), "actual_episodes": len(records),
        "complete_rate": complete_rate, "integrity_failures": failures,
        "primary_target_inpaint_H60": primary_summary,
        "target_blur_H60": blur_summary, "target_black_H60": black_summary,
        "matched_background_inpaint_H60": background_summary,
        "primary_component_variation_count": component_variation,
        "suite_direction_positive": suite_positive,
        "gate_pass": passed, "decision": decision,
        "all_condition_summaries": summaries,
        "interpretation": "This tests outcome-probe dynamic range for a pre-encoder image intervention, not attribution quality.",
    }
    write_json(artifact / "analysis_summary.json", result)
    write_json(artifact / "decision.json", {"status": decision, "gate_pass": passed})
    with (artifact / "paired_effects.csv").open("w", newline="") as stream:
        if effects:
            writer = csv.DictWriter(stream, fieldnames=list(effects[0]))
            writer.writeheader(); writer.writerows(effects)
    with (artifact / "missingness.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("episode_id", "reason"))
        writer.writeheader(); writer.writerows(failures)

    figures = artifact / "figures"
    figures.mkdir(exist_ok=True)
    plotted = ("target_inpaint", "target_blur", "target_black", "background_match_target_inpaint")
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for condition in plotted:
        medians = [summarize(effects, condition, horizon)["median_D"] for horizon in HORIZONS]
        axis.plot(HORIZONS, medians, marker="o", label=condition)
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1, label="gate threshold")
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.set(xlabel="Horizon (control steps)", ylabel="Median D (vanilla - mask)")
    axis.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figures / "target_effect_by_horizon.png", dpi=180); plt.close(fig)

    report = f"""# Image-Space Effect-Existence Gate

## Integrity

- Expected/actual episodes: {len(specs)}/{len(records)}
- Integrity failures: {len(failures)}
- Exact simulator state, six per-replan noise hashes, and image-mask bundle hashes checked pairwise
- Only the first 10-action chunk used the image intervention; later replans were vanilla

## Primary result

- Target inpaint H=60 median D: {primary_summary['median_D']}
- Fraction D > 0.05: {primary_summary['fraction_D_gt_0_05']}
- Per-suite medians: {json.dumps(primary_summary['median_D_by_suite'], sort_keys=True)}
- Per-phase medians: {json.dumps(primary_summary['median_D_by_phase'], sort_keys=True)}
- Component-varying states: {component_variation}/{len(primary)}
- First-chunk action RMSE median: {primary_summary['median_first_chunk_action_rmse']}
- Target blur H=60 median D: {blur_summary['median_D']}
- Target black H=60 median D: {black_summary['median_D']} (OOD sensitivity only)
- Matched-background inpaint H=60 median D: {background_summary['median_D']}

`D = vanilla progress - intervention progress`; positive means image removal reduced progress.

## Decision

`{decision}`

This result answers whether the locked outcome probe has dynamic range under first-chunk pre-encoder image removal. It does not validate attention, IG, activation patching, CFG, or CoreAct guidance.
"""
    (artifact / "report.md").write_text(report)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
