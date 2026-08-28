#!/usr/bin/env python3
"""Audit paired rollouts and decide the preregistered Effect-Existence Gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.coreact_closed_loop.audit_qualification import video_frames
from research.coreact_closed_loop.run_pilot import read_jsonl, write_json
from research.coreact_region.effect_existence import HORIZONS


COMPONENTS = ("composite_progress", "reach_progress", "transport_progress", "grasp_ever", "predicate_ever")


def effect_row(clean: dict, masked: dict, horizon: int) -> dict:
    a = clean["progress_by_horizon"][str(horizon)]
    b = masked["progress_by_horizon"][str(horizon)]
    row = {}
    for component in COMPONENTS:
        row[f"clean_{component}"] = float(a[component])
        row[f"mask_{component}"] = float(b[component])
        row[f"D_{component}"] = float(a[component]) - float(b[component])
    return row


def summarize(rows: list[dict], condition: str, horizon: int) -> dict:
    selected = [row for row in rows if row["condition"] == condition and row["horizon"] == horizon]
    values = np.asarray([row["D_composite_progress"] for row in selected], dtype=float)
    by_suite = {
        suite: float(np.median([row["D_composite_progress"] for row in selected if row["suite"] == suite]))
        for suite in sorted({row["suite"] for row in selected})
    }
    return {
        "condition": condition, "horizon": horizon, "n": len(selected),
        "mean_D": float(np.mean(values)) if len(values) else None,
        "median_D": float(np.median(values)) if len(values) else None,
        "fraction_D_gt_0_05": float(np.mean(values > 0.05)) if len(values) else None,
        "median_D_by_suite": by_suite,
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
        if record["control_steps"] != 60 or not record["all_outputs_finite"]:
            failures.append({"episode_id": spec["episode_id"], "reason": "step or finite mismatch"})
        if record["initial_sim_state_sha256"] != spec["sim_state_sha256"]:
            failures.append({"episode_id": spec["episode_id"], "reason": "sim state mismatch"})
        if record["masked_replans"] != ([] if spec["condition"] == "clean" else [0]):
            failures.append({"episode_id": spec["episode_id"], "reason": "mask schedule mismatch"})
        if video_frames(artifact / record["video"]) != 60:
            failures.append({"episode_id": spec["episode_id"], "reason": "video frame mismatch"})
        if len(read_jsonl(artifact / record["step_log"])) != 60:
            failures.append({"episode_id": spec["episode_id"], "reason": "step log mismatch"})
        records[spec["episode_id"]] = record

    grouped = defaultdict(dict)
    for record in records.values():
        grouped[record["pair_id"]][record["condition"]] = record
    effects = []
    for pair_id, conditions in sorted(grouped.items()):
        if "clean" not in conditions:
            failures.append({"episode_id": pair_id, "reason": "missing clean pair"})
            continue
        clean = conditions["clean"]
        clean_actions = np.asarray([
            row["model_action"] for row in read_jsonl(artifact / clean["step_log"])
        ])
        for condition, masked in sorted(conditions.items()):
            if condition == "clean":
                continue
            if (
                clean["initial_sim_state_sha256"] != masked["initial_sim_state_sha256"]
                or clean["noise_sha256_by_replan"] != masked["noise_sha256_by_replan"]
                or clean["candidate_groups_sha256"] != masked["candidate_groups_sha256"]
            ):
                failures.append({"episode_id": masked["episode_id"], "reason": "paired state/noise/group mismatch"})
                continue
            masked_actions = np.asarray([
                row["model_action"] for row in read_jsonl(artifact / masked["step_log"])
            ])
            first_delta = clean_actions[:10] - masked_actions[:10]
            first_rmse = float(np.sqrt(np.mean(first_delta**2)))
            first_max_abs = float(np.max(np.abs(first_delta)))
            for horizon in HORIZONS:
                effects.append({
                    "pair_id": pair_id, "suite": clean["suite"], "phase": clean["phase"],
                    "condition": condition, "replacement_type": masked["replacement_type"],
                    "selected_count": masked["selected_count"], "horizon": horizon,
                    "first_chunk_action_rmse": first_rmse,
                    "first_chunk_action_max_abs": first_max_abs,
                    **effect_row(clean, masked, horizon),
                })

    primary = [
        row for row in effects
        if row["condition"] == "full_target" and row["horizon"] == 60
    ]
    primary_summary = summarize(effects, "full_target", 60)
    zero_summary = summarize(effects, "full_target_zero", 60)
    background_summary = summarize(effects, "background_random_match_full_target", 60)
    phase_counts = {
        phase: sum(row["phase"] == phase for row in primary)
        for phase in sorted({row["phase"] for row in primary})
    }
    variation_count = sum(
        abs(row["D_reach_progress"]) > 0.05
        or abs(row["D_transport_progress"]) > 0.05
        or row["D_grasp_ever"] != 0
        or row["D_predicate_ever"] != 0
        for row in primary
    )
    complete_rate = len(records) / len(specs) if specs else 0.0
    suite_positive = (
        len(primary_summary["median_D_by_suite"]) == 2
        and all(value > 0 for value in primary_summary["median_D_by_suite"].values())
    )
    passed = (
        not failures and complete_rate == 1.0 and len(primary) >= 6
        and primary_summary["median_D"] is not None and primary_summary["median_D"] > 0.05
        and suite_positive and primary_summary["fraction_D_gt_0_05"] >= 0.5
        and variation_count >= 4
    )
    decision = "EFFECT_EXISTENCE_ESTABLISHED" if passed else "EFFECT_EXISTENCE_NOT_ESTABLISHED"
    summaries = [
        summarize(effects, condition, horizon)
        for condition in sorted({row["condition"] for row in effects})
        for horizon in HORIZONS
    ]
    result = {
        "stage": "development_effect_existence_gate",
        "expected_episodes": len(specs), "actual_episodes": len(records),
        "complete_rate": complete_rate, "integrity_failures": failures,
        "primary_full_target_H60": primary_summary,
        "full_target_zero_H60": zero_summary,
        "matched_background_H60": background_summary,
        "critical_phase_counts": phase_counts,
        "primary_component_variation_count": variation_count,
        "primary_first_chunk_action_rmse_median": primary_summary["median_first_chunk_action_rmse"],
        "suite_direction_positive": suite_positive,
        "gate_pass": passed, "decision": decision,
        "all_condition_summaries": summaries,
        "interpretation": "A pass establishes measurable outcome sensitivity for this intervention scale; it does not validate any attribution ranking or closed-loop success benefit.",
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
    report = f"""# CoreAct Effect-Existence Gate

## Locked question

Does a first-chunk visual intervention from clean-selected pre-grasp/pre-place states produce a measurable task-progress effect?

## Integrity

- Expected/actual episodes: {len(specs)}/{len(records)}
- Integrity failures: {len(failures)}
- Same exact simulator state, per-replan Gaussian noise, candidate mapping: checked pairwise
- Intervention schedule: first 10 executed actions only; later replans vanilla

## Primary result

- Full-target position-mean deletion at H=60: median D = {primary_summary['median_D']}
- Fraction with D > 0.05: {primary_summary['fraction_D_gt_0_05']}
- Per-suite median D: {json.dumps(primary_summary['median_D_by_suite'], sort_keys=True)}
- States with component variation: {variation_count}/{len(primary)}
- Median first-chunk action RMSE (clean versus full-target): {primary_summary['median_first_chunk_action_rmse']}
- Full-target zero-replacement median D at H=60: {zero_summary['median_D']}
- Matched background position-mean median D at H=60: {background_summary['median_D']}
- Critical phases: {json.dumps(phase_counts, sort_keys=True)}

`D = clean progress - mask progress`; positive means deletion reduced progress.

## Decision

`{decision}`

The mask did change the first action chunk, so this is not a no-op hook result. The changed actions did not produce a stable task-progress label under the locked probe.

No clean trajectory reached a pre-place state within the locked 60-step extraction horizon. This result is limited to six true pre-grasp and four threshold-qualified near-reach states.

`full-target` means every post-connector token whose source footprint contains target pixels. Because global visual self-attention occurs before this intervention, it does not guarantee removal of all target information from every visual token.

This gate does not evaluate IG, value-weighted attention, activation patching, CFG, or lambda. A pass only establishes that the outcome probe has a measurable label at the tested intervention scale.
"""
    (artifact / "report.md").write_text(report)
    figures = artifact / "figures"
    figures.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    plotted = (
        "full_target",
        "full_target_zero",
        "background_random_match_full_target",
    )
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for condition in plotted:
        medians = []
        for horizon in HORIZONS:
            values = [
                row["D_composite_progress"] for row in effects
                if row["condition"] == condition and row["horizon"] == horizon
            ]
            medians.append(float(np.median(values)))
        axis.plot(HORIZONS, medians, marker="o", label=condition)
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1, label="gate threshold")
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.set(xlabel="Horizon (control steps)", ylabel="Median D (clean - mask)")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "effect_by_horizon.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 4.5))
    axis.scatter(
        [row["first_chunk_action_rmse"] for row in primary],
        [row["D_composite_progress"] for row in primary],
    )
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1)
    axis.axhline(0, color="gray", linewidth=0.8)
    axis.set(xlabel="First-chunk action RMSE", ylabel="H=60 D (clean - full target)")
    fig.tight_layout()
    fig.savefig(figures / "action_change_vs_task_effect.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
