#!/usr/bin/env python3
"""Evaluate predefined action-consistency condition subsets."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from research.coreact_revision.capture_action_consistency import CONDITION_SUBSETS
from research.coreact_revision.evaluate_region_sign_probe_v3 import aggregate, evaluate

FEATURES = (
    "clean_cross_condition_dispersion",
    "masked_cross_condition_dispersion",
    "masked_minus_clean_dispersion",
    "clean_mask_consensus_distance",
    "intervention_direction_consistency",
    "mean_intervention_norm",
)
GATE = {"precision": 0.60, "recall": 0.60, "average_precision": 0.65}


def load_unique(path: Path) -> dict[tuple, dict]:
    rows = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["task_id"], row["demo_id"], row["frame_id"], row["group_id"])
        if key in rows:
            raise RuntimeError(f"duplicate key at line {line_number}: {key}")
        rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-effects", type=Path, required=True)
    parser.add_argument("--consistency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_effect_keys = set()
    for line in args.raw_effects.read_text().splitlines():
        raw = json.loads(line)
        raw_effect_keys.add((raw["task_id"], raw["demo_id"], raw["frame_id"], raw["group_id"]))
    rows = aggregate(args.raw_effects)
    consistency = load_unique(args.consistency)
    if len(consistency) != 6000:
        raise RuntimeError(f"expected 6000 consistency groups, got {len(consistency)}")
    if len({(r["task_id"], r["demo_id"], r["frame_id"]) for r in consistency.values()}) != 300:
        raise RuntimeError("expected exactly 300 states")

    if set(consistency) != raw_effect_keys:
        raise RuntimeError(
            f"raw key mismatch: missing={len(raw_effect_keys-set(consistency))}, "
            f"extra={len(set(consistency)-raw_effect_keys)}"
        )
    classified_keys = {(r["task_id"], r["demo_id"], r["frame_id"], r["group_id"]) for r in rows}
    if not classified_keys <= set(consistency):
        raise RuntimeError(f"missing {len(classified_keys-set(consistency))} classified keys")

    methods = {}
    for subset, indices in CONDITION_SUBSETS.items():
        names = [f"{subset}_{name}" for name in FEATURES]
        for row in rows:
            source = consistency[(row["task_id"], row["demo_id"], row["frame_id"], row["group_id"])]
            values = [source[name] for name in names]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"nonfinite {subset} feature for {row['group_id']}")
            row.update(dict(zip(names, values, strict=True)))
        metrics = evaluate(rows, names, "task_id")
        metrics["conditions"] = len(indices)
        metrics["velocity_evaluations_per_group"] = 2 * len(indices)
        metrics["every_heldout_task_selects"] = all(fold["predicted_nuisance"] > 0 for fold in metrics["folds"])
        metrics["gate_pass"] = (
            metrics["nuisance_precision"] >= GATE["precision"]
            and metrics["nuisance_recall"] >= GATE["recall"]
            and metrics["nuisance_average_precision"] >= GATE["average_precision"]
            and metrics["every_heldout_task_selects"]
        )
        methods[subset] = metrics

    eligible = [name for name, metrics in methods.items() if metrics["gate_pass"]]
    selected = min(
        eligible,
        key=lambda name: (
            methods[name]["velocity_evaluations_per_group"],
            -methods[name]["nuisance_average_precision"],
            name,
        ),
    ) if eligible else None
    result = {
        "stage": "predefined condition-cost ablation; whole-task heldout development",
        "gate": GATE,
        "methods": methods,
        "selected_subset": selected,
        "eligible_subsets": eligible,
        "online_qualification_authorized": selected is not None,
        "selective_closed_loop_authorized": False,
        "limit": "Offline x_tau remains teacher-forced around demonstration actions; online transfer is unqualified.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
