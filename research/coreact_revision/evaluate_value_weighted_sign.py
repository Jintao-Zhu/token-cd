#!/usr/bin/env python3
"""Task-heldout evaluation of value-norm-weighted attention proxies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.coreact_revision.evaluate_region_sign_probe_v3 import ATTENTION, REGION, aggregate, evaluate


VALUE = ["late_half_value_weighted_attention", "late_half_value_weighted_attention_std", "raw_last_value_weighted_attention"]


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--raw-effects", type=Path, required=True); p.add_argument("--value-scores", type=Path, required=True); p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    rows = aggregate(args.raw_effects)
    scores = {}
    for line in args.value_scores.read_text().splitlines():
        row = json.loads(line); key = (row["task_id"], row["demo_id"], row["frame_id"], row["group_id"])
        if key in scores: raise RuntimeError(f"duplicate value score {key}")
        scores[key] = row
    missing = []
    for row in rows:
        key = (row["task_id"], row["demo_id"], row["frame_id"], row["group_id"])
        if key not in scores: missing.append(key); continue
        row.update({name: float(scores[key][name]) for name in VALUE})
    if missing: raise RuntimeError(f"missing {len(missing)} labeled value scores")
    methods = {
        "attention_only": ATTENTION,
        "value_weighted_only": VALUE,
        "attention_plus_value": ATTENTION + VALUE,
        "region_plus_value": REGION + VALUE,
        "attention_region_plus_value": ATTENTION + REGION + VALUE,
    }
    result = {"stage": "multitask development; leave-one-whole-task-out", "groups": len(rows), "value_score_join_count": len(scores), "methods": {name: evaluate(rows, features, "task_id") for name, features in methods.items()}, "interpretation": "Value norm weighting is a target-free ranking proxy, not causal attribution and not a signed contribution."}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
