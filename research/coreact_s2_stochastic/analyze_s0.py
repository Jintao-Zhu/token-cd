from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(rows: list[dict]) -> dict:
    return {
        "median_pairwise_cosine": float(np.median([row["pairwise_cosine_median"] for row in rows])),
        "mean_pairwise_cosine": float(np.mean([row["pairwise_cosine_mean"] for row in rows])),
        "median_consensus_ratio": float(np.median([row["consensus_ratio"] for row in rows])),
        "positive_to_consensus_fraction": float(np.mean([value > 0 for row in rows for value in row["to_consensus_cosines"]])),
        "median_consensus_to_fixed_w1_cosine": float(np.median([row["consensus_to_fixed_w1_cosine"] for row in rows])),
        "median_residual_norm": float(np.median([value for row in rows for value in row["residual_norms"]])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    files = sorted((artifact / "s0").glob("*.json"))
    if len(files) != 100 or list((artifact / "invalid_units").glob("*.json")):
        raise RuntimeError("S0 incomplete or invalid")
    payloads = [json.loads(path.read_text()) for path in files]
    if any(not item["integrity"]["finite"] or item["integrity"]["all_ones_strong_parity_max_abs"] >= 1e-6 for item in payloads):
        raise RuntimeError("S0 integrity failure")
    candidates = {}
    task_table = []
    for candidate in ("S1", "S2"):
        candidate_rows = [row for item in payloads for row in item["rows"] if row["candidate"] == candidate]
        overall = summarize(candidate_rows)
        passing_tasks = 0
        task_values = []
        for task_id in range(10):
            rows = [row for item in payloads if item["metadata"]["task_id"] == task_id for row in item["rows"] if row["candidate"] == candidate]
            summary = summarize(rows)
            trend = summary["median_pairwise_cosine"] >= 0.50 and summary["median_consensus_ratio"] >= 0.60 and summary["positive_to_consensus_fraction"] >= 0.80
            passing_tasks += int(trend)
            task_values.append({"task": task_id, "candidate": candidate, **summary, "trend_pass": trend})
        gate = overall["median_pairwise_cosine"] >= 0.50 and overall["median_consensus_ratio"] >= 0.60 and overall["positive_to_consensus_fraction"] >= 0.80 and passing_tasks >= 7
        candidates[candidate] = {**overall, "tasks_passing_trend": passing_tasks, "gate_pass": gate}
        task_table.extend(task_values)
    passing = [name for name, value in candidates.items() if value["gate_pass"]]
    selected = None
    if passing:
        if len(passing) == 1:
            selected = passing[0]
        else:
            difference = abs(candidates["S1"]["median_pairwise_cosine"] - candidates["S2"]["median_pairwise_cosine"])
            selected = "S1" if difference < 0.05 else max(passing, key=lambda name: candidates[name]["median_pairwise_cosine"])
    decision = "STOCHASTIC_WEAK_CONSENSUS_NO_GO" if selected is None else "S0_CONSENSUS_PASS_READY_FOR_S1"
    result = {"decision": decision, "selected_candidate": selected, "candidates": candidates, "task_table": task_table, "states": 100, "active_steps": list(range(1, 9)), "masks_per_timestep": 12, "outputs_finite": True}
    (artifact / "s0_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (artifact / "s0_decision.json").write_text(json.dumps({"decision": decision, "selected_candidate": selected}, indent=2) + "\n")
    (artifact / "status.json").write_text(json.dumps({"status": decision}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
