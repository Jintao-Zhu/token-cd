"""Audit the completed baseline and apply the locked task-selection rule."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    manifest = [json.loads(line) for line in (artifact / "baseline_manifest.jsonl").read_text().splitlines() if line]
    results = [json.loads(line) for line in (artifact / "baseline_results.jsonl").read_text().splitlines() if line]
    expected = {row["episode_id"] for row in manifest}
    counts = Counter(row["episode_id"] for row in results)
    actual = set(counts)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    errors = [row["episode_id"] for row in results if row["error"] is not None or row["video_error"] is not None]
    nonfinite = [row["episode_id"] for row in results if row["nonfinite_action"]]
    complete = len(results) == 200 and not duplicates and not missing and not unexpected and not errors and not nonfinite
    if not complete:
        report = {
            "status": "FAIL",
            "result_rows": len(results),
            "duplicates": duplicates,
            "missing": missing,
            "unexpected": unexpected,
            "errors": errors,
            "nonfinite": nonfinite,
        }
        (artifact / "baseline_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise SystemExit("Baseline is incomplete or failed integrity; refusing to select tasks")

    by_task = defaultdict(list)
    descriptions = {}
    for row in results:
        by_task[row["task_id"]].append(bool(row["success"]))
        descriptions[row["task_id"]] = row["task_description"]
    task_rates = {
        task_id: {"successes": sum(values), "episodes": len(values), "success_rate": sum(values) / len(values), "description": descriptions[task_id]}
        for task_id, values in sorted(by_task.items())
    }
    eligible = [task_id for task_id, values in task_rates.items() if 0.40 <= values["success_rate"] <= 0.85]
    eligible.sort(key=lambda task_id: (abs(task_rates[task_id]["success_rate"] - 0.625), task_id))
    selected = eligible[:3]
    report = {
        "status": "PASS" if len(selected) == 3 else "INSUFFICIENT_ELIGIBLE_TASKS",
        "result_rows": len(results),
        "task_rates": task_rates,
        "eligible_task_ids": eligible,
        "selected_task_ids": selected,
        "selection_rule": "success_rate in [0.40, 0.85], then closest to 0.625, then task_id",
        "duplicates": [],
        "missing": [],
        "errors": [],
        "nonfinite": [],
    }
    (artifact / "baseline_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (artifact / "selected_tasks.json").write_text(
        json.dumps({"task_ids": selected, "descriptions": {str(task_id): descriptions[task_id] for task_id in selected}}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": report["status"], "selected_task_ids": selected}))
    if len(selected) != 3:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
