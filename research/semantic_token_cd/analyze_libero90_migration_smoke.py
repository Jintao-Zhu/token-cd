#!/usr/bin/env python3
"""Summarize the bounded LIBERO-90 migration smoke run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()

    manifest = json.loads(a.manifest.read_text())
    rows = []
    for item in manifest["tasks"]:
        task = item["task_name"]
        task_row = {
            "task_id": item["task_id"],
            "task_name": task,
            "instruction": item["instruction"],
            "pcd_reference_text": item["pcd_reference_text"],
            "mapping_status": item["mapping_status"],
            "arms": {},
        }
        for arm in ("vanilla", "matched"):
            record = load_json(a.root / "runs" / arm / task / "episode_000.json")
            if record is None:
                task_row["arms"][arm] = {"status": "missing"}
                continue
            normal_end = bool(
                record.get("success", False)
                or record.get("done", False)
                or int(record.get("steps", 0)) >= int(record.get("max_policy_steps", 0))
            )
            arm_row = {
                "status": "complete",
                "success": bool(record.get("success", False)),
                "steps": int(record.get("steps", 0)),
                "normal_end": normal_end,
                "done": bool(record.get("done", False)),
                "initial_state_sha256": record.get("initial_state_sha256"),
                "runtime_seconds": record.get("runtime_seconds"),
            }
            if arm == "matched":
                trace = record.get("trace") or []
                first = trace[0] if trace else {}
                arm_row.update({
                    "entities": first.get("entities"),
                    "m_matched_first": first.get("m_matched"),
                    "selected_token_ids_first": first.get("selected_token_ids"),
                    "mean_m": record.get("selected_token_count_mean"),
                    "std_m": record.get("selected_token_count_std"),
                    "guided_changed_dims_first": first.get("guided_changed_dims"),
                    "feature_perturbation_relative_first": first.get("feature_perturbation_relative"),
                })
            task_row["arms"][arm] = arm_row
        v = task_row["arms"].get("vanilla", {})
        m = task_row["arms"].get("matched", {})
        task_row["paired_initial_state_match"] = bool(
            v.get("status") == "complete"
            and m.get("status") == "complete"
            and v.get("initial_state_sha256") == m.get("initial_state_sha256")
        )
        rows.append(task_row)

    result = {
        "purpose": "bounded migration smoke validation; not a method-performance comparison",
        "suite": manifest.get("suite"),
        "task_count": len(rows),
        "episode_count": sum(
            1
            for row in rows
            for arm in row["arms"].values()
            if arm.get("status") == "complete"
        ),
        "all_tasks_complete": all(
            row["arms"].get("vanilla", {}).get("status") == "complete"
            and row["arms"].get("matched", {}).get("status") == "complete"
            for row in rows
        ),
        "all_paired_initial_states_match": all(row["paired_initial_state_match"] for row in rows),
        "rows": rows,
    }
    (a.root / "SMOKE_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LIBERO-90 Migration Smoke Results",
        "",
        "This table is an engineering smoke validation, not a method-performance comparison.",
        "",
        "| Task | Init state | Arm | Normal end | Success | Steps | Mean m | Notes |",
        "|---|---:|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        for arm in ("vanilla", "matched"):
            rec = row["arms"][arm]
            if rec.get("status") != "complete":
                lines.append(f"| {row['task_name']} | 0 | {arm} | no | - | - | - | missing result |")
                continue
            mean_m = f"{rec['mean_m']:.2f}" if arm == "matched" and rec.get("mean_m") is not None else "-"
            note = "paired initial state verified" if row["paired_initial_state_match"] else "initial-state mismatch"
            if arm == "matched" and rec.get("entities"):
                note += f"; entities={rec['entities']}"
            lines.append(
                f"| {row['task_name']} | 0 | {arm} | {rec['normal_end']} | {rec['success']} | {rec['steps']} | {mean_m} | {note} |"
            )
    (a.root / "SMOKE_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
