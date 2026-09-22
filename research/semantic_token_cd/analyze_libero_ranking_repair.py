#!/usr/bin/env python3
"""Aggregate and compare the LIBERO ranking-repair matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


def load(root: Path, min_state: int = 20, max_state: int = 49):
    rows = {}
    for path in root.glob("*/episode_*.json"):
        row = json.loads(path.read_text())
        state = int(row.get("init_state_index", row["episode"]))
        if min_state <= state <= max_state:
            rows[(row["task"], state)] = row
    return rows


def paired_stats(a, b, keys):
    a_success = sum(bool(a[k]["success"]) for k in keys)
    b_success = sum(bool(b[k]["success"]) for k in keys)
    rescue = sum((not bool(a[k]["success"])) and bool(b[k]["success"]) for k in keys)
    harm = sum(bool(a[k]["success"]) and (not bool(b[k]["success"])) for k in keys)
    p = binomtest(rescue, rescue + harm, 0.5).pvalue if rescue + harm else 1.0
    return {
        "n": len(keys),
        "a_success": int(a_success),
        "b_success": int(b_success),
        "delta_b_minus_a": int(b_success - a_success),
        "rescue": int(rescue),
        "harm": int(harm),
        "net": int(rescue - harm),
        "p": float(p),
    }


def arm_summary(rows):
    return {
        "n": len(rows),
        "success": int(sum(bool(row["success"]) for row in rows.values())),
        "rate": float(np.mean([bool(row["success"]) for row in rows.values()])) if rows else None,
        "mean_steps": float(np.mean([row["steps"] for row in rows.values()])) if rows else None,
        "mean_m": float(np.mean([row.get("selected_token_count_mean", 0.0) for row in rows.values()])) if rows else None,
    }


def task_table(root: Path, names: list[str], reference: dict):
    output: dict[str, Any] = {"arm_summaries": {}, "tasks": {}}
    data = {name: load(root / name if name not in {"vanilla", "current"} else root / "_refs" / name) for name in names}
    # Load references separately for clarity.
    data["vanilla"] = load(Path("artifacts/libero_official_vanilla_500_v1"))
    data["current"] = load(Path("artifacts/libero_official_matched_500_v1"))
    tasks = sorted({key[0] for values in data.values() for key in values})
    keys_all = sorted(set.intersection(*(set(data[name]) for name in data)))
    for name, rows in data.items():
        output["arm_summaries"][name] = arm_summary(rows)
    for task in tasks:
        keys = [key for key in keys_all if key[0] == task]
        if not keys:
            continue
        output["tasks"][task] = {
            name: int(sum(bool(data[name][key]["success"]) for key in keys))
            for name in data
        }
    return output, data, keys_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--min-state", type=int, default=20)
    p.add_argument("--max-state", type=int, default=49)
    a = p.parse_args()
    root = a.root.resolve()
    arm_names = sorted([path.name for path in root.iterdir() if path.is_dir() and path.name not in {"logs", "__pycache__"}])
    # Include the two reference arms from their existing official roots.
    all_names = ["vanilla", "current"] + arm_names
    out: dict[str, Any] = {"arms": {}, "pairwise_vs_current": {}, "pairwise_vs_vanilla": {}, "tasks": {}}
    data = {"vanilla": load(Path("artifacts/libero_official_vanilla_500_v1"), a.min_state, a.max_state)}
    data["current"] = load(Path("artifacts/libero_official_matched_500_v1"), a.min_state, a.max_state)
    for name in arm_names:
        data[name] = load(root / name, a.min_state, a.max_state)
    keys_all = sorted(set.intersection(*(set(values) for values in data.values())))
    for name, rows in data.items():
        out["arms"][name] = {
            "n": len(rows),
            "success": int(sum(bool(row["success"]) for row in rows.values())),
            "rate": float(np.mean([bool(row["success"]) for row in rows.values()])) if rows else None,
            "mean_steps": float(np.mean([row["steps"] for row in rows.values()])) if rows else None,
            "mean_m": float(np.mean([row.get("selected_token_count_mean", 0.0) for row in rows.values()])) if rows else None,
        }
        if name != "current":
            out["pairwise_vs_current"][name] = paired_stats(data["current"], rows, keys_all) if len(rows) == len(keys_all) else None
        if name != "vanilla":
            out["pairwise_vs_vanilla"][name] = paired_stats(data["vanilla"], rows, keys_all) if len(rows) == len(keys_all) else None
    for task in sorted({key[0] for key in keys_all}):
        keys = [key for key in keys_all if key[0] == task]
        out["tasks"][task] = {name: int(sum(bool(data[name][key]["success"]) for key in keys)) for name in data}
    (root / "RANKING_REPAIR_SUMMARY.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["# LIBERO ranking repair matrix", "", f"Paired states: {len(keys_all)}", "", "| Arm | Success |", "|---|---:|"]
    for name, rec in out["arms"].items():
        lines.append(f"| {name} | {rec['success']} / {rec['n']} |")
    (root / "RANKING_REPAIR_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
