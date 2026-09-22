#!/usr/bin/env python3
"""Compare dev-set attention arms against existing vanilla/current L11 runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(root: Path, min_state: int = 0, max_state: int | None = None) -> dict[tuple[str, int], dict]:
    rows = {}
    for path in root.glob("*/episode_*.json"):
        row = json.loads(path.read_text())
        state = int(row["init_state_index"])
        if state < min_state or (max_state is not None and state > max_state):
            continue
        rows[(row["task"], state)] = row
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla-root", type=Path, required=True)
    p.add_argument("--current-root", type=Path, required=True)
    p.add_argument("--role-root", type=Path, required=True)
    p.add_argument("--deep-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--min-state", type=int, default=0)
    p.add_argument("--max-state", type=int, default=9)
    a = p.parse_args()
    data = {
        "vanilla": load(a.vanilla_root.resolve(), a.min_state, a.max_state),
        "current": load(a.current_root.resolve(), a.min_state, a.max_state),
        "role": load(a.role_root.resolve(), a.min_state, a.max_state),
        "deep": load(a.deep_root.resolve(), a.min_state, a.max_state),
    }
    keys = sorted(set.intersection(*(set(v) for v in data.values())))
    tasks = sorted({key[0] for key in keys})
    out = {"paired_n": len(keys), "tasks": {}}
    lines = [
        "# LIBERO-Spatial attention dev compare",
        "",
        "| Task | n | vanilla | current L11 | role L11 | deep role 23-25 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    totals = {name: 0 for name in data}
    for task in tasks:
        task_keys = [key for key in keys if key[0] == task]
        counts = {
            name: sum(bool(data[name][key]["success"]) for key in task_keys)
            for name in data
        }
        for name, value in counts.items():
            totals[name] += value
        out["tasks"][task] = {
            "n": len(task_keys),
            **{name: int(counts[name]) for name in data},
        }
        lines.append(
            f"| {task} | {len(task_keys)} | {counts['vanilla']} | "
            f"{counts['current']} | {counts['role']} | {counts['deep']} |"
        )
    lines += [
        "",
        "| Arm | Success | Rate |",
        "|---|---:|---:|",
    ]
    for name in ("vanilla", "current", "role", "deep"):
        rate = totals[name] / len(keys) if keys else 0.0
        lines.append(f"| {name} | {totals[name]} / {len(keys)} | {rate:.3f} |")
    out["totals"] = {name: int(value) for name, value in totals.items()}
    root = a.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "DEV_COMPARE_SUMMARY.json").write_text(json.dumps(out, indent=2) + "\n")
    (root / "DEV_COMPARE_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
