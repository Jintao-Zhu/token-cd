#!/usr/bin/env python3
"""CW-LPCD STEP 7: freeze the offline Selection / Confirmation split.

270 states = 9 tasks x 30 seeds (0..29). Per task: seeds 0..14 -> Selection
(15), seeds 15..29 -> Confirmation (15). Total 135 / 135.

Trajectory isolation: each state_id is a single (task, seed) initialization and
is itself its own trajectory (the Pixel-PCD mini harness has exactly one state
per seed per task), so no multi-state trajectory can straddle the split. The
rule is fully deterministic and frozen before any residual is computed.

Run: env/venv/bin/python -m research.cw_lpcd.step7_split --artifact artifacts/cw_lpcd_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research.cw_lpcd.core import TASKS

STATES_LOCK = Path(
    "artifacts/_archive/token_pcd/token_pcd_openvla_simpler_stage_a_v1_20260814/states.lock.jsonl"
)

SPLIT_RULE = "per task: seeds 0..14 -> Selection, seeds 15..29 -> Confirmation (15/15). Each (task, seed) is one trajectory."


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--states-lock", type=Path, default=STATES_LOCK)
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    states = read_jsonl(args.states_lock.resolve())

    by_task: dict[str, list[dict]] = {task: [] for task in TASKS}
    for row in states:
        by_task[row["task"]].append(row)

    selection: list[str] = []
    confirmation: list[str] = []
    per_task: dict[str, dict] = {}
    for task in TASKS:
        rows = sorted(by_task[task], key=lambda r: int(r["seed"]))
        assert len(rows) == 30, f"{task}: expected 30 states, got {len(rows)}"
        sel = [r["state_id"] for r in rows if int(r["seed"]) <= 14]
        conf = [r["state_id"] for r in rows if int(r["seed"]) >= 15]
        assert len(sel) == 15 and len(conf) == 15, f"{task}: {len(sel)}/{len(conf)}"
        selection.extend(sel)
        confirmation.extend(conf)
        per_task[task] = {"selection": sel, "confirmation": conf}

    assert len(selection) == 135 and len(confirmation) == 135
    assert not (set(selection) & set(confirmation)), "selection/confirmation overlap"

    split = {
        "protocol_id": "CW-LPCD-PHASE0-SPLIT-R1",
        "split_rule": SPLIT_RULE,
        "trajectory_isolation_note": (
            "each state_id is a single (task, seed) SIMPLER initialization = one trajectory; "
            "no trajectory spans both splits."
        ),
        "n_selection": len(selection),
        "n_confirmation": len(confirmation),
        "tasks": TASKS,
        "selection_state_ids": sorted(selection),
        "confirmation_state_ids": sorted(confirmation),
        "per_task": per_task,
    }

    canonical = json.dumps(split, sort_keys=True, ensure_ascii=False) + "\n"
    out_json = artifact / "FROZEN_SPLIT.json"
    out_json.write_text(canonical)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (artifact / "FROZEN_SPLIT.sha256").write_text(digest + "  FROZEN_SPLIT.json\n")

    print(json.dumps({"status": "FROZEN", "n_selection": len(selection),
                      "n_confirmation": len(confirmation), "sha256": digest,
                      "split_rule": SPLIT_RULE}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
