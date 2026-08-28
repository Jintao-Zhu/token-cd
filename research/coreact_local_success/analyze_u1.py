from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from research.coreact_local_success.prepare_u1_manifest import ARMS


DIRECTIONS = ARMS[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    episodes = [json.loads(path.read_text()) for path in (artifact / "u1").glob("*.json")]
    if len(episodes) != 4500 or list((artifact / "invalid_units").glob("*.json")):
        raise RuntimeError("U1 incomplete or invalid")
    units: dict[str, dict[str, dict]] = defaultdict(dict)
    for episode in episodes:
        units[episode["unit_id"]][episode["arm"]] = episode
    if len(units) != 500 or any(set(group) != set(ARMS) for group in units.values()):
        raise RuntimeError("U1 pairing failure")
    snapshots: dict[str, list[dict[str, dict]]] = defaultdict(list)
    for group in units.values():
        snapshots[group["Strong"]["snapshot_id"]].append(group)
    if len(snapshots) != 100 or any(len(groups) != 5 for groups in snapshots.values()):
        raise RuntimeError("U1 snapshot clustering failure")
    state_rows, held_pairs, selection_counts = [], [], Counter()
    for snapshot_id, groups in sorted(snapshots.items()):
        groups = sorted(groups, key=lambda group: group["Strong"]["continuation_seed"])
        selection = groups[:3]
        held = groups[3:]
        selection_scores = {arm: float(np.mean([group[arm]["success"] for group in selection])) for arm in DIRECTIONS}
        selected = max(DIRECTIONS, key=lambda arm: (selection_scores[arm], -DIRECTIONS.index(arm)))
        selection_counts[selected] += 1
        strong_held = float(np.mean([group["Strong"]["success"] for group in held]))
        selected_held = float(np.mean([group[selected]["success"] for group in held]))
        row = {"snapshot_id": snapshot_id, "task_id": groups[0]["Strong"]["task_id"], "selected_direction": selected, "strong_held": strong_held, "selected_held": selected_held, "held_utility": selected_held - strong_held}
        for arm in DIRECTIONS:
            row[f"selection_{arm}"] = selection_scores[arm]
            row[f"held_utility_{arm}"] = float(np.mean([group[arm]["success"] for group in held])) - strong_held
        state_rows.append(row)
        for group in held:
            held_pairs.append((int(group["Strong"]["success"]), int(group[selected]["success"]), row["task_id"]))
    diffs = np.asarray([row["held_utility"] for row in state_rows])
    rng = np.random.default_rng(20260817)
    bootstrap = np.asarray([diffs[rng.integers(0, len(diffs), len(diffs))].mean() * 100 for _ in range(10000)])
    strong_success = float(np.mean([pair[0] for pair in held_pairs]))
    selected_success = float(np.mean([pair[1] for pair in held_pairs]))
    rescue = sum(a == 0 and b == 1 for a, b, _ in held_pairs)
    harm = sum(a == 1 and b == 0 for a, b, _ in held_pairs)
    discordant = rescue + harm
    task_rows = []
    for task_id in range(10):
        pairs = [pair for pair in held_pairs if pair[2] == task_id]
        task_rows.append({"task": task_id, "Strong": float(np.mean([pair[0] for pair in pairs])), "Cross_seed_selected": float(np.mean([pair[1] for pair in pairs]))})
    nonworse = sum(row["Cross_seed_selected"] >= row["Strong"] for row in task_rows)
    delta_pp = (selected_success - strong_success) * 100
    confirmed = delta_pp >= 5 and np.quantile(bootstrap, 0.025) > 0 and rescue > harm and nonworse >= 7
    decision = "LOCAL_SUCCESS_HEADROOM_CONFIRMED" if confirmed else "NO_MEANINGFUL_LOCAL_SUCCESS_HEADROOM_IN_TESTED_SUBSPACE"
    result = {"decision": decision, "strong_held_success": strong_success, "cross_seed_selected_success": selected_success, "delta_pp": delta_pp, "snapshot_cluster_ci95_pp": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))], "rescue": rescue, "harm": harm, "mcnemar_p": float(binomtest(min(rescue, harm), discordant).pvalue) if discordant else 1.0, "tasks_nonworse": nonworse, "positive_held_utility_snapshot_fraction": float(np.mean(diffs > 0)), "direction_selection_frequency": dict(selection_counts), "task_rows": task_rows, "snapshots": 100, "held_episodes": 200}
    (artifact / "u1_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (artifact / "final_decision.json").write_text(json.dumps({"decision": decision}, indent=2) + "\n")
    with (artifact / "u1_candidate_utility_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=state_rows[0].keys())
        writer.writeheader()
        writer.writerows(state_rows)
    with (artifact / "u1_task_success.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=task_rows[0].keys())
        writer.writeheader()
        writer.writerows(task_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
