"""Print paired Pi0 vanilla/SHR completion and success statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


TASKS = (
    "google_robot_close_drawer",
    "google_robot_open_drawer",
    "google_robot_move_near",
    "google_robot_pick_coke_can",
    "widowx_carrot_on_plate",
)
ARMS = ("pi0_vanilla", "pi0_shr_harmonic")


def load_arm(root: Path, task: str, arm: str) -> dict[int, bool]:
    values = {}
    for path in (root / "episodes" / task / arm).glob("episode_*_summary.json"):
        row = json.loads(path.read_text())
        if row.get("technical_pass"):
            values[int(row["seed"])] = bool(row["success"])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact", type=Path,
        default=Path("artifacts/pi0_shr_5task_0_299_v1"),
    )
    args = parser.parse_args()
    total = {arm: 0 for arm in ARMS}
    total_n = rescue = harm = 0
    print("task\tpaired_n\tvanilla\tshr\trescue\tharm")
    for task in TASKS:
        arms = {arm: load_arm(args.artifact, task, arm) for arm in ARMS}
        seeds = sorted(set(arms[ARMS[0]]) & set(arms[ARMS[1]]))
        wins = {arm: sum(arms[arm][seed] for seed in seeds) for arm in ARMS}
        task_rescue = sum(
            arms[ARMS[1]][seed] and not arms[ARMS[0]][seed] for seed in seeds
        )
        task_harm = sum(
            arms[ARMS[0]][seed] and not arms[ARMS[1]][seed] for seed in seeds
        )
        for arm in ARMS:
            total[arm] += wins[arm]
        total_n += len(seeds)
        rescue += task_rescue
        harm += task_harm
        rate = lambda value: 100.0 * value / len(seeds) if seeds else 0.0
        print(
            f"{task}\t{len(seeds)}\t{rate(wins[ARMS[0]]):.1f}%\t"
            f"{rate(wins[ARMS[1]]):.1f}%\t{task_rescue}\t{task_harm}"
        )
    rate = lambda value: 100.0 * value / total_n if total_n else 0.0
    print(
        f"ALL\t{total_n}\t{rate(total[ARMS[0]]):.1f}%\t"
        f"{rate(total[ARMS[1]]):.1f}%\t{rescue}\t{harm}"
    )


if __name__ == "__main__":
    main()
