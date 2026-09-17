"""Combine the first-three and remaining-six Prompt-Attn layer evaluations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIRST3 = (
    "google_robot_open_drawer",
    "google_robot_pick_coke_can",
    "google_robot_move_near",
)
REMAINING6 = (
    "google_robot_close_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
TASKS = FIRST3 + REMAINING6
ARMS = ("vanilla", "shr", "prompt_v1", "prompt_single", "prompt_sparse")


def paired(rows: list[dict], base: str, candidate: str) -> dict:
    rescue = sum(not row[base] and row[candidate] for row in rows)
    harm = sum(row[base] and not row[candidate] for row in rows)
    return {"rescue": rescue, "harm": harm, "net": rescue - harm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remaining6-artifact", type=Path, required=True)
    parser.add_argument("--first3-artifact", type=Path, required=True)
    parser.add_argument("--canonical-artifact", type=Path, required=True)
    parser.add_argument("--standard-artifact", type=Path, required=False)
    args = parser.parse_args()

    rows = []
    for task in TASKS:
        candidate_root = args.first3_artifact if task in FIRST3 else args.remaining6_artifact
        for seed in range(100):
            paths = {
                "vanilla": args.canonical_artifact / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json",
                "shr": (
                    args.standard_artifact / "episodes" / task / "standard_shr" / f"episode_{seed:03d}_summary.json"
                    if task in FIRST3
                    else args.canonical_artifact / "episodes" / task / "shr_harmonic" / f"episode_{seed:03d}_summary.json"
                ),
                "prompt_single": candidate_root / "episodes" / task / "prompt_single" / f"episode_{seed:03d}_summary.json",
                "prompt_sparse": candidate_root / "episodes" / task / "prompt_sparse" / f"episode_{seed:03d}_summary.json",
            }
            paths["prompt_v1"] = (
                args.standard_artifact / "episodes" / task / "prompt_attn_shr" / f"episode_{seed:03d}_summary.json"
                if task in FIRST3
                else candidate_root / "episodes" / task / "prompt_v1" / f"episode_{seed:03d}_summary.json"
            )
            if not all(path.exists() for path in paths.values()):
                raise FileNotFoundError([str(path) for path in paths.values() if not path.exists()])
            data = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
            hashes = {
                (item["canonical_snapshot_sha256"], item["initial_state_sha256"], item["initial_rgb_sha256"])
                for item in data.values()
            }
            if len(hashes) != 1:
                raise RuntimeError(f"snapshot pairing failed for {task} seed={seed}")
            if not all(data[arm].get("technical_pass") is True for arm in ("prompt_v1", "prompt_single", "prompt_sparse")):
                raise RuntimeError(f"technical audit failed for {task} seed={seed}")
            rows.append({"task": task, "seed": seed, **{arm: bool(data[arm]["success"]) for arm in ARMS}})

    summaries = {}
    pairings = {}
    for task in (*TASKS, "overall"):
        subset = rows if task == "overall" else [row for row in rows if row["task"] == task]
        summaries[task] = {
            arm: {
                "successes": sum(row[arm] for row in subset),
                "episodes": len(subset),
                "success_rate": sum(row[arm] for row in subset) / len(subset),
            }
            for arm in ARMS
        }
        pairings[task] = {
            f"{candidate}_vs_shr": paired(subset, "shr", candidate)
            for candidate in ("prompt_single", "prompt_sparse")
        }

    result = {
        "protocol": "PROMPT_ATTN_LAYER_SELECTION_ALL9_V1",
        "seeds_per_task": 100,
        "newly_run_for_layer_selection": 2400,
        "layers": {"prompt_single": [11], "prompt_sparse": [11, 14]},
        "summaries": summaries,
        "paired": pairings,
    }
    output = args.remaining6_artifact
    (output / "ALL9_FINAL_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (output / "all9_paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "seed", *ARMS])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Prompt-Attn sparse-layer selection: SIMPLER 9×100",
        "",
        "| Task | Vanilla | SHR | Prompt-v1 | L11 | L11+14 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in (*TASKS, "overall"):
        values = summaries[task]
        lines.append(
            f"| {task} | "
            + " | ".join(
                f"{values[arm]['successes']}/{values[arm]['episodes']} ({100 * values[arm]['success_rate']:.1f}%)"
                for arm in ARMS
            )
            + " |"
        )
    (output / "ALL9_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": True, "overall": summaries["overall"]}))


if __name__ == "__main__":
    main()
