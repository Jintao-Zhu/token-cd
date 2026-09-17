"""Analyze the locked 3x100 four-arm sparse-layer closed-loop experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASKS = ("google_robot_open_drawer", "google_robot_pick_coke_can", "google_robot_move_near")
LABELS = {task: task.removeprefix("google_robot_") for task in TASKS}
ARMS = ("vanilla", "standard_shr", "prompt_v1", "prompt_single", "prompt_sparse")


def paired(rows, base: str, candidate: str) -> dict:
    rescue = sum(not x[base] and x[candidate] for x in rows)
    harm = sum(x[base] and not x[candidate] for x in rows)
    return {"base": base, "candidate": candidate, "rescue": rescue,
            "harm": harm, "net": rescue - harm, "ties": len(rows) - rescue - harm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--layer-artifact", type=Path, required=True)
    parser.add_argument("--closed-loop-v1", type=Path, required=True)
    parser.add_argument("--canonical-artifact", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for task in TASKS:
        for seed in range(100):
            paths = {
                "vanilla": args.canonical_artifact / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json",
                "standard_shr": args.closed_loop_v1 / "episodes" / task / "standard_shr" / f"episode_{seed:03d}_summary.json",
                "prompt_v1": args.closed_loop_v1 / "episodes" / task / "prompt_attn_shr" / f"episode_{seed:03d}_summary.json",
                "prompt_single": args.artifact / "episodes" / task / "prompt_single" / f"episode_{seed:03d}_summary.json",
                "prompt_sparse": args.artifact / "episodes" / task / "prompt_sparse" / f"episode_{seed:03d}_summary.json",
            }
            if not all(path.exists() for path in paths.values()):
                missing = [str(path) for path in paths.values() if not path.exists()]
                raise FileNotFoundError(missing)
            data = {arm: json.loads(path.read_text()) for arm, path in paths.items()}
            hashes = {(x["canonical_snapshot_sha256"], x["initial_state_sha256"], x["initial_rgb_sha256"])
                      for x in data.values()}
            technical = all(data[arm].get("technical_pass") is True for arm in ARMS if arm != "vanilla")
            if len(hashes) != 1 or not technical:
                raise RuntimeError(f"pairing/audit failed: {task} seed={seed}")
            rows.append({"task": task, "seed": seed, **{arm: bool(data[arm]["success"]) for arm in ARMS}})
    summaries = {}; pairings = {}
    for task in list(TASKS) + [None]:
        subset = rows if task is None else [x for x in rows if x["task"] == task]
        key = "overall" if task is None else task
        summaries[key] = {arm: {"successes": sum(x[arm] for x in subset),
                                "episodes": len(subset),
                                "success_rate": sum(x[arm] for x in subset) / len(subset)} for arm in ARMS}
        pairings[key] = {}
        for candidate in ("prompt_single", "prompt_sparse"):
            pairings[key][f"{candidate}_vs_vanilla"] = paired(subset, "vanilla", candidate)
            pairings[key][f"{candidate}_vs_prompt_v1"] = paired(subset, "prompt_v1", candidate)
            pairings[key][f"{candidate}_vs_standard_shr"] = paired(subset, "standard_shr", candidate)
    lock = json.loads((args.layer_artifact / "CANDIDATES_LOCK.json").read_text())
    validation = json.loads((args.layer_artifact / "VALIDATION_LAYER_RESULTS.json").read_text())
    same_state = json.loads((args.layer_artifact / "SAME_STATE_RESULTS.json").read_text())
    result = {
        "protocol_id": "PROMPT_ATTN_LAYER_SELECTION_V1",
        "episodes": {"per_task": 100, "four_arm_total": 1200, "newly_run": 600},
        "layers": {"prompt_v1": lock["prompt_v1"], "prompt_single": lock["prompt_single"],
                   "prompt_sparse": lock["prompt_sparse"]},
        "summaries": summaries, "paired": pairings,
        "offline_validation": validation, "same_state": same_state,
    }
    (args.artifact / "FINAL_RESULTS.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with (args.artifact / "paired_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "seed", *ARMS]); writer.writeheader(); writer.writerows(rows)
    with (args.artifact / "task_results.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["task", "arm", "successes", "episodes", "success_rate"])
        for task, arms in summaries.items():
            for arm, values in arms.items(): writer.writerow([task, arm, values["successes"], values["episodes"], values["success_rate"]])

    lines = ["# Prompt-Attn sparse-layer selection v1", "",
             "Candidates were frozen using 20 exploration episodes before the 10-episode offline validation was read.",
             f"Prompt-Single uses layer `{lock['prompt_single']}`; Prompt-Sparse uses layers `{lock['prompt_sparse']}`.", "",
             "## Closed-loop success", "", "| Task | Vanilla | Standard SHR | Prompt-v1 | Prompt-Single | Prompt-Sparse |", "|---|---:|---:|---:|---:|---:|"]
    for task in list(TASKS) + [None]:
        key = "overall" if task is None else task; name = "Overall" if task is None else LABELS[task]
        values = summaries[key]
        lines.append("| " + name + " | " + " | ".join(
            f"{values[a]['successes']}/{values[a]['episodes']} ({100*values[a]['success_rate']:.1f}%)" for a in ARMS) + " |")
    lines += ["", "## Paired changes", "", "| Scope | Comparison | Rescue | Harm | Net |", "|---|---|---:|---:|---:|"]
    for task in list(TASKS) + [None]:
        key = "overall" if task is None else task; name = "Overall" if task is None else LABELS[task]
        for comparison, values in pairings[key].items():
            lines.append(f"| {name} | {comparison} | {values['rescue']} | {values['harm']} | {values['net']} |")
    lines += ["", "## Interpretation boundary", "",
              "A better target-response heatmap is treated only as a mechanism diagnostic. The conclusion about layer averaging is determined by paired closed-loop success, not by residual magnitude or visual appearance alone."]
    (args.artifact / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"summaries": summaries, "overall_paired": pairings["overall"]}))


if __name__ == "__main__":
    main()
