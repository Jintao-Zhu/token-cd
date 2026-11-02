"""Summarise the paired pi0.5 LIBERO-Long rollout (vanilla vs L6 matched)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def _read(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _binomial_two_sided(successes: int, trials: int) -> float:
    """Exact McNemar test on discordant pairs."""
    if trials == 0:
        return 1.0
    tail = sum(
        math.comb(trials, index) for index in range(0, min(successes, trials - successes) + 1)
    ) / 2**trials
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = _read(args.artifact / "episodes.jsonl")
    if not rows:
        raise SystemExit("no episodes recorded")
    out_path = args.out or (args.artifact / "paired_summary.json")

    paired: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        paired[(int(row["task_id"]), int(row["init_state_id"]))][str(row["arm"])] = row

    arms = sorted({str(row["arm"]) for row in rows})
    per_arm = {}
    for arm in arms:
        arm_rows = [row for row in rows if str(row["arm"]) == arm]
        successes = sum(bool(row["success"]) for row in arm_rows)
        errors = sum(1 for row in arm_rows if row.get("error"))
        per_arm[arm] = {
            "episodes": len(arm_rows),
            "successes": successes,
            "success_rate": successes / len(arm_rows),
            "errors": errors,
            "mean_policy_steps": sum(int(row["policy_steps"]) for row in arm_rows)
            / len(arm_rows),
            "mean_runtime_seconds": sum(float(row["runtime_seconds"]) for row in arm_rows)
            / len(arm_rows),
        }

    complete = {key: value for key, value in paired.items() if len(value) == len(arms)}
    mismatch = []
    for key, value in complete.items():
        hashes = {str(arm): value[arm]["initial_state_sha256"] for arm in arms}
        if len(set(hashes.values())) != 1:
            mismatch.append({"pair": list(key), "hashes": hashes})

    contingency = {arm: {"success": 0, "failure": 0} for arm in arms}
    for value in complete.values():
        for arm in arms:
            contingency[arm]["success" if value[arm]["success"] else "failure"] += 1

    per_task = defaultdict(lambda: {arm: {"episodes": 0, "successes": 0} for arm in arms})
    for (task_id, _), value in complete.items():
        for arm in arms:
            per_task[task_id][arm]["episodes"] += 1
            per_task[task_id][arm]["successes"] += int(bool(value[arm]["success"]))

    summary = {
        "protocol": "PI05_LIBERO_LONG_PAIRED_SUMMARY_V1",
        "artifact": str(args.artifact),
        "arms": arms,
        "episodes_total": len(rows),
        "pairs_complete": len(complete),
        "pairs_incomplete": len(paired) - len(complete),
        "pairing_hash_mismatches": mismatch,
        "per_arm": per_arm,
        "contingency": contingency,
        "per_task": {
            str(task): {
                arm: {
                    **value,
                    "success_rate": value["successes"] / max(1, value["episodes"]),
                }
                for arm, value in table.items()
            }
            for task, table in sorted(per_task.items())
        },
    }

    if {"vanilla", "matched"} <= set(arms):
        only_vanilla = sum(
            1
            for value in complete.values()
            if value["vanilla"]["success"] and not value["matched"]["success"]
        )
        only_matched = sum(
            1
            for value in complete.values()
            if value["matched"]["success"] and not value["vanilla"]["success"]
        )
        discordant = only_vanilla + only_matched
        summary["paired_test"] = {
            "vanilla_only_success": only_vanilla,
            "matched_only_success": only_matched,
            "discordant": discordant,
            "mcnemar_exact_p": _binomial_two_sided(only_matched, discordant),
            "delta_success_rate": per_arm["matched"]["success_rate"]
            - per_arm["vanilla"]["success_rate"],
        }
        selected = [
            float(row["mean_selected_token_count"])
            for row in rows
            if str(row["arm"]) == "matched" and row.get("mean_selected_token_count")
        ]
        if selected:
            summary["matched_budget"] = {
                "mean_selected_tokens": sum(selected) / len(selected),
                "min": min(selected),
                "max": max(selected),
                "replans_counted": len(selected),
            }

    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        f"# pi0.5 LIBERO-Long paired rollout — {args.artifact.name}",
        "",
        f"pairs complete: {len(complete)} / {len(paired)}  (records {len(rows)})",
        "",
        "| arm | episodes | successes | success rate | mean steps | mean s/ep |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in arms:
        stats = per_arm[arm]
        lines.append(
            f"| {arm} | {stats['episodes']} | {stats['successes']} | "
            f"{stats['success_rate']:.3f} | {stats['mean_policy_steps']:.0f} | "
            f"{stats['mean_runtime_seconds']:.1f} |"
        )
    if "paired_test" in summary:
        test = summary["paired_test"]
        lines += [
            "",
            f"delta (matched - vanilla): {test['delta_success_rate']:+.3f}",
            f"discordant: {test['discordant']} "
            f"(vanilla-only {test['vanilla_only_success']}, "
            f"matched-only {test['matched_only_success']}), "
            f"exact McNemar p = {test['mcnemar_exact_p']:.4f}",
        ]
    if mismatch:
        lines += ["", f"WARNING: {len(mismatch)} pairs have different initial states"]
    lines += ["", "## per task", "", "| task | " + " | ".join(arms) + " |", "| --- |" + " --- |" * len(arms)]
    for task, table in sorted(per_task.items()):
        cells = [
            f"{table[arm]['successes']}/{table[arm]['episodes']}" for arm in arms
        ]
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    (args.artifact / "paired_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
