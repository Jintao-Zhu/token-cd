"""Summarise the pi0.5 LIBERO-Plus paired rollout by perturbation category/level."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


CLASSIFICATION = Path(
    "/home/leju-suzhou/zjt_ws/libero_plus_src/libero/libero/benchmark/task_classification.json"
)


def binomial_two_sided(discordant_matched: int, discordant_total: int) -> float:
    if discordant_total == 0:
        return 1.0
    tail = sum(
        math.comb(discordant_total, index)
        for index in range(0, min(discordant_matched, discordant_total - discordant_matched) + 1)
    ) / 2**discordant_total
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    args = parser.parse_args()

    meta = {
        row["name"]: row
        for row in json.loads(CLASSIFICATION.read_text())[args.suite]
    }
    rows = [
        json.loads(line)
        for line in (args.artifact / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    valid = [row for row in rows if not row.get("error")]

    pairs: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in valid:
        pairs[(row["task_name"], int(row["trial"]))][row["arm"]] = row
    complete = {key: value for key, value in pairs.items() if {"vanilla", "matched"} <= set(value)}

    def stats(keys: set[tuple[str, int]]) -> dict:
        table = {}
        for arm in ("vanilla", "matched"):
            values = [complete[key][arm]["success"] for key in keys]
            table[arm] = (sum(values), len(values))
        only_v = sum(
            1 for key in keys if complete[key]["vanilla"]["success"] and not complete[key]["matched"]["success"]
        )
        only_m = sum(
            1 for key in keys if complete[key]["matched"]["success"] and not complete[key]["vanilla"]["success"]
        )
        return {
            "pairs": len(keys),
            "vanilla": table["vanilla"],
            "matched": table["matched"],
            "vanilla_rate": table["vanilla"][0] / max(1, table["vanilla"][1]),
            "matched_rate": table["matched"][0] / max(1, table["matched"][1]),
            "delta": (
                table["matched"][0] - table["vanilla"][0]
            ) / max(1, table["vanilla"][1]),
            "vanilla_only": only_v,
            "matched_only": only_m,
            "p": binomial_two_sided(only_m, only_v + only_m),
        }

    by_category: dict[str, set] = defaultdict(set)
    by_level: dict[int, set] = defaultdict(set)
    for key in complete:
        row = meta.get(key[0], {})
        by_category[str(row.get("category", "unknown"))].add(key)
        by_level[int(row.get("difficulty_level", 0))].add(key)

    overall = stats(set(complete))
    lines = [
        f"# pi0.5 LIBERO-Plus paired rollout — {args.suite}",
        "",
        f"pairs complete: {overall['pairs']} (records {len(rows)}, errors {len(rows) - len(valid)})",
        f"vanilla {overall['vanilla'][0]}/{overall['vanilla'][1]} = {overall['vanilla_rate']:.3f} | "
        f"matched {overall['matched'][0]}/{overall['matched'][1]} = {overall['matched_rate']:.3f} | "
        f"delta {overall['delta']:+.3f} | v-only {overall['vanilla_only']} / m-only "
        f"{overall['matched_only']} | p = {overall['p']:.3f}",
        "",
        "| perturbation | pairs | vanilla | matched | delta | p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for category in sorted(by_category):
        cell = stats(by_category[category])
        lines.append(
            f"| {category} | {cell['pairs']} | {cell['vanilla_rate']:.3f} "
            f"({cell['vanilla'][0]}/{cell['vanilla'][1]}) | {cell['matched_rate']:.3f} "
            f"({cell['matched'][0]}/{cell['matched'][1]}) | {cell['delta']:+.3f} | {cell['p']:.3f} |"
        )
    lines += ["", "| difficulty | pairs | vanilla | matched | delta |", "| --- | --- | --- | --- | --- |"]
    for level in sorted(by_level):
        cell = stats(by_level[level])
        lines.append(
            f"| L{level} | {cell['pairs']} | {cell['vanilla_rate']:.3f} | "
            f"{cell['matched_rate']:.3f} | {cell['delta']:+.3f} |"
        )
    report = "\n".join(lines)
    (args.artifact / "paired_summary.md").write_text(report + "\n")
    (args.artifact / "paired_summary.json").write_text(
        json.dumps(
            {
                "suite": args.suite,
                "overall": overall,
                "by_category": {key: stats(value) for key, value in by_category.items()},
                "by_level": {str(key): stats(value) for key, value in by_level.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(report)


if __name__ == "__main__":
    main()
