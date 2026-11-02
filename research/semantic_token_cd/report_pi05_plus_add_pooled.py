"""Pooled analysis of the distractor (`_add_`) instances across LIBERO-Plus suites."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


CLASSIFICATION = Path(
    "/home/leju-suzhou/zjt_ws/libero_plus_src/libero/libero/benchmark/task_classification.json"
)


def mcnemar(matched_only: int, discordant: int) -> float:
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(0, min(matched_only, discordant - matched_only) + 1)
    ) / 2**discordant
    return min(1.0, 2.0 * tail)


def load_pairs(paths: list[Path]) -> dict[tuple[str, str], dict[str, bool]]:
    pairs: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for artifact in paths:
        episodes = artifact / "episodes.jsonl"
        if not episodes.exists():
            continue
        for line in episodes.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            pairs[(str(row["suite"]), str(row["task_name"]))][str(row["arm"])] = bool(
                row["success"]
            )
    return {key: value for key, value in pairs.items() if len(value) == 2}


def block(name: str, keys, pairs) -> dict:
    total = len(keys)
    if total == 0:
        return {"name": name, "pairs": 0}
    vanilla = sum(1 for key in keys if pairs[key]["vanilla"])
    matched = sum(1 for key in keys if pairs[key]["matched"])
    only_v = sum(1 for key in keys if pairs[key]["vanilla"] and not pairs[key]["matched"])
    only_m = sum(1 for key in keys if pairs[key]["matched"] and not pairs[key]["vanilla"])
    stats = {
        "name": name,
        "pairs": total,
        "vanilla": vanilla,
        "matched": matched,
        "vanilla_rate": vanilla / total,
        "matched_rate": matched / total,
        "delta": (matched - vanilla) / total,
        "vanilla_only": only_v,
        "matched_only": only_m,
        "p": mcnemar(only_m, only_v + only_m),
    }
    print(
        f"{name:34s} {total:4d} 对  vanilla {stats['vanilla_rate']:.3f}  "
        f"matched {stats['matched_rate']:.3f}  Δ {stats['delta']:+.3f}  "
        f"救{only_m}/伤{only_v}  p={stats['p']:.4f}",
        flush=True,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    pairs = load_pairs(args.roots)
    if not pairs:
        raise SystemExit("no paired episodes found")
    classification = json.loads(CLASSIFICATION.read_text())
    level_of = {}
    for suite_rows in classification.values():
        for row in suite_rows:
            level_of[row["name"]] = int(row.get("difficulty_level") or 0)

    keys = sorted(pairs)
    print(f"pooled: {len(keys)} pairs\n", flush=True)
    summary = {"overall": block("pooled (held-out + selection)", keys, pairs)}

    by_suite: dict[str, list] = defaultdict(list)
    by_level: dict[int, list] = defaultdict(list)
    for key in keys:
        by_suite[key[0]].append(key)
        by_level[level_of.get(key[1], 0)].append(key)
    summary["by_suite"] = {suite: block(f"  {suite}", group, pairs) for suite, group in sorted(by_suite.items())}
    print()
    summary["by_level"] = {f"L{level}": block(f"  level {level}", group, pairs) for level, group in sorted(by_level.items())}

    # held-out only (drop libero_10, which was used to pick the layer)
    held_out = [key for key in keys if key[0] != "libero_10"]
    if held_out:
        print()
        summary["held_out_only"] = block("held-out (3 other suites)", held_out, pairs)

    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
