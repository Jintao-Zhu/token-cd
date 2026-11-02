"""Correlate per-task guidance lever with per-task success delta.

Inputs
  --lever     artifacts/<diag>/lever.jsonl        (pi05_lever_diagnostic.py)
  --episodes  one or more paired episodes.jsonl  (vanilla vs matched arms)

Output: per-task table (lever stats vs success delta) plus Pearson/Spearman
correlations, so we can test whether the intervention has physical grip where
it also changes outcomes.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


def load_lever(path: Path) -> dict[int, dict[str, float]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    grouped: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[(int(row["task_id"]), int(row["layer"]))].append(row)
    result: dict[int, dict[str, float]] = {}
    for (task_id, layer), items in grouped.items():
        key = f"L{layer}"
        result.setdefault(task_id, {})[key] = {
            "rel_lever": sum(i["rel_lever"] for i in items) / len(items),
            "lever": sum(i["lever"] for i in items) / len(items),
            "selected_count": sum(i["selected_count"] for i in items) / len(items),
            "fragments": sum(i["fragments"] for i in items) / len(items),
            "largest_fraction": sum(i["largest_fraction"] for i in items if not math.isnan(i["largest_fraction"]))
            / max(1, sum(1 for i in items if not math.isnan(i["largest_fraction"]))),
            "mask_area": sum(i["mask_area"] for i in items) / len(items),
        }
    return result


def load_episodes(paths: list[Path]) -> dict[str, dict[int, dict[str, float]]]:
    """task_id -> {"vanilla": p, "matched": p, "delta": pp} per episodes file."""
    out: dict[str, dict[int, dict[str, float]]] = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        per: dict[int, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
        names: dict[int, str] = {}
        for row in rows:
            task_id = int(row["task_id"])
            arm = str(row["arm"])
            per[task_id][arm][1] += 1
            per[task_id][arm][0] += 1 if row["success"] else 0
            names[task_id] = row.get("task_name") or str(task_id)
        label = path.parent.name
        table = {}
        for task_id, arms in per.items():
            vanilla = arms.get("vanilla", [0, 0])
            matched = arms.get("matched", [0, 0])
            if not vanilla[1] or not matched[1]:
                continue
            pv = vanilla[0] / vanilla[1]
            pm = matched[0] / matched[1]
            table[task_id] = {
                "name": names.get(task_id, str(task_id)),
                "vanilla": pv,
                "matched": pm,
                "delta_pp": 100.0 * (pm - pv),
                "pairs": min(vanilla[1], matched[1]),
            }
        out[label] = table
    return out


def pearson(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return float("nan"), float("nan")
    r = sxy / math.sqrt(sxx * syy)
    # two-sided t approximation via Fisher z (n tiny; reported as a hint only)
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    return r, t


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            out[order[k]] = average
        i = j + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lever", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    lever = load_lever(args.lever)
    runs = load_episodes(args.episodes)
    layers = sorted({key for table in lever.values() for key in table})

    for label, table in runs.items():
        print(f"\n=== {label}")
        header = f"{'task':44s} {'pairs':>5s} {'van':>6s} {'vtcg':>6s} {'delta':>7s}"
        for layer in layers:
            header += f" {layer + ' rel_lever':>16s}"
        print(header)
        for task_id in sorted(table):
            row = table[task_id]
            line = (
                f"{row['name'][:44]:44s} {row['pairs']:5d} {row['vanilla'] * 100:5.1f}% "
                f"{row['matched'] * 100:5.1f}% {row['delta_pp']:+6.1f}pp"
            )
            for layer in layers:
                value = lever.get(task_id, {}).get(layer, {}).get("rel_lever")
                line += f" {100 * value:15.2f}%" if value is not None else f" {'-':>15s}"
            print(line)

        for metric in ("rel_lever", "lever", "selected_count", "mask_area", "fragments", "largest_fraction"):
            for layer in layers:
                xs, ys = [], []
                for task_id in sorted(table):
                    stats = lever.get(task_id, {}).get(layer)
                    if not stats:
                        continue
                    value = stats.get(metric)
                    if value is None or (isinstance(value, float) and math.isnan(value)):
                        continue
                    xs.append(float(value))
                    ys.append(table[task_id]["delta_pp"])
                if len(xs) < 3:
                    continue
                r, t = pearson(xs, ys)
                rs, _ = pearson(ranks(xs), ranks(ys))
                print(f"  corr delta_pp ~ {layer}.{metric:16s} n={len(xs):2d} pearson={r:+.3f} (t={t:+.2f}) spearman={rs:+.3f}")


if __name__ == "__main__":
    main()
