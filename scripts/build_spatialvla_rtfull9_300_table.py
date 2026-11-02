#!/usr/bin/env python3
"""Rebuild the SpatialVLA 9-task / 300-seed / **all RT-full (depth=8)** table.

Why this script exists
----------------------
The 300-seed-per-task SpatialVLA dataset is *assembled* from several artifacts, and two of
them are traps:

* ``spatialvla_drawers_rt_depth0_vanilla_l10_100_199_v1`` - drawer seeds 100-199 but rendered
  at **depth 0**, NOT RT-full.
* ``spatialvla_l10_vanilla_expand9_to300_v1`` - the two drawer tasks are depth 8, but its
  other seven tasks are depth 0, so its non-drawer episodes must be dropped.

The fully RT-full recipe (verified: reproduces the reported table exactly, mean 41.4 -> 54.1):

  seeds 0-99 & 200-299, 7 non-drawer tasks -> spatialvla_l10_vanilla_rtfull7_to300_v1
  seeds 100-199,      7 non-drawer tasks -> remaining7_rtfull_vanilla_100_199_v1 (vanilla)
                                            + l10_matched_rtfull7_n100_199      (l10)
  seeds 0-99 & 200-299, drawer tasks     -> spatialvla_l10_vanilla_expand9_to300_v1
  seeds 100-199,      drawer tasks       -> simpler_l11matched_4task_n100   (vanilla arm)
                                            + l10_vs_l11_4task_n100        (l10_matched arm)

Every episode kept here has renderer.depth == 8 (SAPIEN RT full, spp=32, denoiser on).

Usage:  python build_spatialvla_rtfull9_300_table.py [out.md]
"""
from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

ART = Path(__file__).resolve().parents[1] / "artifacts"

TASKS = [
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_move_near",
    "google_robot_pick_coke_can",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
]

SOURCES = [
    # (artifact, accepted arms, accepted seed range or None)
    ("spatialvla_l10_vanilla_rtfull7_to300_v1", None, None),
    ("spatialvla_remaining7_rtfull_vanilla_100_199_v1", {"vanilla"}, None),
    ("spatialvla_l10_matched_rtfull7_n100_199", {"l10_matched"}, None),
    ("spatialvla_l10_vanilla_expand9_to300_v1", None, None),
    ("spatialvla_simpler_l11matched_4task_n100", {"vanilla"}, (100, 199)),
    ("spatialvla_l10_vs_l11_4task_n100", {"l10_matched"}, (100, 199)),
]


def _depth(summary: dict):
    r = summary.get("renderer")
    return r.get("depth") if isinstance(r, dict) else None


def load() -> tuple[dict, int, int]:
    rec: dict[tuple[str, int], dict[str, bool]] = collections.defaultdict(dict)
    kept = dropped = 0
    for name, arms, seed_range in SOURCES:
        for path in sorted(glob.glob(str(ART / name / "episodes" / "*" / "*" / "episode_*_summary.json"))):
            try:
                d = json.load(open(path))
            except Exception:
                dropped += 1
                continue
            if "success" not in d:
                dropped += 1
                continue
            if _depth(d) != 8:            # <- the whole point of this script
                dropped += 1
                continue
            if arms and d.get("arm") not in arms:
                dropped += 1
                continue
            s = int(d["seed"])
            if seed_range and not (seed_range[0] <= s <= seed_range[1]):
                dropped += 1
                continue
            rec[(d["task"], s)][d["arm"]] = bool(d["success"])
            kept += 1
    return rec, kept, dropped


def main() -> None:
    rec, kept, dropped = load()
    rows = []
    tv = ts = 0
    for task in TASKS:
        v = s = n = rescue = harm = 0
        for (t, _seed), arms in rec.items():
            if t != task or "vanilla" not in arms or "l10_matched" not in arms:
                continue
            n += 1
            v += arms["vanilla"]
            s += arms["l10_matched"]
            if arms["l10_matched"] and not arms["vanilla"]:
                rescue += 1
            if arms["vanilla"] and not arms["l10_matched"]:
                harm += 1
        if n != 300:
            print(f"WARNING: {task} has n={n}, expected 300", file=sys.stderr)
        rows.append((task, v, s, n, rescue, harm))
        tv += v
        ts += s

    total_n = sum(r[3] for r in rows)
    lines = [
        "# SpatialVLA: 9 tasks x 300 seeds x 2 arms, all SAPIEN RT-full (depth=8)",
        "",
        "Dataset **`SPATIALVLA_RTfull9_300_SEEDS_V1`** - canonical record: "
        "`docs/SPATIALVLA_RTfull9_300_SEEDS_V1_DATASET.md`, lock: "
        "`docs/spatialvla_rtfull9_300_seed_dataset_lock.json`.",
        "",
        "Arms: `vanilla` (Base) vs `l10_matched` (Ours, lambda=0.5). "
        "Every episode below has `renderer.depth == 8`, spp=32, denoiser on.",
        "",
        f"Episodes merged: {kept} kept, {dropped} dropped (wrong renderer / wrong arm / unreadable).",
        "",
        "| Task | Base | +Ours | Δ | rescue | harm | net | counts |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for task, v, s, n, r, h in rows:
        lines.append(
            f"| {task.replace('google_robot_', '')} | {v / n * 100:.1f}% | {s / n * 100:.1f}% | "
            f"{(s - v) / n * 100:+.1f} | {r} | {h} | {r - h:+d} | {v}→{s} / {n} |"
        )
    lines.append(
        f"| **MEAN** | **{tv / total_n * 100:.1f}%** | **{ts / total_n * 100:.1f}%** | "
        f"**{(ts - tv) / total_n * 100:+.1f}** | | | | {tv}→{ts} / {total_n} |"
    )
    lines += ["", "Source artifacts (all filtered to `depth == 8`):", ""]
    lines += [f"* `{n}`" for n, _, _ in SOURCES]
    text = "\n".join(lines) + "\n"

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ART.parent / "docs" / "spatialvla_rtfull9_300_seed_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
