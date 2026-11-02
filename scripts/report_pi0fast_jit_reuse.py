#!/usr/bin/env python3
"""Summarize the optimized pi0-FAST arm against the stored RT-full baselines."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
RUN = ROOT / "runs/pi0fast_jit_header_google4_100seed_rtfull_v1"
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
)


def paired(reference: list[bool], candidate: list[bool]) -> dict[str, int]:
    return {
        "rescue": sum(not a and b for a, b in zip(reference, candidate)),
        "harm": sum(a and not b for a, b in zip(reference, candidate)),
    }


def main() -> None:
    rows = []
    all_records = []
    for task in TASKS:
        records = []
        for seed in range(100):
            path = RUN / "comparisons" / task / f"seed{seed:03d}.json"
            if path.is_file():
                records.append(json.loads(path.read_text()))
        strict = [r for r in records if r["strict_pair"]]
        all_records.extend(records)
        row = {
            "task": task,
            "complete": len(records),
            "strict": len(strict),
            "same_seed": {k: sum(r[k] for r in records) for k in (
                "vanilla_success", "old_l6_success", "jit_l6_success")},
            "strict_seed": {k: sum(r[k] for r in strict) for k in (
                "vanilla_success", "old_l6_success", "jit_l6_success")},
            "old_vs_jit_same_seed": paired(
                [r["old_l6_success"] for r in records],
                [r["jit_l6_success"] for r in records],
            ),
            "old_vs_jit_strict": paired(
                [r["old_l6_success"] for r in strict],
                [r["jit_l6_success"] for r in strict],
            ),
            "reference_vanilla_old_rgb_equal": sum(r["vanilla_old_rgb_equal"] for r in records),
            "old_jit_rgb_equal": sum(r["old_jit_rgb_equal"] for r in records),
        }
        rows.append(row)
    strict_all = [r for r in all_records if r["strict_pair"]]
    result = {
        "completed": len(all_records), "planned": 400,
        "strict_pairs": len(strict_all),
        "by_task": rows,
        "same_seed_totals": {k: sum(r[k] for r in all_records) for k in (
            "vanilla_success", "old_l6_success", "jit_l6_success")},
        "strict_totals": {k: sum(r[k] for r in strict_all) for k in (
            "vanilla_success", "old_l6_success", "jit_l6_success")},
        "old_vs_jit_same_seed": paired(
            [r["old_l6_success"] for r in all_records],
            [r["jit_l6_success"] for r in all_records],
        ),
        "old_vs_jit_strict": paired(
            [r["old_l6_success"] for r in strict_all],
            [r["jit_l6_success"] for r in strict_all],
        ),
    }
    (RUN / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# pi0-FAST optimized L6-Matched vs historical RT-full baselines", "",
        f"Completed optimized episodes: **{len(all_records)}/400**. Strict initial RGB and state pairs: **{len(strict_all)}**.", "",
        "| Task | Completed | Strict pairs | Vanilla (same seed) | Old L6 (same seed) | Optimized L6 (same seed) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        c = r["same_seed"]
        lines.append(f"| {r['task']} | {r['complete']}/100 | {r['strict']} | "
                     f"{c['vanilla_success']}/{r['complete']} | "
                     f"{c['old_l6_success']}/{r['complete']} | "
                     f"{c['jit_l6_success']}/{r['complete']} |")
    c = result["same_seed_totals"]
    lines.append(f"| **Total** | **{len(all_records)}/400** | **{len(strict_all)}** | "
                 f"**{c['vanilla_success']}/{len(all_records)}** | "
                 f"**{c['old_l6_success']}/{len(all_records)}** | "
                 f"**{c['jit_l6_success']}/{len(all_records)}** |")
    lines += ["", "Strict paired counts and Rescue/Harm are in SUMMARY.json. "
              "Same-seed comparisons with unequal initial RGB are descriptive only; "
              "in particular historical Move Near Vanilla and old L6 matched RGB in 14/100 seeds.", ""]
    (RUN / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"completed": len(all_records), "strict_pairs": len(strict_all)}))


if __name__ == "__main__":
    main()
