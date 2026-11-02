#!/usr/bin/env python3
"""Nine-task same-seed comparison of optimized pi0-FAST with stored baselines."""
from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
GOOGLE_REF = ROOT / "runs/pi0fast_l6_matched_9task_100seed"
WIDOWX_REF = ROOT / "runs/pi0fast_bridge_widowx_4task_100seed"
GOOGLE_OPT = ROOT / "runs/pi0fast_jit_header_google4_100seed_rtfull_v1"
EXTRA_OPT = ROOT / "runs/pi0fast_jit_header_extra5_100seed_rtfull_v1"
OUT = ROOT / "runs/pi0fast_jit_header_9task_100seed_rtfull_v1"
TASKS = (
    "google_robot_open_drawer", "google_robot_close_drawer",
    "google_robot_pick_coke_can", "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel", "widowx_stack_cube",
)


def exact_mcnemar(rescue: int, harm: int) -> float:
    n = rescue + harm
    if n == 0:
        return 1.0
    k = min(rescue, harm)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    task_rows = []
    all_rows = []
    for task in TASKS:
        widowx = task.startswith("widowx_")
        ref_dir = WIDOWX_REF if widowx else GOOGLE_REF
        opt_dir = EXTRA_OPT if widowx or "apple" in task else GOOGLE_OPT
        opt_arm = "l6_matched_050" if opt_dir == EXTRA_OPT else "l6_jit_header"
        checkpoint = "/home/leju-suzhou/zjt_ws/checkpoints/openpi_fast_finetuned/" + (
            "bridge" if widowx else "fractal"
        )
        rows = []
        for seed in range(100):
            opt_path = opt_dir / "episodes" / task / f"{opt_arm}_seed{seed:03d}.json"
            if not opt_path.is_file():
                continue
            vanilla = json.loads((ref_dir / "episodes" / task / f"vanilla_seed{seed:03d}.json").read_text())
            old = json.loads((ref_dir / "episodes" / task / f"l6_matched_050_seed{seed:03d}.json").read_text())
            new = json.loads(opt_path.read_text())
            if new.get("status") != "complete":
                continue
            if new.get("checkpoint") != checkpoint or new.get("renderer") != {
                "shader_dir": "rt", "rt_max_path_depth": 8,
                "rt_samples_per_pixel": 32, "rt_use_denoiser": True,
            }:
                raise RuntimeError(f"optimized protocol mismatch {task}/{seed}")
            traces = [chunk["trace"] for chunk in new["chunks"]]
            if not all(t.get("jit_header") is True or t.get("guided_fallback") is True for t in traces):
                raise RuntimeError(f"optimized JIT header absent {task}/{seed}")
            hashes = [x["initial_rgb_sha256"] for x in (vanilla, old, new)]
            states = [x["initial_agent_state_sha256"] for x in (vanilla, old, new)]
            row = {
                "task": task, "seed": seed,
                "vanilla": bool(vanilla["success"]),
                "old": bool(old["success"]),
                "optimized": bool(new["success"]),
                "strict_three_way": len(set(hashes)) == 1 and len(set(states)) == 1,
                "strict_old_optimized": hashes[1] == hashes[2] and states[1] == states[2],
                "historical_vanilla_checkpoint_label_exception": (
                    widowx and seed < 4 and vanilla.get("checkpoint", "").endswith("/fractal")
                ),
                "new_guided_fallbacks": new.get("guided_fallbacks", 0),
            }
            rows.append(row)
        all_rows.extend(rows)
        strict = [r for r in rows if r["strict_three_way"]]
        old_strict = [r for r in rows if r["strict_old_optimized"]]
        task_rows.append({
            "task": task, "n": len(rows), "strict_three_way_n": len(strict),
            "strict_old_optimized_n": len(old_strict),
            "same_seed_success": {k: sum(r[k] for r in rows) for k in ("vanilla", "old", "optimized")},
            "strict_three_way_success": {k: sum(r[k] for r in strict) for k in ("vanilla", "old", "optimized")},
            "new_guided_fallbacks": sum(r["new_guided_fallbacks"] for r in rows),
        })
    def compare(records: list[dict], ref: str, candidate: str) -> dict:
        rescue = sum(not r[ref] and r[candidate] for r in records)
        harm = sum(r[ref] and not r[candidate] for r in records)
        return {"rescue": rescue, "harm": harm, "net": rescue - harm,
                "exact_mcnemar_p": exact_mcnemar(rescue, harm)}
    strict_all = [r for r in all_rows if r["strict_three_way"]]
    strict_old = [r for r in all_rows if r["strict_old_optimized"]]
    result = {
        "completed": len(all_rows), "planned": 900,
        "strict_three_way_n": len(strict_all),
        "strict_old_optimized_n": len(strict_old),
        "tasks": task_rows,
        "same_seed_success": {k: sum(r[k] for r in all_rows) for k in ("vanilla", "old", "optimized")},
        "strict_three_way_success": {k: sum(r[k] for r in strict_all) for k in ("vanilla", "old", "optimized")},
        "optimized_vs_vanilla_same_seed": compare(all_rows, "vanilla", "optimized"),
        "optimized_vs_old_same_seed": compare(all_rows, "old", "optimized"),
        "optimized_vs_vanilla_strict_three_way": compare(strict_all, "vanilla", "optimized"),
        "optimized_vs_old_strict": compare(strict_old, "old", "optimized"),
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# pi0-FAST optimized L6-Matched: nine-task RT-full comparison", "",
        f"Optimized episodes: **{len(all_rows)}/900**; strict three-arm initial RGB/state pairs: **{len(strict_all)}**.", "",
        "| Task | Completed | Strict 3-arm pairs | Vanilla | Old L6 | Optimized L6 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in task_rows:
        s = r["same_seed_success"]
        lines.append(f"| {r['task']} | {r['n']}/100 | {r['strict_three_way_n']} | "
                     f"{s['vanilla']}/{r['n']} | {s['old']}/{r['n']} | {s['optimized']}/{r['n']} |")
    s = result["same_seed_success"]
    lines.append(f"| **Total** | **{len(all_rows)}/900** | **{len(strict_all)}** | "
                 f"**{s['vanilla']}/{len(all_rows)}** | **{s['old']}/{len(all_rows)}** | "
                 f"**{s['optimized']}/{len(all_rows)}** |")
    lines += [
        "", "Same-seed rows with different initial RGB are descriptive, not strict paired effects. "
        "Historical Move Near, Apple, Eggplant and Spoon runs contain many such pairs. "
        "For WidowX Vanilla seeds 0–3, the checkpoint path field is mislabeled Fractal; "
        "the historical protocol specifies Bridge and the old L6 rows carry Bridge.",
        "", "See SUMMARY.json for strict-pair success counts, Rescue/Harm and exact McNemar values.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"completed": len(all_rows), "strict_three_way": len(strict_all)}))


if __name__ == "__main__":
    main()
