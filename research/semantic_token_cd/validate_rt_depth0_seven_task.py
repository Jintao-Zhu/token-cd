#!/usr/bin/env python3
"""Validate and summarize the paired 7-task RT-depth0 experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel",
    "widowx_stack_cube",
)
ARMS = ("vanilla", "l11_matched")
PROTOCOL = "SIMPLER_RT_DEPTH0_7TASK_L11_MATCHED_PAIRED_0_299_V1"
RENDERER = "SAPIEN RT depth0, spp=32, denoiser=true"


def read_episode(root: Path, task: str, arm: str, seed: int) -> dict:
    path = root / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    row = json.loads(path.read_text())
    if row.get("protocol_id") != PROTOCOL or row.get("renderer") != RENDERER:
        raise RuntimeError(f"protocol/renderer mismatch: {path}")
    if row.get("task") != task or row.get("arm") != arm or row.get("seed") != seed:
        raise RuntimeError(f"identity mismatch: {path}")
    arrays = path.parent / row["arrays_file"]
    if not arrays.is_file():
        raise FileNotFoundError(arrays)
    if arm == "l11_matched":
        trace = row.get("selector_trace", [])
        if not trace or any(step.get("coverage_exact") is not True for step in trace):
            raise RuntimeError(f"Matched coverage audit failed: {path}")
        if any(abs(float(step.get("lambda", -1)) - 0.5) > 1e-12 for step in trace):
            raise RuntimeError(f"lambda audit failed: {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()

    if args.preflight_only:
        for task in TASKS:
            a, b = (read_episode(root, task, arm, 0) for arm in ARMS)
            if not (a["canonical_snapshot_sha256"] == b["canonical_snapshot_sha256"] and
                    a["initial_state_sha256"] == b["initial_state_sha256"] and
                    a["initial_rgb_sha256"] == b["initial_rgb_sha256"]):
                raise RuntimeError(f"Paired seed-0 identity mismatch: {task}")
        print("PRE-FLIGHT PASS: all seven tasks have paired seed-0 RT-depth0 outputs; snapshot/state/RGB and L11 audit agree.")
        return

    if not args.finalize:
        parser.error("choose --preflight-only or --finalize")

    task_rows = {}
    total_episodes = 0
    for task in TASKS:
        rows = {arm: [] for arm in ARMS}
        for seed in range(300):
            pair = {arm: read_episode(root, task, arm, seed) for arm in ARMS}
            vanilla, matched = pair["vanilla"], pair["l11_matched"]
            for field in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
                if vanilla.get(field) != matched.get(field):
                    raise RuntimeError(f"Paired {field} mismatch: {task}:{seed}")
            for arm in ARMS:
                rows[arm].append(pair[arm])
                total_episodes += 1
        v = sum(bool(r["success"]) for r in rows["vanilla"])
        m = sum(bool(r["success"]) for r in rows["l11_matched"])
        rescue = sum(not bool(a["success"]) and bool(b["success"])
                     for a, b in zip(rows["vanilla"], rows["l11_matched"]))
        harm = sum(bool(a["success"]) and not bool(b["success"])
                   for a, b in zip(rows["vanilla"], rows["l11_matched"]))
        task_rows[task] = {"vanilla_successes": v, "l11_matched_successes": m,
                           "rescue": rescue, "harm": harm,
                           "paired_net": rescue - harm,
                           "vanilla_rate": v / 300, "l11_matched_rate": m / 300}

    if total_episodes != 4200:
        raise RuntimeError(f"Expected 4200 episodes, found {total_episodes}")
    total_v = sum(v["vanilla_successes"] for v in task_rows.values())
    total_m = sum(v["l11_matched_successes"] for v in task_rows.values())
    total_rescue = sum(v["rescue"] for v in task_rows.values())
    total_harm = sum(v["harm"] for v in task_rows.values())
    payload = {"protocol_id": PROTOCOL, "renderer": RENDERER,
               "tasks": list(TASKS), "seeds": list(range(300)),
               "episodes": total_episodes, "per_task": task_rows,
               "overall": {"vanilla_successes": total_v, "l11_matched_successes": total_m,
                           "rescue": total_rescue, "harm": total_harm,
                           "paired_net": total_rescue - total_harm,
                           "vanilla_rate": total_v / total_episodes * 2,
                           "l11_matched_rate": total_m / total_episodes * 2}}
    (root / "FINAL_RESULTS.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# SIMPLER RT-depth0: 7-task Vanilla vs L11-Matched", "",
             f"Protocol: `{PROTOCOL}`  ", f"Renderer: `{RENDERER}`  ",
             "Seeds: 0–299 per task; 7 tasks; 4,200 paired-arm episodes.", "",
             "| Task | Vanilla | L11-Matched | Rescue | Harm | Net |",
             "|---|---:|---:|---:|---:|---:|"]
    for task, values in task_rows.items():
        lines.append(f"| {task} | {values['vanilla_successes']}/300 ({values['vanilla_rate']:.1%}) | "
                     f"{values['l11_matched_successes']}/300 ({values['l11_matched_rate']:.1%}) | "
                     f"{values['rescue']} | {values['harm']} | {values['paired_net']:+d} |")
    lines += ["", f"Total: Vanilla {total_v}/2100 ({total_v/2100:.1%}); "
              f"L11-Matched {total_m}/2100 ({total_m/2100:.1%}); "
              f"Rescue {total_rescue}; Harm {total_harm}; paired net {total_rescue-total_harm:+d}.",
              "", "Pairwise snapshot, simulator-state, and initial-RGB hashes match for every task/seed/arm pair. "
              "Every L11-Matched episode has exact matched coverage and lambda 0.5.", ""]
    (root / "FINAL_RESULTS.md").write_text("\n".join(lines))
    print(f"FINALIZE PASS: validated {total_episodes} episodes; report at {root / 'FINAL_RESULTS.md'}")


if __name__ == "__main__":
    main()
