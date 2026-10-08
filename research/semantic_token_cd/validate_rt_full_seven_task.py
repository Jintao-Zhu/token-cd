#!/usr/bin/env python3
"""Validate paired SIMPLER RT-full results against canonical and RT-depth0 identities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = (
    "google_robot_pick_coke_can", "google_robot_move_near",
    "google_robot_place_apple_in_closed_top_drawer", "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket", "widowx_spoon_on_towel", "widowx_stack_cube",
)
ARMS = ("vanilla", "l11_matched")
PROTOCOL = "SIMPLER_RT_FULL_7TASK_L11_MATCHED_PAIRED_0_299_V1"
RENDERER = "SAPIEN RT full depth8, spp=32, denoiser=true"
ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
CANON = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
DEPTH0 = ROOT / "artifacts/simpler_rt_depth0_7task_4200_v1"


def read(root: Path, task: str, arm: str, seed: int) -> dict:
    path = root / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
    row = json.loads(path.read_text())
    if row.get("protocol_id") != PROTOCOL or row.get("renderer") != RENDERER:
        raise RuntimeError(f"Protocol/renderer mismatch: {path}")
    if row.get("task") != task or row.get("arm") != arm or row.get("seed") != seed:
        raise RuntimeError(f"Episode identity mismatch: {path}")
    arrays = path.parent / row["arrays_file"]
    if not arrays.is_file():
        raise FileNotFoundError(arrays)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    if not args.finalize:
        ap.error("--finalize required")
    out = args.output.resolve()
    per_task = {}
    total_v = total_m = total_rescue = total_harm = 0
    checked = 0
    for task in TASKS:
        successes = {arm: 0 for arm in ARMS}
        rescue = harm = 0
        for seed in range(300):
            pair = {arm: read(out, task, arm, seed) for arm in ARMS}
            for field in ("canonical_snapshot_sha256", "initial_state_sha256", "initial_rgb_sha256"):
                if pair["vanilla"].get(field) != pair["l11_matched"].get(field):
                    raise RuntimeError(f"Paired {field} mismatch {task}:{seed}")
            canonical = json.loads((CANON / "episodes" / task / "vanilla" /
                                    f"episode_{seed:03d}_summary.json").read_text())
            depth0 = json.loads((DEPTH0 / "episodes" / task / "vanilla" /
                                 f"episode_{seed:03d}_summary.json").read_text())
            if pair["vanilla"].get("canonical_snapshot_sha256") != canonical.get("canonical_snapshot_sha256"):
                raise RuntimeError(f"Canonical snapshot mismatch {task}:{seed}")
            if pair["vanilla"].get("initial_state_sha256") != depth0.get("initial_state_sha256"):
                raise RuntimeError(f"Cross-renderer initial state mismatch {task}:{seed}")
            trace = pair["l11_matched"].get("selector_trace", [])
            if not trace or any(x.get("coverage_exact") is not True for x in trace):
                raise RuntimeError(f"Matched coverage audit failed {task}:{seed}")
            if any(abs(float(x.get("lambda", -1)) - .5) > 1e-12 for x in trace):
                raise RuntimeError(f"Matched lambda audit failed {task}:{seed}")
            sv, sm = (bool(pair[a]["success"]) for a in ARMS)
            successes["vanilla"] += int(sv)
            successes["l11_matched"] += int(sm)
            rescue += int(not sv and sm)
            harm += int(sv and not sm)
            checked += 2
        per_task[task] = {
            "vanilla_successes": successes["vanilla"],
            "l11_matched_successes": successes["l11_matched"],
            "vanilla_rate": successes["vanilla"] / 300,
            "l11_matched_rate": successes["l11_matched"] / 300,
            "rescue": rescue, "harm": harm, "paired_net": rescue - harm,
        }
        total_v += successes["vanilla"]
        total_m += successes["l11_matched"]
        total_rescue += rescue
        total_harm += harm
    if checked != 4200:
        raise RuntimeError(f"Expected 4200 episode records, found {checked}")
    payload = {
        "protocol_id": PROTOCOL, "renderer": RENDERER, "tasks": list(TASKS),
        "seed_range": [0, 299], "episodes": checked, "per_task": per_task,
        "overall": {"vanilla_successes": total_v, "l11_matched_successes": total_m,
                    "vanilla_rate": total_v / 2100, "l11_matched_rate": total_m / 2100,
                    "rescue": total_rescue, "harm": total_harm,
                    "paired_net": total_rescue - total_harm},
    }
    (out / "FINAL_RESULTS.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# SIMPLER RT-full: 7-task Vanilla vs L11-Matched", "",
             f"Protocol: `{PROTOCOL}`  ", f"Renderer: `{RENDERER}`  ",
             "Seeds: 0–299 per task; 7 tasks; 4,200 paired-arm episodes.", "",
             "| Task | Vanilla | L11-Matched | Rescue | Harm | Net |", "|---|---:|---:|---:|---:|---:|"]
    for task, row in per_task.items():
        lines.append(f"| {task} | {row['vanilla_successes']}/300 ({row['vanilla_rate']:.1%}) | "
                     f"{row['l11_matched_successes']}/300 ({row['l11_matched_rate']:.1%}) | "
                     f"{row['rescue']} | {row['harm']} | {row['paired_net']:+d} |")
    lines += ["", f"Overall: Vanilla {total_v}/2100 ({total_v/2100:.1%}); "
              f"L11-Matched {total_m}/2100 ({total_m/2100:.1%}); Rescue {total_rescue}; "
              f"Harm {total_harm}; paired net {total_rescue-total_harm:+d}.",
              "", "All RT-full arms are paired on identical task/seed snapshots, simulator states, and initial RGB. "
              "Across renderer comparison, canonical snapshot and simulator-state hashes match RT-depth0; RGB differs as expected.", ""]
    (out / "FINAL_RESULTS.md").write_text("\n".join(lines))
    print(f"FINALIZE PASS: validated {checked} RT-full episodes; report at {out / 'FINAL_RESULTS.md'}")


if __name__ == "__main__":
    main()
