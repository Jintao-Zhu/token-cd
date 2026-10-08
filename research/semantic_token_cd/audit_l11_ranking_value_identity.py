#!/usr/bin/env python3
"""Verify all seed-0..99 Vanilla baselines against this study's renderer."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
CANON = ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2"
RT0 = ROOT / "artifacts/simpler_rt_depth0_7task_4200_v1"
DEPTH = {
    "google_robot_open_drawer": 8, "google_robot_close_drawer": 8,
    "google_robot_pick_coke_can": 0, "google_robot_move_near": 0,
    "widowx_put_eggplant_in_basket": 0, "widowx_spoon_on_towel": 0,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--worker", required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for path in (ROOT / "task1/shim_site", ROOT, PCD):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from research.semantic_token_cd.run_l11_ranking_value_study import make_environment
    from research.semantic_token_cd.distractor_rollout import restore_snapshot, snapshot_sha

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for task in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        env = make_environment(task)
        try:
            baseline_root = CANON if DEPTH[task] == 8 else RT0
            for seed in range(100):
                snapshot = pickle.loads((CANON / "snapshots" / task / f"seed_{seed:03d}.pkl").read_bytes())
                baseline = json.loads((baseline_root / "episodes" / task / "vanilla" / f"episode_{seed:03d}_summary.json").read_text())
                obs, state_hash, rgb_hash = restore_snapshot(env, seed, snapshot)
                instruction = env.unwrapped.get_language_instruction()
                row = {
                    "task": task, "seed": seed, "gpu": args.gpu,
                    "renderer": f"RT depth{DEPTH[task]}, spp=32, denoiser=true",
                    "canonical_snapshot_sha256": snapshot_sha(snapshot),
                    "initial_state_sha256": state_hash,
                    "initial_rgb_sha256": rgb_hash,
                    "instruction": instruction,
                    "checks": {
                        "snapshot": snapshot_sha(snapshot) == baseline.get("canonical_snapshot_sha256"),
                        "state": state_hash == baseline.get("initial_state_sha256"),
                        "rgb": rgb_hash == baseline.get("initial_rgb_sha256"),
                        "instruction": baseline.get("instruction") is None or instruction == baseline.get("instruction"),
                    },
                    "baseline_instruction_available": baseline.get("instruction") is not None,
                }
                row["pass"] = all(row["checks"].values())
                results.append(row)
                if not row["pass"]:
                    raise RuntimeError(f"baseline identity mismatch: {task}:{seed} {row['checks']}")
                if seed in (0, 49, 99):
                    print(json.dumps({"worker": args.worker, "gpu": args.gpu, "task": task,
                                      "seed": seed, "pass": row["pass"], "rgb": rgb_hash}, sort_keys=True), flush=True)
        finally:
            env.close()
    out = output / f"identity_audit_{args.worker}.json"
    out.write_text(json.dumps({"gpu": args.gpu, "n": len(results), "n_pass": sum(r["pass"] for r in results),
                               "results": results}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"worker": args.worker, "n": len(results), "n_pass": sum(r["pass"] for r in results),
                      "output": str(out)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
