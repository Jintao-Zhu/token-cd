"""Clean latency micro-benchmark: vanilla vs r0 vs prune25 vs prune50.

No diagnostic reference forward, single loaded model, identical fixed
observation, torch.cuda.synchronize() around every timed step.
r0 = full pruning machinery but keep-all, so it isolates bookkeeping overhead
from the actual saving of deleting tokens.
"""
from __future__ import annotations

import argparse, gc, json, statistics, sys, time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
PCD_SOURCE = Path("/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source")
CANON = REPO_ROOT / "artifacts/vanilla_recon_shr_canonical_0_299_v2/snapshots"
for p in (str(REPO_ROOT), str(PCD_SOURCE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="google_robot_pick_coke_can")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--arms", default="vanilla,vla_pruner_r0,vla_pruner_prune25,vla_pruner_prune50")
    args = ap.parse_args()

    from research.semantic_token_cd.vla_pruner_formal_rollout import (
        make_environment, load_base_policy, build_policies, restore_and_sha,
    )
    from parallel_inference import get_image_from_maniskill2_obs_dict

    env, _env_id = make_environment(args.task, args.gpu)
    import pickle
    snapshot_path = CANON / args.task / f"seed_{args.seed:03d}.pkl"
    with snapshot_path.open("rb") as fh:
        snapshot = pickle.load(fh)
    obs, _canon, _ssha, _rsha = restore_and_sha(env, args.seed, snapshot)
    image = get_image_from_maniskill2_obs_dict(env, obs)
    inst = env.unwrapped.get_language_instruction()
    print(f"[bench] task={args.task} seed={args.seed} instr={inst!r} img={image.shape}", flush=True)

    base = load_base_policy(args.task)
    arms = args.arms.split(",")
    policies = build_policies(base, args.task, arms)

    out = {}
    for arm in arms:
        pol = policies[arm]
        pol.reset(inst)
        for _ in range(args.warmup):
            pol.step(image, None, inst)
        torch.cuda.synchronize()
        times, keep = [], []
        for _ in range(args.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pol.step(image, None, inst)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
            tr = pol._episode_trace[-1]
            keep.append(tr.get("kept_image_count") or tr.get("num_keep"))
        out[arm] = {
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "p90_ms": sorted(times)[int(0.9 * len(times)) - 1],
            "min_ms": min(times),
            "kept": keep,
        }
        print(f"[bench] {arm:24s} mean={out[arm]['mean_ms']:7.1f}ms "
              f"median={out[arm]['median_ms']:7.1f}ms p90={out[arm]['p90_ms']:7.1f}ms", flush=True)

    v = out["vanilla"]["mean_ms"]
    for arm, r in out.items():
        print(f"[speedup vs vanilla] {arm:24s} {v / r['mean_ms']:.3f}x")
    print("[kept]", {a: sorted(set(v2["kept"]))[:4] for a, v2 in out.items()})
    Path("/home/leju-suzhou/zjt_ws/token-cd/artifacts/vla_pruner_openvla_reproduction").mkdir(parents=True, exist_ok=True)
    with open("/home/leju-suzhou/zjt_ws/token-cd/artifacts/vla_pruner_openvla_reproduction/latency_microbench.json", "a") as f:
        f.write(json.dumps({"task": args.task, "seed": args.seed, "results": out}, indent=1) + "\n")


if __name__ == "__main__":
    main()
