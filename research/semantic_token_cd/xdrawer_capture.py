"""Capture paired top/middle drawer snapshots for the sim-region cross study.

For every seed we store one snapshot per drawer env type (top / middle).  The
two envs are verified to produce the *same* physical reset state (sim state,
agent state, RNG state and rendered RGB all equal), so any emitted state from
one drawer is a valid paired state for the other instruction.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["/home/leju-suzhou/zjt_ws/token-cd",
                "/home/leju-suzhou/zjt_ws/pcd_openvla_simpler_box_31b027e/source"]

from research.semantic_token_cd.xdrawer_protocol import (  # noqa: E402
    DRAWERS, TASK_BY_DRAWER, array_sha, atomic_json, capture_snapshot,
    make_drawer_env, rgb_sha,
)
from research.semantic_token_cd.rollout_pilot import wrapped_observation  # noqa: E402


def parse_seeds(spec: str) -> list[int]:
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def capture(env, seed):
    snapshot = capture_snapshot(env, seed)
    inner = env.unwrapped
    obs = wrapped_observation(env)
    snapshot["drawer_id"] = inner.drawer_id
    snapshot["rgb_sha256"] = rgb_sha(env, obs)
    snapshot["state_sha256"] = array_sha(np.asarray(snapshot["sim_state"]))
    snapshot["agent_repr"] = repr(snapshot["agent_state"])
    snapshot["rng_repr"] = repr(snapshot["rng_state"])
    return snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="artifact root that will hold snapshots/")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    seeds = parse_seeds(args.seeds)
    root = args.root.resolve()
    snap_dir = root / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    snapshots = {}
    for drawer in DRAWERS:
        env, _ = make_drawer_env(drawer, args.gpu)
        task_root = snap_dir / TASK_BY_DRAWER[drawer]
        task_root.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            snap = capture(env, seed)
            if snap["drawer_id"] != drawer:
                raise RuntimeError(f"captured drawer {snap['drawer_id']} != expected {drawer}")
            path = task_root / f"seed_{seed:03d}.pkl"
            with path.open("wb") as handle:
                pickle.dump(snap, handle, protocol=pickle.HIGHEST_PROTOCOL)
            snapshots[(drawer, seed)] = snap
            print(json.dumps({"drawer": drawer, "seed": seed, "saved": str(path),
                              "rgb": snap["rgb_sha256"][:12], "state": snap["state_sha256"][:12]}),
                  flush=True)
        env.close()

    rows = []
    for seed in seeds:
        s_top = snapshots[("top", seed)]
        s_mid = snapshots[("middle", seed)]
        same = {
            "state_equal": s_top["state_sha256"] == s_mid["state_sha256"],
            "agent_equal": s_top["agent_repr"] == s_mid["agent_repr"],
            "rng_equal": s_top["rng_repr"] == s_mid["rng_repr"],
            "rgb_equal": s_top["rgb_sha256"] == s_mid["rgb_sha256"],
        }
        rows.append({"seed": seed, "instruction_top": s_top["instruction"],
                     "instruction_middle": s_mid["instruction"], **same})
        print(json.dumps({"seed": seed, "pairing": same}), flush=True)
        if not all(same.values()):
            raise RuntimeError(f"top/middle reset pairing failed for seed {seed}: {same}")
    manifest = {"seeds": seeds, "pairing_rows": rows,
                "all_pairs_exact": all(all(r[k] for k in ("state_equal", "agent_equal",
                                                          "rng_equal", "rgb_equal")) for r in rows)}
    atomic_json(snap_dir / "pairing_manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
