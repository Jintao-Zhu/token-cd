"""Generic paired CD analysis: SR / Rescue / Harm / Net vs vanilla.

Reads episodes/<task>/<arm>/episode_XXX_summary.json for a shared set of seeds
(present in ALL arms = the intersection), then reports per-task and pooled:

    SR_t(a)  = mean success
    R_t(a)   = #{seed : vanilla fails AND a succeeds}
    H_t(a)   = #{seed : vanilla succeeds AND a fails}
    Net_t(a) = R_t(a) - H_t(a)

Usage:
  <venv>/bin/python research/semantic_token_cd/analyze_paired_cd.py \
    --artifact artifacts/attn_semantic_oracle_mask_v1 \
    --arms kmeans_cd,oracle_cd,oracle_budget_cd
  <venv>/bin/python research/semantic_token_cd/analyze_paired_cd.py \
    --artifact artifacts/attn_local_replace_cd_v1 \
    --arms oracle_attn_cd,local_replace_cd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = (
    "google_robot_move_near",
    "google_robot_close_drawer",
    "google_robot_pick_coke_can",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--arms", type=str, required=True,
                        help="comma-separated CD arm names (vanilla is implicit)")
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    cd_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    all_arms = ["vanilla", *cd_arms]

    def load(task: str, arm: str, seed: int) -> dict | None:
        p = artifact / "episodes" / task / arm / f"episode_{seed:03d}_summary.json"
        if not p.exists():
            return None
        return json.loads(p.read_text())

    out: dict = {"artifact": str(artifact), "cd_arms": cd_arms, "tasks": {}}
    for task in TASKS:
        # discover seed union from vanilla
        vdir = artifact / "episodes" / task / "vanilla"
        if not vdir.exists():
            continue
        seeds = sorted(
            int(f.name.split("_")[1]) for f in vdir.glob("episode_*_summary.json")
        )
        paired = [s for s in seeds if all(load(task, a, s) for a in all_arms)]
        if not paired:
            continue
        rows = []
        for seed in paired:
            succ = {a: bool(load(task, a, seed)["success"]) for a in all_arms}
            rec = {"seed": seed, **{f"{a}_success": succ[a] for a in all_arms}}
            for a in cd_arms:
                if not succ["vanilla"] and succ[a]:
                    rec[f"{a}_rescue"] = 1
                    rec[f"{a}_harm"] = 0
                elif succ["vanilla"] and not succ[a]:
                    rec[f"{a}_rescue"] = 0
                    rec[f"{a}_harm"] = 1
                else:
                    rec[f"{a}_rescue"] = 0
                    rec[f"{a}_harm"] = 0
            rows.append(rec)

        n = len(paired)
        t = {
            "n_paired": n,
            "n_total_vanilla": len(seeds),
            "sr": {a: float(sum(r[f"{a}_success"] for r in rows) / n) for a in all_arms},
        }
        for a in cd_arms:
            R = sum(r[f"{a}_rescue"] for r in rows)
            H = sum(r[f"{a}_harm"] for r in rows)
            t[a] = {"rescue": R, "harm": H, "net": R - H}
        out["tasks"][task] = t

    # pooled summary
    for a in cd_arms:
        total_R = sum(t[a]["rescue"] for t in out["tasks"].values())
        total_H = sum(t[a]["harm"] for t in out["tasks"].values())
        n_units = sum(t["n_paired"] for t in out["tasks"].values())
        sr_vals = [t["sr"][a] for t in out["tasks"].values()]
        out[f"pooled_{a}"] = {
            "rescue": total_R, "harm": total_H, "net": total_R - total_H,
            "mean_sr": float(sum(sr_vals) / len(sr_vals)) if sr_vals else None,
        }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
