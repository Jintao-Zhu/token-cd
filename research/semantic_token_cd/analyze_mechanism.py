"""Mechanism analysis for the close_drawer reversal.

Combines three data sources to answer the (a)-vs-(b) question the user posed:

    why does semantic_attn_k8_l8_15 (block Action Query -> selected handle tokens)
    RESCUE close_drawer, while semantic_merge_k8_eta100 (collapse handle tokens to a
    prototype) HARM it?

  (a) attn-blocking removes *misleading* handle evidence  -> gripper AIMs the handle/
      panel more accurately (better approach, still a grasp/pull);
  (b) attn-blocking forces a *push-the-panel* strategy     -> gripper stops reaching
      for the handle and presses the front face shut;
  (c) merge-collapse misaligns the fine approach          -> grasp/push off by a few
      cm, drawer left ajar.

Inputs:
  - replay_close_drawer/seed_XXX.json  : per-step tcp pose, drawer qpos, handle/front
      world pos, gripper action (dim6), full action, per arm.
  - replay_close_drawer/token_coverage.json : handle/front projected token cells vs
      selected cells (Q1).
  - flip_matrix/flip_matrix.json       : per-seed outcomes + first-divergence info.

Output: mechanism_analysis.json + printed table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
ARTIFACT = REPO_ROOT / "artifacts/attn_semantic_merge_k8_v1"
REPLAY = ARTIFACT / "replay_close_drawer"
VANILLA, ATTN, MERGE = "vanilla", "semantic_attn_k8_l8_15", "semantic_merge_k8_eta100"

ATTN_RESCUE = [200, 210, 211, 226, 236, 248, 253, 258, 260, 270, 277, 286, 287, 294]
MERGE_HARM = [205, 214, 215, 221, 223, 233, 241, 271, 281, 297]


def norm(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.linalg.norm(a - b))


def arm_summary(arm):
    """Extract discriminating physical quantities from one arm's recorded traj."""
    traj = arm["traj"]
    hp0 = np.asarray(traj[0]["handle_p"], float)   # handle at t=0 (drawer pre-motion)
    fp0 = np.asarray(traj[0]["front_p"], float)    # front panel at t=0
    d_handle = [norm(r["tcp_p"], hp0) for r in traj]
    d_front = [norm(r["tcp_p"], fp0) for r in traj]
    ih = int(np.argmin(d_handle))
    it = int(np.argmin(d_front))
    close = sum(1 for r in traj if r["gripper_action"] < -0.5)
    open_ = sum(1 for r in traj if r["gripper_action"] > 0.5)
    neutral = sum(1 for r in traj if abs(r["gripper_action"]) <= 0.5)
    return {
        "n_steps": arm["n_steps"],
        "qpos_final": arm["qpos_final"],
        "qpos_min": arm["qpos_min"],
        "contact_t": arm["contact_t"],
        "min_d_handle": min(d_handle),
        "min_d_front": min(d_front),
        "tcp_z_at_minh": float(traj[ih]["tcp_p"][2]),
        "tcp_z_at_minf": float(traj[it]["tcp_p"][2]),
        "handle_z0": float(hp0[2]),
        "front_z0": float(fp0[2]),
        "n_close": close, "n_open": open_, "n_neutral": neutral,
    }


def strategy(s):
    if s["n_close"] > s["n_open"]:
        return "grasp"
    if s["n_open"] > s["n_close"]:
        return "push_open"
    return "neutral"


def load_outcomes():
    fm = json.load(open(ARTIFACT / "flip_matrix" / "flip_matrix.json"))
    out = {}
    for r in fm["rows"]:
        if r["task"] == "google_robot_close_drawer":
            out[r["seed"]] = {
                "attn_outcome": r["attn_outcome"],
                "merge_outcome": r["merge_outcome"],
                "success": r["success"],
                "cos_r": r.get("cos_r_attn_r_merge_step0"),
                "first_div": r.get("attn_vs_vanilla", {}).get("first_div_timestep"),
            }
    return out


def main():
    outcomes = load_outcomes()
    cov = {r["seed"]: r for r in json.load(open(REPLAY / "token_coverage.json"))}

    seeds = sorted(int(f.stem.split("_")[1]) for f in REPLAY.glob("seed_*.json"))
    rows = {}
    for seed in seeds:
        d = json.load(open(REPLAY / f"seed_{seed:03d}.json"))
        arms = {a: arm_summary(d["arms"][a]) for a in (VANILLA, ATTN, MERGE)}
        for a in arms:
            arms[a]["strategy"] = strategy(arms[a])
        c = cov.get(seed, {})
        rows[seed] = {
            "warnings": d.get("warnings", []),
            "outcomes": outcomes.get(seed),
            "arms": arms,
            "handle_in_sel": c.get("handle_in_sel"),
            "front_in_sel": c.get("front_in_sel"),
            "handle_cell_dist": c.get("handle_cell_dist_to_sel"),
            "selected_entities": c.get("entities"),
        }

    (REPLAY / "mechanism_analysis.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n")

    # ---- printed tables ----
    def hdr(grp, seeds_):
        print(f"\n{'='*110}\n{grp}  (n={len(seeds_)})\n{'='*110}")
        print(f"{'seed':>4} {'warn':>4} {'out':>6} | "
              f"{'arm':>6} {'strat':>9} {'qT':>6} {'qmin':>6} {'ct':>3} "
              f"{'dHndl':>7} {'dFront':>7} {'z@h':>6} {'z@f':>6} {'hZ':>6} {'fZ':>6} "
              f"{'ncl':>3} {'nop':>3} | {'hInSel':>7} {'fInSel':>7} {'hDist':>5}")
        for seed in seeds_:
            r = rows[seed]
            w = "M" if r["warnings"] else ""
            o = r["outcomes"]
            for a in (VANILLA, ATTN, MERGE):
                s = r["arms"][a]
                c = r.get("handle_in_sel")
                print(f"{seed:>4} {w:>4} {o['attn_outcome'] if a==ATTN else (o['merge_outcome'] if a==MERGE else 'v'):>6} | "
                      f"{a[:6]:>6} {s['strategy']:>9} {s['qpos_final']:>6.3f} {s['qpos_min']:>6.3f} "
                      f"{str(s['contact_t']):>3} {s['min_d_handle']:>7.3f} {s['min_d_front']:>7.3f} "
                      f"{s['tcp_z_at_minh']:>6.3f} {s['tcp_z_at_minf']:>6.3f} {s['handle_z0']:>6.3f} {s['front_z0']:>6.3f} "
                      f"{s['n_close']:>3} {s['n_open']:>3} | {str(c):>7} {str(r.get('front_in_sel')):>7} "
                      f"{str(r.get('handle_cell_dist')):>5}")

    # split into clean vs mismatch
    def split(seeds_):
        clean = [s for s in seeds_ if not rows[s]["warnings"]]
        mism = [s for s in seeds_ if rows[s]["warnings"]]
        return clean, mism

    ar_clean, ar_mism = split(ATTN_RESCUE)
    mh_clean, mh_mism = split(MERGE_HARM)
    hdr("ATTN-RESCUE (vanilla fail -> attn success)  CLEAN", ar_clean)
    hdr("ATTN-RESCUE  MISMATCH (flag)", ar_mism)
    hdr("MERGE-HARM (vanilla success -> merge ajar)  CLEAN", mh_clean)
    hdr("MERGE-HARM  MISMATCH (flag)", mh_mism)

    # ---- aggregate: strategy flip + aim deltas ----
    def agg(seeds_):
        n_grasp_v = sum(1 for s in seeds_ if rows[s]["arms"][VANILLA]["strategy"] == "grasp")
        n_grasp_a = sum(1 for s in seeds_ if rows[s]["arms"][ATTN]["strategy"] == "grasp")
        n_grasp_m = sum(1 for s in seeds_ if rows[s]["arms"][MERGE]["strategy"] == "grasp")
        dd_h = np.mean([rows[s]["arms"][ATTN]["min_d_handle"] - rows[s]["arms"][VANILLA]["min_d_handle"] for s in seeds_])
        dd_f = np.mean([rows[s]["arms"][ATTN]["min_d_front"] - rows[s]["arms"][VANILLA]["min_d_front"] for s in seeds_])
        dd_hm = np.mean([rows[s]["arms"][MERGE]["min_d_handle"] - rows[s]["arms"][VANILLA]["min_d_handle"] for s in seeds_])
        dd_fm = np.mean([rows[s]["arms"][MERGE]["min_d_front"] - rows[s]["arms"][VANILLA]["min_d_front"] for s in seeds_])
        return n_grasp_v, n_grasp_a, n_grasp_m, dd_h, dd_f, dd_hm, dd_fm

    for name, seeds_ in (("ATTN-RESCUE-clean", ar_clean), ("MERGE-HARM-clean", mh_clean)):
        nv, na, nm, dh, df, dhm, dfm = agg(seeds_)
        print(f"\n[{name}] grasp-strategy count  vanilla={nv} attn={na} merge={nm} (n={len(seeds_)})")
        print(f"  attn-vs-vanilla: min_d_handle Δ={dh:+.3f} m   min_d_front Δ={df:+.3f} m   "
              f"(neg = attn gets CLOSER)")
        print(f"  merge-vs-vanilla: min_d_handle Δ={dhm:+.3f} m   min_d_front Δ={dfm:+.3f} m")

    print(f"\nwrote {REPLAY / 'mechanism_analysis.json'}")


if __name__ == "__main__":
    main()
