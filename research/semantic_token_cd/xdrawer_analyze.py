"""Aggregate the sim-region cross-instruction drawer experiment.

Reads episode summaries produced by xdrawer_rollout and reports:
  - pairing consistency (same initial state/RGB across top/middle for a seed)
  - technical audits for the shr arms
  - success / paired Rescue-Harm for target vs other regions
  - behavioral metrics: first-contacted drawer, per-drawer contact step,
    final drawer displacement, success
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.semantic_token_cd.xdrawer_protocol import (
    DRAWERS, INSTRUCTION_BY_DRAWER, OPEN_QPOS_M, PROTOCOL, SEEDS,
    TASK_BY_DRAWER,
)

ARMS = ("vanilla", "shr_target", "shr_other")


def load_summary(root: Path, drawer: str, arm: str, seed: int):
    path = root / "episodes" / TASK_BY_DRAWER[drawer] / arm / f"episode_{seed:03d}_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="rollout artifact (episodes/) root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="100-119")
    args = ap.parse_args()
    runs = args.runs.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    seed_list = []
    for part in args.seeds.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            seed_list.extend(range(lo, hi + 1))
        elif part:
            seed_list.append(int(part))
    seed_list = sorted(set(seed_list))

    ep = {(d, a, s): load_summary(runs, d, a, s) for d in DRAWERS for a in ARMS for s in seed_list}
    missing = [(d, a, s) for (d, a, s), v in ep.items() if v is None]
    if missing:
        print(json.dumps({"missing": missing}), flush=True)
        raise SystemExit(2)

    # ---- pairing consistency ----
    pairing = {"checked": 0, "state_mismatch": [], "rgb_mismatch": []}
    audits = {"checked": 0, "failures": []}
    for d in DRAWERS:
        for a in ARMS:
            for s in seed_list:
                e = ep[(d, a, s)]
                if a in ("shr_target", "shr_other"):
                    audits["checked"] += 1
                    ad = e.get("shr_audit") or {}
                    if not ad.get("technical_pass"):
                        audits["failures"].append({"drawer": d, "arm": a, "seed": s, "audit": ad})
                pairing["checked"] += 1
                other_d = "middle" if d == "top" else "top"
                ref = ep[(other_d, a, s)]
                if e["initial_state_sha256"] != ref["initial_state_sha256"]:
                    pairing["state_mismatch"].append((d, a, s))
                if e["initial_rgb_sha256"] != ref["initial_rgb_sha256"]:
                    pairing["rgb_mismatch"].append((d, a, s))

    # ---- per (drawer, arm) aggregates + paired rescue/harm ----
    per_drawer = {}
    for d in DRAWERS:
        arm_stat = {}
        rescue = {"shr_target": {"rescue": 0, "harm": 0}, "shr_other": {"rescue": 0, "harm": 0}}
        cross = {"same": 0, "target_only": 0, "other_only": 0, "neither": 0}
        contact_target_first = {"vanilla": 0, "shr_target": 0, "shr_other": 0}
        opened = {"vanilla": 0, "shr_target": 0, "shr_other": 0}
        for a in ARMS:
            rows = [ep[(d, a, s)] for s in seed_list]
            success = [r["success"] for r in rows]
            arm_stat[a] = {
                "n": len(rows), "success": int(sum(success)),
                "success_rate": float(np.mean(success)) if rows else None,
                "mean_steps": float(np.mean([r["control_steps"] for r in rows])),
                "mean_region_size": float(np.mean([r["region_mean_sizes"].get(d, 0) for r in rows])),
                "region_overlap_mean": float(np.mean([r["region_mean_sizes"].get("overlap", 0) for r in rows])),
                "contact_step": {
                    "target": [r["contact_step"].get(d) for r in rows],
                    "other": [r["contact_step"].get("middle" if d == "top" else "top") for r in rows],
                },
                "final_qpos_target": {
                    "mean": float(np.mean([r["final_drawer_qpos"][d] for r in rows])),
                    "max": float(np.max([r["final_drawer_qpos"][d] for r in rows])),
                },
                "final_qpos_other": {
                    "mean": float(np.mean([r["final_drawer_qpos"]["middle" if d == "top" else "top"] for r in rows])),
                },
                "mean_eef_handle_dist_first10": None,
            }
            for r in rows:
                contact_order = r.get("contact_order", [])
                if contact_order and contact_order[0] == d:
                    contact_target_first[a] += 1
                if r["final_drawer_qpos"][d] >= OPEN_QPOS_M:
                    opened[a] += 1
        for s in seed_list:
            v = ep[(d, "vanilla", s)]["success"]
            for a in ("shr_target", "shr_other"):
                ok = ep[(d, a, s)]["success"]
                if ok and not v:
                    rescue[a]["rescue"] += 1
                elif (not ok) and v:
                    rescue[a]["harm"] += 1
            t = ep[(d, "shr_target", s)]["success"]
            o = ep[(d, "shr_other", s)]["success"]
            if t == o:
                cross["same" if t else "neither"] += 1
            elif t:
                cross["target_only"] += 1
            else:
                cross["other_only"] += 1
        per_drawer[d] = {
            "instruction": INSTRUCTION_BY_DRAWER[d],
            "arms": arm_stat,
            "paired_rescue_harm": rescue,
            "target_vs_other_success_pattern": cross,
            "first_contact_is_target": contact_target_first,
            "target_opened_at_end": opened,
        }

    report = {
        "protocol_id": PROTOCOL,
        "seeds": seed_list,
        "pairing": pairing,
        "audits": audits,
        "per_drawer": per_drawer,
    }
    (out / "closed_loop_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "pairing": {"checked": pairing["checked"], "state_mismatch": len(pairing["state_mismatch"]),
                    "rgb_mismatch": len(pairing["rgb_mismatch"])},
        "audits": {"checked": audits["checked"], "failures": len(audits["failures"])},
    }, indent=2), flush=True)
    for d in DRAWERS:
        print(f"\n== drawer target: {d} ({INSTRUCTION_BY_DRAWER[d]}) ==")
        for a in ARMS:
            st = per_drawer[d]["arms"][a]
            print(f"  {a:12s} success {st['success']:3d}/{st['n']:3d}  "
                  f"mean_steps {st['mean_steps']:5.1f}  region_size {st['mean_region_size']:4.1f}  "
                  f"overlap {st['region_overlap_mean']:4.1f}  "
                  f"final_target_qpos_mean {st['final_qpos_target']['mean']:.3f} max {st['final_qpos_target']['max']:.3f}  "
                  f"first_contact_target {per_drawer[d]['first_contact_is_target'][a]:3d}")
        print("  paired rescue/harm:", json.dumps(per_drawer[d]["paired_rescue_harm"]))
        print("  target-vs-other pattern:", json.dumps(per_drawer[d]["target_vs_other_success_pattern"]))


if __name__ == "__main__":
    main()
