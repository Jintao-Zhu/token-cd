"""Summarize the v2 fixed-mask calibration arms and apply the three-hypothesis
interpretation locked in the mechanism-audit protocol.

Comparison table per task + total over the 22 locked calibration scenes:
  control | l11_k64_t05 (dynamic v1) | v2_fix_k64_t05 | v2_fix_k109_t05
Plus dim0 mechanism statistics (|D|, a_m, tau*a_m, max A_unimportant,
sum_{v in D} A[v]) from each step trace.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

WS = Path("/home/leju-suzhou/zjt_ws/token-cd")
ROOT = WS / "artifacts/dtp_openvla_calibration_v1/closed_loop/episodes"
OUT = WS / "artifacts/dtp_openvla_calibration_v1/closed_loop/V2_AUDIT.json"
TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")
ARMS = ("control", "l11_k64_t05", "v2_fix_k64_t05", "v2_fix_k109_t05")


def load(task: str, arm: str) -> dict:
    rows = {}
    d = ROOT / task / arm
    if not d.exists():
        return rows
    for f in sorted(d.glob("episode_*_summary.json")):
        s = json.loads(f.read_text())
        rows[int(s["seed"])] = s
    return rows


def succ(rows):
    return sum(1 for r in rows.values() if r["success"])


def prune_fired_summary(rows):
    n = len(rows)
    if not n:
        return None
    fired = sum(1 for r in rows.values() if r["prune_activation_steps"] > 0)
    steps = sum(r["control_steps"] for r in rows.values())
    fired_steps = sum(r["prune_activation_steps"] for r in rows.values())
    return {"n_episodes": n, "episodes_with_any_prune": fired,
            "total_steps": steps, "steps_with_any_prune": fired_steps,
            "frac_steps_pruned": round(fired_steps / steps, 4) if steps else 0.0}


def dim0_stats(rows):
    """Aggregate the per-step dim0 mechanism fields across episodes."""
    keys = ("a_m", "tau_times_a_m", "max_A_unimportant", "sum_A_pruned")
    acc = {k: [] for k in keys}
    fired = []
    unfired = []
    n_dim0 = 0
    n_fired = 0
    total_pruned_dim0 = 0
    for r in rows.values():
        for row in r.get("trace", []):
            d0 = row.get("dim0_trace") or []
            for t in d0:
                n_dim0 += 1
                pruned = t.get("pruned") or []
                if pruned:
                    n_fired += 1
                    total_pruned_dim0 += len(pruned)
                    fired.append(t)
                else:
                    unfired.append(t)
                for k in keys:
                    if k in t:
                        acc[k].append(t[k])
    recorded = bool(n_dim0)
    out = {
        "recorded": recorded,
        "n_dim0_steps": n_dim0,
        "fired_dim0_steps": n_fired,
        "total_pruned_tokens_dim0": total_pruned_dim0,
        "mean|D|_when_fired": round(float(np.mean([len(t["pruned"]) for t in fired])), 3) if fired else None,
        "fired_a_m": {"min": float(np.min([t["a_m"] for t in fired])),
                      "mean": float(np.mean([t["a_m"] for t in fired])),
                      "max": float(np.max([t["a_m"] for t in fired]))} if fired else None,
        "unfired_a_m": {"min": float(np.min([t["a_m"] for t in unfired])),
                        "mean": float(np.mean([t["a_m"] for t in unfired])),
                        "max": float(np.max([t["a_m"] for t in unfired]))} if unfired else None,
        "max_A_unimp_fired_mean": float(np.mean([t["max_A_unimportant"] for t in fired])) if fired else None,
        "sum_A_pruned_fired_mean": float(np.mean([t["sum_A_pruned"] for t in fired])) if fired else None,
    }
    return out


def paired(cand: dict, base: dict):
    """Per-scene rescue/harm counts of `cand` relative to `base`."""
    rescue = harm = unchanged = 0
    detail = {}
    for seed, cr in base.items():
        if seed not in cand:
            continue
        cs, ks = cr["success"], cand[seed]["success"]
        if not cs and ks:
            rescue += 1
            tag = "rescue"
        elif cs and not ks:
            harm += 1
            tag = "harm"
        else:
            unchanged += 1
            tag = "unchanged"
        detail[str(seed)] = tag
    return {"rescue": rescue, "harm": harm, "unchanged": unchanged,
            "net": rescue - harm, "detail": detail}


def main():
    data = {}
    for task in TASKS:
        loaded = {arm: load(task, arm) for arm in ARMS}
        data[task] = {
            "seeds": sorted(loaded["control"].keys()),
            "success": {arm: succ(loaded[arm]) for arm in ARMS},
            "prune": {arm: prune_fired_summary(loaded[arm]) for arm in ARMS},
            "dim0": {arm: dim0_stats(loaded[arm]) for arm in ARMS},
        }
    # totals across tasks
    merged = {arm: {} for arm in ARMS}
    for task in TASKS:
        for arm in ARMS:
            for seed, s in load(task, arm).items():
                merged[arm][(task, seed)] = s
    total = {
        "success": {arm: succ(merged[arm]) for arm in ARMS},
        "prune": {arm: prune_fired_summary(merged[arm]) for arm in ARMS},
        "dim0": {arm: dim0_stats(merged[arm]) for arm in ARMS},
    }
    pair_tbl = {}
    for key, (a, b) in {
        "dynamic_k64_vs_control": ("l11_k64_t05", "control"),
        "fixed_k64_vs_control": ("v2_fix_k64_t05", "control"),
        "fixed_k109_vs_control": ("v2_fix_k109_t05", "control"),
        "fixed_k64_vs_dynamic_k64": ("v2_fix_k64_t05", "l11_k64_t05"),
        "fixed_k109_vs_fixed_k64": ("v2_fix_k109_t05", "v2_fix_k64_t05"),
    }.items():
        pair_tbl[key] = {"candidate": a, "base": b,
                         "total": paired(merged[a], merged[b]),
                         "per_task": {t: paired(load(t, a), load(t, b)) for t in TASKS}}
    payload = {"arms": ARMS, "per_task": data, "total": total, "paired": pair_tbl}
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # console view
    print(f"{'task':>28} | ctrl | dyn64 | fix64 | fix109")
    for task in TASKS:
        d = data[task]
        print(f"{task:>28} | {d['success']['control']:>4} | {d['success']['l11_k64_t05']:>5} | "
              f"{d['success']['v2_fix_k64_t05']:>5} | {d['success']['v2_fix_k109_t05']:>6}")
    t = total["success"]
    print(f"{'TOTAL':>28} | {t['control']:>4} | {t['l11_k64_t05']:>5} | "
          f"{t['v2_fix_k64_t05']:>5} | {t['v2_fix_k109_t05']:>6}")
    print("\npaired (rescue/harm/net):")
    for key in ("dynamic_k64_vs_control", "fixed_k64_vs_control",
                "fixed_k109_vs_control", "fixed_k64_vs_dynamic_k64",
                "fixed_k109_vs_fixed_k64"):
        p = pair_tbl[key]["total"]
        print(f"  {key:>26}: rescue={p['rescue']} harm={p['harm']} net={p['net']:+d}  "
              f"(success {total['success'][pair_tbl[key]['candidate']]} vs "
              f"{total['success'][pair_tbl[key]['base']]})")
    print("\ndim0 firing (fired dim0 steps / total dim0 steps):")
    for arm in ARMS:
        s = total["dim0"][arm]
        tag = "recorded" if s["recorded"] else "NOT-RECORDED (pre-dim0-trace run)"
        print(f"  {arm:>16}: {s['fired_dim0_steps']}/{s['n_dim0_steps']} [{tag}] "
              f"mean|D|={s['mean|D|_when_fired']} fired_a_m_mean={s['fired_a_m'] and s['fired_a_m']['mean']} "
              f"unfired_a_m_mean={s['unfired_a_m'] and s['unfired_a_m']['mean']}")


if __name__ == "__main__":
    main()
