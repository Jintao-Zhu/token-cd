"""Aggregate the VLA-Pruner FORMAL 1200-episode closed-loop run.

Reads artifacts/vla_pruner_openvla_reproduction/formal_1200/episodes/... and the
frozen CONFIG_LOCK.json, then writes:
  episode_results.csv / paired_results.csv / task_summary.csv
  latency_summary.csv / temporal_summary.csv
  technical_audit.json / FINAL_REPORT.md

Run when the 1200 technical_pass episodes are complete:
  python research/semantic_token_cd/vla_pruner_formal_analyze.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/home/leju-suzhou/zjt_ws/token-cd")
ROOT = REPO_ROOT / "artifacts/vla_pruner_openvla_reproduction/formal_1200"
PROTOCOL = "VLA_PRUNER_OPENVLA_REPRODUCTION_V1"
ARMS = ("vanilla", "vla_pruner_prune25", "vla_pruner_prune50")
PRUNE_ARMS = ARMS[1:]
TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")


def chi2_sf_1df(x: float) -> float:
    """chi-square survival at 1 dof = erfc(sqrt(x/2))."""
    from math import erfc
    if x <= 0:
        return 1.0
    return erfc(math.sqrt(x / 2.0))


def mcnemar(b: int, c: int) -> dict:
    """b = harm (vanilla ok, prune fail), c = rescue (vanilla fail, prune ok)."""
    if b + c == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p_two_sided": 1.0, "significant": False}
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    p = chi2_sf_1df(chi2)
    return {"b": b, "c": c, "chi2": round(float(chi2), 3),
            "p_two_sided": float(p), "significant": bool(p < 0.05)}


def load_episodes() -> list[dict]:
    out = []
    for arm in ARMS:
        for f in sorted((ROOT / "episodes").glob(f"*/{arm}/episode_*_summary.json")):
            out.append(json.loads(f.read_text()))
    return out


def ep_rows(episodes, arm, task=None, seed=None):
    return [e for e in episodes if e["arm"] == arm
            and (task is None or e["task"] == task)
            and (seed is None or e["seed"] == seed)]


def main() -> None:
    lock = json.loads((ROOT / "CONFIG_LOCK.json").read_text())
    episodes = load_episodes()
    expect = 3 * 100 * len(TASKS)
    if len(episodes) != expect:
        raise RuntimeError(f"found {len(episodes)} episodes, expected {expect}; run formal to completion")
    # Pairing integrity.
    missing = []
    for task in TASKS:
        for seed in range(100):
            for arm in ARMS:
                if not any(e["task"] == task and e["seed"] == seed and e["arm"] == arm
                           for e in episodes):
                    missing.append((task, seed, arm))
    if missing:
        raise RuntimeError(f"missing paired episodes: {missing[:5]} ...")

    technical = {
        "all_technical_pass": all(bool(e["technical_pass"]) for e in episodes),
        "n": len(episodes),
        "fails": [{"task": e["task"], "seed": e["seed"], "arm": e["arm"]}
                  for e in episodes if not e["technical_pass"]],
        "by_worker": {},
        "rounds_used": {},
    }
    for e in episodes:
        technical["by_worker"].setdefault(e["worker_id"], 0)
        technical["by_worker"][e["worker_id"]] += 1
        technical["rounds_used"].setdefault(str(e.get("round_no", 1)), 0)
        technical["rounds_used"][str(e.get("round_no", 1))] += 1
    technical["retry_failures_logged"] = 0
    for f in (ROOT / "logs").glob("technical_failures_*.jsonl"):
        technical["retry_failures_logged"] += sum(1 for _ in f.open())

    # ---- episode_results.csv ------------------------------------------------
    rows = []
    for e in episodes:
        s = e.get("prune_stats", {})
        rows.append({
            "task": e["task"], "seed": e["seed"], "arm": e["arm"],
            "success": bool(e["success"]), "failure_reason": e["failure_reason"],
            "control_steps": int(e["control_steps"]),
            "runtime_seconds": round(float(e["runtime_seconds"]), 2),
            "technical_pass": bool(e["technical_pass"]),
            "worker_id": e["worker_id"],
            "first_divergence_step": ("" if s.get("first_divergence_step") is None
                                      else int(s["first_divergence_step"])),
            "total_token_flips": ("" if s.get("total_token_flips") is None
                                  else int(s["total_token_flips"])),
            "flip_steps": ("" if s.get("flip_steps") is None else int(s["flip_steps"])),
            "mean_raw_l1": ("" if s.get("mean_raw_l1") is None else round(float(s["mean_raw_l1"]), 4)),
            "mean_raw_l2": ("" if s.get("mean_raw_l2") is None else round(float(s["mean_raw_l2"]), 4)),
            "prune_activation_steps": int(s.get("prune_activation_steps", 0)),
            "mean_kept_active": ("" if s.get("mean_kept_image_count_active") is None
                                 else round(float(s["mean_kept_image_count_active"]), 2)),
        })
    with (ROOT / "episode_results.csv").open("w", newline="") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- task/arm summary -----------------------------------------------------
    def agg(arm, task=None):
        rs = ep_rows(episodes, arm, task)
        st = [r.get("prune_stats", {}) for r in rs]
        runtimes = [float(r["runtime_seconds"]) for r in rs]
        steps = [int(r["control_steps"]) for r in rs]
        return {
            "n": len(rs),
            "success": sum(bool(r["success"]) for r in rs),
            "control_steps_total": sum(steps),
            "runtime_seconds_total": round(sum(runtimes), 1),
            "mean_episode_seconds": round(float(np.mean(runtimes)), 2),
            "mean_env_steps": round(float(np.mean(steps)), 2),
            "mean_seconds_per_env_step": round(sum(runtimes) / max(1, sum(steps)), 3),
            "mean_prune_activation_steps": round(float(np.mean(
                [s.get("prune_activation_steps", 0) for s in st])), 2),
            "mean_kept_active": (round(float(np.mean([s["mean_kept_image_count_active"]
                                                      for s in st if s.get("mean_kept_image_count_active") is not None])), 2)
                                 if any(s.get("mean_kept_image_count_active") is not None for s in st) else None),
        }

    task_summary_rows = []
    per_task = {}
    for task in TASKS:
        per_task[task] = {a: agg(a, task) for a in ARMS}
        for a in ARMS:
            v = per_task[task][a]
            task_summary_rows.append({"task": task, "arm": a, "n": v["n"], "success": v["success"],
                                      "mean_env_steps": v["mean_env_steps"],
                                      "mean_episode_seconds": v["mean_episode_seconds"],
                                      "mean_seconds_per_env_step": v["mean_seconds_per_env_step"]})
    with (ROOT / "task_summary.csv").open("w", newline="") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=list(task_summary_rows[0]))
        w.writeheader()
        w.writerows(task_summary_rows)
    totals = {a: agg(a) for a in ARMS}

    # ---- paired results + rescue/harm + McNemar ------------------------------
    pairs = []
    disc = {"vla_pruner_prune25": {"b": 0, "c": 0}, "vla_pruner_prune50": {"b": 0, "c": 0}}
    disc_task = {t: {"vla_pruner_prune25": {"b": 0, "c": 0},
                     "vla_pruner_prune50": {"b": 0, "c": 0}} for t in TASKS}
    for task in TASKS:
        for seed in range(100):
            v = next(e for e in episodes if e["task"] == task and e["seed"] == seed and e["arm"] == "vanilla")
            for arm in PRUNE_ARMS:
                c = next(e for e in episodes if e["task"] == task and e["seed"] == seed and e["arm"] == arm)
                if c["success"] and not v["success"]:
                    kind = "rescue"
                    disc[arm]["c"] += 1
                    disc_task[task][arm]["c"] += 1
                elif not c["success"] and v["success"]:
                    kind = "harm"
                    disc[arm]["b"] += 1
                    disc_task[task][arm]["b"] += 1
                else:
                    kind = "unchanged"
                sd = c.get("prune_stats", {})
                pairs.append({"task": task, "seed": seed, "arm": arm,
                              "vanilla_success": bool(v["success"]), "arm_success": bool(c["success"]),
                              "pair": kind,
                              "first_divergence_step": ("" if sd.get("first_divergence_step") is None
                                                        else int(sd["first_divergence_step"])),
                              "total_token_flips": ("" if sd.get("total_token_flips") is None
                                                    else int(sd["total_token_flips"]))})
    with (ROOT / "paired_results.csv").open("w", newline="") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=list(pairs[0]))
        w.writeheader()
        w.writerows(pairs)
    mcnemar_all = {a: mcnemar(disc[a]["b"], disc[a]["c"]) for a in PRUNE_ARMS}
    mcnemar_task = {t: {a: mcnemar(disc_task[t][a]["b"], disc_task[t][a]["c"])
                        for a in PRUNE_ARMS} for t in TASKS}

    # ---- latency summary (inference ms / env step from arrays) ----------------
    lat_rows = []
    for task in TASKS:
        for arm in ARMS:
            rs = ep_rows(episodes, arm, task)
            inf_ms, pol_ms, env_ms, ref_ms = [], [], [], []
            for r in rs:
                ap = ROOT / "episodes" / task / arm / f"episode_{r['seed']:03d}_arrays.npz"
                z = np.load(ap)
                inf_ms.append(np.asarray(z["inferred_ms"], dtype=np.float64))
                pol_ms.append(np.asarray(z["policy_ms"], dtype=np.float64))
                env_ms.append(np.asarray(z["env_ms"], dtype=np.float64))
                ref_ms.append(np.asarray(z["ref_ms"], dtype=np.float64))
            inf = float(np.mean([m.mean() for m in inf_ms]))
            pol = float(np.mean([m.mean() for m in pol_ms]))
            env = float(np.mean([m.mean() for m in env_ms]))
            ref = float(np.mean([m.mean() for m in ref_ms]))
            lat_rows.append({"task": task, "arm": arm, "n": len(rs),
                             "mean_model_inference_ms_per_env_step": round(inf, 1),
                             "mean_policy_ms_per_step": round(pol, 1),
                             "mean_ref_ms_per_step": round(ref, 1),
                             "mean_env_step_ms": round(env, 1),
                             "mean_episode_seconds": per_task[task][arm]["mean_episode_seconds"]})
    with (ROOT / "latency_summary.csv").open("w", newline="") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=list(lat_rows[0]))
        w.writeheader()
        w.writerows(lat_rows)

    # ---- temporal summary -------------------------------------------------------
    temp_rows = []
    for task in TASKS:
        for arm in PRUNE_ARMS:
            rs = ep_rows(episodes, arm, task)
            st = [r.get("prune_stats", {}) for r in rs]
            temp_rows.append({
                "task": task, "arm": arm, "n": len(rs),
                "mean_prune_activation_steps": round(float(np.mean([s.get("prune_activation_steps", 0) for s in st])), 2),
                "mean_history_used_steps": round(float(np.mean([s.get("history_used_steps", 0) for s in st])), 2),
                "mean_kept_active": round(float(np.mean([s["mean_kept_image_count_active"]
                                                         for s in st if s.get("mean_kept_image_count_active") is not None])), 2),
                "mean_overlap_active": (round(float(np.mean([s["guide_topk_overlap_active"]
                                                             for s in st if s.get("guide_topk_overlap_active") is not None])), 2)
                                        if any(s.get("guide_topk_overlap_active") is not None for s in st) else None),
                "redundancy_steps_total": int(sum(s.get("overlap_used_redundancy_steps", 0) for s in st)),
                "mean_phase_early_flips": round(float(np.mean([s.get("phase_early_flips", 0) for s in st])), 2),
                "mean_phase_middle_flips": round(float(np.mean([s.get("phase_middle_flips", 0) for s in st])), 2),
                "mean_phase_late_flips": round(float(np.mean([s.get("phase_late_flips", 0) for s in st])), 2),
            })
    with (ROOT / "temporal_summary.csv").open("w", newline="") as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=list(temp_rows[0]))
        w.writeheader()
        w.writerows(temp_rows)

    # ---- latency: own inference vs diagnostic reference --------------------------
    lat_pool = {}
    for arm in ARMS:
        pol, ref, env, rt, steps = [], [], [], [], []
        for task in TASKS:
            for r in ep_rows(episodes, arm, task):
                z = np.load(ROOT / "episodes" / task / arm / f"episode_{r['seed']:03d}_arrays.npz")
                pol.append(float(np.asarray(z["policy_ms"], dtype=np.float64).mean()))
                ref.append(float(np.asarray(z["ref_ms"], dtype=np.float64).mean()))
                env.append(float(np.asarray(z["env_ms"], dtype=np.float64).mean()))
                rt.append(float(r["runtime_seconds"]))
                steps.append(int(r["control_steps"]))
        n = len(rt)
        projected = [rt[i] - ref[i] * steps[i] / 1000.0 for i in range(n)]
        lat_pool[arm] = {
            "n": n,
            "policy_ms_per_step": round(float(np.mean(pol)), 1),
            "ref_ms_per_step": round(float(np.mean(ref)), 1),
            "env_ms_per_step": round(float(np.mean(env)), 1),
            "mean_episode_seconds": round(float(np.mean(rt)), 2),
            "projected_episode_seconds_no_ref": round(float(np.mean(projected)), 2),
        }
    v_own = lat_pool["vanilla"]["policy_ms_per_step"]
    v_proj = lat_pool["vanilla"]["projected_episode_seconds_no_ref"]
    for a in PRUNE_ARMS:
        lat_pool[a]["inference_speedup"] = round(v_own / lat_pool[a]["policy_ms_per_step"], 3)
        lat_pool[a]["projected_e2e_speedup_no_ref"] = round(
            v_proj / lat_pool[a]["projected_episode_seconds_no_ref"], 3)
        lat_pool[a]["e2e_with_diagnostic_ref_speedup"] = round(
            lat_pool["vanilla"]["mean_episode_seconds"] / lat_pool[a]["mean_episode_seconds"], 3)

    (ROOT / "technical_audit.json").write_text(json.dumps(technical, indent=2, sort_keys=True) + "\n")

    # ---- FINAL_REPORT ----------------------------------------------------------------
    def srate(v):
        return f"{v['success']}/{v['n']} ({100.0*v['success']/v['n']:.1f}%)"

    lines = [
        "# VLA-Pruner reproduction — FORMAL 1200 closed-loop report",
        "",
        f"protocol: {PROTOCOL} / stage formal_1200 (frozen config in CONFIG_LOCK.json)",
        f"episodes: {len(episodes)}/1200 technical_pass = {technical['all_technical_pass']}",
        "",
        "## 1. Overall success rate",
        "| arm | success |",
        "|---|---|",
    ]
    for a in ARMS:
        lines.append(f"| {a} | {srate(totals[a])} |")
    lines.append("")
    lines.append("## 2. Per-task success")
    lines.append("| task | vanilla | prune25 | prune50 |")
    lines.append("|---|---|---|---|")
    for t in TASKS:
        lines.append(f"| {t} | {srate(per_task[t]['vanilla'])} | {srate(per_task[t]['vla_pruner_prune25'])} | {srate(per_task[t]['vla_pruner_prune50'])} |")
    lines.append("")
    lines.append("## 3. Paired rescue / harm + McNemar (vs current-harness vanilla)")
    lines.append("| arm | rescue | harm | unchanged | McNemar p |")
    lines.append("|---|---|---|---|---|")
    for a in PRUNE_ARMS:
        mm = mcnemar_all[a]
        n = totals["vanilla"]["n"]
        lines.append(f"| {a} | {mm['c']} | {mm['b']} | {n - mm['c'] - mm['b']} | {mm['p_two_sided']:.4f} |")
    lines.append("")
    lines.append("### per-task discordant pairs")
    for t in TASKS:
        parts = []
        for a in PRUNE_ARMS:
            mm = mcnemar_task[t][a]
            parts.append(f"{a}: rescue {mm['c']} / harm {mm['b']} / p={mm['p_two_sided']:.3f}")
        lines.append(f"- {t}: " + "; ".join(parts))
    lines.append("")
    lines.append("## 4. Latency (measured on GPU2/3, this harness)")
    lines.append("| arm | own inference ms/env-step | own-inference speedup | env.step ms | "
                 "projected end-to-end speedup (diagnostic ref excluded) | end-to-end with diagnostic ref (paired cost) |")
    lines.append("|---|---|---|---|---|---|")
    for a in ARMS:
        if a == "vanilla":
            lines.append(f"| {a} | {lat_pool[a]['policy_ms_per_step']} | 1.000 | {lat_pool[a]['env_ms_per_step']} | 1.000 | 1.000 |")
        else:
            lp = lat_pool[a]
            lines.append(f"| {a} | {lp['policy_ms_per_step']} | {lp['inference_speedup']} | "
                         f"{lp['env_ms_per_step']} | {lp['projected_e2e_speedup_no_ref']} | "
                         f"{lp['e2e_with_diagnostic_ref_speedup']} |")
    lines.append("")
    lines.append("`own inference` = the arm's own 7-token generate per env step, incl. VLA-Pruner's "
                 "attention/history bookkeeping; `env.step ms` = simulator; `projected end-to-end` = "
                 "episode wall-clock after subtracting the diagnostic vanilla reference forward "
                 "(which exists only to measure same-obs action deltas, not part of VLA-Pruner itself).")
    lines.append("")
    lines.append("## 4b. Action-change / temporal behaviour")
    lines.append("| arm | first divergence step (mean / median) | flip steps (mean) | mean raw L1 | mean overlap active |")
    lines.append("|---|---|---|---|---|")
    for a in PRUNE_ARMS:
        fds, fls, l1s, ovs = [], [], [], []
        for r in ep_rows(episodes, a):
            st = r.get("prune_stats", {})
            if st.get("first_divergence_step") is not None:
                fds.append(st["first_divergence_step"])
            if st.get("flip_steps") is not None:
                fls.append(st["flip_steps"])
            if st.get("mean_raw_l1") is not None:
                l1s.append(st["mean_raw_l1"])
            if st.get("guide_topk_overlap_active") is not None:
                ovs.append(st["guide_topk_overlap_active"])
        lines.append(f"| {a} | {np.mean(fds):.2f} / {int(np.median(fds))} | {np.mean(fls):.1f} | "
                     f"{np.mean(l1s):.3f} | {np.mean(ovs):.1f} |")
    lines.append("")
    lines.append("## 5. Temporal / pruning sanity")
    lines.append("| arm | mean prune activation steps | mean kept (active) | mean overlap active |")
    lines.append("|---|---|---|---|")
    for a in PRUNE_ARMS:
        rows = [r for r in temp_rows if r["arm"] == a]
        n = float(np.mean([r["mean_prune_activation_steps"] for r in rows]))
        k = float(np.mean([r["mean_kept_active"] for r in rows]))
        o = float(np.mean([r["mean_overlap_active"] for r in rows if r["mean_overlap_active"] is not None]))
        lines.append(f"| {a} | {n:.2f} | {k:.2f} | {o:.2f} |")
    lines.append("")
    lines.append("## 6. Conclusions (pre-registered interpretation)")
    vanilla_sr = totals["vanilla"]["success"] / totals["vanilla"]["n"]
    p25_sr = totals["vla_pruner_prune25"]["success"] / totals["vla_pruner_prune25"]["n"]
    p50_sr = totals["vla_pruner_prune50"]["success"] / totals["vla_pruner_prune50"]["n"]
    delta_ok = abs(p25_sr - vanilla_sr) < 0.03 and abs(p50_sr - vanilla_sr) < 0.03
    best_sr = max(p25_sr, p50_sr)
    best_p = min(mcnemar_all[a]["p_two_sided"] for a in PRUNE_ARMS)
    best_net = max(mcnemar_all[a]["c"] - mcnemar_all[a]["b"] for a in PRUNE_ARMS)
    gain = (best_sr - vanilla_sr >= 0.03) and (best_p < 0.05) and (best_net > 0)
    faster = min(lat_pool[a]["projected_e2e_speedup_no_ref"] for a in PRUNE_ARMS) > 1.03
    churn = max(mcnemar_all[a]["b"] + mcnemar_all[a]["c"] for a in PRUNE_ARMS) / totals["vanilla"]["n"]
    if gain:
        verdict = "A: VLA-Pruner SR clearly above vanilla with Rescue > Harm (closed-loop gain reproduced)."
    elif delta_ok and faster:
        verdict = ("B: SR ≈ vanilla and measured latency clearly lower "
                   "(keeps closed-loop performance and reduces cost).")
    elif delta_ok and churn > 0.05:
        verdict = ("C: SR ≈ vanilla but Rescue/Harm are large and offsetting "
                   "(pruning clearly changes the policy; no net gain, no measured speedup).")
    elif p25_sr < vanilla_sr - 0.03 and p50_sr < vanilla_sr - 0.03:
        verdict = ("D: VLA-Pruner SR clearly below vanilla on this domain "
                   "(no lossless transfer); stop ratio tuning, revisit HH/temporal-attention route.")
    else:
        verdict = ("C2: SR closest to vanilla with offsetting Rescue/Harm; mixed per-task effects, "
                   "no significant difference (see McNemar).")
    lines.append(f"- **Verdict: {verdict}**")
    lines.append("")
    lines.append("## 7. Raw artifacts")
    lines.append("- `episode_results.csv`, `paired_results.csv`, `task_summary.csv`, "
                 "`latency_summary.csv`, `temporal_summary.csv`, `technical_audit.json`")
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    print((ROOT / "FINAL_REPORT.md").read_text())


if __name__ == "__main__":
    main()
