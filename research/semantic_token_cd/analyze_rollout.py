#!/usr/bin/env python3
"""SEMANTIC_ENTITY_CD_ROLLOUT_PILOT_V1 — analysis + report.

Reads episodes/{task}/{arm}/episode_{seed}_summary.json, computes:
  - per-arm SR (global + per-task)
  - Rescue/Harm (paired vanilla vs entity_cd on identical snapshots)
  - Gates A (ΔSR>+3pp non-single-task) / B (Rescue/Harm>1) / C (≥7/9 tasks non-worse)
  - STOP conditions (ΔSR<+1pp / task collapse -10pp)
  - alignment-vs-success (mean_language_score high vs low bucket)
Writes aggregate.json, decision.json, CSVs, REPORT_ZH.md.

Run (after rollout completes):
  cd /home/leju-suzhou/zjt_ws/token-cd
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/analyze_rollout.py \
    --artifact artifacts/semantic_entity_cd_rollout_pilot_v1
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ARMS = ("vanilla", "entity_cd_025", "entity_cd_050")
LAMBDA_LABEL = {"entity_cd_025": "λ=0.25", "entity_cd_050": "λ=0.50"}


def _f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def load_episodes(art: Path) -> dict:
    episodes = {}
    for task_dir in sorted((art / "episodes").iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for arm in ARMS:
            arm_dir = task_dir / arm
            if not arm_dir.is_dir():
                continue
            for f in sorted(arm_dir.glob("episode_*_summary.json")):
                seed = int(f.stem.split("_")[1])
                episodes[(task, seed, arm)] = json.loads(f.read_text())
    return episodes


def sr(episodes, arm, task=None, seeds=None):
    vals = [e["success"] for (t, s, a), e in episodes.items()
            if a == arm and (task is None or t == task) and (seeds is None or s in seeds)]
    return (float(np.mean(vals)), len(vals)) if vals else (float("nan"), 0)


def paired(episodes, cd_arm):
    """Rescue / Harm / same counts, paired over (task, seed)."""
    rescue = harm = fail_fail = succ_succ = 0
    pairs = []
    keys = sorted({(t, s) for (t, s, a) in episodes if a == "vanilla"})
    for t, s in keys:
        van = episodes.get((t, s, "vanilla"))
        cd = episodes.get((t, s, cd_arm))
        if van is None or cd is None:
            continue
        vs, cs = bool(van["success"]), bool(cd["success"])
        if not vs and cs:
            rescue += 1
        elif vs and not cs:
            harm += 1
        elif not vs and not cs:
            fail_fail += 1
        else:
            succ_succ += 1
        pairs.append((t, s, vs, cs))
    return {"rescue": rescue, "harm": harm, "fail_fail": fail_fail, "succ_succ": succ_succ,
            "pairs": pairs, "n": len(pairs),
            "ratio": (rescue / harm) if harm > 0 else (float("inf") if rescue > 0 else float("nan"))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    a = p.parse_args()
    art = a.artifact.resolve()
    eps = load_episodes(art)
    tasks = sorted({t for (t, s, arm) in eps})
    n_tasks = len(tasks)
    n_seeds = len({s for (t, s, arm) in eps if t == tasks[0] and arm == "vanilla"}) if tasks else 0
    print(json.dumps({"loaded": len(eps), "tasks": n_tasks, "seeds_per_task": n_seeds}))

    # ---- SR per arm ----
    sr_global = {arm: sr(eps, arm) for arm in ARMS}
    sr_task = {}
    for arm in ARMS:
        sr_task[arm] = {t: sr(eps, arm, task=t)[0] for t in tasks}

    # ---- Rescue / Harm ----
    rh = {arm: paired(eps, arm) for arm in ("entity_cd_025", "entity_cd_050")}

    # ---- per-task ΔSR ----
    delta = {}
    for arm in ("entity_cd_025", "entity_cd_050"):
        delta[arm] = {t: sr_task[arm][t] - sr_task["vanilla"][t] for t in tasks}

    # ---- Gates ----
    van_sr = sr_global["vanilla"][0]
    gates = {}
    for arm in ("entity_cd_025", "entity_cd_050"):
        cdsr = sr_global[arm][0]
        d = cdsr - van_sr
        improving_tasks = [t for t in tasks if delta[arm][t] > 0.0]
        single_task = len(improving_tasks) < 2
        gateA = (d > 0.03) and (not single_task)
        gateB = rh[arm]["ratio"] > 1.0
        n_nw = sum(1 for t in tasks if delta[arm][t] >= -0.01)  # non-worse (not drop >1pp)
        gateC = n_nw >= max(1, round(7 / 9 * n_tasks))  # ≥7/9 tasks non-worse
        collapsed = [t for t in tasks if delta[arm][t] <= -0.10]
        stop_no_gain = d < 0.01
        gates[arm] = {
            "delta_SR": d, "SR_vanilla": van_sr, "SR_cd": cdsr,
            "improving_tasks": improving_tasks, "single_task_driven": single_task,
            "A": {"PASS": bool(gateA), "delta": d, "non_single_task": not single_task},
            "B": {"PASS": bool(gateB), "rescue": rh[arm]["rescue"], "harm": rh[arm]["harm"],
                  "ratio": rh[arm]["ratio"]},
            "C": {"PASS": bool(gateC), "non_worse_tasks": n_nw, "of": n_tasks,
                  "threshold": max(1, round(7 / 9 * n_tasks))},
            "collapsed_tasks": collapsed,
            "stop_no_gain": bool(stop_no_gain),
        }

    # ---- alignment-vs-success (mean_language_score median split per CD arm) ----
    align_vs = {}
    for arm in ("entity_cd_025", "entity_cd_050"):
        scores = [e.get("mean_language_score") for e in eps.values() if e["arm"] == arm]
        scores = [x for x in scores if x is not None]
        if scores:
            med = float(np.median(scores))
            hi = [e["success"] for e in eps.values() if e["arm"] == arm and e.get("mean_language_score", -1) >= med]
            lo = [e["success"] for e in eps.values() if e["arm"] == arm and e.get("mean_language_score", -1) < med]
            align_vs[arm] = {
                "median_lang_score": med,
                "n_high": len(hi), "sr_high": float(np.mean(hi)) if hi else None,
                "n_low": len(lo), "sr_low": float(np.mean(lo)) if lo else None,
            }

    # ---- verdict ----
    verdicts = {}
    for arm in ("entity_cd_025", "entity_cd_050"):
        g = gates[arm]
        passed = g["A"]["PASS"] and g["B"]["PASS"] and g["C"]["PASS"]
        stopped = g["stop_no_gain"] or bool(g["collapsed_tasks"])
        verdicts[arm] = "PASS_TO_ROLLOUT" if (passed and not stopped) else "STOP_ENTITY_CD_ROLLOUT_NO_GO"

    aggregate = {
        "experiment": "SEMANTIC_ENTITY_CD_ROLLOUT_PILOT_V1",
        "n_tasks": n_tasks, "n_seeds_per_task": n_seeds, "n_episodes_per_arm": sr_global["vanilla"][1],
        "SR": {arm: sr_global[arm][0] for arm in ARMS},
        "rescue_harm": {arm: {k: rh[arm][k] for k in ("rescue", "harm", "fail_fail", "succ_succ", "ratio", "n")}
                        for arm in ("entity_cd_025", "entity_cd_050")},
        "task_SR": {t: {arm: sr_task[arm][t] for arm in ARMS} for t in tasks},
        "task_delta": {arm: delta[arm] for arm in ("entity_cd_025", "entity_cd_050")},
        "gates": gates,
        "alignment_vs_success": align_vs,
        "verdict": verdicts,
    }
    (art / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")

    decision = {
        "experiment": "SEMANTIC_ENTITY_CD_ROLLOUT_PILOT_V1",
        "status": verdicts["entity_cd_050"],  # primary: higher λ
        "status_by_arm": verdicts,
        "date": "2026-08-24", "type": "pilot",
        "SR_vanilla": van_sr,
        "SR_cd_025": sr_global["entity_cd_025"][0], "SR_cd_050": sr_global["entity_cd_050"][0],
        "rescue_harm_025": rh["entity_cd_025"]["ratio"], "rescue_harm_050": rh["entity_cd_050"]["ratio"],
        "core_question": "PCD alignment 0.586 的自动 semantic counterfactual 是否跨过闭环最低有效性门槛？",
    }
    (art / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    # ---- CSVs ----
    with (art / "task_breakdown.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "vanilla_SR", "cd025_SR", "cd050_SR", "cd025_delta", "cd050_delta"])
        for t in tasks:
            w.writerow([t, sr_task["vanilla"][t], sr_task["entity_cd_025"][t], sr_task["entity_cd_050"][t],
                        delta["entity_cd_025"][t], delta["entity_cd_050"][t]])
    with (art / "rescue_harm.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "rescue", "harm", "fail_fail", "succ_succ", "ratio"])
        for arm in ("entity_cd_025", "entity_cd_050"):
            w.writerow([arm, rh[arm]["rescue"], rh[arm]["harm"], rh[arm]["fail_fail"],
                        rh[arm]["succ_succ"], rh[arm]["ratio"]])

    # ---- REPORT_ZH.md ----
    lines = []
    lines.append("# SEMANTIC_ENTITY_CD_ROLLOUT_PILOT_V1 — 闭环 pilot")
    lines.append("")
    lines.append(f"> **Verdict: `{decision['status']}`**  · 日期 2026-08-24 · pilot（非 confirmation）")
    lines.append("> 核心问题：Phase-2B language-selected semantic entity negative branch（PCD alignment 0.586）是否已跨过闭环最低有效性门槛？")
    lines.append("")
    lines.append("## 0. 一句话结论")
    lines.append("")
    d25, d50 = delta["entity_cd_025"], delta["entity_cd_050"]
    best_arm = "entity_cd_050" if sr_global["entity_cd_050"][0] >= sr_global["entity_cd_025"][0] else "entity_cd_025"
    lines.append(f"- **SR**：vanilla **{_f(van_sr)}** → entity_cd λ=0.25 **{_f(sr_global['entity_cd_025'][0])}** "
                 f"（Δ {_f(sr_global['entity_cd_025'][0]-van_sr, 3)}）→ λ=0.50 **{_f(sr_global['entity_cd_050'][0])}** "
                 f"（Δ {_f(sr_global['entity_cd_050'][0]-van_sr, 3)}）")
    for arm in ("entity_cd_025", "entity_cd_050"):
        r = rh[arm]
        lines.append(f"- **{LAMBDA_LABEL[arm]} Rescue/Harm**：rescue **{r['rescue']}** / harm **{r['harm']}** "
                     f"（ratio **{_f(r['ratio'], 3)}**；fail_fail {r['fail_fail']}，succ_succ {r['succ_succ']}）")
    lines.append("")
    lines.append("## 1. Gates")
    lines.append("")
    lines.append("| Arm | A: ΔSR>+3pp 非单任务 | B: Rescue/Harm>1 | C: ≥7/9 任务不劣 | Verdict |")
    lines.append("|---|---|---|---|---|")
    for arm in ("entity_cd_025", "entity_cd_050"):
        g = gates[arm]
        a = "PASS" if g["A"]["PASS"] else "FAIL"
        b = "PASS" if g["B"]["PASS"] else "FAIL"
        c = "PASS" if g["C"]["PASS"] else "FAIL"
        lines.append(f"| {LAMBDA_LABEL[arm]} | {a}（Δ {_f(g['A']['delta'],3)}） | {b}（{_f(g['B']['ratio'],3)}） | "
                     f"{c}（{g['C']['non_worse_tasks']}/{g['C']['of']}） | {verdicts[arm]} |")
    lines.append("")
    lines.append("## 2. Task-wise SR")
    lines.append("")
    lines.append("| task | vanilla | cd λ=0.25 | cd λ=0.50 | Δ0.25 | Δ0.50 |")
    lines.append("|---|---|---|---|---|---|")
    for t in tasks:
        lines.append(f"| {t} | {_f(sr_task['vanilla'][t])} | {_f(sr_task['entity_cd_025'][t])} | "
                     f"{_f(sr_task['entity_cd_050'][t])} | {_f(d25[t],3)} | {_f(d50[t],3)} |")
    lines.append("")
    lines.append("## 3. 对齐-成功关系（mean_language_score 中位分割）")
    lines.append("")
    lines.append("| Arm | median lang | SR 高对齐 | SR 低对齐 | n高/n低 |")
    lines.append("|---|---|---|---|---|")
    for arm in ("entity_cd_025", "entity_cd_050"):
        av = align_vs.get(arm)
        if av:
            lines.append(f"| {LAMBDA_LABEL[arm]} | {_f(av['median_lang_score'],4)} | "
                         f"{_f(av['sr_high'])} | {_f(av['sr_low'])} | {av['n_high']}/{av['n_low']} |")
    lines.append("")
    lines.append("## 4. 结论")
    lines.append("")
    lines.append(f"**{decision['status']}**")
    lines.append("")
    lines.append(f"- 若 PASS_TO_ROLLOUT：Phase-2B semantic entity branch 在 0.586 对齐下已产生净闭环收益 → 该线从机制分析升级为可用方法。")
    lines.append(f"- 若 STOP：0.586 未跨过闭环最低门槛 → Semantic Token-CD 线正式关闭，结论完整。")
    lines.append("")
    lines.append("**产出**：aggregate.json · decision.json · task_breakdown.csv · rescue_harm.csv · REPORT_ZH.md")
    (art / "REPORT_ZH.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({"verdict": decision["status"], "SR_vanilla": van_sr,
                      "SR_cd_025": sr_global["entity_cd_025"][0], "SR_cd_050": sr_global["entity_cd_050"][0],
                      "rh_025": rh["entity_cd_025"]["ratio"], "rh_050": rh["entity_cd_050"]["ratio"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
