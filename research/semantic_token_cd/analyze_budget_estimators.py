"""Offline stage: calibrate gamma*/tau* to match Matched's mean budget, then compare the three estimators.

gamma*/tau* are chosen ONLY to match the mean m on calibration states - no success labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")


def load(root: Path):
    rows = []
    for task in TASKS:
        for f in sorted((root / "estimators" / task).glob("seed_*.json")):
            d = json.loads(f.read_text())
            for st in d["states"]:
                rows.append({"task": task, "seed": d["seed"], "step": st["step"],
                             "progress": st["progress"], "m_matched": st["m_matched"],
                             "rel": st["m_relative_by_gamma"], "spec": st["m_spectral_by_tau"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    args = ap.parse_args()
    root = args.artifact.resolve()
    rows = load(root)
    if not rows:
        raise SystemExit("no estimator states found")

    mean_matched = float(np.mean([r["m_matched"] for r in rows]))
    gammas = sorted(rows[0]["rel"].keys(), key=float)
    taus = sorted(rows[0]["spec"].keys(), key=float)
    g_stat = {g: (np.mean([r["rel"][g] for r in rows]), g) for g in gammas}
    t_stat = {t: (np.mean([r["spec"][t] for r in rows]), t) for t in taus}
    gamma_star = min(gammas, key=lambda g: abs(g_stat[g][0] - mean_matched))
    tau_star = min(taus, key=lambda t: abs(t_stat[t][0] - mean_matched))

    for r in rows:
        r["m_relative"] = r["rel"][gamma_star]
        r["m_spectral"] = r["spec"][tau_star]

    def spearman(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
            return float("nan"), float("nan")
        r, p = stats.spearmanr(a, b)
        return float(r), float(p)

    def desc(v):
        v = np.asarray(v, float)
        return dict(mean=float(v.mean()), std=float(v.std()), median=float(np.median(v)),
                    min=float(v.min()), max=float(v.max()),
                    p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)))

    L = ["# Budget estimator comparison (offline)", "",
         f"状态数: **{len(rows)}**（4 任务 × 50 seeds × 4 progress）", "",
         f"- 校准目标: 让 Relative / Spectral 的平均预算 ≈ Matched 的 **{mean_matched:.2f}**",
         f"- **gamma\\* = {gamma_star}** → 平均 m = {g_stat[gamma_star][0]:.2f}",
         f"- **tau\\* = {tau_star}** → 平均 m = {t_stat[tau_star][0]:.2f}", "",
         "## 三个估计量的分布", "",
         "| 估计量 | mean | std | median | min | max | P10 | P90 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, key in (("Matched", "m_matched"), ("Relative", "m_relative"), ("Spectral", "m_spectral")):
        d = desc([r[key] for r in rows])
        L.append(f"| {name} | {d['mean']:.2f} | {d['std']:.2f} | {d['median']:.1f} | "
                 f"{d['min']:.0f} | {d['max']:.0f} | {d['p10']:.0f} | {d['p90']:.0f} |")

    L += ["", "## 与 Matched 的 Spearman 相关", "", "| 范围 | n | rho(Matched, Relative) | rho(Matched, Spectral) |", "|---|---:|---:|---:|"]
    def rowset(sub):
        a = [r["m_matched"] for r in sub]
        rr, _ = spearman(a, [r["m_relative"] for r in sub])
        rs, _ = spearman(a, [r["m_spectral"] for r in sub])
        return len(sub), rr, rs
    n, rr, rs = rowset(rows)
    L.append(f"| **总体** | {n} | **{rr:.3f}** | **{rs:.3f}** |")
    for t in TASKS:
        sub = [r for r in rows if r["task"] == t]
        n, rr, rs = rowset(sub)
        L.append(f"| {t.replace('google_robot_','')} | {n} | {rr:.3f} | {rs:.3f} |")

    # episode-internal centred
    L += ["", "## episode 内中心化（去掉任务/场景造成的假相关）", "",
          "| 对比 | rho | p |", "|---|---:|---:|"]
    for name, key in (("Matched vs Relative", "m_relative"), ("Matched vs Spectral", "m_spectral")):
        a, b = [], []
        for task in TASKS:
            for seed in sorted(set(r["seed"] for r in rows if r["task"] == task)):
                sub = sorted([r for r in rows if r["task"] == task and r["seed"] == seed], key=lambda r: r["step"])
                if len(sub) < 2: continue
                ma = np.array([r["m_matched"] for r in sub], float)
                mb = np.array([r[key] for r in sub], float)
                a.extend(ma - ma.mean()); b.extend(mb - mb.mean())
        rr, pp = spearman(a, b)
        L.append(f"| {name} | **{rr:.3f}** | {pp:.4f} |")

    # per-episode correlation average
    L += ["", "## 每条 episode 内的相关（逐条算再汇总）", "",
          "| 对比 | 逐条 rho 均值 | 中位 | 正相关比例 |", "|---|---:|---:|---:|"]
    for name, key in (("Matched vs Relative", "m_relative"), ("Matched vs Spectral", "m_spectral")):
        cs = []
        for task in TASKS:
            for seed in sorted(set(r["seed"] for r in rows if r["task"] == task)):
                sub = sorted([r for r in rows if r["task"] == task and r["seed"] == seed], key=lambda r: r["step"])
                if len(sub) < 4: continue
                rr, _ = spearman([r["m_matched"] for r in sub], [r[key] for r in sub])
                if np.isfinite(rr): cs.append(rr)
        cs = np.array(cs)
        L.append(f"| {name} | {cs.mean():.3f} | {np.median(cs):.3f} | {(cs>0).mean()*100:.0f}% |")

    out = {"n_states": len(rows), "mean_matched": mean_matched,
           "gamma_star": gamma_star, "tau_star": tau_star,
           "mean_relative": float(np.mean([r["m_relative"] for r in rows])),
           "mean_spectral": float(np.mean([r["m_spectral"] for r in rows])),
           "dist": {k: desc([r[k] for r in rows]) for k in ("m_matched", "m_relative", "m_spectral")},
           "rho_pooled": {"relative": spearman([r["m_matched"] for r in rows], [r["m_relative"] for r in rows]),
                          "spectral": spearman([r["m_matched"] for r in rows], [r["m_spectral"] for r in rows])},
           "gamma_scan": {g: g_stat[g][0] for g in gammas},
           "tau_scan": {t: t_stat[t][0] for t in taus},
           "calibration": {"gamma_star": gamma_star, "tau_star": tau_star}}
    (root / "OFFLINE_RESULTS.json").write_text(json.dumps(out, indent=1) + "\n")
    (root / "OFFLINE_REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
