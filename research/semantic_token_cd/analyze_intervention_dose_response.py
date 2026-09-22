"""Analyze the 800-state intervention-dose response scan.

Questions:
  * does one unit r=m/m_matched produce comparable perturbation across states?
  * does Fixed34 over/under-intervene as a function of the state's matched budget?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

TASKS = ("google_robot_open_drawer", "google_robot_close_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near")
SHORT = {t: t.replace("google_robot_", "") for t in TASKS}
METRICS = ("D_feat", "D_action", "D_support")
FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)


def load_rows(root: Path):
    rows = []
    for task in TASKS:
        for path in sorted((root / "dose" / task).glob("seed_*.json")):
            d = json.loads(path.read_text())
            for st in d["states"]:
                curve = {float(x["m_ratio"]): x for x in st["curve"]}
                # Source tags are more robust than rounded ratios for small m.
                by_source = {}
                for x in st["curve"]:
                    for source in x.get("sources", []):
                        by_source[source] = x
                row = {"task": task, "seed": int(d["seed"]), "step": int(st["step"]),
                       "m0": int(st["matched_m"]), "entity_m": int(st["entity_m"]), "curve": st["curve"]}
                for factor in FACTORS:
                    row[f"rel_{factor:g}"] = by_source.get(f"relative_{factor:g}")
                row["fixed34"] = by_source.get("fixed34")
                row["entity"] = by_source.get("entity")
                rows.append(row)
    if len(rows) != 800:
        raise RuntimeError(f"expected 800 states, got {len(rows)}")
    return rows


def mean_metric(rows, key, metric):
    vals = [r[key][metric] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def group_rows(rows, task=None):
    sub = [r for r in rows if task is None or r["task"] == task]
    sub = sorted(sub, key=lambda r: r["m0"])
    n = len(sub); a = n // 3; b = 2 * n // 3
    return {"small": sub[:a], "middle": sub[a:b], "large": sub[b:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    args = ap.parse_args()
    root = args.artifact.resolve()
    rows = load_rows(root)
    lines = ["# Intervention-dose response analysis", "",
             "800 states, each scanned with 5 relative budgets plus Fixed34 and Entity-TopP.", "",
             "Metrics: `D_feat` visual feature change, `D_action` action JS divergence, `D_support` clean-action log-probability drop.", ""]

    # Overall monotonicity and curve collapse by matched-m tercile.
    lines += ["## Matched-relative response", "",
              "| r = m/m0 | Small | Middle | Large |", "|---|---:|---:|---:|"]
    report = {"groups": {}, "overall": {}}
    for metric in METRICS:
        lines += [f"### {metric}", "", "| r | Small | Middle | Large |", "|---|---:|---:|---:|"]
        groups = group_rows(rows)
        for factor in FACTORS:
            key = f"rel_{factor:g}"
            vals = {g: mean_metric(sub, key, metric) for g, sub in groups.items()}
            lines.append(f"| {factor:g} | {vals['small']:.4f} | {vals['middle']:.4f} | {vals['large']:.4f} |")
            report["groups"].setdefault(metric, {})[f"{factor:g}"] = vals
        lines.append("")

    # Fixed34 and Entity: actual perturbation and dispersion across m0 groups.
    lines += ["## Fixed34 / Entity vs Matched scale", "",
              "| Budget | Metric | Overall mean | Overall std | Small mean | Middle mean | Large mean |", "|---|---:|---:|---:|---:|---:|---:|"]
    groups = group_rows(rows)
    report["budgets"] = {}
    for key, label in (("rel_1", "Matched r=1"), ("fixed34", "Fixed34"), ("entity", "Entity-TopP")):
        for metric in METRICS:
            vals = [r[key][metric] for r in rows if r.get(key) is not None]
            means = {g: mean_metric(sub, key, metric) for g, sub in groups.items()}
            lines.append(f"| {label} | {metric} | {np.mean(vals):.4f} | {np.std(vals):.4f} | {means['small']:.4f} | {means['middle']:.4f} | {means['large']:.4f} |")
            report["budgets"].setdefault(label, {})[metric] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)), **means,
            }
    # Per-task matched-r response at r=1: the cleanest collapse check.
    lines += ["", "## r=1 by task and matched-m tercile", "", "| Task | Size | m0 mean | D_feat | D_action | D_support |", "|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        groups = group_rows(rows, task)
        for g, sub in groups.items():
            lines.append(f"| {SHORT[task]} | {g} | {np.mean([r['m0'] for r in sub]):.1f} | "
                         f"{mean_metric(sub, 'rel_1', 'D_feat'):.4f} | "
                         f"{mean_metric(sub, 'rel_1', 'D_action'):.4f} | "
                         f"{mean_metric(sub, 'rel_1', 'D_support'):.4f} |")

    # Direct correlation of Fixed34/Entity perturbation with m0:
    # negative => small-m states over-intervened, large-m states under-intervened.
    lines += ["", "## Budget-vs-state trend", "", "| Budget | Metric | Spearman rho vs m0 |", "|---|---:|---:|"]
    for key, label in (("fixed34", "Fixed34"), ("entity", "Entity-TopP")):
        for metric in METRICS:
            x = [r["m0"] for r in rows if r.get(key) is not None]
            y = [r[key][metric] for r in rows if r.get(key) is not None]
            rho = stats.spearmanr(x, y).statistic if len(set(x)) > 1 else float("nan")
            lines.append(f"| {label} | {metric} | {rho:.4f} |")
            report["budgets"].setdefault(label, {})[metric]["rho_vs_m0"] = float(rho)

    (root / "DOSE_RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "DOSE_REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
