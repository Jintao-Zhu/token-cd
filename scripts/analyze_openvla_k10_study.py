#!/usr/bin/env python3
"""Final paired analysis for the real K=10 run (lambda 0.5 and 1.0).

Every K=10 episode is paired with the SAME (task, seed) episode of the grid reference,
so all contrasts are within-pair. Reported for each lambda:
  * successes/denominator and percent per task and pooled
  * versus matched K8 (the primary contrast) and versus fixed016 (an under-budget reference)
  * rescue/harm, McNemar exact p, Holm within the lambda's declared family
  * task-stratified paired bootstrap CI of the success-rate difference
  * measured mean m_t, i.e. how much budget K=10 actually gave away

Outputs: docs/OPENVLA_K10_2LAMBDA_FINAL.{json,md} and
         docs/figures/openvla_k10_2lambda.png
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
K10 = REPO / "artifacts/openvla_k10_2lambda_200_299_v1"
GRID = REPO / "artifacts/openvla_lambda_budget_grid_200_299_v1"
DOCS = REPO / "docs"
FIGDIR = DOCS / "figures"
TASKS = ["google_robot_close_drawer", "google_robot_open_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near"]
SHORT = {"google_robot_close_drawer": "close_drawer", "google_robot_open_drawer": "open_drawer",
         "google_robot_pick_coke_can": "pick_coke_can", "google_robot_move_near": "move_near"}
CONDITIONS = [
    ("050", 0.5, "matched_k10_l050", [("matched", "matched_l050"), ("fixed016", "fixed016_l050")]),
    ("100", 1.0, "matched_k10_l100", [("matched", "matched_l100"), ("fixed016", "fixed016_l100")]),
]
EXPECTED = 400  # per arm: 4 tasks x 100 seeds


def scan(job: tuple) -> tuple:
    path, task, arm = job
    rows = {}
    for summary in glob.glob(str(Path(path) / "episodes" / task / arm / "episode_*_summary.json")):
        data = json.loads(Path(summary).read_text())
        rows[int(data["seed"])] = {"success": bool(data["success"]),
                                   "m_t": data.get("mean_selected_token_count"),
                                   "steps": int(data.get("steps", 0))}
    return task, arm, rows


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def contrast(arm: dict, reference: dict, tasks: list) -> dict:
    rescue = harm = both = neither = 0
    strata = []
    for task in tasks:
        seeds = sorted(set(arm.get(task, {})) & set(reference.get(task, {})))
        if not seeds:
            continue
        a = np.array([arm[task][s]["success"] for s in seeds], dtype=np.float64)
        b = np.array([reference[task][s]["success"] for s in seeds], dtype=np.float64)
        strata.append((a, b))
        for s in seeds:
            x, y = arm[task][s]["success"], reference[task][s]["success"]
            if x and not y:
                rescue += 1
            elif y and not x:
                harm += 1
            elif x and y:
                both += 1
            else:
                neither += 1
    total = rescue + harm + both + neither
    pvalue = (float(binomtest(min(rescue, harm), rescue + harm, 0.5, alternative="two-sided").pvalue)
              if rescue + harm else 1.0)
    return {"n_pairs": total, "both_success": both, "both_failure": neither,
            "rescue_by_arm": rescue, "harm_to_arm": harm, "net": rescue - harm,
            "arm_successes": both + rescue, "reference_successes": both + harm,
            "arm_rate": (both + rescue) / total if total else None,
            "reference_rate": (both + harm) / total if total else None,
            "diff_pp": 100.0 * (rescue - harm) / total if total else None,
            "mcnemar_exact_p": pvalue, "_strata": strata}


def bootstrap(strata: list, rng: np.random.Generator, draws: int) -> dict:
    if not strata:
        return {"observed_pp": None, "ci_low_pp": None, "ci_high_pp": None}
    observed = float(np.mean([a.mean() - b.mean() for a, b in strata])) * 100.0
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        samples[index] = np.mean([a[rng.integers(0, len(a), len(a))].mean()
                                  - b[rng.integers(0, len(b), len(b))].mean()
                                  for a, b in strata]) * 100.0
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"observed_pp": observed, "ci_low_pp": float(low), "ci_high_pp": float(high),
            "draws": draws}


def mean_m_t(rows: dict, tasks: list) -> dict:
    out = {}
    for task in tasks:
        values = [v["m_t"] for v in rows.get(task, {}).values() if v.get("m_t")]
        out[task] = float(np.mean(values)) if values else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261016)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    jobs = []
    for _, _, arm, references in CONDITIONS:
        for task in TASKS:
            jobs.append((K10, task, arm))
        for _, ref in references:
            for task in TASKS:
                jobs.append((GRID, task, ref))
    started = time.time()
    data: dict[tuple[str, str], dict] = {}
    with mp.Pool(args.jobs) as pool:
        for task, arm, rows in pool.imap_unordered(scan, jobs):
            data[(arm, task)] = rows
    print(f"scanned {len(jobs)} cells in {time.time() - started:.1f}s", flush=True)

    payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "generator": "scripts/analyze_openvla_k10_study.py", "conditions": {}}
    for tag, strength, arm, references in CONDITIONS:
        arm_rows = {task: data.get((arm, task), {}) for task in TASKS}
        coverage = {SHORT[t]: len(arm_rows[t]) for t in TASKS}
        entry = {"lambda": strength, "arm": arm, "coverage": coverage,
                 "complete": all(len(arm_rows[t]) == 100 for t in TASKS),
                 "expected_per_task": 100,
                 "arm_successes": sum(int(v["success"]) for t in TASKS for v in arm_rows[t].values()),
                 "arm_total": sum(len(arm_rows[t]) for t in TASKS),
                 "arm_mean_m_t": mean_m_t(arm_rows, TASKS), "contrasts": {}}
        entry["arm_rate"] = (entry["arm_successes"] / entry["arm_total"]) if entry["arm_total"] else None
        for name, ref_arm in references:
            reference_rows = {task: data.get((ref_arm, task), {}) for task in TASKS}
            result = contrast(arm_rows, reference_rows, TASKS)
            strata = result.pop("_strata")
            result["bootstrap"] = bootstrap(strata, rng, args.bootstrap)
            result["reference"] = name
            result["reference_arm"] = ref_arm
            result["reference_successes_total"] = sum(
                len(v) for v in reference_rows.values())
            result["reference_mean_m_t"] = mean_m_t(reference_rows, TASKS)
            result["per_task"] = {}
            for task in TASKS:
                per = contrast({task: arm_rows[task]}, {task: reference_rows[task]}, [task])
                per.pop("_strata")
                result["per_task"][SHORT[task]] = per
            entry["contrasts"][name] = result
        # Holm within this lambda's family (matched + fixed016 contrasts)
        keys = list(entry["contrasts"])
        for key, adjusted in zip(keys, holm([entry["contrasts"][k]["mcnemar_exact_p"] for k in keys])):
            entry["contrasts"][key]["holm_p"] = adjusted
            entry["contrasts"][key]["holm_family_size"] = len(keys)
        payload["conditions"][tag] = entry

    payload["caveats"] = [
        "K=10 is evaluated on the same seeds 200-299 as the reference, and was chosen after "
        "seeing the probe: this is an exploratory hyper-parameter check, not a pre-registered "
        "confirmatory test.",
        "K changes the matched union size, so this run changes the budget as well as the "
        "clustering; the measured m_t columns quantify how much budget was given away.",
        "Non-significance is not equivalence: every contrast is reported with its observed "
        "difference and a task-stratified paired bootstrap interval.",
        "The fixed016 contrast is a secondary, descriptive reference for the under-budget "
        "regime; the primary contrast is against matched K8 at the same lambda.",
    ]
    if args.write:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "OPENVLA_K10_2LAMBDA_FINAL.json").write_text(
            json.dumps(payload, indent=2, default=float) + "\n")
        report(payload)
        figure(payload, FIGDIR / "openvla_k10_2lambda.png")
        print("wrote docs/OPENVLA_K10_2LAMBDA_FINAL.{json,md} and docs/figures/openvla_k10_2lambda.png")
    else:
        print(json.dumps({tag: {"complete": c["complete"], "n": c["arm_total"],
                                "arm_successes": c["arm_successes"],
                                "vs_matched": {k: v for k, v in
                                               c["contrasts"]["matched"].items()
                                               if k in ("n_pairs", "rescue_by_arm", "harm_to_arm",
                                                        "diff_pp", "mcnemar_exact_p", "holm_p")}}
                          for tag, c in payload["conditions"].items()}, indent=2, default=float))
    return 0


def report(payload: dict) -> None:
    lines = ["# K=10 Matched at lambda 0.5 and 1.0: paired results against the grid reference",
             "",
             f"Generated {payload['generated_at']}. Every row is paired on identical (task, seed).",
             ""]
    for tag, entry in payload["conditions"].items():
        lines += [f"## lambda = {entry['lambda']}", "",
                  f"coverage per task: {entry['coverage']} (expected 100 each, complete: "
                  f"**{entry['complete']}**)", "",
                  f"K=10 arm: {entry['arm_successes']}/{entry['arm_total']} "
                  f"({100 * entry['arm_rate']:.1f}%)" if entry["arm_rate"] is not None else "n/a",
                  "", "| contrast | pairs | K=10 | reference | rescue | harm | diff (pp) | "
                  "bootstrap 95% CI | McNemar p | Holm p |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, c in entry["contrasts"].items():
            boot = c["bootstrap"]
            lines.append(
                f"| vs {name} | {c['n_pairs']} | {c['arm_successes']} | {c['reference_successes']} | "
                f"{c['rescue_by_arm']} | {c['harm_to_arm']} | {c['diff_pp']:+.2f} | "
                f"[{boot['ci_low_pp']:+.2f}, {boot['ci_high_pp']:+.2f}] | "
                f"{c['mcnemar_exact_p']:.4g} | {c['holm_p']:.4g} |")
        lines += ["", "per task (versus matched at the same lambda):", "",
                  "| task | n | K=10 | matched | net | m_t K=10 | m_t matched |",
                  "|---|---|---|---|---|---|---|"]
        primary = entry["contrasts"]["matched"]
        for task, per in primary["per_task"].items():
            lines.append(f"| {task} | {per['n_pairs']} | {per['arm_successes']} | "
                         f"{per['reference_successes']} | {per['net']:+d} | "
                         f"{entry['arm_mean_m_t'][[k for k, v in SHORT.items() if v == task][0]]:.1f} | "
                         f"{primary['reference_mean_m_t'][[k for k, v in SHORT.items() if v == task][0]]:.1f} |")
        lines.append("")
    lines += ["## Caveats", ""] + [f"- {note}" for note in payload["caveats"]]
    lines += ["", "Figure: `docs/figures/openvla_k10_2lambda.png`.", ""]
    (DOCS / "OPENVLA_K10_2LAMBDA_FINAL.md").write_text("\n".join(lines))


def figure(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = list(payload["conditions"].values())
    figure_, axes = plt.subplots(1, len(conditions), figsize=(7.2 * len(conditions), 6.0), squeeze=False)
    for axis, entry in zip(axes[0], conditions):
        tasks = list(entry["contrasts"]["matched"]["per_task"])
        positions = np.arange(len(tasks))
        width = 0.36
        arm_values, ref_values, arm_err, ref_err = [], [], [], []
        for task in tasks:
            per = entry["contrasts"]["matched"]["per_task"][task]
            n = per["n_pairs"] or 1
            a = 100.0 * per["arm_successes"] / n
            b = 100.0 * per["reference_successes"] / n
            arm_values.append(a)
            ref_values.append(b)
            arm_err.append(100.0 * np.sqrt(a / 100 * (1 - a / 100) / n))
            ref_err.append(100.0 * np.sqrt(b / 100 * (1 - b / 100) / n))
        axis.bar(positions - width / 2, arm_values, width, yerr=arm_err, capsize=3,
                 color="#a02c2c", label="K=10")
        axis.bar(positions + width / 2, ref_values, width, yerr=ref_err, capsize=3,
                 color="#1f4e9c", label="matched K8")
        for position, a, b in zip(positions, arm_values, ref_values):
            axis.text(position, max(a, b) + 2.0, f"{a - b:+.0f}pp", ha="center", fontsize=8)
        axis.set_xticks(positions, tasks, fontsize=9, rotation=12)
        axis.set_ylabel("success rate (%)")
        axis.set_ylim(0, 100)
        contrast = entry["contrasts"]["matched"]
        boot = contrast["bootstrap"]
        axis.set_title(f"lambda = {entry['lambda']}   n={contrast['n_pairs']} paired\n"
                       f"pooled diff {contrast['diff_pp']:+.1f}pp "
                       f"[{boot['ci_low_pp']:+.1f}, {boot['ci_high_pp']:+.1f}], "
                       f"McNemar p={contrast['mcnemar_exact_p']:.3f}", fontsize=11)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=9)
    figure_.suptitle("Does K=10 help? Real closed-loop K=10 versus matched K8 at the same lambda "
                     "(4 Google tasks, seeds 200-299)", fontsize=12.5)
    figure_.tight_layout(rect=(0, 0, 1, 0.92))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figure_.savefig(path, dpi=165)
    plt.close(figure_)


if __name__ == "__main__":
    raise SystemExit(main())
