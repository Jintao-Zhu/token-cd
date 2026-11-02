#!/usr/bin/env python3
"""Analysis + figure for the OpenVLA budget study with KMeans granularity K=8/16/32.

Every method is evaluated on the SAME task x seed grid (seeds 200-299), so all
comparisons are paired on (task, seed).  Reused arms are certified by
CONTROL_REUSE_AUDIT.json; the two new arms come from the KMeans sweep artifact.

Outputs (docs/, docs/figures/):
    OPENVLA_KMEANS_BUDGET_9TASK_TABLE.md/.tsv/.csv
    OPENVLA_KMEANS_BUDGET_9TASK_FINAL.json/.md
    figures/openvla_kmeans_budget_9task.png

Honesty rules enforced by construction:
  * every cell is successes/denominator plus percent; incomplete coverage is labelled
  * non-significance is never reported as equivalence
  * primary family (6 tests) and secondary family (18 tests) are Holm-corrected separately
  * the report states that K was explored on the same evaluation seeds

Usage:
    source scripts/activate_simpler_local.sh
    python scripts/analyze_openvla_kmeans_budget_study.py [--write] [--bootstrap 10000]
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "artifacts/openvla_lambda1_extension_200_299_v1"
CONTROLS = REPO / "artifacts/openvla_budget_rule_9task_200_299_v1"
KMEANS = REPO / "artifacts/openvla_kmeans_budget_9task_200_299_v1"
DOCS = REPO / "docs"
FIGDIR = DOCS / "figures"
SEEDS = range(200, 300)
TASKS = [
    "google_robot_open_drawer", "google_robot_close_drawer", "google_robot_pick_coke_can",
    "google_robot_move_near", "google_robot_place_apple_in_closed_top_drawer",
    "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
    "widowx_spoon_on_towel", "widowx_stack_cube",
]
TASK_SHORT = {
    "google_robot_open_drawer": "open_drawer",
    "google_robot_close_drawer": "close_drawer",
    "google_robot_pick_coke_can": "pick_coke_can",
    "google_robot_move_near": "move_near",
    "google_robot_place_apple_in_closed_top_drawer": "place_apple",
    "widowx_carrot_on_plate": "carrot",
    "widowx_put_eggplant_in_basket": "eggplant",
    "widowx_spoon_on_towel": "spoon",
    "widowx_stack_cube": "stack_cube",
}
GOOGLE5 = [t for t in TASKS if t.startswith("google_robot")]
WIDOWX4 = [t for t in TASKS if t.startswith("widowx")]

# (method key, artifact, arm directory, display label, family)
METHODS = [
    ("k8", REFERENCE, "lambda_100", "Matched K8 (baseline)", "kmeans"),
    ("k16", KMEANS, "matched_k16_l100", "Matched K16", "kmeans"),
    ("k32", KMEANS, "matched_k32_l100", "Matched K32", "kmeans"),
    ("topp75", CONTROLS, "top_p075_l100", "TopP 0.75", "topp"),
    ("topp80", CONTROLS, "top_p080_l100", "TopP 0.80", "topp"),
    ("topp85", CONTROLS, "top_p085_l100", "TopP 0.85", "topp"),
    ("topp90", CONTROLS, "top_p090_l100", "TopP 0.90", "topp"),
    ("fixed016", CONTROLS, "fixed016_l100", "Fixed 16", "fixed"),
    ("fixed024", CONTROLS, "fixed024_l100", "Fixed 24", "fixed"),
    ("fixed032", CONTROLS, "fixed032_l100", "Fixed 32", "fixed"),
    ("fixed040", CONTROLS, "fixed040_l100", "Fixed 40", "fixed"),
    ("fixed048", CONTROLS, "fixed048_l100", "Fixed 48", "fixed"),
    ("fixed056", CONTROLS, "fixed056_l100", "Fixed 56", "fixed"),
    ("fixed064", CONTROLS, "fixed064_l100", "Fixed 64", "fixed"),
]
# Primary: does finer clustering change anything relative to the incumbent K8 and the
# two strongest attention-mass baselines.  Secondary: the full Fixed/TopP sweep.
PRIMARY = [("k16", "k8"), ("k16", "topp85"), ("k16", "topp90"),
           ("k32", "k8"), ("k32", "topp85"), ("k32", "topp90")]
SECONDARY = [(new, ref) for new in ("k16", "k32")
             for ref in ("fixed016", "fixed024", "fixed032", "fixed040", "fixed048",
                         "fixed056", "fixed064", "topp75", "topp80")]


def scan_cell(job: tuple) -> tuple:
    key, artifact, arm, task = job
    directory = artifact / "episodes" / task / arm
    rows = {}
    for seed in SEEDS:
        path = directory / f"episode_{seed:03d}_summary.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:                                              # noqa: BLE001
            continue
        rows[seed] = {
            "success": bool(data.get("success")),
            "steps": int(data.get("steps", 0)),
            "m_t": data.get("mean_selected_token_count"),
            "runtime": data.get("runtime_seconds"),
            "protocol": data.get("protocol_id"),
            "technical": data.get("technical_pass") is True,
        }
    return key, task, rows


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        value = (len(pvalues) - rank) * pvalues[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_test(arm: dict[int, dict], ref: dict[int, dict], seeds: list[int]) -> dict:
    """McNemar on discordant pairs plus a task-stratified paired bootstrap CI."""
    rescue = harm = both = neither = 0
    for seed in seeds:
        a, b = arm[seed]["success"], ref[seed]["success"]
        if a and not b:
            rescue += 1
        elif b and not a:
            harm += 1
        elif a and b:
            both += 1
        else:
            neither += 1
    discordant = rescue + harm
    if discordant:
        pvalue = float(binomtest(min(rescue, harm), discordant, 0.5, alternative="two-sided").pvalue)
    else:
        pvalue = 1.0
    return {"n_pairs": len(seeds), "both_success": both, "both_failure": neither,
            "rescue_by_arm": rescue, "harm_to_arm": harm, "net": rescue - harm,
            "arm_successes": both + rescue, "reference_successes": both + harm,
            "rate_difference_pp": 100.0 * (rescue - harm) / len(seeds) if seeds else None,
            "mcnemar_exact_p": pvalue}


def stratified_bootstrap(arm: dict[str, dict[int, dict]], ref: dict[str, dict[int, dict]],
                         tasks: list[str], rng: np.random.Generator, draws: int) -> dict:
    """Resample seeds WITHIN each task, so the task mix is held fixed."""
    per_task = {}
    for task in tasks:
        seeds = sorted(set(arm[task]) & set(ref[task]))
        a = np.array([arm[task][s]["success"] for s in seeds], dtype=np.float64)
        b = np.array([ref[task][s]["success"] for s in seeds], dtype=np.float64)
        per_task[task] = (a, b)
    obs = float(np.mean([a.mean() - b.mean() for a, b in per_task.values()])) * 100.0
    if draws <= 0:
        return {"observed_pp": obs, "ci_low_pp": None, "ci_high_pp": None, "draws": 0}
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        diffs = []
        for a, b in per_task.values():
            pick = rng.integers(0, len(a), len(a))
            diffs.append(a[pick].mean() - b[pick].mean())
        samples[index] = np.mean(diffs) * 100.0
    return {"observed_pp": obs,
            "ci_low_pp": float(np.percentile(samples, 2.5)),
            "ci_high_pp": float(np.percentile(samples, 97.5)),
            "draws": draws}


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    half = z * np.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def make_figure(table: dict, comparisons: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [m[0] for m in METHODS]
    labels = {m[0]: m[3] for m in METHODS}
    family = {m[0]: m[4] for m in METHODS}
    colours = {"kmeans": "#1f4e9c", "topp": "#c8641c", "fixed": "#6d6d6d"}
    pooled = {key: (sum(table[key][t]["success"] for t in TASKS),
                    sum(table[key][t]["n"] for t in TASKS)) for key in order}

    figure, axes = plt.subplots(1, 2, figsize=(15.0, 7.2))
    axis = axes[0]
    y = np.arange(len(order))[::-1]
    for position, key in zip(y, order):
        successes, total = pooled[key]
        rate = 100.0 * successes / total if total else 0.0
        low, high = wilson(successes, total)
        axis.barh(position, rate, color=colours[family[key]], alpha=0.85,
                  edgecolor="black" if key == "k8" else "none",
                  linewidth=1.8 if key == "k8" else 0.0)
        axis.errorbar(rate, position,
                      xerr=[[rate - 100 * low], [100 * high - rate]],
                      fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
        axis.text(rate + 1.2, position, f"{successes}/{total}  {rate:.1f}%",
                  va="center", fontsize=8.5)
    axis.set_yticks(y, [labels[k] for k in order], fontsize=9)
    axis.set_xlabel("pooled success rate (%) on 9 tasks x 100 seeds")
    axis.set_xlim(0, 68)
    axis.set_title("(a) Pooled success by budget rule (Wilson 95% CI)", fontsize=11)
    axis.grid(axis="x", alpha=0.25)

    axis = axes[1]
    lookup = {(c["arm"], c["reference"]): c for c in comparisons}
    entries = [(key, "k8") for key in order if key != "k8"]
    positions = np.arange(len(entries))[::-1]
    for position, (key, reference) in zip(positions, entries):
        comparison = lookup.get((key, reference))
        if comparison is None:
            continue
        observed = comparison["bootstrap"]["observed_pp"]
        low = comparison["bootstrap"]["ci_low_pp"]
        high = comparison["bootstrap"]["ci_high_pp"]
        axis.errorbar(observed, position,
                      xerr=[[observed - low], [high - observed]] if low is not None else None,
                      fmt="o", color=colours[family[key]], capsize=3, markersize=5)
        marker = " *" if comparison["holm_p"] < 0.05 else ""
        axis.text(high + 0.4 if high is not None else observed, position,
                  f"{observed:+.1f}pp{marker}", va="center", fontsize=8.5)
    axis.axvline(0, color="black", linewidth=1.0)
    axis.set_yticks(positions, [labels[k] for k, _ in entries], fontsize=9)
    axis.set_xlabel("paired difference vs Matched K8 (percentage points),\n"
                    "task-stratified paired bootstrap 95% CI")
    axis.set_title("(b) Every rule against the K8 incumbent on identical task/seed pairs",
                   fontsize=11)
    axis.grid(axis="x", alpha=0.25)
    axis.text(0.99, 0.02, "* Holm-adjusted p < 0.05 in its declared family",
              transform=axis.transAxes, ha="right", va="bottom", fontsize=8, style="italic")

    figure.suptitle("OpenVLA lambda=1.0: KMeans granularity (8/16/32) versus attention-mass "
                    "and fixed budgets, 9 SIMPLER tasks, seeds 200-299", fontsize=12.5)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    jobs = [(key, artifact, arm, task) for key, artifact, arm, _, _ in METHODS for task in TASKS]
    started = time.time()
    raw: dict[str, dict[str, dict[int, dict]]] = defaultdict(dict)
    with mp.Pool(args.jobs) as pool:
        for key, task, rows in pool.imap_unordered(scan_cell, jobs):
            raw[key][task] = rows
    print(f"scanned {len(jobs)} cells in {time.time() - started:.1f}s", flush=True)

    table, coverage = {}, {}
    for key, _, _, label, family in METHODS:
        table[key] = {}
        for task in TASKS:
            rows = raw[key].get(task, {})
            successes = sum(int(r["success"]) for r in rows.values())
            table[key][task] = {
                "n": len(rows), "success": successes,
                "rate": (successes / len(rows)) if rows else None,
                "mean_m_t": (float(np.mean([r["m_t"] for r in rows.values() if r["m_t"]]))
                             if any(r["m_t"] for r in rows.values()) else None),
                "mean_runtime_seconds": (float(np.mean([r["runtime"] for r in rows.values()
                                                        if r["runtime"]]))
                                         if any(r["runtime"] for r in rows.values()) else None),
                "all_technical_pass": all(r["technical"] for r in rows.values()) if rows else False,
            }
        coverage[key] = {task: len(raw[key].get(task, {})) for task in TASKS}
    complete = all(count == 100 for arm in coverage.values() for count in arm.values())

    rng = np.random.default_rng(20261016)
    comparisons = []
    for family_name, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)):
        pvalues = []
        staged = []
        for arm_key, reference_key in pairs:
            pool_arm = {t: raw[arm_key][t] for t in TASKS}
            pool_ref = {t: raw[reference_key][t] for t in TASKS}
            shared = {t: sorted(set(pool_arm[t]) & set(pool_ref[t])) for t in TASKS}
            flat_arm = {t: {s: pool_arm[t][s] for s in shared[t]} for t in TASKS}
            flat_ref = {t: {s: pool_ref[t][s] for s in shared[t]} for t in TASKS}
            merged_arm = {s: row for t in TASKS for s, row in flat_arm[t].items()}
            merged_ref = {s: row for t in TASKS for s, row in flat_ref[t].items()}
            seeds = sorted(merged_arm)
            test = paired_test(merged_arm, merged_ref, seeds)
            test.update({"arm": arm_key, "reference": reference_key, "family": family_name,
                         "arm_label": dict((m[0], m[3]) for m in METHODS)[arm_key],
                         "reference_label": dict((m[0], m[3]) for m in METHODS)[reference_key],
                         "shared_pairs_per_task": {TASK_SHORT[t]: len(shared[t]) for t in TASKS},
                         "per_task": {}})
            for task in TASKS:
                if shared[task]:
                    test["per_task"][TASK_SHORT[task]] = paired_test(flat_arm[task], flat_ref[task],
                                                                    shared[task])
            staged.append(test)
            pvalues.append(test["mcnemar_exact_p"])
        for test, adjusted in zip(staged, holm(pvalues)):
            test["holm_p"] = adjusted
            test["holm_family_size"] = len(pvalues)
        comparisons.extend(staged)

    bootstrapped = []
    for test in comparisons:
        arm_key, reference_key = test["arm"], test["reference"]
        bootstrap = stratified_bootstrap({t: raw[arm_key][t] for t in TASKS},
                                        {t: raw[reference_key][t] for t in TASKS},
                                        TASKS, rng, args.bootstrap)
        test["bootstrap"] = bootstrap
        test["rate_difference_pp_flat"] = test["rate_difference_pp"]
    bootstrapped = comparisons

    pooled = {}
    for key, _, _, label, family in METHODS:
        successes = sum(table[key][t]["success"] for t in TASKS)
        total = sum(table[key][t]["n"] for t in TASKS)
        low, high = wilson(successes, total)
        pooled[key] = {"label": label, "family": family, "successes": successes, "n": total,
                       "rate": successes / total if total else None,
                       "wilson_low": low, "wilson_high": high,
                       "google5": [sum(table[key][t]["success"] for t in GOOGLE5),
                                   sum(table[key][t]["n"] for t in GOOGLE5)],
                       "widowx4": [sum(table[key][t]["success"] for t in WIDOWX4),
                                   sum(table[key][t]["n"] for t in WIDOWX4)]}

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "scripts/analyze_openvla_kmeans_budget_study.py",
        "complete_coverage": complete,
        "coverage": coverage,
        "tasks": TASKS,
        "seeds": [SEEDS[0], SEEDS[-1]],
        "methods": [{"key": k, "label": l, "family": f, "artifact": str(a.relative_to(REPO)),
                     "arm": arm} for k, a, arm, l, f in METHODS],
        "pooled": pooled,
        "per_task": {k: {TASK_SHORT[t]: table[k][t] for t in TASKS} for k in table},
        "comparisons": bootstrapped,
        "budget_diagnostics": {k: {TASK_SHORT[t]: table[k][t]["mean_m_t"] for t in TASKS} for k in table},
        "caveats": [
            "K8/16/32 are evaluated on the same seeds 200-299 that define the K8 reference, "
            "so K is an exploratory hyper-parameter choice here, not an independently confirmed one.",
            "K is the number of clusters, not a token count: a different K changes the matched "
            "union size, so this ablation moves the budget rule rather than preserving m_t.",
            "Non-significance is not equivalence; every non-significant comparison is reported "
            "with its observed difference and interval.",
            "Reused arms are certified only for integrity/completeness by CONTROL_REUSE_AUDIT.json.",
            "Closed-loop mean m_t values come from divergent trajectories and are not same-state "
            "causal evidence about budget or cost.",
            "Holm families: primary = 6 tests, secondary = 18 tests, corrected separately.",
        ],
    }

    if args.write:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "OPENVLA_KMEANS_BUDGET_9TASK_FINAL.json").write_text(
            json.dumps(payload, indent=2, default=float) + "\n")
        write_table(table, pooled, complete)
        write_report(payload)
        make_figure(table, bootstrapped, FIGDIR / "openvla_kmeans_budget_9task.png")
        print("wrote docs/OPENVLA_KMEANS_BUDGET_9TASK_FINAL.{json,md}, "
              "docs/OPENVLA_KMEANS_BUDGET_9TASK_TABLE.{md,tsv,csv}, "
              "docs/figures/openvla_kmeans_budget_9task.png")
    else:
        print(json.dumps({"complete": complete, "pooled": {k: [v["successes"], v["n"]]
                                                          for k, v in pooled.items()}}, indent=2))
    return 0


def write_table(table: dict, pooled: dict, complete: bool) -> None:
    """Tasks as rows, methods as columns, every cell successes/denominator and percent."""
    header = ["task", "n"] + [m[3] for m in METHODS]
    rows = []
    for task in TASKS:
        row = [TASK_SHORT[task], "100"]
        for key, _, _, _, _ in METHODS:
            cell = table[key][task]
            rate = 100.0 * cell["rate"] if cell["rate"] is not None else float("nan")
            row.append(f"{cell['success']}/{cell['n']} ({rate:.1f}%)")
        rows.append(row)
    for label, keys in (("Google (5 tasks)", GOOGLE5), ("WidowX (4 tasks)", WIDOWX4)):
        row = [label, str(100 * len(keys))]
        for key, _, _, _, _ in METHODS:
            successes = sum(table[key][t]["success"] for t in keys)
            total = sum(table[key][t]["n"] for t in keys)
            row.append(f"{successes}/{total} ({100.0 * successes / total:.1f}%)" if total else "n/a")
        rows.append(row)
    row = ["All (9 tasks)", "900"]
    for key, _, _, _, _ in METHODS:
        entry = pooled[key]
        if entry["n"]:
            row.append(f"{entry['successes']}/{entry['n']} ({100.0 * entry['rate']:.1f}%)")
        else:
            row.append(f"{entry['successes']}/{entry['n']} (n/a)")
    rows.append(row)

    tsv = DOCS / "OPENVLA_KMEANS_BUDGET_9TASK_TABLE.tsv"
    with tsv.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)
    with (DOCS / "OPENVLA_KMEANS_BUDGET_9TASK_TABLE.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)

    def block(title: str, keys: list[str]) -> str:
        lines = ["| task | n | " + " | ".join(pooled[k]["label"] for k in keys) + " |",
                 "|---|---|" + "---|" * len(keys)]
        for task in TASKS:
            cells = []
            for k in keys:
                cell = table[k][task]
                rate = 100.0 * cell["rate"] if cell["rate"] is not None else float("nan")
                cells.append(f"{cell['success']}/{cell['n']} ({rate:.1f}%)")
            lines.append(f"| {TASK_SHORT[task]} | 100 | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    kmeans_keys = [k for k, _, _, _, f in METHODS if f == "kmeans"]
    topp_keys = [k for k, _, _, _, f in METHODS if f == "topp"]
    fixed_keys = [k for k, _, _, _, f in METHODS if f == "fixed"]
    text = [
        "# OpenVLA lambda=1.0 budget rules: KMeans granularity vs TopP vs Fixed",
        "",
        f"Coverage complete: **{complete}** | seeds 200-299 | successes/denominator and percent.",
        "",
        "## KMeans granularity with the attention-mass baselines",
        "",
        block("kmeans+topp", kmeans_keys + topp_keys),
        "",
        "## KMeans granularity with the fixed budgets",
        "",
        block("kmeans+fixed", kmeans_keys + fixed_keys),
        "",
        "## Pooled",
        "",
        "| method | Google 5 tasks | WidowX 4 tasks | all 9 tasks |",
        "|---|---|---|---|",
    ]
    for key, _, _, _, _ in METHODS:
        entry = pooled[key]
        g, gn = entry["google5"]
        w, wn = entry["widowx4"]
        google = f"{g}/{gn} ({100.0 * g / gn:.1f}%)" if gn else f"{g}/{gn} (n/a)"
        widowx = f"{w}/{wn} ({100.0 * w / wn:.1f}%)" if wn else f"{w}/{wn} (n/a)"
        total = (f"{entry['successes']}/{entry['n']} ({100.0 * entry['rate']:.1f}%)"
                 if entry["n"] else f"{entry['successes']}/{entry['n']} (n/a)")
        text.append(f"| {entry['label']} | {google} | {widowx} | {total} |")
    (DOCS / "OPENVLA_KMEANS_BUDGET_9TASK_TABLE.md").write_text("\n".join(text) + "\n")


def write_report(payload: dict) -> None:
    pooled = payload["pooled"]
    lines = [
        "# OpenVLA KMeans-granularity budget study (lambda=1.0, 9 tasks, seeds 200-299)",
        "",
        f"Generated {payload['generated_at']} by `{payload['generator']}`.",
        f"Coverage complete: **{payload['complete_coverage']}** "
        "(any incomplete cell is visible in the table; nothing is silently dropped).",
        "",
        "## Pooled success",
        "",
        "| method | family | successes/denominator | percent | Wilson 95% |",
        "|---|---|---|---|---|",
    ]
    for key, entry in pooled.items():
        percent = f"{100.0 * entry['rate']:.2f}%" if entry["n"] else "n/a"
        wilson_text = (f"[{100 * entry['wilson_low']:.1f}, {100 * entry['wilson_high']:.1f}]"
                       if entry["n"] else "n/a")
        lines.append(f"| {entry['label']} | {entry['family']} | {entry['successes']}/{entry['n']} | "
                     f"{percent} | {wilson_text} |")
    lines += ["", "## Paired comparisons on identical (task, seed) pairs", "",
              "Rescue = the first method succeeds where the second fails; harm = the reverse. "
              "Holm is applied within each declared family (primary 6, secondary 18).", "",
              "| first | second | family | pairs | first | second | rescue | harm | diff (pp) | "
              "bootstrap 95% CI | McNemar p | Holm p |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for test in payload["comparisons"]:
        boot = test["bootstrap"]
        lines.append(
            f"| {test['arm_label']} | {test['reference_label']} | {test['family']} | "
            f"{test['n_pairs']} | {test['arm_successes']} | {test['reference_successes']} | "
            f"{test['rescue_by_arm']} | {test['harm_to_arm']} | "
            f"{boot['observed_pp']:+.2f} | "
            f"[{boot['ci_low_pp']:+.2f}, {boot['ci_high_pp']:+.2f}] | "
            f"{test['mcnemar_exact_p']:.4g} | {test['holm_p']:.4g} |")
    lines += ["", "## Mean selected token count m_t (episode-balanced, closed loop)", "",
              "| task | " + " | ".join(entry["label"] for entry in pooled.values()) + " |",
              "|---|" + "---|" * len(pooled)]
    for task, short in TASK_SHORT.items():
        cells = []
        for key in pooled:
            value = payload["budget_diagnostics"][key][short]
            cells.append("n/a" if value is None else f"{value:.1f}")
        lines.append(f"| {short} | " + " | ".join(cells) + " |")
    lines += ["", "## Caveats that must travel with these numbers", ""]
    lines += [f"- {note}" for note in payload["caveats"]]
    lines += ["", "Figure: `docs/figures/openvla_kmeans_budget_9task.png` "
                  "(pooled rates with Wilson CI; paired differences vs K8 with bootstrap CI).", ""]
    (DOCS / "OPENVLA_KMEANS_BUDGET_9TASK_FINAL.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
