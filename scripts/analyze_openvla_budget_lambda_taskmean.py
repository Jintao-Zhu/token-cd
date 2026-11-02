#!/usr/bin/env python3
"""Composition-corrected analysis of the lambda x budget rule grid and the
task-mean (no per-step adaptation) ablation.

Covers, with paired task/seed design throughout:

  A. grid: 5 budget rules x lambda {0.5, 1.0} x 4 Google tasks x seeds 200-299
       - rates per (rule, lambda) with counts/denominators
       - per-rule paired contrast against the matched rule, WITHIN each lambda
       - within-rule lambda effect (0.5 -> 1.0)
       - the lambda x budget-rule INTERACTION itself, estimated directly instead of
         inferring it from "significant here, not significant there"
  B. task-mean ablation at both lambdas on the same grid tasks/seeds
  C. independent seed-range replication of the lambda=0.5 task-mean contrast (seeds 100-199)

Outputs: docs/OPENVLA_BUDGET_LAMBDA_TASKMEAN_FINAL.{json,md},
         docs/figures/openvla_budget_lambda_taskmean.png

Usage:
    source scripts/activate_simpler_local.sh
    python scripts/analyze_openvla_budget_lambda_taskmean.py [--write] [--bootstrap 10000]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "artifacts/openvla_lambda_budget_grid_200_299_v1"
TASKMEAN_NEW = REPO / "artifacts/openvla_taskmean_2lambda_200_299_v1"
OLD_MATCHED = REPO / "artifacts/prompt_attn_l11_token_count_v1"
OLD_TASKMEAN = REPO / "artifacts/l11_taskmean_budget_100_199_v1"
DOCS = REPO / "docs"
FIGDIR = DOCS / "figures"

GRID_TASKS = ["google_robot_open_drawer", "google_robot_close_drawer",
              "google_robot_pick_coke_can", "google_robot_move_near"]
SHORT = {"google_robot_open_drawer": "open_drawer", "google_robot_close_drawer": "close_drawer",
         "google_robot_pick_coke_can": "pick_coke_can", "google_robot_move_near": "move_near"}
RULES = [("matched", "Matched", "matched_l{lam}"),
         ("fixed16", "Fixed 16", "fixed016_l{lam}"),
         ("fixed48", "Fixed 48", "fixed048_l{lam}"),
         ("random48", "Random 48", "random_fixed48_l{lam}"),
         ("topp90", "TopP 0.90", "top_p090_l{lam}")]
LAMBDAS = [("050", 0.5), ("100", 1.0)]
SEEDS_200_299 = list(range(200, 300))
SEEDS_100_199 = list(range(100, 200))


def scan(job: tuple) -> tuple:
    source, task, arm, seeds = job
    directory = source / "episodes" / task / arm
    rows = {}
    for seed in seeds:
        path = directory / f"episode_{seed:03d}_summary.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:                                              # noqa: BLE001
            continue
        rows[seed] = {"success": bool(data.get("success")),
                      "steps": int(data.get("steps", 0)),
                      "m_t": data.get("mean_selected_token_count")}
    return source.name, task, arm, rows


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return (0.0, 0.0)
    z = 1.959963984540054
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    half = z * np.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar(one: dict, two: dict, tasks: list[str]) -> dict:
    rescue = harm = both = neither = 0
    for task in tasks:
        for seed in sorted(set(one[task]) & set(two[task])):
            a, b = one[task][seed]["success"], two[task][seed]["success"]
            if a and not b:
                rescue += 1
            elif b and not a:
                harm += 1
            elif a and b:
                both += 1
            else:
                neither += 1
    total = rescue + harm + both + neither
    pvalue = (float(binomtest(min(rescue, harm), rescue + harm, 0.5, alternative="two-sided").pvalue)
              if rescue + harm else 1.0)
    return {"n_pairs": total, "both_success": both, "both_failure": neither,
            "rescue_by_first": rescue, "harm_to_first": harm, "net": rescue - harm,
            "first_successes": both + rescue, "second_successes": both + harm,
            "first_rate": (both + rescue) / total if total else None,
            "second_rate": (both + harm) / total if total else None,
            "diff_pp": 100.0 * (rescue - harm) / total if total else None,
            "mcnemar_exact_p": pvalue}


def bootstrap_difference(one: dict, two: dict, tasks: list[str],
                         rng: np.random.Generator, draws: int) -> dict:
    """Task-stratified paired bootstrap of the difference in success rate."""
    strata = []
    for task in tasks:
        seeds = sorted(set(one[task]) & set(two[task]))
        if not seeds:
            continue
        strata.append((np.array([one[task][s]["success"] for s in seeds], dtype=np.float64),
                       np.array([two[task][s]["success"] for s in seeds], dtype=np.float64)))
    if not strata:
        return {"observed_pp": None, "ci_low_pp": None, "ci_high_pp": None, "p_two_sided": None}
    observed = float(np.mean([a.mean() - b.mean() for a, b in strata])) * 100.0
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        samples[index] = np.mean([a[rng.integers(0, len(a), len(a))].mean()
                                  - b[rng.integers(0, len(b), len(b))].mean() for a, b in strata]) * 100.0
    low, high = np.percentile(samples, [2.5, 97.5])
    below = float(np.mean(samples <= 0.0))
    above = float(np.mean(samples >= 0.0))
    return {"observed_pp": observed, "ci_low_pp": float(low), "ci_high_pp": float(high),
            "p_two_sided": min(1.0, 2 * min(below, above)), "draws": draws}


def interaction(matched: dict, rule: dict, reference05: dict, reference1: dict,
                tasks: list[str], rng: np.random.Generator, draws: int) -> dict:
    """I = (matched - rule) at lambda=0.5  minus  (matched - rule) at lambda=1.0.

    Both terms use the SAME seeds and tasks, so I is a paired interaction contrast.
    """
    cells = []
    for task in tasks:
        seeds = sorted(set(matched[task]) & set(rule[task]))
        for seed in seeds:
            observed = ((matched[task][seed]["success"] - rule[task][seed]["success"])
                        - (reference05[task][seed]["success"] - reference1[task][seed]["success"]))
            cells.append((task, observed))
    if not cells:
        return {"observed_pp": None, "ci_low_pp": None, "ci_high_pp": None, "p_two_sided": None}
    per_task = defaultdict(list)
    for task, value in cells:
        per_task[task].append(value)
    arrays = {task: np.array(values, dtype=np.float64) for task, values in per_task.items()}
    observed = float(np.mean([values.mean() for values in arrays.values()])) * 100.0
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        samples[index] = np.mean([values[rng.integers(0, len(values), len(values))].mean()
                                  for values in arrays.values()]) * 100.0
    low, high = np.percentile(samples, [2.5, 97.5])
    below = float(np.mean(samples <= 0.0))
    above = float(np.mean(samples >= 0.0))
    return {"observed_pp": observed, "ci_low_pp": float(low), "ci_high_pp": float(high),
            "p_two_sided": min(1.0, 2 * min(below, above)), "draws": draws,
            "definition": "(matched - rule) at lambda=0.5 minus (matched - rule) at lambda=1.0"}


def pooled_counts(rows: dict, tasks: list[str]) -> tuple[int, int]:
    successes = sum(int(rows[task][seed]["success"]) for task in tasks for seed in rows[task])
    total = sum(len(rows[task]) for task in tasks)
    return successes, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    jobs = []
    for rule, _, pattern in RULES:
        for lambda_tag, _ in LAMBDAS:
            for task in GRID_TASKS:
                jobs.append((GRID, task, pattern.format(lam=lambda_tag), SEEDS_200_299))
    for lambda_tag, _ in LAMBDAS:
        for task in GRID_TASKS:
            jobs.append((TASKMEAN_NEW, task, f"taskmean_l{lambda_tag}", SEEDS_200_299))
    for task in GRID_TASKS:
        jobs.append((OLD_MATCHED, task, "l11_matched", SEEDS_100_199))
        jobs.append((OLD_TASKMEAN, task, "taskmean", SEEDS_100_199))

    started = time.time()
    data: dict[tuple[str, str, str], dict] = {}
    with mp.Pool(args.jobs) as pool:
        for source, task, arm, rows in pool.imap_unordered(scan, jobs):
            data[(source, task, arm)] = rows
    print(f"scanned {len(jobs)} cells in {time.time() - started:.1f}s", flush=True)

    def frame(source: str, pattern: str, lambda_tag: str) -> dict:
        return {task: data.get((source, task, pattern.format(lam=lambda_tag)), {}) for task in GRID_TASKS}

    grid = {}
    for rule, label, pattern in RULES:
        for lambda_tag, lambda_value in LAMBDAS:
            grid[(rule, lambda_tag)] = frame(GRID.name, pattern, lambda_tag)

    rates = []
    for rule, label, _ in RULES:
        for lambda_tag, lambda_value in LAMBDAS:
            rows = grid[(rule, lambda_tag)]
            successes, total = pooled_counts(rows, GRID_TASKS)
            low, high = wilson(successes, total)
            rates.append({"rule": rule, "label": label, "lambda": lambda_value,
                          "successes": successes, "n": total,
                          "rate": successes / total if total else None,
                          "wilson_low": low, "wilson_high": high,
                          "per_task": {SHORT[t]: [sum(int(rows[t][s]["success"]) for s in rows[t]),
                                                  len(rows[t])] for t in GRID_TASKS},
                          "complete": all(len(rows[t]) == 100 for t in GRID_TASKS)})

    # --- per-rule contrast against matched, within each lambda (Holm over rules) ---
    within_lambda = []
    for lambda_tag, lambda_value in LAMBDAS:
        staged, pvalues = [], []
        matched_rows = grid[("matched", lambda_tag)]
        for rule, label, _ in RULES:
            if rule == "matched":
                continue
            rows = grid[(rule, lambda_tag)]
            contrast = mcnemar(matched_rows, rows, GRID_TASKS)
            contrast.update({"rule": rule, "rule_label": label, "lambda": lambda_value,
                             "reference": "matched",
                             "bootstrap": bootstrap_difference(matched_rows, rows, GRID_TASKS,
                                                               rng, args.bootstrap)})
            staged.append(contrast)
            pvalues.append(contrast["mcnemar_exact_p"])
        for contrast, adjusted in zip(staged, holm(pvalues)):
            contrast["holm_p"] = adjusted
            contrast["holm_family_size"] = len(pvalues)
        within_lambda.extend(staged)

    # --- within-rule lambda effect (Holm over rules) ---
    lambda_effect, pvalues = [], []
    for rule, label, _ in RULES:
        half, full = grid[(rule, "050")], grid[(rule, "100")]
        effect = mcnemar(full, half, GRID_TASKS)
        effect.update({"rule": rule, "rule_label": label,
                       "direction": "lambda=1.0 minus lambda=0.5",
                       "bootstrap": bootstrap_difference(full, half, GRID_TASKS, rng, args.bootstrap)})
        lambda_effect.append(effect)
        pvalues.append(effect["mcnemar_exact_p"])
    for effect, adjusted in zip(lambda_effect, holm(pvalues)):
        effect["holm_p"] = adjusted
        effect["holm_family_size"] = len(pvalues)

    # --- the interaction itself (Holm over rules) ---
    interactions, pvalues = [], []
    for rule, label, _ in RULES:
        if rule == "matched":
            continue
        value = interaction(grid[("matched", "050")], grid[(rule, "050")],
                            grid[("matched", "100")], grid[(rule, "100")],
                            GRID_TASKS, rng, args.bootstrap)
        value.update({"rule": rule, "rule_label": label, "against": "matched"})
        interactions.append(value)
        pvalues.append(value["p_two_sided"] if value["p_two_sided"] is not None else 1.0)
    for value, adjusted in zip(interactions, holm(pvalues)):
        value["holm_p"] = adjusted
        value["holm_family_size"] = len(pvalues)

    # --- B: task-mean ablation on the same grid tasks/seeds ---
    taskmean_new = {}
    for lambda_tag, lambda_value in LAMBDAS:
        arm = frame(TASKMEAN_NEW.name, "taskmean_l{lam}", lambda_tag)
        matched_rows = grid[("matched", lambda_tag)]
        contrast = mcnemar(arm, matched_rows, GRID_TASKS)
        contrast.update({"lambda": lambda_value, "arm": "taskmean", "reference": "matched",
                         "complete": all(len(arm[t]) == 100 for t in GRID_TASKS),
                         "coverage": {SHORT[t]: len(arm[t]) for t in GRID_TASKS},
                         "bootstrap": bootstrap_difference(arm, matched_rows, GRID_TASKS,
                                                           rng, args.bootstrap),
                         "per_task": {SHORT[t]: mcnemar({t: arm[t]}, {t: matched_rows[t]}, [t])
                                      for t in GRID_TASKS if arm[t]}})
        taskmean_new[lambda_tag] = contrast
    staged = [taskmean_new[tag] for tag, _ in LAMBDAS]
    for contrast, adjusted in zip(staged, holm([c["mcnemar_exact_p"] for c in staged])):
        contrast["holm_p"] = adjusted
        contrast["holm_family_size"] = len(staged)

    # --- C: independent seed-range replication at lambda=0.5 ---
    old_matched = {task: data.get((OLD_MATCHED.name, task, "l11_matched"), {}) for task in GRID_TASKS}
    old_taskmean = {task: data.get((OLD_TASKMEAN.name, task, "taskmean"), {}) for task in GRID_TASKS}
    replication = mcnemar(old_taskmean, old_matched, GRID_TASKS)
    replication.update({"lambda": 0.5, "seeds": [100, 199], "arm": "taskmean", "reference": "matched",
                        "bootstrap": bootstrap_difference(old_taskmean, old_matched, GRID_TASKS,
                                                          rng, args.bootstrap),
                        "per_task": {SHORT[t]: mcnemar({t: old_taskmean[t]}, {t: old_matched[t]}, [t])
                                     for t in GRID_TASKS if old_taskmean[t] and old_matched[t]},
                        "source_arms": {"arm": "l11_taskmean_budget_100_199_v1:taskmean",
                                        "reference": "prompt_attn_l11_token_count_v1:l11_matched"}})

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "scripts/analyze_openvla_budget_lambda_taskmean.py",
        "design": {"tasks": GRID_TASKS, "seeds": [200, 299], "lambdas": [v for _, v in LAMBDAS],
                   "rules": [r for r, _, _ in RULES],
                   "pairing": "every contrast is paired on identical (task, seed)"},
        "rates": rates,
        "within_lambda_vs_matched": within_lambda,
        "within_rule_lambda_effect": lambda_effect,
        "interaction": interactions,
        "taskmean_new": taskmean_new,
        "taskmean_replication_100_199": replication,
        "caveats": [
            "The interaction is estimated directly as a paired contrast; it is never inferred "
            "from one comparison being significant and another not.",
            "Non-significance is not equivalence: every non-significant contrast is reported with "
            "its observed difference and a task-stratified paired bootstrap interval.",
            "The task-mean arm uses constants computed from the lambda=1.0 Matched reference on the "
            "SAME seeds 200-299 and reuses them at lambda=0.5. It is therefore a post-hoc "
            "mean-matched ablation, not an independently calibrated baseline, and at lambda=0.5 it "
            "is not a same-lambda mean-matched control.",
            "Random 48 changes both the count rule and the token content, so its contrast with "
            "Matched mixes both; content is isolated only by Fixed 48 versus Random 48.",
            "Closed-loop m_t values come from divergent trajectories and are not same-state "
            "causal evidence.",
            "The 100-199 replication runs on an older renderer/protocol than the 200-299 grid for "
            "pick and move, so it is a directional replication of the lambda=0.5 task-mean effect, "
            "not a bitwise continuation of the same experiment.",
        ],
    }

    if args.write:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "OPENVLA_BUDGET_LAMBDA_TASKMEAN_FINAL.json").write_text(
            json.dumps(payload, indent=2, default=float) + "\n")
        write_report(payload)
        figure(payload, FIGDIR / "openvla_budget_lambda_taskmean.png")
        print("wrote docs/OPENVLA_BUDGET_LAMBDA_TASKMEAN_FINAL.{json,md} and "
              "docs/figures/openvla_budget_lambda_taskmean.png")
    else:
        print(json.dumps({"rates": [{k: r[k] for k in ("rule", "lambda", "successes", "n")}
                                    for r in rates],
                          "interaction": [{k: v[k] for k in ("rule", "observed_pp", "p_two_sided")}
                                          for v in interactions],
                          "taskmean": {tag: [c["first_successes"], c["second_successes"], c["n_pairs"]]
                                       for tag, c in taskmean_new.items()},
                          "replication": [replication["first_successes"], replication["second_successes"],
                                          replication["n_pairs"]]}, indent=2))
    return 0


def figure(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_, axes = plt.subplots(1, 3, figsize=(19.5, 6.6))
    colours = {"matched": "#1f4e9c", "fixed16": "#c8641c", "fixed48": "#2e8b57",
               "random48": "#a02c2c", "topp90": "#7d3ca8"}
    labels = {r["rule"]: r["label"] for r in payload["rates"]}

    axis = axes[0]
    width = 0.36
    for index, (lambda_tag, lambda_value) in enumerate([("050", 0.5), ("100", 1.0)]):
        positions = np.arange(len(RULES)) + (index - 0.5) * width
        values, errors = [], []
        for rule, _, _ in RULES:
            entry = next(r for r in payload["rates"]
                         if r["rule"] == rule and r["lambda"] == lambda_value)
            rate = 100.0 * entry["rate"]
            values.append(rate)
            errors.append([rate - 100 * entry["wilson_low"], 100 * entry["wilson_high"] - rate])
        axis.bar(positions, values, width, yerr=np.array(errors).T, capsize=3,
                 color=[colours[r] for r, _, _ in RULES],
                 alpha=1.0 if index else 0.55,
                 label=f"lambda={lambda_value}", edgecolor="black", linewidth=0.6)
        for position, value in zip(positions, values):
            axis.text(position, value + 1.0, f"{value:.1f}", ha="center", fontsize=8)
    axis.set_xticks(np.arange(len(RULES)), [labels[r] for r, _, _ in RULES], fontsize=9)
    axis.set_ylabel("pooled success rate (%)")
    axis.set_ylim(0, 75)
    axis.set_title("(a) 4 Google tasks x 100 seeds\npale = lambda 0.5, solid = lambda 1.0",
                   fontsize=10.5)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1]
    entries = payload["interaction"]
    positions = np.arange(len(entries))[::-1]
    for position, entry in zip(positions, entries):
        axis.errorbar(entry["observed_pp"], position,
                      xerr=[[entry["observed_pp"] - entry["ci_low_pp"]],
                            [entry["ci_high_pp"] - entry["observed_pp"]]],
                      fmt="o", color=colours[entry["rule"]], capsize=3, markersize=6)
        star = " *" if entry["holm_p"] < 0.05 else ""
        axis.text(entry["ci_high_pp"] + 0.6, position,
                  f"{entry['observed_pp']:+.1f}pp (p={entry['p_two_sided']:.3f}){star}",
                  va="center", fontsize=8)
    axis.axvline(0, color="black", linewidth=1.0)
    axis.set_yticks(positions, [labels[e["rule"]] for e in entries], fontsize=9)
    axis.set_xlabel("interaction (pp): (Matched - rule) at 0.5  minus  at 1.0")
    axis.set_xlim(-8, 26)
    axis.set_title("(b) Directly estimated lambda x budget-rule interaction\n"
                   "positive = the rule's disadvantage shrinks at higher lambda", fontsize=10.5)
    axis.grid(axis="x", alpha=0.25)

    axis = axes[2]
    bars = [("lambda 0.5\n(seeds 200-299)", payload["taskmean_new"]["050"]),
            ("lambda 1.0\n(seeds 200-299)", payload["taskmean_new"]["100"]),
            ("lambda 0.5\n(seeds 100-199)", payload["taskmean_replication_100_199"])]
    positions = np.arange(len(bars))
    for position, (name, contrast) in zip(positions, bars):
        difference = contrast["diff_pp"]
        low = contrast["bootstrap"]["ci_low_pp"]
        high = contrast["bootstrap"]["ci_high_pp"]
        axis.bar(position, difference, 0.55,
                 color="#a02c2c" if difference < 0 else "#2e8b57", alpha=0.85)
        axis.errorbar(position, difference,
                      yerr=[[difference - low], [high - difference]], fmt="none",
                      ecolor="black", capsize=4)
        axis.text(position, difference + (1.0 if difference >= 0 else -2.2),
                  f"{contrast['first_successes']}/{contrast['n_pairs']} vs "
                  f"{contrast['second_successes']}/{contrast['n_pairs']}\n"
                  f"{difference:+.1f}pp  p={contrast['mcnemar_exact_p']:.3f}",
                  ha="center", fontsize=8)
    axis.axhline(0, color="black", linewidth=1.0)
    axis.set_xticks(positions, [name for name, _ in bars], fontsize=8.5)
    axis.set_ylabel("Task-mean minus Matched (pp)")
    axis.set_ylim(-22, 14)
    axis.set_title("(c) Removing per-step budget adaptation:\n"
                   "task-mean constant versus Matched, both lambdas + seed replication",
                   fontsize=10.5)
    axis.grid(axis="y", alpha=0.25)

    figure_.suptitle("Budget-rule ablations on OpenVLA: does per-step budget adaptation pay off, "
                     "and does the answer depend on lambda?", fontsize=12.5)
    figure_.tight_layout(rect=(0, 0, 1, 0.93))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figure_.savefig(path, dpi=165)
    plt.close(figure_)


def write_report(payload: dict) -> None:
    lines = [
        "# Reloaded budget-rule analysis: lambda x rule interaction and the task-mean ablation",
        "",
        f"Generated {payload['generated_at']} by `{payload['generator']}`.",
        "Design: 4 Google tasks, seeds 200-299, 5 budget rules, lambda 0.5 and 1.0; "
        "every contrast is paired on identical (task, seed).",
        "",
        "## Pooled success rates (successes/denominator, percent, Wilson 95% CI)",
        "",
        "| rule | lambda | successes/denominator | percent | Wilson 95% | complete |",
        "|---|---|---|---|---|---|",
    ]
    for entry in payload["rates"]:
        lines.append(f"| {entry['label']} | {entry['lambda']} | "
                     f"{entry['successes']}/{entry['n']} | {100.0 * entry['rate']:.2f}% | "
                     f"[{100 * entry['wilson_low']:.1f}, {100 * entry['wilson_high']:.1f}] | "
                     f"{entry['complete']} |")
    lines += ["", "## Each rule against Matched, within each lambda", "",
              "Rescue = the rule succeeds where Matched fails. Holm is applied over the four "
              "non-matched rules within each lambda.", "",
              "| lambda | rule | pairs | rule | matched | rescue | harm | diff (pp) | "
              "bootstrap 95% CI | McNemar p | Holm p |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for contrast in payload["within_lambda_vs_matched"]:
        boot = contrast["bootstrap"]
        lines.append(f"| {contrast['lambda']} | {contrast['rule_label']} | {contrast['n_pairs']} | "
                     f"{contrast['first_successes']} | {contrast['second_successes']} | "
                     f"{contrast['rescue_by_first']} | {contrast['harm_to_first']} | "
                     f"{boot['observed_pp']:+.2f} | "
                     f"[{boot['ci_low_pp']:+.2f}, {boot['ci_high_pp']:+.2f}] | "
                     f"{contrast['mcnemar_exact_p']:.4g} | {contrast['holm_p']:.4g} |")
    lines += ["", "## Within-rule lambda effect (lambda 1.0 minus lambda 0.5)", "",
              "| rule | pairs | lambda 1.0 | lambda 0.5 | rescue | harm | diff (pp) | "
              "bootstrap 95% CI | McNemar p | Holm p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for effect in payload["within_rule_lambda_effect"]:
        boot = effect["bootstrap"]
        lines.append(f"| {effect['rule_label']} | {effect['n_pairs']} | {effect['first_successes']} | "
                     f"{effect['second_successes']} | {effect['rescue_by_first']} | "
                     f"{effect['harm_to_first']} | {boot['observed_pp']:+.2f} | "
                     f"[{boot['ci_low_pp']:+.2f}, {boot['ci_high_pp']:+.2f}] | "
                     f"{effect['mcnemar_exact_p']:.4g} | {effect['holm_p']:.4g} |")
    lines += ["", "## The interaction itself", "",
              "I = (Matched - rule) at lambda 0.5 minus (Matched - rule) at lambda 1.0, computed on "
              "the same task/seed cells and bootstrapped within task. Positive means the rule's "
              "disadvantage relative to Matched is smaller at higher lambda.", "",
              "| rule | I (pp) | bootstrap 95% CI | bootstrap p | Holm p |", "|---|---|---|---|---|"]
    for entry in payload["interaction"]:
        lines.append(f"| {entry['rule_label']} | {entry['observed_pp']:+.2f} | "
                     f"[{entry['ci_low_pp']:+.2f}, {entry['ci_high_pp']:+.2f}] | "
                     f"{entry['p_two_sided']:.4g} | {entry['holm_p']:.4g} |")
    lines += ["", "## Task-mean constant instead of per-step adaptation", "",
              "| setting | pairs | task-mean | matched | rescue | harm | diff (pp) | "
              "bootstrap 95% CI | McNemar p | Holm p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lambda_tag in ("050", "100"):
        contrast = payload["taskmean_new"][lambda_tag]
        boot = contrast["bootstrap"]
        lines.append(f"| lambda {contrast['lambda']} (seeds 200-299) | {contrast['n_pairs']} | "
                     f"{contrast['first_successes']} | {contrast['second_successes']} | "
                     f"{contrast['rescue_by_first']} | {contrast['harm_to_first']} | "
                     f"{boot['observed_pp']:+.2f} | "
                     f"[{boot['ci_low_pp']:+.2f}, {boot['ci_high_pp']:+.2f}] | "
                     f"{contrast['mcnemar_exact_p']:.4g} | {contrast['holm_p']:.4g} |")
    replication = payload["taskmean_replication_100_199"]
    boot = replication["bootstrap"]
    lines.append(f"| lambda 0.5 (seeds 100-199, older protocol) | {replication['n_pairs']} | "
                 f"{replication['first_successes']} | {replication['second_successes']} | "
                 f"{replication['rescue_by_first']} | {replication['harm_to_first']} | "
                 f"{boot['observed_pp']:+.2f} | "
                 f"[{boot['ci_low_pp']:+.2f}, {boot['ci_high_pp']:+.2f}] | "
                 f"{replication['mcnemar_exact_p']:.4g} | n/a (separate replication) |")
    lines += ["", "## Per-task task-mean contrasts (descriptive; no separate correction claimed)", ""]
    for lambda_tag in ("050", "100"):
        contrast = payload["taskmean_new"][lambda_tag]
        lines.append(f"**lambda {contrast['lambda']}**: " + "; ".join(
            f"{task} {entry['first_successes']}-{entry['second_successes']} "
            f"(net {entry['net']:+d}, p={entry['mcnemar_exact_p']:.3f})"
            for task, entry in contrast["per_task"].items()) + "")
    lines += ["", "## Caveats that must travel with these numbers", ""]
    lines += [f"- {note}" for note in payload["caveats"]]
    lines += ["", "Figure: `docs/figures/openvla_budget_lambda_taskmean.png`.", ""]
    (DOCS / "OPENVLA_BUDGET_LAMBDA_TASKMEAN_FINAL.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
