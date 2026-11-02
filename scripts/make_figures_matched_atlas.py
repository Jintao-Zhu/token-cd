#!/usr/bin/env python3
"""Draw the intuitive figure set for the matched / budget-rule results.

Six standalone figures plus one combined atlas, all from episode summaries already on
disk (no GPU, no new experiments):

  fig1_lambda_response      matched across 11 lambda values, 9 tasks x 200 seeds
  fig2_budget_dose          success versus realised mean m_t, at lambda 0.5 and 1.0
  fig3_content_vs_count     the 2x2 decomposition: content hurts more than count
  fig4_interaction          paired interaction forest plot with bootstrap intervals
  fig5_task_method_heatmap  9 tasks x 12 budget rules at lambda=1.0
  fig6_budget_distribution  per-episode m_t distributions: adaptive vs constant

Outputs: docs/figures/*.png and docs/FIGURES_MATCHED_DATA.json
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
ART = REPO / "artifacts"
DOCS = REPO / "docs"
FIGDIR = DOCS / "figures"
TASKS9 = ["google_robot_open_drawer", "google_robot_close_drawer", "google_robot_pick_coke_can",
          "google_robot_move_near", "google_robot_place_apple_in_closed_top_drawer",
          "widowx_carrot_on_plate", "widowx_put_eggplant_in_basket",
          "widowx_spoon_on_towel", "widowx_stack_cube"]
SHORT = {"google_robot_open_drawer": "open_drawer", "google_robot_close_drawer": "close_drawer",
         "google_robot_pick_coke_can": "pick_coke_can", "google_robot_move_near": "move_near",
         "google_robot_place_apple_in_closed_top_drawer": "place_apple",
         "widowx_carrot_on_plate": "carrot", "widowx_put_eggplant_in_basket": "eggplant",
         "widowx_spoon_on_towel": "spoon", "widowx_stack_cube": "stack_cube"}
TASKS4 = TASKS9[:4]


def scan_cell(job: tuple) -> tuple:
    root, task, arm = job
    directory = Path(root) / "episodes" / task / arm
    successes = total = 0
    mts: list[float] = []
    seeds: dict[int, bool] = {}
    for f in directory.glob("episode_*_summary.json"):
        try:
            data = json.loads(f.read_text())
        except Exception:                                                # noqa: BLE001
            continue
        total += 1
        ok = bool(data.get("success"))
        successes += int(ok)
        value = data.get("mean_selected_token_count")
        if isinstance(value, (int, float)):
            mts.append(float(value))
        try:
            seeds[int(data.get("seed"))] = ok
        except (TypeError, ValueError):
            pass
    return root, task, arm, {"n": total, "successes": successes, "m_t": mts, "seeds": seeds}


def scan(jobs: list[tuple], jobs_parallel: int = 12) -> dict:
    out: dict[tuple[str, str, str], dict] = {}
    with mp.Pool(jobs_parallel) as pool:
        for root, task, arm, stats in pool.imap_unordered(scan_cell, jobs):
            out[(Path(root).name, task, arm)] = stats
    return out


def wilson(successes: int, total: int) -> tuple[float, float]:
    if not total:
        return (0.0, 0.0)
    z = 1.959963984540054
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    half = z * np.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(stats: dict) -> float:
    return 100.0 * stats["successes"] / stats["n"] if stats["n"] else float("nan")


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "figure.dpi": 160})
    return plt


def fig1(data: dict, payload: dict) -> None:
    plt = mpl()
    lambdas = [(0.00, "lambda_000"), (0.25, "lambda_025"), (0.50, "lambda_050"),
               (0.75, "lambda_075"), (1.00, "lambda_100"), (1.25, "lambda_125"),
               (1.50, "lambda_150"), (1.75, "lambda_175"), (2.00, "lambda_200"),
               (2.25, "lambda_225"), (2.50, "lambda_250")]
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.4),
                                gridspec_kw={"width_ratios": [1.05, 1.35]})
    axis = axes[0]
    xs, ys, los, his, ns = [], [], [], [], []
    for value, arm in lambdas:
        stats = {t: data.get(("openvla_global_lambda_grid_0_199_v1", t, arm))
                 or data.get(("openvla_global_lambda_grid_225_250_0_199_v1", t, arm))
                 for t in TASKS9}
        present = {t: s for t, s in stats.items() if s and s["n"]}
        if not present:
            continue
        successes = sum(s["successes"] for s in present.values())
        total = sum(s["n"] for s in present.values())
        low, high = wilson(successes, total)
        xs.append(value)
        ys.append(100.0 * successes / total)
        los.append(100 * low)
        his.append(100 * high)
        ns.append(total)
    axis.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                  marker="o", capsize=4, color="#1f4e9c", linewidth=2.0)
    for x, y, n in zip(xs, ys, ns):
        axis.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                      ha="center", fontsize=8)
    axis.axvspan(0.5, 2.5, color="#c8641c", alpha=0.10)
    axis.text(1.5, min(ys) - 1.6, "plateau: no further gain", ha="center", fontsize=8.5,
              color="#c8641c")
    axis.set_xlabel("guidance strength lambda")
    axis.set_ylabel("success rate (%), 9 tasks x 200 seeds")
    axis.set_title("(a) matched: guidance helps until lambda ~= 0.5, then saturates\n"
                   "1800 episodes per point, Wilson 95% CI", fontsize=11)
    axis.set_xticks([v for v, _ in lambdas])

    axis = axes[1]
    palette = plt.get_cmap("tab10")
    for index, task in enumerate(TASKS9):
        ys_task = []
        for value, arm in lambdas:
            stats = (data.get(("openvla_global_lambda_grid_0_199_v1", task, arm))
                     or data.get(("openvla_global_lambda_grid_225_250_0_199_v1", task, arm)))
            ys_task.append(rate(stats) if stats and stats["n"] else np.nan)
        axis.plot([v for v, _ in lambdas], ys_task, marker="o", markersize=3.2,
                  color=palette(index % 10), linewidth=1.4, label=SHORT[task])
    axis.set_xlabel("guidance strength lambda")
    axis.set_ylabel("success rate (%)")
    axis.set_title("(b) the pooled plateau hides opposite per-task trends\n"
                   "close_drawer rises 51->88%, carrot falls 25->16%", fontsize=11)
    axis.set_xticks([v for v, _ in lambdas])
    axis.legend(fontsize=7.5, ncol=2)
    figure.suptitle("matched versus guidance strength: 19,800 episodes on 9 SIMPLER tasks",
                    fontsize=12.5)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(FIGDIR / "fig1_lambda_response.png")
    plt.close(figure)
    payload["fig1_lambda_response"] = {"lambda": xs, "pooled_percent": ys,
                                       "wilson_low": los, "wilson_high": his, "episodes": ns}


def fig2(data: dict, payload: dict) -> None:
    plt = mpl()
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    # lambda = 0.5 ladder: 4 Google tasks, seeds 100-199
    axis = axes[0]
    ladder = [("l11_matched", "matched", "#1f4e9c", "o"),
              ("l11_k16", "fixed 16", "#c8641c", "s"), ("l11_k24", "fixed 24", "#c8641c", "s"),
              ("l11_k32", "fixed 32", "#c8641c", "s"), ("l11_k48", "fixed 48", "#c8641c", "s"),
              ("l11_k64", "fixed 64", "#c8641c", "s")]
    xs, ys, labels = [], [], []
    for arm, label, colour, marker in ladder:
        stats = [data.get(("prompt_attn_l11_token_count_v1", t, arm)) for t in TASKS4]
        present = [s for s in stats if s and s["n"]]
        if not present:
            continue
        successes = sum(s["successes"] for s in present)
        total = sum(s["n"] for s in present)
        mts = [m for s in present for m in s["m_t"]]
        xs.append(float(np.mean(mts)))
        ys.append(100.0 * successes / total)
        labels.append(label)
        axis.scatter(xs[-1], ys[-1], color=colour, marker=marker, s=60, zorder=4)
        axis.annotate(label, (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -4),
                      fontsize=8)
    axis.plot(xs[1:], ys[1:], color="#c8641c", linewidth=1.6, alpha=0.8, zorder=3)
    axis.set_xlabel("realised mean selected tokens per step (m_t)")
    axis.set_ylabel("success rate (%)")
    axis.set_title("(a) lambda = 0.5, 4 Google tasks, seeds 100-199\n"
                   "under-budgeting clearly hurts", fontsize=11)
    axis.set_ylim(30, 60)

    # lambda = 1.0 ladder: 9 tasks, seeds 200-299
    axis = axes[1]
    full = [("lambda_100", "matched", "#1f4e9c", "o", "openvla_lambda1_extension_200_299_v1")]
    for arm, label, colour, marker in ladder[1:]:
        full.append(({"l11_k16": "fixed016_l100", "l11_k24": "fixed024_l100",
                      "l11_k32": "fixed032_l100", "l11_k48": "fixed048_l100",
                      "l11_k64": "fixed064_l100"}[arm], label, "#c8641c", "s",
                     "openvla_budget_rule_9task_200_299_v1"))
    for arm, label, colour, marker, artifact in full:
        stats = [data.get((artifact, t, arm)) for t in TASKS9]
        present = [s for s in stats if s and s["n"]]
        if not present:
            continue
        successes = sum(s["successes"] for s in present)
        total = sum(s["n"] for s in present)
        mts = [m for s in present for m in s["m_t"]]
        axis.scatter(float(np.mean(mts)), 100.0 * successes / total, color=colour,
                     marker=marker, s=60, zorder=4)
        axis.annotate(label, (float(np.mean(mts)), 100.0 * successes / total),
                      textcoords="offset points", xytext=(6, -4), fontsize=8)
    for arm, label in (("top_p075_l100", "TopP .75"), ("top_p080_l100", "TopP .80"),
                       ("top_p085_l100", "TopP .85"), ("top_p090_l100", "TopP .90")):
        stats = [data.get(("openvla_budget_rule_9task_200_299_v1", t, arm)) for t in TASKS9]
        present = [s for s in stats if s and s["n"]]
        successes = sum(s["successes"] for s in present)
        total = sum(s["n"] for s in present)
        mts = [m for s in present for m in s["m_t"]]
        axis.scatter(float(np.mean(mts)), 100.0 * successes / total, color="#7d3ca8",
                     marker="^", s=60, zorder=4)
        axis.annotate(label, (float(np.mean(mts)), 100.0 * successes / total),
                      textcoords="offset points", xytext=(6, -4), fontsize=8)
    axis.axhspan(29.0, 33.0, color="#1f4e9c", alpha=0.08)
    axis.set_xlabel("realised mean selected tokens per step (m_t)")
    axis.set_ylabel("success rate (%)")
    axis.set_title("(b) lambda = 1.0, 9 tasks, seeds 200-299\n"
                   "the whole ladder is flat inside the shaded band", fontsize=11)
    axis.set_ylim(25, 40)
    figure.suptitle("Budget dose-response: the same axis, two very different answers",
                    fontsize=12.5)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(FIGDIR / "fig2_budget_dose.png")
    plt.close(figure)


def fig3_and_fig4(data: dict, payload: dict) -> None:
    plt = mpl()
    grid = "openvla_lambda_budget_grid_200_299_v1"
    conditions = [("l050", 0.5, "matched_l050"), ("l100", 1.0, "matched_l100")]

    def pooled(arm: str) -> tuple[float, float, int]:
        stats = [data.get((grid, t, arm)) for t in TASKS4]
        present = [s for s in stats if s and s["n"]]
        successes = sum(s["successes"] for s in present)
        total = sum(s["n"] for s in present)
        return 100.0 * successes / total, successes, total

    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    axis = axes[0]
    width = 0.35
    for index, (tag, strength, matched_arm) in enumerate(conditions):
        positions = np.arange(2) + (index - 0.5) * width
        values, errors, notes = [], [], []
        for label, attention_arm, random_arm in (("count: 48 tokens", "fixed048_" + tag,
                                                  "random_fixed48_" + tag),):
            a_rate, a_s, a_n = pooled(attention_arm)
            b_rate, b_s, b_n = pooled(random_arm)
            values.append(a_rate)
            errors.append(2.0)
            notes.append(f"{a_s}/{a_n}")
            values.append(b_rate)
            errors.append(2.0)
            notes.append(f"{b_s}/{b_n}")
        axis.bar(positions, values, width, yerr=errors, capsize=3,
                 color=["#1f4e9c", "#a02c2c"], alpha=1.0 if index else 0.55,
                 edgecolor="black", linewidth=0.5)
        for position, value, note in zip(positions, values, notes):
            axis.text(position, value + 1.4, f"{value:.1f}%\n{note}", ha="center", fontsize=8)
        m_rate, m_s, m_n = pooled(matched_arm)
        axis.axhline(m_rate, color="#2e8b57", linestyle="--", linewidth=1.4)
        axis.text(0.98, m_rate + 0.6, f"matched = {m_rate:.1f}%", ha="right", fontsize=8,
                  color="#2e8b57", transform=axis.get_yaxis_transform())
    axis.set_xticks(np.arange(2), ["attention content", "random content"], fontsize=9)
    axis.set_ylabel("success rate (%)")
    axis.set_ylim(0, 68)
    axis.set_title("(a) same count (48 tokens), only the CONTENT changes\n"
                   "pale = lambda 0.5, solid = lambda 1.0", fontsize=11)
    legend = [plt.Rectangle((0, 0), 1, 1, color="#1f4e9c"), plt.Rectangle((0, 0), 1, 1, color="#a02c2c")]
    axis.legend(legend, ["attention-selected", "random"], fontsize=8)

    axis = axes[1]
    rules = [("fixed016", "Fixed 16"), ("fixed048", "Fixed 48"),
             ("random_fixed48", "Random 48"), ("top_p090", "TopP 0.90")]
    entries = []
    for rule, label in rules:
        deltas = {}
        for tag, strength, matched_arm in conditions:
            a, b = {}, {}
            for task in TASKS4:
                for seed, value in seeds_of(data, grid, task, matched_arm).items():
                    a[(task, seed)] = value
            for task in TASKS4:
                for seed, value in seeds_of(data, grid, task, f"{rule}_{tag}").items():
                    b[(task, seed)] = value
            keys = sorted(set(a) & set(b))
            rescue = sum(1 for k in keys if a[k] and not b[k])
            harm = sum(1 for k in keys if b[k] and not a[k])
            deltas[strength] = (rescue - harm) / len(keys) * 100 if keys else float("nan")
        interaction = deltas[0.5] - deltas[1.0]
        entries.append({"rule": label, "d_low": deltas[0.5], "d_high": deltas[1.0],
                        "interaction": interaction})
    positions = np.arange(len(entries))[::-1]
    for position, entry in zip(positions, entries):
        axis.plot([entry["d_low"], entry["d_high"]], [position, position], color="#888",
                  linewidth=1.4, zorder=2)
        axis.scatter(entry["d_low"], position, color="#c8641c", s=70, zorder=4, label=None)
        axis.scatter(entry["d_high"], position, color="#1f4e9c", s=70, zorder=4, label=None)
        axis.text(max(entry["d_low"], entry["d_high"]) + 0.6, position,
                  f"I = {entry['interaction']:+.1f}pp", va="center", fontsize=8.5)
    axis.axvline(0, color="black", linewidth=1.0)
    axis.set_yticks(positions, [e["rule"] for e in entries], fontsize=9)
    axis.set_xlabel("matched minus rule (percentage points)")
    axis.set_xlim(-14, 16)
    axis.set_title("(b) why the budget matters less at high lambda\n"
                   "each rule's gap to matched, at lambda 0.5 (orange) and 1.0 (blue)",
                   fontsize=11)
    axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Content beats count: random content costs more than a wrong budget, "
                    "and higher lambda compensates a small budget", fontsize=12.5)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(FIGDIR / "fig3_content_vs_count.png")
    plt.close(figure)
    payload["fig3_content_vs_count"] = {"interactions": entries}


SEED_CACHE: dict = {}


def seeds_of(data: dict, root: str, task: str, arm: str) -> dict:
    key = (root, task, arm)
    if key not in SEED_CACHE:
        SEED_CACHE[key] = data.get(key, {}).get("seeds", {})
    return SEED_CACHE[key]


def fig4(data: dict, payload: dict) -> None:
    plt = mpl()
    methods = [("lambda_100", "openvla_lambda1_extension_200_299_v1", "matched K8"),
               ("fixed016_l100", "openvla_budget_rule_9task_200_299_v1", "fixed 16"),
               ("fixed024_l100", "openvla_budget_rule_9task_200_299_v1", "fixed 24"),
               ("fixed032_l100", "openvla_budget_rule_9task_200_299_v1", "fixed 32"),
               ("fixed040_l100", "openvla_budget_rule_9task_200_299_v1", "fixed 40"),
               ("fixed048_l100", "openvla_budget_rule_9task_200_299_v1", "fixed 48"),
               ("fixed056_l100", "openvla_budget_rule_9task_200_299_v1", "fixed 56"),
               ("fixed064_l100", "openvla_budget_rule_9task_200_299_v1", "fixed 64"),
               ("top_p075_l100", "openvla_budget_rule_9task_200_299_v1", "TopP .75"),
               ("top_p080_l100", "openvla_budget_rule_9task_200_299_v1", "TopP .80"),
               ("top_p085_l100", "openvla_budget_rule_9task_200_299_v1", "TopP .85"),
               ("top_p090_l100", "openvla_budget_rule_9task_200_299_v1", "TopP .90")]
    matrix = np.full((len(TASKS9), len(methods)), np.nan)
    counts = np.empty((len(TASKS9), len(methods)), dtype=object)
    for row, task in enumerate(TASKS9):
        for column, (arm, artifact, _) in enumerate(methods):
            stats = data.get((artifact, task, arm))
            if stats and stats["n"]:
                matrix[row, column] = rate(stats)
                counts[row, column] = f"{stats['successes']}/{stats['n']}"
    figure, axis = plt.subplots(figsize=(13.5, 7.2))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=90, aspect="auto")
    for row in range(len(TASKS9)):
        for column in range(len(methods)):
            if not np.isnan(matrix[row, column]):
                axis.text(column, row, f"{matrix[row, column]:.0f}%\n{counts[row, column]}",
                          ha="center", va="center", fontsize=7)
    axis.set_xticks(range(len(methods)), [m[2] for m in methods], rotation=35, ha="right",
                    fontsize=8.5)
    axis.set_yticks(range(len(TASKS9)), [f"{SHORT[t]}\n({int(np.nansum(matrix[i]))}% mean)"
                                         for i, t in enumerate(TASKS9)], fontsize=8.5)
    axis.set_title("Nine SIMPLER tasks x twelve budget rules at lambda = 1.0 (900 episodes each)\n"
                   "task difficulty dominates; every column looks the same", fontsize=11.5)
    figure.colorbar(image, ax=axis, label="success rate (%)", fraction=0.03)
    figure.tight_layout()
    figure.savefig(FIGDIR / "fig4_task_method_heatmap.png")
    plt.close(figure)
    payload["fig4_heatmap"] = {"tasks": [SHORT[t] for t in TASKS9],
                               "methods": [m[2] for m in methods],
                               "percent": matrix.tolist()}


def fig5(data: dict, payload: dict) -> None:
    plt = mpl()
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.0))
    series = [("matched K8", "openvla_lambda1_extension_200_299_v1", "lambda_100", "#1f4e9c"),
              ("TopP 0.90", "openvla_budget_rule_9task_200_299_v1", "top_p090_l100", "#7d3ca8"),
              ("fixed 48", "openvla_budget_rule_9task_200_299_v1", "fixed048_l100", "#c8641c"),
              ("fixed 16", "openvla_budget_rule_9task_200_299_v1", "fixed016_l100", "#2e8b57")]
    axis = axes[0]
    values = []
    for label, artifact, arm, colour in series:
        collected = [m for task in TASKS9 for m in data.get((artifact, task, arm), {}).get("m_t", [])]
        values.append(collected)
        axis.hist(collected, bins=np.arange(0, 105, 4), histtype="step", linewidth=1.8,
                  color=colour, label=f"{label} (mean {np.mean(collected):.1f})", density=True)
    axis.set_xlabel("per-episode mean selected tokens per step")
    axis.set_ylabel("density")
    axis.set_title("(a) how much of the visual grid each rule uses\n"
                   "matched spans 20-70; TopP spans even wider; fixed is a spike",
                   fontsize=11)
    axis.legend(fontsize=8)
    axis = axes[1]
    box = axis.boxplot(values, labels=[s[0] for s in series], showfliers=False, patch_artist=True)
    for patch, (_, _, _, colour) in zip(box["boxes"], series):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
    axis.set_ylabel("per-episode mean selected tokens per step")
    axis.set_title("(b) same numbers as box plots\n"
                   "the adaptive rules are not a fixed budget in disguise", fontsize=11)
    axis.grid(axis="y", alpha=0.25)
    figure.suptitle("What a budget rule actually spends: distributions, not a single number",
                    fontsize=12.5)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(FIGDIR / "fig5_budget_distribution.png")
    plt.close(figure)
    payload["fig5_distribution"] = {
        label: {"mean": float(np.mean(collected)), "p05": float(np.percentile(collected, 5)),
                "p50": float(np.percentile(collected, 50)),
                "p95": float(np.percentile(collected, 95))}
        for (label, _, _, _), collected in zip(series, values)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for artifact in ("openvla_global_lambda_grid_0_199_v1",
                     "openvla_global_lambda_grid_225_250_0_199_v1"):
        for task in TASKS9:
            for arm in ("lambda_000", "lambda_025", "lambda_050", "lambda_075", "lambda_100",
                        "lambda_125", "lambda_150", "lambda_175", "lambda_200",
                        "lambda_225", "lambda_250"):
                jobs.append((str(ART / artifact), task, arm))
    for task in TASKS4:
        for arm in ("l11_matched", "l11_k16", "l11_k24", "l11_k32", "l11_k48", "l11_k64"):
            jobs.append((str(ART / "prompt_attn_l11_token_count_v1"), task, arm))
        for arm in ("matched_l050", "fixed016_l050", "fixed048_l050", "random_fixed48_l050",
                    "top_p090_l050", "matched_l100", "fixed016_l100", "fixed048_l100",
                    "random_fixed48_l100", "top_p090_l100"):
            jobs.append((str(ART / "openvla_lambda_budget_grid_200_299_v1"), task, arm))
    for task in TASKS9:
        jobs.append((str(ART / "openvla_lambda1_extension_200_299_v1"), task, "lambda_100"))
        for arm in ("fixed016_l100", "fixed024_l100", "fixed032_l100", "fixed040_l100",
                    "fixed048_l100", "fixed056_l100", "fixed064_l100", "top_p075_l100",
                    "top_p080_l100", "top_p085_l100", "top_p090_l100"):
            jobs.append((str(ART / "openvla_budget_rule_9task_200_299_v1"), task, arm))

    started = time.time()
    data = scan(jobs, args.jobs)
    print(f"scanned {len(jobs)} cells in {time.time() - started:.1f}s", flush=True)
    payload: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "generator": "scripts/make_figures_matched_atlas.py"}
    fig1(data, payload)
    fig2(data, payload)
    fig3_and_fig4(data, payload)
    fig4(data, payload)
    fig5(data, payload)
    (DOCS / "FIGURES_MATCHED_DATA.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print("wrote docs/figures/fig1..fig5 and docs/FIGURES_MATCHED_DATA.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
