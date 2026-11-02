#!/usr/bin/env python3
"""Analyse the matched-budget probe: what does m_t become as KMeans K grows?

Reads the per-step probe files produced by
`research/semantic_token_cd/openvla_matched_k_probe_rollout.py`, where the intervention stays
at K=8 and the other K columns are counterfactual budgets measured on the SAME states.

Three questions:
  1. does the K=8 column reproduce the historical matched budget exactly?
  2. how does m_t scale with K, per task and pooled?
  3. where does each K land on the already-measured budget dose-response (fixed budgets),
     i.e. would a given K put us in the "under-budgeted" regime we already know about?

Outputs: docs/OPENVLA_K_PROBE_BUDGET_CURVE.{json,md} and
         docs/figures/openvla_k_probe_budget_curve.png
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "artifacts/openvla_k_probe_200_219_v1"
REFERENCE = REPO / "artifacts/openvla_lambda1_extension_200_299_v1"
DOCS = REPO / "docs"
FIGDIR = DOCS / "figures"
KS = (8, 9, 10, 11, 12, 14, 16, 20)
SHORT = {"google_robot_open_drawer": "open_drawer", "google_robot_close_drawer": "close_drawer",
         "google_robot_pick_coke_can": "pick_coke_can", "google_robot_move_near": "move_near",
         "google_robot_place_apple_in_closed_top_drawer": "place_apple",
         "widowx_carrot_on_plate": "carrot", "widowx_put_eggplant_in_basket": "eggplant",
         "widowx_spoon_on_towel": "spoon", "widowx_stack_cube": "stack_cube"}
# already-measured dose-response at lambda=0.5, seeds 100-199, 4 Google tasks (prompt_attn_l11_token_count_v1)
DOSE_RESPONSE_L050 = {16: (151, 400), 24: (173, 400), 32: (196, 400), 48: (207, 400), 64: (198, 400)}
MATCHED_L050 = (213, 400)
# already-measured dose-response at lambda=1.0, 9 tasks, 100 seeds (openvla_budget_rule_9task_200_299_v1)
DOSE_RESPONSE_L100 = {16: (289, 900), 24: (267, 900), 32: (267, 900), 40: (275, 900),
                      48: (285, 900), 56: (296, 900), 64: (279, 900)}


def read_probe(path: str) -> tuple:
    data = json.loads(Path(path).read_text())
    task = data["task"]
    seed = data["seed"]
    steps = {}
    for row in data.get("per_step", []):
        if "error" in row:
            return task, seed, None, row["error"]
        steps[row["step"]] = {k: row.get(f"m_k{k}") for k in KS}
    return task, seed, {"success": data.get("success"), "steps": steps,
                        "summary": data.get("summary"), "n_steps": data.get("steps")}, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()

    files = sorted(glob.glob(str(PROBE / "episodes/*/lambda_100/episode_*_probe.json")))
    if not files:
        print("no probe files yet")
        return 1
    with mp.Pool(args.jobs) as pool:
        results = pool.map(read_probe, files)
    errors = [{"task": t, "seed": s, "error": e} for t, s, d, e in results if e]
    episodes = {}
    for task, seed, data, error in results:
        if error:
            continue
        episodes.setdefault(task, {})[seed] = data
    print(f"probe episodes: {sum(len(v) for v in episodes.values())} across {len(episodes)} tasks, "
          f"{len(errors)} unreadable")

    per_task = {}
    for task, by_seed in sorted(episodes.items()):
        entry = {}
        for k in KS:
            step_values = [row[k] for data in by_seed.values() for row in data["steps"].values()
                           if row.get(k) is not None]
            episode_means = [float(np.mean([row[k] for row in data["steps"].values()
                                            if row.get(k) is not None]))
                             for data in by_seed.values()]
            entry[k] = {"step_level_mean": float(np.mean(step_values)),
                        "step_level_median": float(np.median(step_values)),
                        "step_level_p05": float(np.percentile(step_values, 5)),
                        "step_level_p95": float(np.percentile(step_values, 95)),
                        "episode_level_mean": float(np.mean(episode_means)),
                        "n_steps": len(step_values), "n_episodes": len(by_seed)}
        per_task[task] = entry

    pooled = {}
    for k in KS:
        values = []
        episode_means = []
        for task, by_seed in episodes.items():
            for data in by_seed.values():
                row_values = [row[k] for row in data["steps"].values() if row.get(k) is not None]
                values.extend(row_values)
                if row_values:
                    episode_means.append(float(np.mean(row_values)))
        pooled[k] = {"step_level_mean": float(np.mean(values)),
                     "step_level_median": float(np.median(values)),
                     "episode_level_mean": float(np.mean(episode_means)),
                     "n_steps": len(values), "n_episodes": len(episode_means)}

    # K=8 reproduction check against the historical matched reference
    reproduction = {"checked": 0, "exact": 0, "mismatches": []}
    for task, by_seed in episodes.items():
        for seed, data in by_seed.items():
            path = REFERENCE / "episodes" / task / "lambda_100" / f"episode_{seed:03d}_summary.json"
            if not path.is_file():
                continue
            reference = json.loads(path.read_text())
            reproduction["checked"] += 1
            reference_m = [int(row.get("m_t", -1)) for row in reference["selector_trace"]]
            probe_m = [data["steps"][step].get(8) for step in sorted(data["steps"])]
            if reference_m == probe_m:
                reproduction["exact"] += 1
            else:
                first = next((i for i, (a, b) in enumerate(zip(reference_m, probe_m)) if a != b), None)
                reproduction["mismatches"].append({"task": task, "seed": seed,
                                                   "steps_reference": len(reference_m),
                                                   "steps_probe": len(probe_m), "first_diff": first})

    def predict_k(mean_m: float, curve: dict, matched: tuple) -> dict:
        points = sorted(curve)
        if mean_m <= points[0]:
            low, high = points[0], points[0]
        elif mean_m >= points[-1]:
            low, high = points[-1], points[-1]
        else:
            high = next(p for p in points if p >= mean_m)
            low = max(p for p in points if p <= mean_m)
        def rate(point):
            return curve[point][0] / curve[point][1]
        if low == high:
            rate_low = rate_high = rate(low)
        else:
            weight = (mean_m - low) / (high - low)
            rate_low, rate_high = rate(low), rate(high)
            return {"between": [low, high], "interpolated_rate": rate_low + weight * (rate_high - rate_low),
                    "matched_rate": matched[0] / matched[1]}
        return {"nearest": low, "interpolated_rate": rate_low, "matched_rate": matched[0] / matched[1]}

    projection = {}
    for k in KS:
        projection[k] = {
            "pooled_mean_m_t": pooled[k]["step_level_mean"],
            "lambda050_4google": predict_k(pooled[k]["step_level_mean"], DOSE_RESPONSE_L050, MATCHED_L050),
            "lambda100_9task": predict_k(pooled[k]["step_level_mean"], DOSE_RESPONSE_L100, (281, 900)),
        }

    payload = {
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "scripts/analyze_openvla_k_probe.py",
        "artifact": str(PROBE.relative_to(REPO)),
        "ks": list(KS),
        "episodes_per_task": {SHORT[t]: len(v) for t, v in episodes.items()},
        "pooled": pooled,
        "per_task": {SHORT[t]: v for t, v in per_task.items()},
        "k8_reproduction": reproduction,
        "projection_onto_measured_dose_response": projection,
        "dose_response_sources": {
            "lambda050_4_google_seeds100_199": "artifacts/prompt_attn_l11_token_count_v1 (fixed budgets)",
            "lambda100_9task_seeds200_299": "artifacts/openvla_budget_rule_9task_200_299_v1 (fixed budgets)",
        },
        "errors": errors,
        "caveats": [
            "The probe records counterfactual budgets on the SAME states; only the K=8 column was "
            "executed. Budget comparisons across K are therefore within-state, but success-rate "
            "implications are NOT measured here.",
            "The projection line is an interpolation onto fixed-budget arms measured at lambda=0.5 "
            "(4 Google tasks, seeds 100-199) and lambda=1.0 (9 tasks, seeds 200-299). Fixed budgets "
            "are constant across steps while the matched budget varies, so the projection is an "
            "orientation aid, not a prediction with a confidence interval.",
            "At lambda=1.0 the nine-task study found every fixed budget statistically "
            "indistinguishable from matched, so budget size alone should not be expected to move "
            "lambda=1.0 results much.",
        ],
    }

    if args.write:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "OPENVLA_K_PROBE_BUDGET_CURVE.json").write_text(
            json.dumps(payload, indent=2, default=float) + "\n")
        report(payload)
        figure(payload, FIGDIR / "openvla_k_probe_budget_curve.png")
        print("wrote docs/OPENVLA_K_PROBE_BUDGET_CURVE.{json,md} and "
              "docs/figures/openvla_k_probe_budget_curve.png")
    else:
        print(json.dumps({"pooled": {k: round(v["step_level_mean"], 2) for k, v in pooled.items()},
                          "k8_reproduction": reproduction}, indent=2))
    return 0


def report(payload: dict) -> None:
    lines = [
        "# Matched budget versus KMeans granularity: measured m_t(K) on identical states",
        "",
        f"Generated {payload['generated_at']}. Intervention held at K=8; other K columns are "
        "counterfactual budgets on the same states.",
        "",
        f"Episodes per task: {payload['episodes_per_task']}",
        "",
        "## K=8 reproduction",
        "",
        f"checked {payload['k8_reproduction']['checked']} episodes, exact match on every step: "
        f"**{payload['k8_reproduction']['exact']}**",
        "",
        "## Pooled mean selected tokens per step",
        "",
        "| K | mean m_t | median | episode-level mean | steps | episodes |",
        "|---|---|---|---|---|---|",
    ]
    for k in payload["ks"]:
        entry = payload["pooled"][str(k)] if str(k) in payload["pooled"] else payload["pooled"][k]
        lines.append(f"| {k} | {entry['step_level_mean']:.2f} | {entry['step_level_median']:.1f} | "
                     f"{entry['episode_level_mean']:.2f} | {entry['n_steps']} | {entry['n_episodes']} |")
    tasks = list(payload["per_task"])
    lines += ["", "## Per task, mean m_t per step", "",
              "| task | " + " | ".join(f"K{k}" for k in payload["ks"]) + " |",
              "|---|" + "---|" * len(payload["ks"])]
    for task in tasks:
        entry = payload["per_task"][task]
        cells = []
        for k in payload["ks"]:
            value = entry.get(str(k), entry.get(k))
            cells.append(f"{value['step_level_mean']:.1f}" if value else "n/a")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines += ["", "## Where each K lands on the already-measured budget dose-response", "",
              "| K | pooled m_t | lambda 0.5 (4 Google tasks, fixed-budget ladder) | "
              "lambda 1.0 (9 tasks, fixed-budget ladder) |", "|---|---|---|---|"]
    for k in payload["ks"]:
        entry = payload["projection_onto_measured_dose_response"][str(k)] if \
            str(k) in payload["projection_onto_measured_dose_response"] else \
            payload["projection_onto_measured_dose_response"][k]
        half = entry["lambda050_4google"]
        full = entry["lambda100_9task"]
        half_text = (f"interpolated {100 * half['interpolated_rate']:.1f}% vs matched "
                     f"{100 * half['matched_rate']:.1f}%"
                     + (f" (between {half['between'][0]} and {half['between'][1]})"
                        if "between" in half else f" (nearest {half.get('nearest')})"))
        full_text = (f"interpolated {100 * full['interpolated_rate']:.1f}% vs matched "
                     f"{100 * full['matched_rate']:.1f}%"
                     + (f" (between {full['between'][0]} and {full['between'][1]})"
                        if "between" in full else f" (nearest {full.get('nearest')})"))
        lines.append(f"| {k} | {entry['pooled_mean_m_t']:.1f} | {half_text} | {full_text} |")
    lines += ["", "## Caveats", ""]
    lines += [f"- {note}" for note in payload["caveats"]]
    lines += ["", "Figure: `docs/figures/openvla_k_probe_budget_curve.png`.", ""]
    (DOCS / "OPENVLA_K_PROBE_BUDGET_CURVE.md").write_text("\n".join(lines))


def figure(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    axis = axes[0]
    ks = payload["ks"]
    for task in payload["per_task"]:
        values = [payload["per_task"][task][str(k)]["step_level_mean"]
                  if str(k) in payload["per_task"][task] else
                  payload["per_task"][task][k]["step_level_mean"] for k in ks]
        axis.plot(ks, values, marker="o", markersize=3.5, linewidth=1.2, alpha=0.75, label=task)
    pooled = [payload["pooled"][str(k)]["step_level_mean"] if str(k) in payload["pooled"]
              else payload["pooled"][k]["step_level_mean"] for k in ks]
    axis.plot(ks, pooled, marker="s", color="black", linewidth=2.2, label="pooled")
    axis.axhline(16, color="#c8641c", linestyle="--", linewidth=1.2)
    axis.text(max(ks), 16.8, "Fixed 16 (known under-budget arm)", ha="right", fontsize=8,
              color="#c8641c")
    axis.set_xlabel("KMeans K (cluster count)")
    axis.set_ylabel("mean selected tokens per step")
    axis.set_xticks(list(ks))
    axis.set_title("(a) Cluster count mechanically sets the budget\n"
                   "intervention held at K=8; other columns are same-state counterfactuals",
                   fontsize=10.5)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7.5, ncol=2)

    axis = axes[1]
    curve = {16: 151 / 400, 24: 173 / 400, 32: 196 / 400, 48: 207 / 400, 64: 198 / 400}
    xs = sorted(curve)
    axis.plot(xs, [100 * curve[x] for x in xs], marker="o", color="#1f4e9c",
              label="fixed budgets, lambda 0.5")
    axis.axhline(100 * 213 / 400, color="#1f4e9c", linestyle=":", linewidth=1.2,
                 label="matched, lambda 0.5 (53.2%)")
    for k in ks:
        entry = (payload["projection_onto_measured_dose_response"][str(k)]
                 if str(k) in payload["projection_onto_measured_dose_response"]
                 else payload["projection_onto_measured_dose_response"][k])
        axis.scatter([entry["pooled_mean_m_t"]], [100 * entry["lambda050_4google"]["interpolated_rate"]],
                     marker="v", s=42, color="#a02c2c", zorder=5)
        axis.annotate(f"K{k}", (entry["pooled_mean_m_t"],
                               100 * entry["lambda050_4google"]["interpolated_rate"]),
                      textcoords="offset points", xytext=(4, 4), fontsize=8, color="#a02c2c")
    axis.set_xlabel("mean selected tokens per step")
    axis.set_ylabel("success rate (%)")
    axis.set_title("(b) Each K mapped onto the measured lambda=0.5 budget ladder\n"
                   "red = where that K's budget sits; orientation only, not a prediction",
                   fontsize=10.5)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    figure_.suptitle("Is K=10 a finer estimate or just a smaller budget?", fontsize=12.5)
    figure_.tight_layout(rect=(0, 0, 1, 0.93))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figure_.savefig(path, dpi=165)
    plt.close(figure_)


if __name__ == "__main__":
    raise SystemExit(main())
