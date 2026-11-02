#!/usr/bin/env python3
"""Moderator analysis for the TopP90-vs-matched contrast.

Two questions the pooled number cannot answer on its own:
  1. does the effect depend on SCENE COMPLEXITY (proxied by how much budget matched
     itself needs on that episode, and by the TopP90/matched budget ratio)?
  2. is the seed-range difference (200-299 vs 100-199) still there once those
     covariates are accounted for, i.e. is "seed range" standing in for something else?

Method: on DISCORDANT pairs only (one arm succeeds, the other fails) fit a logistic
regression of "which arm won" on the covariates, plus per-quartile stratified effects
with task-stratified bootstrap intervals. Free of GPU work.

Outputs: docs/OPENVLA_TOPP90_MODERATORS.{json,md}, docs/figures/fig6_topp90_moderators.png
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
FIGDIR = DOCS / "figures"
TASKS = ["google_robot_close_drawer", "google_robot_open_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near"]
SHORT = {"google_robot_close_drawer": "close", "google_robot_open_drawer": "open",
         "google_robot_pick_coke_can": "pick", "google_robot_move_near": "move"}
RANGES = [
    ("200-299", "openvla_budget_rule_9task_200_299_v1", "top_p090_l100",
     "openvla_lambda1_extension_200_299_v1", "lambda_100"),
    ("100-199", "openvla_topp90_repl_100_199_v1", "top_p090_l100",
     "openvla_topp90_repl_100_199_v1", "matched_l100"),
]


def load(root: str, arm: str) -> dict:
    out = {}
    for path in glob.glob(str(REPO / "artifacts" / root / "episodes" / "*" / arm
                                / "episode_*_summary.json")):
        data = json.loads(Path(path).read_text())
        out.setdefault(data["task"], {})[int(data["seed"])] = {
            "success": bool(data["success"]),
            "m_t": data.get("mean_selected_token_count"),
            "steps": int(data.get("steps", 0)),
        }
    return out


def collect() -> list[dict]:
    pairs = []
    for label, a_root, a_arm, b_root, b_arm in RANGES:
        top, matched = load(a_root, a_arm), load(b_root, b_arm)
        for task in TASKS:
            for seed in sorted(set(top.get(task, {})) & set(matched.get(task, {}))):
                t, m = top[task][seed], matched[task][seed]
                if t["m_t"] is None or m["m_t"] is None:
                    continue
                pairs.append({
                    "range": label, "task": task, "seed": seed,
                    "top_success": int(t["success"]), "matched_success": int(m["success"]),
                    "top_mt": float(t["m_t"]), "matched_mt": float(m["m_t"]),
                    "budget_ratio": float(t["m_t"]) / float(m["m_t"]) if m["m_t"] else np.nan,
                    "steps": t["steps"],
                })
    return pairs


def effect(rows: list[dict], draws: int, rng: np.random.Generator) -> dict:
    rescue = sum(1 for r in rows if r["top_success"] and not r["matched_success"])
    harm = sum(1 for r in rows if r["matched_success"] and not r["top_success"])
    n = len(rows)
    if not n:
        return {"n": 0}
    pvalue = (float(binomtest(min(rescue, harm), rescue + harm, 0.5, alternative="two-sided").pvalue)
              if rescue + harm else 1.0)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    samples = np.empty(draws)
    for i in range(draws):
        acc = []
        for values in by_task.values():
            index = rng.integers(0, len(values), len(values))
            picked = [values[j] for j in index]
            res = sum(1 for r in picked if r["top_success"] and not r["matched_success"])
            ha = sum(1 for r in picked if r["matched_success"] and not r["top_success"])
            acc.append((res - ha) / len(picked))
        samples[i] = np.mean(acc) * 100.0
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"n": n, "top_successes": sum(r["top_success"] for r in rows),
            "matched_successes": sum(r["matched_success"] for r in rows),
            "rescue": rescue, "harm": harm, "diff_pp": 100.0 * (rescue - harm) / n,
            "ci_low_pp": float(low), "ci_high_pp": float(high), "p": pvalue,
            "mean_matched_mt": float(np.mean([r["matched_mt"] for r in rows])),
            "mean_top_mt": float(np.mean([r["top_mt"] for r in rows]))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()
    rng = np.random.default_rng(20261016)
    pairs = collect()
    payload = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "generator": "scripts/moderate_openvla_topp90.py",
               "n_pairs_total": len(pairs),
               "pooled": effect(pairs, args.bootstrap, rng),
               "by_range": {label: effect([r for r in pairs if r["range"] == label],
                                          args.bootstrap, rng) for label, *_ in RANGES},
               "by_task": {SHORT[t]: effect([r for r in pairs if r["task"] == t],
                                            args.bootstrap, rng) for t in TASKS}}

    # ---- stratified by the matched budget (scene/state complexity proxy) ----
    values = np.array([r["matched_mt"] for r in pairs])
    edges = np.percentile(values, [0, 25, 50, 75, 100])
    strata = []
    for index in range(4):
        low, high = edges[index], edges[index + 1]
        subset = [r for r in pairs
                  if (r["matched_mt"] >= low and (r["matched_mt"] < high or index == 3))]
        entry = effect(subset, args.bootstrap, rng)
        entry.update({"matched_mt_from": float(low), "matched_mt_to": float(high)})
        strata.append(entry)
    payload["by_matched_budget_quartile"] = strata

    # same, by how much MORE budget TopP90 used
    ratio = np.array([r["budget_ratio"] for r in pairs])
    redges = np.percentile(ratio, [0, 25, 50, 75, 100])
    ratio_strata = []
    for index in range(4):
        low, high = redges[index], redges[index + 1]
        subset = [r for r in pairs
                  if (r["budget_ratio"] >= low and (r["budget_ratio"] < high or index == 3))]
        entry = effect(subset, args.bootstrap, rng)
        entry.update({"budget_ratio_from": float(low), "budget_ratio_to": float(high)})
        ratio_strata.append(entry)
    payload["by_budget_ratio_quartile"] = ratio_strata

    # ---- discordant-only logistic regression: which arm wins ----
    discordant = [r for r in pairs if r["top_success"] != r["matched_success"]]
    design, labels, y = [], [], []
    for r in discordant:
        row = [r["matched_mt"], r["budget_ratio"], r["steps"]]
        row += [1.0 if r["task"] == t else 0.0 for t in TASKS[1:]]
        row += [1.0 if r["range"] == "100-199" else 0.0]
        design.append(row)
        y.append(1 if r["top_success"] else 0)
    labels = ["matched_m_t", "budget_ratio", "steps"] + \
             [f"task={SHORT[t]}" for t in TASKS[1:]] + ["range=100-199"]
    regression = None
    if len(set(y)) == 2 and len(y) > len(labels) + 5:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X = np.asarray(design, dtype=np.float64)
        scaler = StandardScaler().fit(X[:, :3])
        X[:, :3] = scaler.transform(X[:, :3])
        model = LogisticRegression(max_iter=5000, C=1.0).fit(X, np.asarray(y))
        # Wald p-values via the Fisher information
        probability = model.predict_proba(X)[:, 1]
        weights = probability * (1 - probability)
        information = X.T @ (X * weights[:, None])
        try:
            covariance = np.linalg.inv(information)
            standard = np.sqrt(np.diag(covariance))
            z = model.coef_[0] / standard
            pvalues = 2 * (1 - norm.cdf(np.abs(z)))
        except np.linalg.LinAlgError:
            standard = np.full(len(labels), np.nan)
            pvalues = np.full(len(labels), np.nan)
        regression = {"n_discordant": len(y), "top_wins": int(sum(y)),
                      "coefficients": [{"term": name, "coef": float(coef), "se": float(se),
                                        "p": float(p)} for name, coef, se, p in
                                       zip(labels, model.coef_[0], standard, pvalues)],
                      "note": "positive coefficient = more likely that TopP90 wins the pair; "
                              "the first three covariates were standardised"}
    payload["discordant_logistic"] = regression
    payload["verdict"] = {
        "pooled_pp": payload["pooled"]["diff_pp"],
        "pooled_ci": [payload["pooled"]["ci_low_pp"], payload["pooled"]["ci_high_pp"]],
        "range_gap_pp": payload["by_range"]["200-299"]["diff_pp"]
                        - payload["by_range"]["100-199"]["diff_pp"],
        "reading": "The pooled 800-pair estimate is the headline. If the stratified effects move "
                   "monotonically with a covariate, the effect is scene-dependent; if not, the "
                   "seed-range gap is not explained by these covariates.",
    }
    if args.write:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "OPENVLA_TOPP90_MODERATORS.json").write_text(
            json.dumps(payload, indent=2, default=float) + "\n")
        report(payload)
        figure(payload, FIGDIR / "fig6_topp90_moderators.png")
        print("wrote docs/OPENVLA_TOPP90_MODERATORS.{json,md} and docs/figures/fig6_topp90_moderators.png")
    else:
        print(json.dumps({"pooled": payload["pooled"], "by_range": payload["by_range"],
                          "strata": [(round(s.get("matched_mt_from", 0), 1),
                                      round(s.get("matched_mt_to", 0), 1),
                                      round(s.get("diff_pp", 0), 2), s.get("n"))
                                     for s in strata]}, indent=2, default=float))
    return 0


def report(payload: dict) -> None:
    lines = ["# TopP90 vs matched：调节变量分析（合并 800 对）", "",
             f"生成 {payload['generated_at']}。两臂同 task/seed 配对，仅用已有数据。", "",
             "## 合并与分段", "",
             "| 分段 | 对数 | TopP90 | matched | 差值 | bootstrap 95% CI | p |",
             "|---|---|---|---|---|---|---|"]
    for name, entry in [("合并（两段）", payload["pooled"])] + \
                       [(f"种子段 {k}", v) for k, v in payload["by_range"].items()] + \
                       [(f"任务 {k}", v) for k, v in payload["by_task"].items()]:
        lines.append(f"| {name} | {entry['n']} | {entry['top_successes']} "
                     f"({100 * entry['top_successes'] / entry['n']:.1f}%) | "
                     f"{entry['matched_successes']} "
                     f"({100 * entry['matched_successes'] / entry['n']:.1f}%) | "
                     f"{entry['diff_pp']:+.2f}pp | "
                     f"[{entry['ci_low_pp']:+.1f}, {entry['ci_high_pp']:+.1f}] | {entry['p']:.3f} |")
    lines += ["", "## 按 matched 自身需要的预算分层（场景/状态复杂度代理）", "",
              "| matched m_t 区间 | 对数 | 差值 | 95% CI | p |", "|---|---|---|---|---|"]
    for s in payload["by_matched_budget_quartile"]:
        lines.append(f"| {s['matched_mt_from']:.1f} – {s['matched_mt_to']:.1f} | {s['n']} | "
                     f"{s['diff_pp']:+.2f}pp | [{s['ci_low_pp']:+.1f}, {s['ci_high_pp']:+.1f}] | "
                     f"{s['p']:.3f} |")
    lines += ["", "## 按 TopP90/matched 预算比分层", "",
              "| 预算比区间 | 对数 | 差值 | 95% CI | p |", "|---|---|---|---|---|"]
    for s in payload["by_budget_ratio_quartile"]:
        lines.append(f"| {s['budget_ratio_from']:.2f} – {s['budget_ratio_to']:.2f} | {s['n']} | "
                     f"{s['diff_pp']:+.2f}pp | [{s['ci_low_pp']:+.1f}, {s['ci_high_pp']:+.1f}] | "
                     f"{s['p']:.3f} |")
    regression = payload.get("discordant_logistic")
    if regression:
        lines += ["", f"## 不一致对的 logistic 回归（n={regression['n_discordant']}，"
                      f"TopP90 赢 {regression['top_wins']} 对）", "",
                  "| 项 | 系数 | SE | p |", "|---|---|---|---|"]
        for term in regression["coefficients"]:
            lines.append(f"| {term['term']} | {term['coef']:+.3f} | {term['se']:.3f} | "
                         f"{term['p']:.3f} |")
        lines += ["", f"正系数 = TopP90 更可能赢下该对。{regression['note']}"]
    lines += ["", "## 读数", "",
              f"- 合并 800 对：**{payload['verdict']['pooled_pp']:+.2f}pp**，"
              f"CI [{payload['verdict']['pooled_ci'][0]:+.1f}, {payload['verdict']['pooled_ci'][1]:+.1f}]",
              f"- 两个种子段之差：**{payload['verdict']['range_gap_pp']:+.2f}pp**",
              f"- {payload['verdict']['reading']}",
              "", "图：`docs/figures/fig6_topp90_moderators.png`。", ""]
    (DOCS / "OPENVLA_TOPP90_MODERATORS.md").write_text("\n".join(lines))


def figure(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_, axes = plt.subplots(1, 3, figsize=(18.5, 5.8))
    axis = axes[0]
    groups = [("200-299", payload["by_range"]["200-299"]),
              ("100-199", payload["by_range"]["100-199"]),
              ("pooled", payload["pooled"])]
    positions = np.arange(len(groups))
    values = [g[1]["diff_pp"] for g in groups]
    errors = [[g[1]["diff_pp"] - g[1]["ci_low_pp"] for g in groups],
              [g[1]["ci_high_pp"] - g[1]["diff_pp"] for g in groups]]
    axis.bar(positions, values, 0.5, yerr=errors, capsize=4,
             color=["#c8641c", "#1f4e9c", "#444444"])
    for position, group in zip(positions, groups):
        axis.text(position, group[1]["diff_pp"] + (1.0 if group[1]["diff_pp"] >= 0 else -2.2),
                  f"{group[1]['diff_pp']:+.2f}pp\nn={group[1]['n']}", ha="center", fontsize=8.5)
    axis.axhline(0, color="black", linewidth=1.0)
    axis.set_xticks(positions, [g[0] for g in groups])
    axis.set_ylabel("TopP90 − matched (pp)")
    axis.set_ylim(-10, 12)
    axis.set_title("(a) the headline is the pooled 800 pairs", fontsize=10.5)

    axis = axes[1]
    strata = payload["by_matched_budget_quartile"]
    positions = np.arange(len(strata))
    values = [s["diff_pp"] for s in strata]
    axis.errorbar(values, positions,
                  xerr=[[s["diff_pp"] - s["ci_low_pp"] for s in strata],
                        [s["ci_high_pp"] - s["diff_pp"] for s in strata]],
                  fmt="o", color="#a02c2c", capsize=4)
    axis.axvline(0, color="black", linewidth=1.0)
    axis.set_yticks(positions, [f"m_t {s['matched_mt_from']:.0f}–{s['matched_mt_to']:.0f}\n"
                                f"(n={s['n']})" for s in strata], fontsize=8)
    axis.set_xlabel("TopP90 − matched (pp)")
    axis.set_title("(b) by matched's own budget\n(proxy for scene/state complexity)", fontsize=10.5)
    axis.set_xlim(-20, 20)

    axis = axes[2]
    strata = payload["by_budget_ratio_quartile"]
    positions = np.arange(len(strata))
    values = [s["diff_pp"] for s in strata]
    axis.errorbar(values, positions,
                  xerr=[[s["diff_pp"] - s["ci_low_pp"] for s in strata],
                        [s["ci_high_pp"] - s["diff_pp"] for s in strata]],
                  fmt="s", color="#7d3ca8", capsize=4)
    axis.axvline(0, color="black", linewidth=1.0)
    axis.set_yticks(positions, [f"ratio {s['budget_ratio_from']:.2f}–{s['budget_ratio_to']:.2f}\n"
                                f"(n={s['n']})" for s in strata], fontsize=8)
    axis.set_xlabel("TopP90 − matched (pp)")
    axis.set_title("(c) by how much more budget TopP90 used", fontsize=10.5)
    axis.set_xlim(-20, 20)

    figure_.suptitle("Is the TopP90 effect scene-dependent, or is it noise?", fontsize=12.5)
    figure_.tight_layout(rect=(0, 0, 1, 0.92))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figure_.savefig(path, dpi=165)
    plt.close(figure_)


if __name__ == "__main__":
    raise SystemExit(main())
