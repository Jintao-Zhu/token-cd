#!/usr/bin/env python3
"""Verdict analysis for the TopP90-vs-matched replication (lambda=1.0, seeds 100-199).

The two arms ran in the SAME process per chunk, so the pairing is internal and the
process history is shared. Output: successes/denominator + percent per task and pooled,
rescue/harm, McNemar exact p, task-stratified paired bootstrap CI, and an explicit
side-by-side with the two earlier measurements of the same contrast.

Outputs: docs/OPENVLA_TOPP90_REPLICATION_FINAL.{json,md}
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts/openvla_topp90_repl_100_199_v1"
DOCS = REPO / "docs"
TASKS = ["google_robot_close_drawer", "google_robot_open_drawer",
         "google_robot_pick_coke_can", "google_robot_move_near"]
SHORT = {"google_robot_close_drawer": "close_drawer", "google_robot_open_drawer": "open_drawer",
         "google_robot_pick_coke_can": "pick_coke_can", "google_robot_move_near": "move_near"}
# earlier measurements of the same contrast, quoted here for the verdict table
EARLIER = [
    ("λ=1.0, seeds 200-299, 4 Google 任务", 226, 201, 400, 0.0274,
     "网格实验（原始 p；Holm(8)=0.164）—— 9 任务实验的 4 任务子集，属同一份证据"),
    ("λ=1.0, seeds 200-299, 全部 9 任务", 293, 281, 900, 0.4604,
     "加入 WidowX 与 apple 后差异缩小到 +1.33pp"),
    ("λ=0.5, seeds 100-199, 4 Google 任务", 191, 213, 400, 0.0651,
     "旧管线（TopP 无 m_t 记录），方向相反：TopP90 明显更差"),
]


def load(job: tuple) -> tuple:
    arm, task = job
    rows = {}
    for path in glob.glob(str(ART / "episodes" / task / arm / "episode_*_summary.json")):
        data = json.loads(Path(path).read_text())
        rows[int(data["seed"])] = {"success": bool(data["success"]),
                                   "m_t": data.get("mean_selected_token_count"),
                                   "task": data.get("task")}
    return arm, task, rows


def contrast(a: dict, b: dict, tasks: list) -> dict:
    rescue = harm = both = neither = 0
    strata = []
    for task in tasks:
        seeds = sorted(set(a.get(task, {})) & set(b.get(task, {})))
        if not seeds:
            continue
        strata.append((np.array([a[task][s]["success"] for s in seeds], dtype=np.float64),
                       np.array([b[task][s]["success"] for s in seeds], dtype=np.float64)))
        for s in seeds:
            x, y = a[task][s]["success"], b[task][s]["success"]
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
    return {"n_pairs": total, "top_p_successes": both + rescue, "matched_successes": both + harm,
            "rescue_by_top_p": rescue, "harm_to_top_p": harm, "net": rescue - harm,
            "top_p_rate": (both + rescue) / total if total else None,
            "matched_rate": (both + harm) / total if total else None,
            "diff_pp": 100.0 * (rescue - harm) / total if total else None,
            "mcnemar_exact_p": pvalue, "_strata": strata}


def bootstrap(strata, rng, draws):
    if not strata:
        return {"observed_pp": None, "ci_low_pp": None, "ci_high_pp": None}
    observed = float(np.mean([a.mean() - b.mean() for a, b in strata])) * 100.0
    samples = np.empty(draws)
    for i in range(draws):
        samples[i] = np.mean([a[rng.integers(0, len(a), len(a))].mean()
                              - b[rng.integers(0, len(b), len(b))].mean() for a, b in strata]) * 100.0
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"observed_pp": observed, "ci_low_pp": float(low), "ci_high_pp": float(high),
            "draws": draws}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()
    rng = np.random.default_rng(20261016)
    jobs = [(arm, task) for arm in ("matched_l100", "top_p090_l100") for task in TASKS]
    data: dict[str, dict] = {"matched_l100": {}, "top_p090_l100": {}}
    with mp.Pool(args.jobs) as pool:
        for arm, task, rows in pool.imap_unordered(load, jobs):
            data[arm][task] = rows
    top, matched = data["top_p090_l100"], data["matched_l100"]
    coverage = {SHORT[t]: [len(top.get(t, {})), len(matched.get(t, {}))] for t in TASKS}
    complete = all(v[0] == 100 and v[1] == 100 for v in coverage.values())
    result = contrast(top, matched, TASKS)
    strata = result.pop("_strata")
    result["bootstrap"] = bootstrap(strata, rng, args.bootstrap)
    result["per_task"] = {}
    for task in TASKS:
        per = contrast({task: top.get(task, {})}, {task: matched.get(task, {})}, [task])
        per.pop("_strata")
        result["per_task"][SHORT[task]] = per

    def mean_mt(rows):
        values = [v["m_t"] for task in rows for v in rows[task].values() if v.get("m_t")]
        return float(np.mean(values)) if values else None

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "scripts/analyze_openvla_topp90_repl.py",
        "design": "matched_l100 and top_p090_l100 on seeds 100-199, 4 Google tasks, both arms in "
                  "the same process per chunk; independent of the seeds-200-299 evidence",
        "complete": complete, "coverage": coverage,
        "pooled": result, "m_t": {"top_p090": mean_mt(top), "matched": mean_mt(matched)},
        "earlier_measurements": [{"setting": s, "top_p": a, "matched": b, "n": n, "p": p, "note": note}
                                 for s, a, b, n, p, note in EARLIER],
        "verdict_rule": "If TopP90 wins again here with a CI excluding zero, the seeds-200-299 "
                        "advantage is corroborated. If the difference is near zero or negative, the "
                        "advantage was multiplicity noise.",
        "caveats": [
            "Seeds 100-199 are an independent seed range for THIS contrast at lambda=1.0, but the "
            "same states were used by older lambda=0.5 studies; that does not affect this test.",
            "Both arms were run in one process per chunk, so the pairs share process history.",
            "A single seed range still cannot separate 'TopP90 is better on Google tasks' from "
            "'this seed range happens to favour it'; the verdict combines three measurements.",
        ],
    }
    if args.write:
        (DOCS / "OPENVLA_TOPP90_REPLICATION_FINAL.json").write_text(
            json.dumps(payload, indent=2, default=float) + "\n")
        lines = ["# TopP90 vs matched 判决实验：λ=1.0，seeds 100–199（独立种子段）", "",
                 f"生成 {payload['generated_at']}。两臂在同一进程内成对运行。",
                 f"覆盖完整：**{complete}**  {coverage}", "",
                 "## 本次结果", "",
                 "| 条件 | 对数 | TopP90 | matched | 净 | 救/坏 | 差值 | bootstrap 95% CI | McNemar p |",
                 "|---|---|---|---|---|---|---|---|---|"]
        b = result["bootstrap"]
        lines.append(f"| 合并 | {result['n_pairs']} | {result['top_p_successes']} "
                     f"({100 * result['top_p_rate']:.1f}%) | {result['matched_successes']} "
                     f"({100 * result['matched_rate']:.1f}%) | {result['net']:+d} | "
                     f"{result['rescue_by_top_p']}/{result['harm_to_top_p']} | "
                     f"{result['diff_pp']:+.2f}pp | [{b['ci_low_pp']:+.2f}, {b['ci_high_pp']:+.2f}] | "
                     f"{result['mcnemar_exact_p']:.4f} |")
        for task, per in result["per_task"].items():
            lines.append(f"| {task} | {per['n_pairs']} | {per['top_p_successes']} | "
                         f"{per['matched_successes']} | {per['net']:+d} | "
                         f"{per['rescue_by_top_p']}/{per['harm_to_top_p']} | "
                         f"{per['diff_pp']:+.2f}pp | — | {per['mcnemar_exact_p']:.4f} |")
        lines += ["", f"实测预算：TopP90 平均 m_t **{payload['m_t']['top_p090']:.1f}** vs "
                      f"matched **{payload['m_t']['matched']:.1f}**", "",
                  "## 与先前三次测量的对照", "",
                  "| 设置 | TopP90 | matched | 差值 | p | 说明 |", "|---|---|---|---|---|---|"]
        for s, a, bb, n, p, note in EARLIER:
            lines.append(f"| {s} | {a}/{n} | {bb}/{n} | {100 * (a - bb) / n:+.2f}pp | {p:.4f} | {note} |")
        lines.append(f"| **本次（seeds 100–199, λ=1.0）** | {result['top_p_successes']}/"
                     f"{result['n_pairs']} | {result['matched_successes']}/{result['n_pairs']} | "
                     f"**{result['diff_pp']:+.2f}pp** | {result['mcnemar_exact_p']:.4f} | 独立种子段 |")
        lines += ["", "## 判据", "", f"- {payload['verdict_rule']}", "", "## 限制", ""]
        lines += [f"- {note}" for note in payload["caveats"]]
        (DOCS / "OPENVLA_TOPP90_REPLICATION_FINAL.md").write_text("\n".join(lines) + "\n")
        print("wrote docs/OPENVLA_TOPP90_REPLICATION_FINAL.{json,md}")
    else:
        print(json.dumps({"complete": complete, "n": result["n_pairs"],
                          "top_p": result["top_p_successes"], "matched": result["matched_successes"],
                          "diff_pp": result["diff_pp"], "p": result["mcnemar_exact_p"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
