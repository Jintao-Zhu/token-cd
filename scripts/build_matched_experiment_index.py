#!/usr/bin/env python3
"""Build ONE document cataloguing every matched-rule experiment in this repository.

The catalogue is generated from the filesystem (so counts are exact), grouped by the
question each experiment asks, and paired with a curated section of results that were
verified in analysis runs (with their source artefact and report file).

Outputs:
    docs/ALL_MATCHED_EXPERIMENTS_INDEX.md    main document
    docs/ALL_MATCHED_EXPERIMENTS_INDEX.tsv   machine-readable catalogue
    docs/ALL_MATCHED_EXPERIMENTS_INDEX.json  full detail
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
DOCS = REPO / "docs"

THEMES = [
    ("smoke",
     "G. 冒烟 / 预检 / 审计（不构成结论）",
     ("smoke", "preflight", "reproduction", "video_audit")),
    ("budget_rule_comparison",
     "A. 数量规则之争：matched vs 固定预算 / TopP / 随机 / 任务均值 / K 粒度",
     ("budget_rule", "budget_grid", "token_count", "top_p", "taskmean", "kmeans", "k10", "k_probe")),
    ("budget_quantity",
     "B. 数量维度的因果消融（把 matched 的预算改大改小、或打乱它与状态的对应）",
     ("dose_response", "causal_pilot", "smallscale", "state_coupling", "uniform16")),
    ("cross_model",
     "E. 跨模型：同一 matched 规则换架构",
     ("spatialvla", "pi0")),
    ("content_selection",
     "D. 内容/空间选择：选哪些 token（数量保持不变）",
     ("spatial_v", "geometry_selector", "factorization", "patch_ablation", "spatial")),
    ("generalisation",
     "F. 跨设置泛化：7 任务 × 300 seeds、两种渲染",
     ("7task", "4200", "drawers_0_299")),
    ("lambda_response",
     "C. 引导强度 λ：matched 随 λ 的行为",
     ("lambda", "guidance_strength")),
]
MODEL_HINTS = [("spatialvla", "SpatialVLA"), ("pi0_fractal", "pi0-FAST"),
               ("pi0fast", "pi0-FAST"), ("openvla", "OpenVLA"), ("seven_card", "OpenVLA (7-card)"),
               ("simpler", "OpenVLA"), ("prompt_attn", "OpenVLA"), ("l11", "OpenVLA")]
STATUS_OVERRIDE = {
    "openvla_k10_2lambda_200_299_v1": "进行中（真跑 K=10）",
    "openvla_kmeans_budget_9task_200_299_v1": "已停止（用户在漂移检出后叫停，主实验 0/1800）",
    "openvla_kmeans_budget_smoke_v1": "已停止（冒烟 7/9 后叫停）",
    "openvla_k_probe_200_219_v1": "已停止（13/180 后叫停，数据保留）",
}


def is_matched_arm(name: str) -> bool:
    return ("matched" in name) or name.startswith("lambda_")


def theme_of(artifact: str) -> str:
    for key, _, needles in THEMES:
        if any(n in artifact for n in needles):
            return key
    return "other"


def model_of(artifact: str) -> str:
    for needle, model in MODEL_HINTS:
        if needle in artifact:
            return model
    return "unknown"


def scan() -> list[dict]:
    records = []
    for artifact in sorted(p for p in ARTIFACTS.iterdir() if p.is_dir()):
        episodes = artifact / "episodes"
        if not episodes.is_dir():
            continue
        arms: dict[str, dict] = defaultdict(lambda: {"episodes": 0, "tasks": set(), "seeds": set()})
        all_arms = defaultdict(int)
        for arm_dir in episodes.glob("*/*"):
            if not arm_dir.is_dir():
                continue
            files = list(arm_dir.glob("episode_*_summary.json"))
            if not files:
                continue
            task, arm = arm_dir.parts[-2], arm_dir.parts[-1]
            all_arms[arm] += len(files)
            if not is_matched_arm(arm):
                continue
            entry = arms[arm]
            entry["episodes"] += len(files)
            entry["tasks"].add(task)
            for f in files:
                m = re.search(r"episode_(\d+)_summary", f.name)
                if m:
                    entry["seeds"].add(int(m.group(1)))
        if not arms:
            continue
        lock, lock_name = {}, None
        for name in ("EXPERIMENT_LOCK.json", "CONFIG_LOCK.json"):
            if (artifact / name).is_file():
                lock_name = name
                try:
                    lock = json.loads((artifact / name).read_text())
                except Exception:                                        # noqa: BLE001
                    lock = {}
                break
        arm_records = []
        for arm, entry in sorted(arms.items()):
            span = (max(entry["seeds"]) - min(entry["seeds"]) + 1) if entry["seeds"] else 0
            expected = len(entry["tasks"]) * span
            arm_records.append({
                "arm": arm, "episodes": entry["episodes"],
                "n_tasks": len(entry["tasks"]), "n_seeds": len(entry["seeds"]),
                "seed_lo": min(entry["seeds"]) if entry["seeds"] else None,
                "seed_hi": max(entry["seeds"]) if entry["seeds"] else None,
                "grid_expected_if_dense": expected,
                "complete_dense_grid": entry["episodes"] == expected,
            })
        lambdas = sorted({round(v, 3) for v in (lock.get("lambda") or {}).values()
                          if isinstance(v, (int, float))}) if isinstance(lock.get("lambda"), dict) else []
        seeds_all = sorted({s for entry in arms.values() for s in entry["seeds"]})
        records.append({
            "artifact": artifact.name,
            "path": str(artifact.relative_to(REPO)),
            "theme": theme_of(artifact.name),
            "model": model_of(artifact.name),
            "protocol_id": lock.get("protocol_id", ""),
            "lock_file": lock_name,
            "matched_episodes": sum(entry["episodes"] for entry in arms.values()),
            "matched_arms": arm_records,
            "other_arm_episodes": sum(n for a, n in all_arms.items() if not is_matched_arm(a)),
            "other_arms": sorted(a for a in all_arms if not is_matched_arm(a)),
            "tasks": sorted({t for entry in arms.values() for t in entry["tasks"]}),
            "seeds": [seeds_all[0], seeds_all[-1]] if seeds_all else [None, None],
            "lambdas": lambdas,
            "dense": all(a["complete_dense_grid"] for a in arm_records),
        })
    return records


# Verified in this session: numbers recomputed from the episode summaries, never quoted
# from memory.  Each block names the artefact it came from.
VERIFIED = [
    ("V1", "OpenVLA，9 任务，λ=1.0：matched 与 12 种预算配置",
     "`artifacts/openvla_lambda1_extension_200_299_v1` + `artifacts/openvla_budget_rule_9task_200_299_v1`",
     [["matched (K8)", "281/900", "31.2%", "基线"],
      ["fixed 16 / 24 / 32 / 40", "289 / 267 / 267 / 275 of 900", "32.1 / 29.7 / 29.7 / 30.6%", "全部 Holm 不显著"],
      ["fixed 48 / 56 / 64", "285 / 296 / 279 of 900", "31.7 / 32.9 / 31.0%", "全部 Holm 不显著"],
      ["TopP 0.75 / 0.80 / 0.85 / 0.90", "266 / 264 / 280 / 293 of 900", "29.6 / 29.3 / 31.1 / 32.6%", "全部 Holm 不显著"]],
     "不显著 ≠ 等效；九任务上没有任何预算规则被区分出来"),
    ("V2", "λ × 预算规则交互（4 Google 任务 × 100 seeds × 2 λ）",
     "`artifacts/openvla_lambda_budget_grid_200_299_v1`，报告 `docs/OPENVLA_BUDGET_LAMBDA_TASKMEAN_FINAL.md`",
     [["matched λ=0.5 / λ=1.0", "205/400 / 201/400", "51.25% / 50.25%", "自身对 λ 无显著变化（净 −4, p=0.79）"],
      ["fixed16 λ=0.5 / λ=1.0", "165/400 / 206/400", "41.25% / 51.50%", "λ 效应净 +41，Holm 0.001"],
      ["fixed48 λ=0.5 / λ=1.0", "199/400 / 202/400", "49.75% / 50.50%", "λ 效应净 +3，Holm 1.0"],
      ["random48 λ=0.5 / λ=1.0", "157/400 / 189/400", "39.25% / 47.25%", "λ 效应净 +32，Holm 0.014"],
      ["TopP0.90 λ=0.5 / λ=1.0", "201/400 / 226/400", "50.25% / 56.50%", "λ 效应净 +25，Holm 0.086"],
      ["交互量 I（vs matched）", "fixed16 +11.25pp、random48 +9.00、TopP90 +7.25、fixed48 +1.75",
       "CI [3.75,18.75] / [1.50,16.25] / [−0.50,15.00] / [−5.50,9.00]", "Holm 0.021 / 0.058 / 0.130 / 0.661"]],
     "唯一 Holm 显著的交互是「预算过小」的代价随 λ 缩小；这不能推广成「数量维度失效」"),
    ("V3", "去掉逐步自适应（任务均值常数）对 matched 的消融",
     "`artifacts/openvla_taskmean_2lambda_200_299_v1`（新，800 集）+ `artifacts/l11_taskmean_budget_100_199_v1`（旧，400 集）",
     [["taskmean λ=0.5 vs matched", "182/400 vs 205/400", "−5.75pp", "p=0.054，Holm 0.108（不显著）"],
      ["taskmean λ=1.0 vs matched", "199/400 vs 201/400", "−0.50pp", "p=0.93"],
      ["复现：λ=0.5，seeds 100–199", "185/400 vs 213/400", "−7.00pp", "p=0.012（另一 seed 区间显著）"]],
     "常数来自 λ=1 参考且用了同一批评估 seed，属于事后均值匹配消融；同一效应在两个 seed 区间一个显著一个不显著"),
    ("V4", "λ=0.5、seeds 100–199 的固定预算阶梯 vs matched（旧实验）",
     "`artifacts/prompt_attn_l11_token_count_v1`（matched 与 5 个固定档）",
     [["matched", "213/400", "53.25%", "同一批 task/seed"],
      ["fixed 16 / 24 / 32 / 48 / 64", "151 / 173 / 196 / 207 / 198 of 400", "37.75 / 43.25 / 49.00 / 51.75 / 49.50%",
       "预算过小时明显吃亏，48 档接近 matched"]],
     "这条阶梯是后来解释「K 调大 ⇒ 预算变小 ⇒ 变差」的参照系"),
    ("V5", "KMeans 粒度 K=16/32（漂移检出后停止，仅冒烟证据）",
     "`artifacts/openvla_kmeans_budget_smoke_v1`（7 集）",
     [["close_drawer/seed200 平均 m_t", "K8 43.5 / K16 8.2 / K32 6.5", "—", "K8 冒烟与历史逐 step 完全相同（113/113）"],
      ["pick_coke_can/seed200 平均 m_t", "K8 39.8 / K16 16.1 / K32 4.1", "—", "下降比 1/K 更陡"],
      ["主实验 1,800 集", "0/1800", "—", "启动前叫停，未产生主结果"]],
     "3 任务 × 1 seed，只能证明「K 改变的是预算规模」，不能证明效能"),
    ("V6", "同状态反事实探针：m_t 随 K 的曲线（提前停止）",
     "`artifacts/openvla_k_probe_200_219_v1`（13 集 + 1 集 preflight）",
     [["close_drawer", "K8 39.2 → K9 30.1 → K10 24.3 → K12 15.8 → K16 9.8", "K10/K8 = 0.62", "4 集"],
      ["open_drawer", "K8 36.2 → K9 42.3 → K10 37.1 → K12 16.8 → K16 7.4", "K10/K8 = 1.02", "3 集"],
      ["eggplant", "K8 29.5 → K9 29.2 → K10 27.7 → K12 26.6 → K16 23.6", "K10/K8 = 0.94", "3 集"]],
     "K=10 对预算的压缩幅度**依任务而异**；仅 3–4 集/任务，属初步"),
    ("V7", "真跑 K=10（进行中，λ=0.5 与 λ=1.0）",
     "`artifacts/openvla_k10_2lambda_200_299_v1`，与网格 matched 同 task/seed 配对",
     [["λ=1.0（396 对）", "K10 184/396 vs matched 199/396", "−3.79pp", "p=0.199，区间约 [−9.2, +1.6] 含 0"],
      ["λ=0.5（286 对，缺 pick）", "K10 160/286 vs matched 165/286", "−1.75pp", "p=0.63"],
      ["实测预算", "m_t 24.8–26.2 vs matched 33.6–35.5", "约 −27%", "预算确实被压小"]],
     "两档 λ 都略负但都不显著；move_near 在两个 λ 上都最负（−9 / −8）。**仍在跑**"),
    ("V8", "复用对照的完整性审计（10,800 集）",
     "`artifacts/openvla_kmeans_budget_9task_200_299_v1/CONTROL_REUSE_AUDIT.json`",
     [["K8 参考", "900/900", "technical_pass 全真", "—"],
      ["11 个对照臂", "9,900/9,900", "technical_pass 全真", "排除 20 集诊断臂"],
      ["配对身份", "900/900 个 (task,seed)", "初始 SHA + 指令一致", "0 失败"]],
     "只证明数据完整可复用，不是效能结论"),
]

KNOWN_ISSUES = [
    "**早期任务均值被污染**：某张任务表的 `ref_mt` 曾把 fixed64 的值混进 matched 均值，报出 45–57；"
    "正确值约为 close 35.2 / open 37.0 / pick 29.3 / move 33.3。未使用的其他任务常数（apple 57 等）仍是污染值，不得引用。",
    "**进程历史影响轨迹**：seed 分块（`[200]`、`[201–210]`…）不同会导致同一初始状态走出不同轨迹；"
    "比较必须用相同分块，且不能用「初始 SHA 相同」推断轨迹相同。",
    "**渲染器差异**：旧种子段（100–199）的 pick/move 与 200–299 不是同一渲染设置，跨种子段比较只能当方向性复现。",
    "**不显著 ≠ 等效**：多处曾用「打平」「差异消失」「不敏感」描述非显著结果，已作废；"
    "现在的表述一律给出观测差值与区间。",
    "**λ 只有离散测点**：「matched 对 λ 不敏感」这类说法在本仓库中没有被检验过（只测了若干 λ 点）。",
    "**K 不是粒度旋钮**：簇大小约 256/K，改 K 同时改预算；K=16/32 的 1,800 集因此没有跑（漂移在冒烟阶段被检出）。",
    "**随机内容对照的正确用法**：内容维度的干净对照是 fixed48 vs random48（同数量、不同内容）；"
    "matched vs random48 同时改变数量规则与内容，不能单独归因。",
]

REPORTS = [
    "docs/OPENVLA_KMEANS_BUDGET_9TASK_PLAN.md", "docs/OPENVLA_KMEANS_BUDGET_9TASK_TABLE.md",
    "docs/OPENVLA_KMEANS_BUDGET_9TASK_FINAL.md", "docs/OPENVLA_BUDGET_LAMBDA_TASKMEAN_FINAL.md",
    "docs/OPENVLA_K10_2LAMBDA_FINAL.md", "docs/OPENVLA_K_PROBE_BUDGET_CURVE.md",
    "docs/OPENVLA_BUDGET_RULE_9TASK_FINAL.md", "docs/OPENVLA_LAMBDA_BUDGET_GRID_FINAL.md",
    "docs/OPENVLA_BUDGET_RULE_REPRODUCIBILITY_FINDING.md",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    records = scan()
    by_theme = defaultdict(list)
    for record in records:
        by_theme[record["theme"]].append(record)

    total = sum(r["matched_episodes"] for r in records)
    lines = ["# matched 实验总目录（全部数据结果汇总）", "",
             f"生成时间 {time.strftime('%Y-%m-%d %H:%M:%S%z')} · 生成脚本 "
             f"`scripts/build_matched_experiment_index.py` · 直接扫描 `artifacts/` 得到", "",
             "## 0. 概览", "",
             f"- 含 matched（或 matched 变体）臂的 artifact：**{len(records)} 个**",
             f"- matched 干预总集数：**{total:,} 集**",
             f"- 覆盖模型：{', '.join(sorted({r['model'] for r in records}))}",
             f"- 单实验规模：{min(r['matched_episodes'] for r in records)} – "
             f"{max(r['matched_episodes'] for r in records):,} 集",
             f"- 跑满稠密网格的 artifact：{sum(1 for r in records if r['dense'])} / {len(records)}"
             "（其余为部分覆盖或已提前停止）", "",
             "## 1. 已核实的关键结果（数字均在本会话中从 episode summary 重算）", ""]
    for code, title, source, rows, note in VERIFIED:
        lines += [f"### {code}. {title}", "", f"来源：{source}", "",
                  "| 条件 | 成功/总数 | 比例或效应 | 备注 |", "|---|---|---|---|"]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", f"**解读与限制**：{note}", ""]

    lines += ["## 2. 完整目录（按主题分组）", ""]
    for key, title, _ in THEMES + [("other", "H. 未归类", ())]:
        group = by_theme.get(key)
        if not group:
            continue
        lines += [f"### {title}", "",
                  "| artifact | 模型 | 任务 | seeds | matched 集数 | matched 臂 | 对照臂数 | 稠密 | 状态 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for record in sorted(group, key=lambda r: -r["matched_episodes"]):
            status = STATUS_OVERRIDE.get(record["artifact"], "完成" if record["dense"] else "部分覆盖")
            arms = ", ".join(a["arm"] for a in record["matched_arms"])
            if len(arms) > 46:
                arms = arms[:43] + "…"
            lines.append(f"| `{record['artifact']}` | {record['model']} | {len(record['tasks'])} | "
                         f"{record['seeds'][0]}–{record['seeds'][1]} | {record['matched_episodes']:,} | "
                         f"{arms} | {len(record['other_arms'])} | {'是' if record['dense'] else '否'} | {status} |")
        lines.append("")
    lines += ["## 3. 已停止 / 未跑满的实验（不要当结论引用）", ""]
    for record in records:
        if record["artifact"] in STATUS_OVERRIDE or not record["dense"]:
            reason = STATUS_OVERRIDE.get(record["artifact"], "网格未填满")
            lines.append(f"- `{record['artifact']}`：{reason}（matched {record['matched_episodes']:,} 集）")
    lines += ["", "## 4. 数据质量与注意事项（引用本目录数字前必读）", ""]
    lines += [f"- {issue}" for issue in KNOWN_ISSUES]
    lines += ["", "## 5. 相关详细报告", ""]
    for report in REPORTS:
        path = REPO / report
        lines.append(f"- [`{report}`](<{report}>) — {'存在' if path.is_file() else '尚未生成'}")
    lines += ["", "## 6. 方法与口径说明", "",
              "- 「matched 集数」= 该 artifact 中所有 matched 类臂（名字含 `matched` 或以 `lambda_` 开头）"
              "的 episode summary 文件数，直接数文件得到。",
              "- 「稠密」= 每个臂的集数等于（任务数 × seed 跨度），用来发现缺块；"
              "它不判断该实验是否本来就不需要跑满。",
              "- 只有第 1 节列出的结果是本会话中重算并核对过的；其他 artifact 我只给出规模与设置，"
              "**不替它们填结论**。", ""]
    text = "\n".join(lines) + "\n"

    if args.write:
        DOCS.mkdir(parents=True, exist_ok=True)
        (DOCS / "ALL_MATCHED_EXPERIMENTS_INDEX.md").write_text(text)
        (DOCS / "ALL_MATCHED_EXPERIMENTS_INDEX.json").write_text(
            json.dumps({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "total_matched_episodes": total, "artifacts": records,
                        "verified": [{"code": c, "title": t, "source": s, "rows": r, "note": n}
                                     for c, t, s, r, n in VERIFIED],
                        "known_issues": KNOWN_ISSUES}, indent=2, ensure_ascii=False) + "\n")
        with (DOCS / "ALL_MATCHED_EXPERIMENTS_INDEX.tsv").open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["artifact", "theme", "model", "protocol_id", "matched_episodes",
                             "n_matched_arms", "n_other_arms", "tasks", "n_tasks", "seed_lo",
                             "seed_hi", "lambdas", "dense_grid", "status", "path"])
            for record in sorted(records, key=lambda r: -r["matched_episodes"]):
                writer.writerow([record["artifact"], record["theme"], record["model"],
                                 record["protocol_id"], record["matched_episodes"],
                                 len(record["matched_arms"]), len(record["other_arms"]),
                                 ";".join(record["tasks"]), len(record["tasks"]),
                                 record["seeds"][0], record["seeds"][1],
                                 ";".join(str(v) for v in record["lambdas"]),
                                 record["dense"],
                                 STATUS_OVERRIDE.get(record["artifact"],
                                                     "complete" if record["dense"] else "partial"),
                                 record["path"]])
        print(f"wrote docs/ALL_MATCHED_EXPERIMENTS_INDEX.{{md,json,tsv}} "
              f"({len(records)} artifacts, {total:,} matched episodes)")
    else:
        print(json.dumps({"artifacts": len(records), "matched_episodes": total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
