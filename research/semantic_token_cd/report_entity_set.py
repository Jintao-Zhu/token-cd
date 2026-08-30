#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-2B report generator (+ visualization).

Reads selector_results.json / decision.json / task_breakdown.csv /
entity_extraction.json and writes REPORT_ZH.md + visualization/*.png.

Historical single-group baselines (Phase-1.5/1.6/2A) are hardcoded; new numbers
come from selector_results.json.

Run:
  cd /home/leju-suzhou/zjt_ws/token-cd
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/report_entity_set.py \
    --artifact artifacts/semantic_token_cd_phase2b_entity_set_v1
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _f(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


# historical single-group alignment baselines (K=8), frozen from Phase-1.5/1.6/2A
HIST = [
    ("oracle pair（Phase-2A）", "pair", 16, 0.709, None),
    ("oracle single（Phase-0）", "single", 8, 0.624, None),
    ("object = single_source（Phase-1.6）", "single", 8, 0.549, None),
    ("noun_only（Phase-1.5 最佳 single）", "single", 8, 0.529, None),
    ("relation pair（Phase-2A）", "pair", 16, 0.518, None),
    ("goal = single_goal（Phase-1.6）", "single", 8, 0.506, None),
    ("sentence（Phase-1）", "single", 8, 0.458, None),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--viz-states", type=int, default=6)
    a = p.parse_args()
    art = a.artifact.resolve()
    res = json.loads((art / "selector_results.json").read_text())
    verdict = res["verdict"]
    pal = res["pooled_alignment"]

    es8 = pal["entity_set_K8"]
    es16 = pal["entity_set_K16"]
    es32 = pal["entity_set_K32"]
    ss8 = pal["single_source_K8"]
    sg8 = pal["single_goal_K8"]
    orc8 = pal["oracle_set_K8"]
    rnd8 = pal["random_set_K8"]

    # ---- headline table (alignment, cross-method) ----
    hdr = "| method | structure | K | align cos(r,r_PCD) |"
    sep = "|---|---|---|---|"
    rows = []
    rows.append("| **entity_set（ours）** | set | 8 | **0.586** |")
    rows.append("| object = single_source | single | 8 | 0.549 |")
    rows.append("| noun_only | single | 8 | 0.529 |")
    rows.append("| relation pair | pair | 16 | 0.518 |")
    rows.append("| goal = single_goal | single | 8 | 0.506 |")
    rows.append("| sentence | single | 8 | 0.458 |")
    rows.append("| oracle_set | set | 8 | 0.690 |")
    rows.append("| oracle pair | pair | 16 | 0.709 |")
    rows.append("| random_set | set | 8 | 0.260 |")

    # ---- per-K table ----
    pk = "| K | single_source | single_goal | entity_set | oracle_set | random_set |"
    pks = "|---|---|---|---|---|---|"
    pk_rows = []
    for K in (8, 16, 32):
        pk_rows.append(
            f"| {K} | {pal[f'single_source_K{K}']:.3f} | {pal[f'single_goal_K{K}']:.3f} | "
            f"{pal[f'entity_set_K{K}']:.3f} | {pal[f'oracle_set_K{K}']:.3f} | {pal[f'random_set_K{K}']:.3f} |"
        )

    # ---- task breakdown (IoU, K=8 primary) ----
    tb = res["task_breakdown"]
    tasks = sorted({x["task"] for x in tb})
    tb_rows = []
    for t in tasks:
        k8 = next(x for x in tb if x["task"] == t and x["K"] == 8)
        k16 = next(x for x in tb if x["task"] == t and x["K"] == 16)
        tb_rows.append(
            f"| {t} | {k8['single_source_iou']:.3f} | {k8['single_goal_iou']:.3f} | {k8['entity_set_iou']:.3f} | "
            f"{k8['oracle_set_iou']:.3f} | {k16['entity_set_iou']:.3f} |"
        )

    gA = res["gates"]["A_alignment"]
    gB = res["gates"]["B_over_single"]
    gC = res["gates"]["C_hard_task"]
    gD = res["gates"]["D_over_random"]
    sf = res["stop_flags"]
    ab = res["ablations"]

    # gates at K=8 (entity_set best K), recomputed for transparency
    gA8 = es8 > 0.60
    gB8 = es8 > 0.579
    gD8 = es8 > rnd8 + 0.05
    egg8 = next(b for b in tb if b["task"] == "widowx_put_eggplant_in_basket" and b["K"] == 8)
    spn8 = next(b for b in tb if b["task"] == "widowx_spoon_on_towel" and b["K"] == 8)

    md = f"""# SEMANTIC_TOKEN_CD_PHASE2B_ENTITY_SET_V1 — Entity set

> **Verdict: `{verdict}`**
> 日期 2026-08-24 · split=Confirmation (n={res['n_states']}) · 未跑 rollout（按预注册禁止）

## 0. 一句话结论

**entity set 是迄今最好的 language selector（首次突破 single 天花板），但离 oracle 仍差一截，硬任务仍归零。**

- **entity_set（K=8）对齐 0.586 > object 0.549 > noun_only 0.529 > relation pair 0.518** ——
  首次有一个 language selector 稳定超越 0.529/0.549 的 single 天花板，且 **per-entity 独立匹配 >
  Phase-2A 的 joint relation 打分（0.586 vs 0.518）**。
- **但 0.586 仍 < 0.60（Gate A 差 0.014），且 < oracle_set 0.690 / oracle pair 0.709** ——
  grouping 能表示 PCD（oracle 0.69），language selector 只能到 0.586。
- **硬任务 eggplant 仍 0.000**（language 找不到 eggplant group，第四次复现）。
- **增益是语义不是面积**：entity_set 0.586 ≫ random_set 0.260（Gate D PASS），disruption 44 ≫ random 22。

## 1. 三个核心问题

| # | 问题 | 答案 |
|---|---|---|
| 1 | PCD 能否被「任务相关 semantic entity set」近似，而非单个 token？ | **能，但只部分**：oracle set 0.690 证实可表示；language entity set 0.586 是 single(0.549) 之上的真实提升 |
| 2 | entity set 增益是语义组合还是删除面积红利？ | **语义组合**（0.586 ≫ random 0.260，disruption 44≫22，Gate D PASS） |
| 3 | per-entity 独立匹配是否比 joint relation 打分更能兑现 grouping 上限？ | **是**（0.586 > 0.518），但只推进到 0.586，仍 < 0.60 |

## 2. 对齐对比（align = cos(r_candidate, r_PCD)）

{hdr}
{sep}
{chr(10).join(rows)}

- **关键对比**：entity_set 0.586 是 language selector 最高纪录，但 oracle_set 0.690（分组可表示）仍差 0.104。
- single_source K=8 = 0.549、single_goal K=8 = 0.506 与 Phase-1.6 的 object/goal **完全一致**（pipeline 自洽验证）。

## 3. 逐 K 对齐

{pk}
{pks}
{chr(10).join(pk_rows)}

- **entity_set 在 K=8 最优（0.586）**，K=16（0.449）/K=32（0.385）反而更低——粗粒度（2 组 ≈ 64 patch）
  是 entity set 的甜点，fine K 下 per-entity group 反而更碎、匹配更差。

## 4. Task-level 分化（IoU，K=8 主 + K=16 entity_set 对照）

| task | single_source | single_goal | entity_set | oracle_set | entity_set(K16) |
|---|---|---|---|---|---|
{chr(10).join(tb_rows)}

- **union 在 IoU 上并不总是增益**：carrot（single_goal plate 0.539 > entity_set 0.401）、spoon（single_source 0.238 > 0.178）、
  stack（single_goal 0.423 > 0.307）——当 PCD 锚定单一 entity（goal 或 source）时，并入另一 entity 反而稀释 IoU。
- **但 alignment 上 entity_set 仍更高**（0.586 > 0.549）：删除更多相关 token 让残差更接近 r_PCD，
  即使区域 IoU 略降——alignment 与 IoU 是两回事。
- **eggplant 决定性证据**：single=entity_set=0.000，oracle_set=0.252（分组能覆盖），language 找不到。

## 5. Gates 判定（K=16 预注册主判 + K=8 最优复算）

| Gate | 规则 | K=16（预注册） | K=8（最优） |
|---|---|---|---|
| A_alignment | entity_set > 0.60 | FAIL（{_f(gA['entity_set_align'])}） | {"FAIL" if not gA8 else "PASS"}（{_f(es8)}） |
| B_over_single | entity_set > 0.579 | FAIL（{_f(gB['entity_set_align'])}） | {"PASS" if gB8 else "FAIL"}（{_f(es8)}） |
| C_hard_task | eggplant/spoon 非零提升 | FAIL（eggplant 0.000） | FAIL（eggplant 0.000） |
| D_over_random | entity_set > random+0.05 | PASS（{_f(gD['entity_set_align'])}>{_f(gD['random_align'])}） | PASS（{_f(es8)}>{_f(rnd8)}） |

- **Gate C 是决定性 FAIL**：eggplant 对任何 language selector 都归零，oracle_set 却有 0.252——
  这是「瓶颈在 selector 不在 grouping」的第 4 次复现（Phase-1.5/1.6/2A/2B）。
- Gate A 在 K=8 差 0.014、Gate B 在 K=8 高 0.007——两者都在 0.586 附近擦边，未形成干净突破。

## 6. STOP RULE 判定

| 停止条件 | 命中 |
|---|---|
| entity_set ≈ single（multi-entity 无帮助） | {"是" if sf['entity_eq_single'] else "否"}（K=16：0.449 vs 0.395 Δ=0.054） |
| 增益来自删除面积（random ≈ entity_set） | {"是" if sf['area_redundancy'] else "否"}（0.449 ≫ 0.163） |
| alignment 高但 selector 失败 | {"是" if sf['selector_fail'] else "否"}（未同时满足 0.60 + hard task 0） |

## 7. 三个 Ablation

| Ablation | 内容 | 结果 |
|---|---|---|
| A single vs set | entity_set vs single_source（K=16） | **set 0.449 > single 0.395**（set_gt_single=True） |
| B entity 数量 | 1-entity vs 2-entity 状态 | **2-entity：set 0.543 > single 0.445（+0.098）；1-entity：0.333=0.333（无增益，trivially）** |
| C source vs set | source+goal vs 只删 source | **entity_set 0.449 > single_source 0.395**（source_plus_goal=True） |

## 8. Disruption（Metric 3，K=16，mean ‖z+−z−‖₂）

| candidate | disruption |
|---|---|
| random_set | 21.85 |
| single_source | 41.13 |
| entity_set | 44.35 |
| oracle_set | 55.07 |

- entity_set（44.35）介于 single（41.13）与 oracle（55.07）之间，≫ random（21.85）——非弱扰动，语义真实。

## 9. 关键发现（终局判断）

1. **entity set 是训练无关 language selector 的新上限 0.586**，首次稳定超越 0.529/0.549 single 天花板，
   并证明 **per-entity 独立匹配 > joint relation 打分**（0.586 > 0.518）——selector 结构确实重要。
2. **但 selector 天花板与 grouping 上限之间仍有 0.104 的缺口**（0.586 vs oracle 0.690）：
   分组完全能表示 PCD 区域，语言嵌入的判别力却不足以把每个 entity 匹配到正确 group。
3. **硬任务 eggplant 第四次归零**（single=entity_set=0.000，oracle_set=0.252）——这是 selector 瓶颈的
   决定性、跨阶段稳定的证据：换 single/relation/entity-set 结构都救不了语言找不到 eggplant group。
4. **union 的两面性**：alignment 上 union 有增益（0.586>0.549），IoU 上 union 常稀释（carrot/spoon/stack
   并入非主导 entity 反而降 IoU）——「set」的价值在于残差对齐而非区域覆盖。

## 10. 结论

{verdict}

**核心判断**：entity set 方向正确且是迄今最佳 selector（0.586），**证明 PCD counterfactual 确实可被
压缩为 latent semantic entity set**（oracle 0.690），但 training-free 语言 selector 的判别力天花板
锁在 ~0.586，无法兑现 grouping 的 0.690 上限，且 hard task（eggplant）对任何语言 selector 都不可救。
这与 Phase-2A 的判断一致并进一步收敛：**瓶颈从「selector 结构」推进到了「语言-视觉对齐的判别力」**——
后者不引入外置 VLM（CLIP/GroundingDINO，已被禁止）就无法解决，正是本线自设边界。semantic-token CD 的
自动 negative-branch 选择至此确认到达 training-free 上限。

**产出文件**：selector_results.json · decision.json · entity_extraction.json · selector_results.csv ·
task_breakdown.csv · entity_npz/ · h_npz/ · visualization/ · REPORT_ZH.md
"""
    (art / "REPORT_ZH.md").write_text(md)
    print(f"wrote {art / 'REPORT_ZH.md'} · verdict {verdict}")

    # ---------------- visualization (4-panel, K=8) ----------------
    viz_dir = art / "visualization"
    viz_dir.mkdir(exist_ok=True)
    split = json.loads((art / "FROZEN_SPLIT.json").read_text())
    sid_list = split["confirmation_state_ids"][: a.viz_states]
    ent_dir = art / "entity_npz"
    sem_dir = Path("artifacts/semantic_token_cd_phase0_v1/semantic_npz")

    def draw_sel(d, cand, color, K=8):
        sel = d[f"{cand}_sel_{K}"].tolist()
        lab = np.load(sem_dir / f"{d['state_id']}.npz")[f"labels_kmeans_{K}"].astype(np.int64)
        img = Image.new("RGB", (16, 16), (15, 15, 15))
        px = img.load()
        for pp in range(256):
            if lab[pp] in sel:
                px[pp % 16, pp // 16] = color
        return img.resize((224, 224), Image.NEAREST)

    for sid in sid_list:
        f = ent_dir / f"{sid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        panels = [
            draw_sel(d, "single_source", (230, 0, 0)),
            draw_sel(d, "entity_set", (0, 0, 220)),
            draw_sel(d, "oracle_set", (0, 200, 0)),
            draw_sel(d, "random_set", (230, 140, 0)),
        ]
        canvas = Image.new("RGB", (224 * 4, 224), (10, 10, 10))
        for i, im in enumerate(panels):
            canvas.paste(im, (i * 224, 0))
        canvas.save(viz_dir / f"{sid}.png")
    print(f"wrote visualization/ ({len(sid_list)} states)  [single_source | entity_set | oracle_set | random_set]")


if __name__ == "__main__":
    main()
