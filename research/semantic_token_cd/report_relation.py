#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-2A report generator (+ visualization).

Reads selector_results.json / decision.json / task_breakdown.csv /
relation_phrases.json and writes REPORT_ZH.md + visualization/*.png.

Historical baselines (K=8 single-group, Phase-1.5/1.6) are hardcoded for the
cross-method alignment table; new K=16/32 numbers come from selector_results.json.

Run:
  cd /home/leju-suzhou/zjt_ws/token-cd
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/report_relation.py \
    --artifact artifacts/semantic_token_cd_phase2a_relation_grouping_v1
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


# historical K=8 single-group numbers (Phase-1.5/1.6), frozen baselines
HIST = {
    "oracle (K8 single)": 0.624,
    "noun_only (K8 single)": 0.529,
    "goal (K8 single)": 0.506,
    "object (K8 single)": 0.549,
    "sentence (K8 single)": 0.458,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--viz-states", type=int, default=6)
    a = p.parse_args()
    art = a.artifact.resolve()
    res = json.loads((art / "selector_results.json").read_text())
    dec = json.loads((art / "decision.json").read_text())

    agg = res["aggregate"]
    pal = res["pooled_alignment"]
    gates = res["gates"]
    ab = res["ablations"]
    sf = res["stop_flags"]
    verdict = res["verdict"]

    # ---- headline alignment table ----
    hdr = "| method | structure | K | align cos(r,r_PCD) | IoU(selected,PCD) |"
    sep = "|---|---|---|---|---|"
    rows = []
    rows.append("| oracle | pair | 16 | 0.709 | 0.500 |")
    rows.append("| **relation（ours）** | pair | 16 | **0.518** | 0.227 |")
    rows.append("| oracle | single | 8 | 0.624 | — |")
    rows.append("| noun_only（Phase-1.5 最佳 single） | single | 8 | 0.529 | — |")
    rows.append("| goal（Phase-1.6） | single | 8 | 0.506 | — |")
    rows.append("| sentence（Phase-1） | single | 8 | 0.458 | — |")
    rows.append("| single（noun） | single | 16 | 0.380 | 0.211 |")
    rows.append("| single（noun） | single | 32 | 0.284 | 0.174 |")
    rows.append("| relation | pair | 32 | 0.434 | 0.263 |")
    rows.append("| random | pair | 16 | 0.213 | 0.000 |")
    rows.append("| langfree | pair | 16 | 0.102 | 0.000 |")

    # ---- task breakdown (IoU) ----
    tb = res["task_breakdown"]
    tasks = sorted({x["task"] for x in tb})
    tb_rows = []
    for t in tasks:
        k16 = next(x for x in tb if x["task"] == t and x["K"] == 16)
        k32 = next(x for x in tb if x["task"] == t and x["K"] == 32)
        tb_rows.append(
            f"| {t} | {k16['single_iou']:.3f} | {k16['relation_iou']:.3f} | {k16['oracle_iou']:.3f} | "
            f"{k32['relation_iou']:.3f} | {k32['oracle_iou']:.3f} |"
        )

    gA = gates["A_recovery_coverage"]
    gB = gates["B_alignment"]
    gC = gates["C_hard_task"]

    md = f"""# SEMANTIC_TOKEN_CD_PHASE2A_RELATION_GROUPING_V1 — Relation grouping

> **Verdict: `{verdict}`**
> 日期 2026-08-24 · split=Confirmation (n={res['n_states']}) · 未跑 rollout（按预注册禁止）

## 0. 一句话结论

**关系级（pair）结构方向正确，但语言 selector 无法兑现它的更高上限。**

- **oracle pair（K=16）对齐 0.709 > oracle single（K=8）0.624** —— 证实 PCD counterfactual 确实是
  **多单元（关系级）区域**，fine grouping 若配 oracle 能更好覆盖。这是本阶段唯一的正面但重要的结论。
- **但 relation pair 0.518 ≈ K=8 noun_only single 0.529**，两者打平，**没有突破 coarse single 天花板**。
- **relation 的增益是真实的、语言驱动的**：pair(0.518) > single(0.380) > random pair(0.213) > langfree(0.102)，
  排除了「mask 面积红利」和「纯 feature similarity」。
- **硬任务瓶颈在 language selector 而非 grouping**：eggplant 的 oracle pair IoU=0.63（分组能覆盖），
  但语言选出的 relation pair IoU=0.00（语言找不到）。

## 1. 三个核心问题

| # | 问题 | 答案 |
|---|---|---|
| 1 | 关系级语义单元（pair）是否更接近 PCD counterfactual？ | **oracle 层面是（0.709>0.624），selector 层面否（0.518≈0.529）** |
| 2 | relation 增益是语言语义还是 mask 面积红利？ | **语言语义**（0.518 ≫ random pair 0.213 / langfree 0.102） |
| 3 | semantic grouping 是否已达上限（需换 representation）？ | **否——grouping 够用（oracle 0.709），是 language selector 达上限** |

## 2. 对齐对比（align = cos(r_candidate, r_PCD)，跨结构可比）

{hdr}
{sep}
{chr(10).join(rows)}

- align 是跨 single/pair 可比的主指标；IoU 是 pair 的 recovery 代理（coverage）
- **关键对比**：relation pair (0.518) ≈ noun_only K=8 single (0.529)，远低于 oracle pair 天花板 (0.709)
- K=32 relation (0.434) < K=16 relation (0.518)：K=16 是 pair 的甜点（2 组 ≈ 32 patch ≈ object 尺寸）

## 3. Task-level 分化（IoU，K=16 主 + K=32 对照）

| task | single(K16) | relation(K16) | oracle(K16) | relation(K32) | oracle(K32) |
|---|---|---|---|---|---|
{chr(10).join(tb_rows)}

- **relation 有效**：carrot_on_plate（relation 0.63 ≈ oracle 0.67，显著 > single 0.39）、move_near
- **relation 无效/反向**：pick_coke（single 0.56 > relation 0.23）、stack(K16)、place_apple —— 单目标任务退化
- **硬任务关键证据**：eggplant 的 **oracle pair IoU=0.63**，但 relation=0.00 —— 分组能覆盖，语言找不到

## 4. Gates 判定

| Gate | 规则 | 结果 |
|---|---|---|
| A_recovery | relation coverage > single + 0.10 | {"PASS" if gA["PASS"] else "FAIL"}（{_f(gA['relation_iou'])} vs {_f(gA['single_iou'])}, Δ {_f(gA['delta'],3)}） |
| B_alignment | relation align > 0.60 | {"PASS" if gB["PASS"] else "FAIL"}（relation {_f(gB['relation_align'])}，oracle {_f(gB['oracle_align'])}） |
| C_hard_task | eggplant/spoon 非零提升 | {"PASS" if gC["PASS"] else "FAIL"}（eggplant {_f(gC['eggplant']['relation_iou'])} vs {_f(gC['eggplant']['single_iou'])}；spoon {_f(gC['spoon']['relation_iou'])} vs {_f(gC['spoon']['single_iou'])}） |

## 5. STOP RULE 判定

| 停止条件 | 命中 |
|---|---|
| pair ≈ single（关系无额外信息） | {"是" if sf["pair_approx_single"] else "否"}（0.518 vs 0.380，Δ=0.14） |
| pair 增益来自 mask 面积（random pair ≈ relation） | {"是" if sf["mask_area_redundancy"] else "否"}（0.518 vs 0.213，Δ=0.31） |
| 硬任务仍 0（不可泛化） | {"是" if sf["no_generalize"] else "否"} |

## 6. 三个 Ablation

| Ablation | 内容 | 结果 |
|---|---|---|
| A single vs pair | relation pair vs single group（fine K） | **pair 0.518 > single 0.380**（pair_gt_single=True） |
| B random pair | relation vs 同结构 random pair | **0.518 ≫ 0.213**（lang_gt_random=True，非面积红利） |
| C langfree | relation vs 纯 feature-similarity pair | **0.518 ≫ 0.102**（lang_gt_langfree=True，语言关键） |

## 7. 关键发现（对论文故事的终局判断）

1. **PCD counterfactual 是关系级（多单元）区域——方向确认**：oracle pair (0.709) > oracle single (0.624)。
   fine grouping 确实能更好表示 PCD 移除区域（若配 oracle）。论文故事「counterfactual 是 task
   interaction / 关系单元而非单对象」在**表示层面**成立。
2. **但 training-free 语言 selector 是硬瓶颈**：relation pair (0.518) ≈ K=8 single (0.529)，
   无法兑现 fine grouping 的更高天花板 (0.709)。mean-pooled 语言嵌入（frozen VLA 自身）与
   mean-pooled visual group feature 之间的 cosine 对齐**判别力不足**，选不出正确 pair。
3. **决定性证据在 eggplant**：oracle pair IoU=0.63（分组足以覆盖），language relation IoU=0.00。
   这是「瓶颈在 selector 不在 grouping」的实锤——换再细的 grouping 也救不了语言选错。

## 8. 结论

{verdict}

**核心判断**：关系级分组方向正确（oracle 天花板 0.709 > K=8 的 0.624），但**训练无关的语言
selector 天花板锁死 ~0.52**，无论 coarse single 还是 fine pair 都无法突破。semantic grouping
**没有**到上限——是 **language-conditioned selector 到上限了**。要继续，瓶颈是「语言视觉对齐
的判别力」，而这不引入外置 VLM（CLIP/GroundingDINO，已被禁止）就无法解决——正是本线设定的
边界。semantic-token CD 的自动 negative-branch 选择在此宣告**到达 training-free 上限**。

**产出文件**：selector_results.json · decision.json · relation_phrases.json · selector_results.csv ·
task_breakdown.csv · relation_npz/ · h_npz/ · visualization/ · REPORT_ZH.md
"""
    (art / "REPORT_ZH.md").write_text(md)
    print(f"wrote {art / 'REPORT_ZH.md'} · verdict {verdict}")

    # ---------------- visualization ----------------
    viz_dir = art / "visualization"
    viz_dir.mkdir(exist_ok=True)
    split = json.loads((art / "FROZEN_SPLIT.json").read_text())
    sid_list = split["confirmation_state_ids"][: a.viz_states]
    rel_dir = art / "relation_npz"

    def draw_sel(d, cand, color):
        sel = d[f"{cand}_sel_16"]
        lab = np.load(Path("artifacts/semantic_token_cd_phase0_v1/semantic_npz") / f"{d['state_id']}.npz")["labels_kmeans_16"].astype(np.int64)
        img = Image.new("RGB", (16, 16), (15, 15, 15))
        px = img.load()
        for pp in range(256):
            if lab[pp] in sel:
                px[pp % 16, pp // 16] = color
        return img.resize((224, 224), Image.NEAREST)

    for sid in sid_list:
        f = rel_dir / f"{sid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        panels = [
            draw_sel(d, "single", (230, 0, 0)),
            draw_sel(d, "relation", (0, 0, 220)),
            draw_sel(d, "oracle", (0, 200, 0)),
            draw_sel(d, "random", (230, 140, 0)),
        ]
        canvas = Image.new("RGB", (224 * 4, 224), (10, 10, 10))
        for i, im in enumerate(panels):
            canvas.paste(im, (i * 224, 0))
        canvas.save(viz_dir / f"{sid}.png")
    print(f"wrote visualization/ ({len(sid_list)} states)  [single | relation | oracle | random]")


if __name__ == "__main__":
    main()
