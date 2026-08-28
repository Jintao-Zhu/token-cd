#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-1.6 report generator (+ lightweight visualization).

Reads selector_results.json / decision.json / task_breakdown.csv /
goal_phrase.json and writes REPORT_ZH.md + visualization/*.png.

Run:
  cd /data/docker/dev_zjt/data/code
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/report_goal_selector.py \
    --artifact artifacts/semantic_token_cd_phase1_6_goal_selector_v1
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

    sel_order = ("oracle", "noun_only", "goal_phrase", "goal", "object", "sentence", "wrong_goal", "random")
    sel_label = {
        "oracle": "oracle（IoU 上界）", "noun_only": "noun_only（object+goal 全名词）",
        "goal_phrase": "goal_phrase（prep+goal）", "goal": "goal（目标位置名词）",
        "object": "object（直接宾语）", "sentence": "sentence（整句基线）",
        "wrong_goal": "wrong_goal（轮换替换）", "random": "random",
    }
    sel_rows = []
    for s in sel_order:
        r = agg[s]
        sel_rows.append(f"| {sel_label[s]} | {r['oracle_recovery']:.3f} | {pal[s]:.3f} | {r['iou_median']:.3f} |")

    tb_rows = []
    for tb in res["task_breakdown"]:
        tb_rows.append(
            f"| {tb['task']} | {tb['sentence']:.3f} | {tb['object']:.3f} | {tb['noun_only']:.3f} | "
            f"{tb['goal']:.3f} | {tb['goal_phrase']:.3f} | {tb['wrong_goal']:.3f} |"
        )

    gA = gates["A_recovery_goal_vs_noun"]
    gB = gates["B_alignment_goal_vs_noun"]
    gC = gates["C_failed_task_rescue"]

    md = f"""# SEMANTIC_TOKEN_CD_PHASE1_6_GOAL_SELECTOR_V1 — Goal-conditioned selector

> **Verdict: `{verdict}`**
> 日期 2026-08-24 · split=Confirmation (n={res['n_states']}) · 未跑 rollout（按预注册禁止）

## 0. 一句话结论

**「PCD 移除的是 goal condition 而不是 object」这一诊断被证实；但「goal-only 是最优 selector」被否决。**

- **goal（目标位置名词）65.9% > object（直接宾语）53.3%** —— PCD counterfactual 锚定在目标位置，不是被操纵对象。
  这是本阶段最重要的**正面**发现。
- **wrong_goal control 通过**：goal 65.9% ≫ wrong_goal 20.0% ≫ random 9.6% —— goal 语义是**特定且真实**的，不是「任意名词都指向显著组」的伪信号。
- **但 goal-only 不敌 object+goal**：goal 65.9% < noun_only（全名词）71.1%，Gate A 要求 +5pp 反而 -5.2pp。
  PCD 移除的区域**横跨 goal+object 两个名词**，单纯 goal 不是最优 ground。
- **硬任务仍救不活**：eggplant / spoon 对**所有**语言 variant 都归零（K-means@8 分组本身无法分离小/歧义对象）。

## 1. 三个核心问题（协议要求回答）

| # | 问题 | 答案 |
|---|---|---|
| 1 | PCD 删除的是 object 还是 goal condition？ | **goal 为主**（goal 65.9% > object 53.3%），但完整区域 = goal + object（noun_only 71.1% 最高） |
| 2 | goal phrase grounding 是否优于 object grounding？ | **优于 object-only**（65.9% > 53.3%），**不优于 object+goal**（65.9% < 71.1%） |
| 3 | semantic token CD 能否自动选择 negative branch？ | **否**（goal 65.9% 离 oracle 100% 远，硬任务 0%） |

## 2. Selector 对比（Metric 1 recovery / 2 alignment / IoU）

| selector | top-1 oracle recovery | align cos(r,r_PCD) | IoU(selected,obj) |
|---|---|---|---|
{chr(10).join(sel_rows)}

- oracle recovery = top-1 命中 Phase-0 oracle group 比例（random 理论 12.5%，实测 9.6%）
- align = cos(r_selected, r_PCD)，mean over states×7（oracle 上界 0.624）
- 注：goal recovery 高于 object，但 goal align (0.506) **低于** object align (0.549) —— alignment 与 recovery 再次脱钩（同 Phase-1 发现）

## 3. Task-level 分化（Metric 3）

| task | sentence | object | noun_only | goal | goal_phrase | wrong_goal |
|---|---|---|---|---|---|---|
{chr(10).join(tb_rows)}

- **goal 强**：stack_cube（goal `yellow block` 1.00）、place_apple（1.00）、pick_coke（0.93）、carrot（goal `plate` 0.87 vs object `carrot` 0.13）、open_drawer（0.80）
- **goal 救不活**：put_eggplant_in_basket（goal `yellow basket` 0.00）、spoon_on_towel（goal `towel` 0.00）
- **wrong_goal 异常点**：spoon 的 wrong_goal（`yellow basket`）0.53 > 其正确 goal（`towel`）0.00；stack 的 wrong_goal（`blue plastic bottle`）0.80 —— 说明部分对象名词共享「objectness」子空间，wrong-goal control 在多对象任务上有污染

## 4. Gates 判定（Goal vs Object=noun_only 71.1%）

| Gate | 规则 | 结果 |
|---|---|---|
| A_recovery | goal > noun_only + 0.05 | {"PASS" if gA["PASS"] else "FAIL"}（goal {_f(gA['goal'])} vs noun_only {_f(gA['noun_only'])}, Δ {_f(gA['delta'],3)}） |
| B_alignment | goal align > noun_only | {"PASS" if gB["PASS"] else "FAIL"}（goal {_f(gB['goal'])} vs noun_only {_f(gB['noun_only'])}） |
| C_failed_rescue | eggplant/spoon 至少一个 goal ≥ 0.50 | {"PASS" if gC["PASS"] else "FAIL"}（eggplant {_f(gC['failed_goal']['widowx_put_eggplant_in_basket'])}, spoon {_f(gC['failed_goal']['widowx_spoon_on_towel'])}) |

## 5. STOP RULE 判定

| 停止条件 | 命中 |
|---|---|
| goal ≈ object（selector 已到天花板） | {"是" if sf["goal_approx_object"] else "否"}（Δ=5.2pp > 5pp） |
| goal 提升 recovery 但 alignment 不提升 | {"是" if sf["recovery_no_align"] else "否"} |
| goal 仍无法处理多对象任务（grouping 不足） | {"是" if sf["multiobject_unresolved"] else "否"} |

## 6. 三个 Ablation（本阶段新增证据）

| Ablation | 内容 | 结果 |
|---|---|---|
| A1 goal vs object | goal(plate) vs object(carrot) | goal **65.9%** > object 53.3%（goal_gt_object=True）—— **PCD 是 goal 条件** |
| A2 phrase 长度 | noun(plate) vs phrase(on plate) vs full(整句) | noun 65.9% ≈ phrase 66.7% ≫ full 52.6% —— **介词无关，去掉动词/object 才是增益来源** |
| A3 wrong-goal | goal 轮换替换 | goal 65.9% ≫ wrong_goal 20.0%（goal_gt_wrong=True）—— **goal 语义真实且特定** |

## 7. 关键发现（对论文故事的影响）

1. **「counterfactual 单元是 goal 不是 object」成立**（A1 + A3 双重确认）。这是本线最硬的正面结果：
   `put carrot on plate` 移除的是 `plate`（goal 0.87）而非 `carrot`（object 0.13）。论文故事可写成
   **「VLA 的正确 counterfactual 单元是任务目标条件（goal），不是视觉对象（object）」**。
2. **但 goal-only 不是最优 selector**（Gate A/B FAIL）：goal 65.9% < noun_only 71.1%。PCD 移除区域
   实际横跨 goal + object 两个名词（noun_only 最接近）。「goal-conditioned semantic CD」若要成立，
   其 negative branch 应基于 **goal + object 全名词**，而非 goal 单一名词。
3. **selector 天花板 ~70% + 硬任务 0%**（Gate C FAIL）：eggplant/spoon 小对象任务 K-means@8 分组
   无法干净分离，任何语言 variant（goal/object/noun_only/wrong_goal）都归零。语言 grounding 的
   **信息上限被分组质量封死**，不是语言公式能救的。
4. **wrong-goal control 揭示 objectness 子空间污染**：spoon 的 wrong_goal（`yellow basket` 0.53）
   > 正确 goal（`towel` 0.00）。部分对象名词在 embedding 空间共享「objectness」方向，goal 特异性
   在**歧义/小对象**任务上会失效——这解释了为什么 wrong_goal 整体 20% 仍 > random 9.6%。

## 8. 结论

{verdict}

**核心判断**：本阶段**证实了 PCD 的 counterfactual 是 goal condition（不是 object）**——这是比
object-centric 更正确的诊断，且 wrong-goal control 证明该语义是真实的。但 **goal-only selector
不敌 object+goal 全名词 grounding（65.9% < 71.1%），且硬任务仍 0%**，不满足进入 rollout 的 gates。
semantic-token CD 的自动 negative-branch 选择**仍不具备可靠性**：语言 grounding 天花板约 70%，
被 K-means@8 分组质量封死。

**产出文件**：selector_results.json · decision.json · goal_phrase.json · selector_results.csv ·
task_breakdown.csv · visualization/ · REPORT_ZH.md
"""
    (art / "REPORT_ZH.md").write_text(md)
    print(f"wrote {art / 'REPORT_ZH.md'} · verdict {verdict}")

    # ---------------- lightweight visualization ----------------
    viz_dir = art / "visualization"
    viz_dir.mkdir(exist_ok=True)
    preds = {}
    with (art / "selector_results.csv").open() as fh:
        for r in csv.DictReader(fh):
            preds.setdefault(r["state_id"], {})[r["selector"]] = int(r["group"])

    split = json.loads((art / "FROZEN_SPLIT.json").read_text())
    sid_list = split["confirmation_state_ids"][: a.viz_states]
    feat_dir = Path("artifacts/semantic_token_cd_phase1_selector_v1/group_features_npz")

    def pick_panel(lab, g, color):
        img = Image.new("RGB", (16, 16), (15, 15, 15))
        px = img.load()
        for pp in range(256):
            if lab[pp] == g:
                px[pp % 16, pp // 16] = color
        return img.resize((224, 224), Image.NEAREST)

    def mask_panel(lab, obj):
        img = Image.new("RGB", (16, 16), (15, 15, 15))
        px = img.load()
        for pp in range(256):
            if pp in obj:
                px[pp % 16, pp // 16] = (255, 255, 255)
        return img.resize((224, 224), Image.NEAREST)

    for sid in sid_list:
        f = feat_dir / f"{sid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        lab = d["labels"].astype(np.int64)
        obj = set(int(x) for x in d["object_ids"].tolist())
        oracle = int(d["g_obj_idx"][0])
        pred = preds.get(sid, {})
        panels = [
            mask_panel(lab, obj),
            pick_panel(lab, oracle, (0, 200, 0)),
            pick_panel(lab, pred.get("object", oracle), (230, 0, 0)),
            pick_panel(lab, pred.get("goal", oracle), (0, 0, 220)),
            pick_panel(lab, pred.get("noun_only", oracle), (230, 140, 0)),
        ]
        canvas = Image.new("RGB", (224 * 5, 224), (10, 10, 10))
        for i, im in enumerate(panels):
            canvas.paste(im, (i * 224, 0))
        canvas.save(viz_dir / f"{sid}.png")
    print(f"wrote visualization/ ({len(sid_list)} states)  [mask | oracle | object | goal | noun_only]")


if __name__ == "__main__":
    main()
