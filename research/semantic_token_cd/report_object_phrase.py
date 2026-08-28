#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-1.5 report generator (+ lightweight visualization).

Reads selector_results.json / decision.json / task_breakdown.csv /
phrase_extraction.json and writes REPORT_ZH.md + visualization/*.png.

Run:
  cd /data/docker/dev_zjt/data/code
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/report_object_phrase.py \
    --artifact artifacts/semantic_token_cd_phase1_5_object_grounding_v1
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
    ph = json.loads((art / "phrase_extraction.json").read_text())

    agg = res["aggregate"]
    pal = res["pooled_alignment"]
    gates = res["gates"]
    sf = res["stop_flags"]
    verdict = res["verdict"]

    # ---- selector comparison table ----
    sel_order = ("oracle", "noun_only", "last_noun", "first_noun", "sentence", "random")
    sel_label = {
        "oracle": "oracle（IoU 上界）", "noun_only": "noun_only（所有名词）",
        "last_noun": "last_noun（容器/位置）", "first_noun": "first_noun（直接宾语）",
        "sentence": "sentence（整句基线）", "random": "random",
    }
    sel_rows = []
    for s in sel_order:
        r = agg[s]
        sel_rows.append(
            f"| {sel_label[s]} | {r['oracle_recovery']:.3f} | {pal[s]:.3f} | {r['iou_median']:.3f} |"
        )

    # ---- task breakdown table ----
    tb_rows = []
    for tb in res["task_breakdown"]:
        tb_rows.append(
            f"| {tb['task']} | {tb['n_states']} | {tb['sentence']:.3f} | {tb['first_noun']:.3f} | "
            f"{tb['last_noun']:.3f} | {tb['noun_only']:.3f} |"
        )

    gA = gates["A_recovery"]
    gB = gates["B_alignment"]
    gC = gates["C_failed_task_rescue"]

    md = f"""# SEMANTIC_TOKEN_CD_PHASE1_5_OBJECT_GROUNDING_V1 — Object Phrase Grounding Selector

> **Verdict: `{verdict}`**
> 日期 2026-08-23 · split=Confirmation (n={res['n_states']}) · 未跑 rollout（按预注册禁止）

## 0. 一句话结论

「把整句指令换成**对象名词短语**来 ground PCD 的 semantic group」这一核心假设被**证伪**：

- **first_noun（直接宾语 = 被操纵对象）几乎无增益**：53.3% vs 整句 52.6%（+0.7pp，Gate A 要求 +10pp）。
- **被 PCD 移除的区域锚定在「目标位置/容器」，不是被操纵对象本身**：last_noun（容器）65.9% **>** first_noun（对象）53.3%。
  在 `put carrot on plate` 里 `plate`(0.87) ≫ `carrot`(0.13)；在 `stack green block on yellow block` 里 `yellow block`(1.0) > `green block`(0.53)。
- 唯一真实增益是 **noun_only（保留所有名词、去掉动词/介词/冠词）= 71.1%**（+18.5pp），但它来自「去掉动词」而非「隔离对象」，
  且对两个失败任务（eggplant / spoon）仍为 0。

## 1. 四个核心问题

| # | 问题 | 答案 |
|---|---|---|
| 1 | object-level language grounding 是否优于 sentence-level？ | **部分**（noun_only 71.1% 优于 52.6%，但 first_noun 53.3% 不优于） |
| 2 | 「直接宾语 = 被移除对象」这个前提成立吗？ | **不成立**（容器/位置 noun 是更强的 ground） |
| 3 | semantic token selector 是否接近 oracle？ | **否**（71.1% vs oracle 100%，且 2 个任务仍 0%） |
| 4 | 是否值得进入 closed-loop Semantic-CD？ | **否**（预注册 NO-GO） |

## 2. Selector 对比（Metric 1 recovery / 2 alignment / IoU）

| selector | top-1 oracle recovery | align cos(r,r_PCD) | IoU(selected,obj) |
|---|---|---|---|
{chr(10).join(sel_rows)}

- oracle recovery = top-1 命中 Phase-0 oracle group 的比例（random 理论 12.5%，实测 9.6%）
- align = cos(r_selected, r_PCD)，mean over states×7（oracle 上界 0.624）
- IoU = 选中组与 object mask 的交并比中位数

## 3. Task-level 分化（Metric 3 / STEP 5）

| task | n | sentence | first_noun | last_noun | noun_only |
|---|---|---|---|---|---|
{chr(10).join(tb_rows)}

- **noun_only 大幅提升**：stack_cube (0.33→1.00)、place_apple (0.60→1.00)、move_near (0.60→0.93)、open_drawer (0.53→0.80)
- **first_noun 反向**：carrot_on_plate 里 `carrot`(0.13) 远低于 `plate`(0.87) —— 直接宾语是错的 ground
- **两任务救不活**：put_eggplant_in_basket 全 0；spoon_on_towel 仅 first_noun 0.20，其余 0

## 4. Gates 判定（first_noun = 预注册「object phrase」）

| Gate | 规则 | 结果 |
|---|---|---|
| A_recovery | first_noun > sentence + 0.10 | {"PASS" if gA["PASS"] else "FAIL"}（{_f(gA['first_noun'])} vs {_f(gA['sentence'])}, Δ {_f(gA['delta'],3)}） |
| B_alignment | first_noun align > sentence | {"PASS" if gB["PASS"] else "FAIL"}（{_f(gB['first_noun'])} > {_f(gB['sentence'])}） |
| C_failed_rescue | eggplant/spoon 至少一个 first_noun ≥ 0.50 | {"PASS" if gC["PASS"] else "FAIL"}（eggplant {_f(gC['failed_first_noun']['widowx_put_eggplant_in_basket'])}, spoon {_f(gC['failed_first_noun']['widowx_spoon_on_towel'])}） |

## 5. STOP RULE 判定

| 停止条件 | 命中 |
|---|---|
| first_noun ≈ sentence（对象名词稀释**不是**主要问题） | {"是" if sf["dilution_not_main"] else "否"}（Δ=0.007 ≤ 0.05） |
| recovery 提升但 alignment 不提升 | {"是" if sf["recovery_no_align"] else "否"} |
| alignment 提升但任务太少（泛化不足） | {"是" if sf["align_few_tasks"] else "否"} |

## 6. 关键发现（对论文故事的影响）

1. **「对象名词稀释」假设证伪**：first_noun ≈ sentence（+0.7pp）。被整句 mean-pool 稀释的**不是**对象名词，
   而是动词/介词（`put`/`move`/`into`/`on`）。真正的增益来自 noun_only 去掉动词（+18.5pp）。
2. **PCD counterfactual 锚定在「目标位置」，不是被操纵对象**：last_noun(65.9%) > first_noun(53.3%)。
   这是对 Phase-0 「semantic group = PCD 移除区域」理解的**修正**——移除的往往是 goal location（plate/yellow block/basket），
   而非 grasped object（carrot/green block/eggplant）。论文若写「object grounding」需要重新定义 object = goal location。
3. **selector 天花板 ~70%，且硬任务救不活**：noun_only 71.1% 离 oracle 100% 仍远；eggplant/spoon 小对象任务
   K-means@8 组本身就无法干净分离（Phase-0 已知），语言再准也定位不到。

## 7. 结论

{verdict}

**核心判断**：sentence→object-phrase 的升级方向**未通过预注册 gates**。object noun 隔离无效、
「对象=直接宾语」前提错误、硬任务不可救。semantic-token selector 的语言 grounding 天花板约 70%，
无法作为「训练无关 CD」的可靠 negative-branch 构造器。

**产出文件**：selector_results.json · decision.json · phrase_extraction.json · selector_results.csv ·
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

    def pick_panel(lab, obj, g, color):
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
            pick_panel(lab, obj, oracle, (0, 200, 0)),
            pick_panel(lab, obj, pred.get("first_noun", oracle), (230, 0, 0)),
            pick_panel(lab, obj, pred.get("last_noun", oracle), (0, 0, 220)),
            pick_panel(lab, obj, pred.get("noun_only", oracle), (230, 140, 0)),
        ]
        canvas = Image.new("RGB", (224 * 5, 224), (10, 10, 10))
        for i, im in enumerate(panels):
            canvas.paste(im, (i * 224, 0))
        canvas.save(viz_dir / f"{sid}.png")
    print(f"wrote visualization/ ({len(sid_list)} states)  [mask | oracle | first_noun | last_noun | noun_only]")


if __name__ == "__main__":
    main()
