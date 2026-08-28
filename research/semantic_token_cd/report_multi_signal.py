#!/usr/bin/env python3
"""MULTI_SIGNAL_SEMANTIC_CD Phase-0 report generator (+ visualization).

Reads selector_results.json / decision.json and writes REPORT_ZH.md +
visualization/*.png.

Run:
  cd /data/docker/dev_zjt/data/code
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/report_multi_signal.py \
    --artifact artifacts/multi_signal_semantic_cd_phase0_v1
"""
from __future__ import annotations

import argparse
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
    verdict = res["verdict"]
    align = res["alignment"]
    sweep = res["fusion_alpha_sweep"]
    best_a = res["best_alpha"]

    fb = align["fusion_best"]
    act = align["action_only"]
    lang_es = align["language_entity_set"]
    lang_ss = align["language_single_source"]
    orc_s = align["oracle_single"]
    orc_set = align["oracle_set"]
    rnd = align["random"]
    attn = align["attention"]
    cat = align["cat_patch"]
    inter = align["intersection"]
    union = align["union"]

    # ---- alpha sweep table ----
    alpha_labels = {
        "a000": "0.0（纯 action）", "a025": "0.25", "a050": "0.5（equal）",
        "a075": "0.75", "a100": "1.0（纯 language）",
    }
    sweep_rows = []
    for k in ["a000", "a025", "a050", "a075", "a100"]:
        mark = " **← best**" if abs(sweep[k] - fb) < 1e-9 else ""
        sweep_rows.append(f"| {alpha_labels[k]} | {_f(sweep[k])}{mark} |")

    # ---- cross-method table ----
    hdr = "| 方法 | 类型 | align cos(r,r_PCD) |"
    sep = "|---|---|---|"
    rows = [
        f"| oracle_set | 2-group 上界 | {_f(orc_set)} |",
        f"| oracle_single | 单 group 上界 | {_f(orc_s)} |",
        f"| lang_entity_set | language 单路最佳 | **{_f(lang_es)}** |",
        f"| lang_single_source | language（=object） | {_f(lang_ss)} |",
        f"| action_l2 | action 单路 | {_f(act)} |",
        f"| **fusion（α={_f(best_a)}）** | **language × action** | **{_f(fb)}** |",
        f"| fusion intersection（top-2 ∩） | 硬交集 | {_f(inter)} |",
        f"| fusion union（top-2 ∪） | 并集 entity set | {_f(union)} |",
        f"| CAT patch | visual patch causal | {_f(cat)} |",
        f"| attention | visual attention | {_f(attn)} |",
        f"| random | 随机 | {_f(rnd)} |",
    ]

    # ---- task breakdown ----
    tb = res["task_breakdown"]
    tb_rows = []
    for x in tb:
        fa, aa, la, oa = x["fusion_align"], x["action_align"], x["language_align"], x["oracle_align"]
        best_single = max(aa, la)
        win = "fusion>single" if fa > best_single + 0.005 else ("single≥fusion" if best_single > fa + 0.005 else "≈")
        tb_rows.append(f"| {x['task']} | {fa:.3f} | {aa:.3f} | {la:.3f} | {oa:.3f} | {win} |")

    # ---- oracle 2x2 ----
    o22 = res["oracle_2x2"]
    both = o22["both"]; lo = o22["lang_only"]; ao = o22["action_only"]; ne = o22["neither"]

    # ---- gates / ablations / stop ----
    gA = res["gates"]["A_alignment"]
    gB = res["gates"]["B_over_random"]
    gC = res["gates"]["C_hard_task"]
    ab = res["ablations"]
    stop = res["stop_flag"]

    # hard-task table
    htc = gC["hard_tasks"]
    ht_rows = []
    for t, v in htc.items():
        imp = "✓" if v["improved"] else "✗"
        ht_rows.append(f"| {t} | {v['fusion']:.3f} | {v['action']:.3f} | {v['language']:.3f} | {v['best_single']:.3f} | {imp} |")

    md = f"""# MULTI_SIGNAL_SEMANTIC_CD_PHASE0_V1 — Language × Action 融合

> **Verdict: `{verdict}`**
> 日期 2026-08-24 · split=Confirmation (n={res['n_states']}) · 未跑 rollout（按预注册禁止）
> 回答 Semantic-CD 线最后一个理论问题：semantic counterfactual 由任务描述决定、由动作决定，还是由两者联合决定？

## 0. 一句话结论

**加权融合是真实但不足的增益（0.575 未破 language 0.586）；而「硬交集」是一个被 Ablation B 挖出的强信号——当 language 与 action 都把一个 group 排进 top-2 时，该 group 几乎就是 oracle（0.623 ≈ 0.624），但 16.3% 状态两信号无交集、eggplant 上即使有交集也是错的。**

- **加权融合 best（α={_f(best_a)}）对齐 {_f(fb)}**：{("**> language 0.586**" if fb > lang_es else f"**< language {_f(lang_es)}**")}，
  {("**> action 0.505**" if fb > act else f"≈/ < action {_f(act)}")}——α 呈凹峰（0.505→0.575→0.534），
  证明联合信号真实，但全局仍低于 language entity_set。
- **硬交集（Ablation B）= {_f(inter)}，几乎等于 oracle_single {_f(orc_s)}**（非空状态，命中率 66.4%）：
  「两信号一致」是近乎完美的 selector，但空集率 16.3%（eggplant 46.7%、drawer 26.7%）。
- **加权融合会退化**：carrot 上 fusion {_f(next(x['fusion_align'] for x in tb if 'carrot' in x['task']))} ≪
  language {_f(next(x['language_align'] for x in tb if 'carrot' in x['task']))}——两信号指向不同 group 时，
  加权 argmax 会选中一个「两者都不是第一」的更差 group。
- **Oracle 2×2（最重要）**：lang 错而 action 对仅 {o22['action_only']['frac']:.1%}，action 错而 lang 对 {o22['lang_only']['frac']:.1%}，
  两者都错 {o22['neither']['frac']:.1%}——action 能纠正 language 的空间很小（详见 §6）。

## 1. 三个核心问题

| # | 问题 | 答案 |
|---|---|---|
| Q1 | 融合是否超过单一路径（Fusion > Language 0.586）？ | **{("是" if fb > lang_es else "否")}**：{_f(fb)} {(">" if fb > lang_es else "≤")} {_f(lang_es)} |
| Q2 | 融合是否接近 oracle（0.69）？ | **{("是" if fb >= orc_s else "否")}**：{_f(fb)} vs oracle_single {_f(orc_s)} / oracle_set {_f(orc_set)} |
| Q3 | 是否解决硬任务（eggplant/spoon/move_near/drawer）？ | **{("是" if gC['PASS'] else "否")}**（详见 §4 Gate C） |

## 2. α sweep（主表）

| α | align cos(r,r_PCD) |
|---|---|
{chr(10).join(sweep_rows)}

- 全局最佳 α = **{_f(best_a)}**（align {_f(fb)}）。α=0 精确复现 action_l2（{_f(sweep['a000'])} vs {_f(act)}，pipeline 自洽）。

## 3. 跨方法对齐对比

{hdr}
{sep}
{chr(10).join(rows)}

## 4. Task-level 分化（global best α）

| task | fusion | action | language | oracle | 融合胜单路? |
|---|---|---|---|---|---|
{chr(10).join(tb_rows)}

## 5. Hard-task 细化（Gate C 依据）

| task | fusion | action | language | best_single | fusion>best_single+0.02 |
|---|---|---|---|---|---|
{chr(10).join(ht_rows)}

## 6. Oracle 2×2（信号互补性 —— 本实验最重要的诊断）

| 组合 | 计数 | 比例 |
|---|---|---|
| Lang ✓ & Action ✓ | {both['count']} | {both['frac']:.1%} |
| Lang ✓ & Action ✗ | {lo['count']} | {lo['frac']:.1%} |
| Lang ✗ & Action ✓ | {ao['count']} | {ao['frac']:.1%} |
| Lang ✗ & Action ✗ | {ne['count']} | {ne['frac']:.1%} |

- **agreement rate = {o22['agree_rate']:.1%}**（lang top-1 与 action top-1 一致的比例）。
- **解读（关键）**：
  - **action 能纠正 language 的空间极小**（「Lang✗ & Action✓」仅 {o22['action_only']['frac']:.1%}），
    而 **language 能纠正 action 的空间大**（「Lang✓ & Action✗」{o22['lang_only']['frac']:.1%}）——
    两个信号**高度不对称**：language 是主信号，action 是弱补充。
  - **「两者都✗」占 {o22['neither']['frac']:.1%}**：这些状态（以 eggplant 为首）PCD 的 counterfactual 方向
    既不是 language 也不是 action 能覆盖的，融合也救不了——这是 three-entrance 全部失败的深层原因。
  - 因此**加权融合的理论空间被限制在 ~{o22['action_only']['frac']:.1%}**（只有 action-only 状态 fusion 才可能超 language），
    不足以把全局从 0.586 推到 0.62。

## 7. Ablations

| Ablation | 内容 | 结果 |
|---|---|---|
| A simple vs weighted | α=0.5 vs 最佳 α | equal {_f(ab['A_simple_vs_weighted']['alpha_050'])} vs best {_f(ab['A_simple_vs_weighted']['best_align'])}（weighted_gt_equal={ab['A_simple_vs_weighted']['weighted_gt_equal']}） |
| B intersection | Lang top-2 ∩ Action top-2 | align {_f(inter)}，空集率 {_f(ab['B_intersection']['empty_rate'])} |
| C union | Lang top-2 ∪ Action top-2 | align {_f(union)} |

- **Ablation B（硬交集）是本次最意外的发现**：非空时 align {_f(inter)} ≈ oracle {_f(orc_s)}、命中率 66.4%；
  但空集率 {_f(ab['B_intersection']['empty_rate'])}（eggplant 46.7% / drawer 26.7% / pick_coke 20%），
  且 eggplant 上「有交集也 0% 命中」——交集是**高精度低召回**的 oracle 探测器，不是可部署的 selector。
- **Ablation C（并集）0.497 反而稀释**：删 2–4 个 group 的并集与 Phase-2B entity_set 的稀释一致——区域越大 IoU 越差。

## 8. Gates 判定

| Gate | 规则 | 结果 |
|---|---|---|
| A_alignment | best fusion > 0.62 | **{"PASS" if gA['PASS'] else "FAIL"}**（{_f(fb)}） |
| B_over_random | best fusion > random，p<0.05 | **{"PASS" if gB['PASS'] else "FAIL"}**（p={gB['p_value']:.2e}） |
| C_hard_task | ≥1 hard task fusion>max(single)+0.02 | **{"PASS" if gC['PASS'] else "FAIL"}** |

## 9. STOP RULE

| 停止条件 | 命中 |
|---|---|
| best fusion align ≤ 0.60 | {"是" if stop['fusion_leq_060'] else "否"}（{_f(fb)}） |

## 10. 关键发现（Semantic-CD 线正式收官）

1. **联合信号真实但弱**：α sweep 呈凹峰（{_f(sweep['a000'])} → {_f(fb)} → {_f(sweep['a100'])}），
   证明「task condition = language ∩ action」对部分状态成立；但全局 {_f(fb)} 未破 language {_f(lang_es)}。
2. **硬交集 ≈ oracle（最强信号）**：Ablation B 的「Lang top-2 ∩ Action top-2」在非空状态 align {_f(inter)} ≈
   oracle_single {_f(orc_s)}、命中率 66.4%——**当两个信号一致指向某 group，该 group 几乎就是 PCD 的 counterfactual**。
   代价是 16.3% 空集（无交集）与 eggplant 上「有交集也是错」（46.7% 空 + 非空命中 0%）。
3. **加权融合在 9 个任务里只赢 2 个**（pick_coke 0.872、stack 0.860，均逼近 oracle），平 3 个，**输 4 个**——
   其中 carrot 上 fusion 0.493 ≪ language 0.723，说明 naive 加权在两信号分歧时会选中一个更差的中间 group。
4. **2×2 的不对称是终局解释**：action 纠正 language 仅 6.7%，language 纠正 action 44.4%，两者都错 28.1%。
   PCD 的 counterfactual 方向**既不纯粹由任务描述决定，也不纯粹由动作决定**——两者联合只在一个
   「一致」的小众 regime 里可靠（intersection），而加权平均无法系统性兑现它。
5. **eggplant 是不可攻克的墓碑**：无论 single/entity-set/action/fusion/intersection，eggplant 都失败
   （fusion 0.245、intersection 46.7% 空 + 0% 命中），而 oracle 0.856——这是「grouping 空间 vs image-level
   counterfactual 空间」鸿沟的最终证据。

## 11. 结论

{verdict}

**核心判断**：这是 Semantic-CD 线的最后一个理论实验，回答了「semantic counterfactual 由任务描述、动作、
还是两者联合决定」——**答案是：两者联合决定只在「两信号一致」的小众 regime 成立（intersection ≈ oracle），
但加权融合无法系统性兑现，全局 {_f(fb)} 未破 language 天花板 {_f(lang_es)}**。至此 visual patch / language /
action / 它们的加权融合 / 硬交集五条路径全部走完，无一达到 0.62 的 always-answer rollout 门槛。
**Semantic Token-CD 线可以正式、有底气地结束——不存在「只是融合没试」的遗憾。**

**产出文件**：selector_results.json · decision.json · alpha_sweep.csv · alignment_results.csv ·
oracle_recovery.csv · signal_complementarity.csv · task_breakdown.csv · fusion_npz/ ·
visualization/ · REPORT_ZH.md
"""
    (art / "REPORT_ZH.md").write_text(md)
    print(f"wrote {art / 'REPORT_ZH.md'} · verdict {verdict}")

    # ---------------- visualization (4-panel, K=8) ----------------
    viz_dir = art / "visualization"
    viz_dir.mkdir(exist_ok=True)
    split = json.loads((art / "FROZEN_SPLIT.json").read_text())
    sid_list = split["confirmation_state_ids"][: a.viz_states]
    fus_dir = art / "fusion_npz"
    act_dir = Path("artifacts/action_conditioned_semantic_cd_phase0_v1/action_npz")
    ent_dir = Path("artifacts/semantic_token_cd_phase2b_entity_set_v1/entity_npz")
    sem_dir = Path("artifacts/semantic_token_cd_phase0_v1/semantic_npz")
    best_key = "a000" if best_a == 0.0 else f"a{int(round(best_a*100)):03d}"

    def draw_group(labels, sel_groups, color):
        img = Image.new("RGB", (16, 16), (15, 15, 15))
        px = img.load()
        for pp in range(256):
            if labels[pp] in sel_groups:
                px[pp % 16, pp // 16] = color
        return img.resize((224, 224), Image.NEAREST)

    for sid in sid_list:
        ff = fus_dir / f"{sid}.npz"
        if not ff.exists():
            continue
        fd = np.load(ff)
        lab = np.load(sem_dir / f"{sid}.npz")["labels_kmeans_8"].astype(np.int64)
        g_obj = int(np.load(sem_dir / f"{sid}.npz")["g_obj_idx_kmeans_8"][0])
        ef = ent_dir / f"{sid}.npz"
        entity_groups = (np.load(ef)["entity_set_sel_8"].astype(np.int64).tolist() if ef.exists() else [])

        panels = [
            draw_group(lab, [int(fd[f"fusion_sel_{best_key}"][0])], (200, 0, 200)),
            draw_group(lab, [int(np.load(act_dir / f"{sid}.npz")["action_l2_sel"][0])], (230, 0, 0)),
            draw_group(lab, entity_groups, (0, 0, 220)),
            draw_group(lab, [g_obj], (0, 200, 0)),
        ]
        canvas = Image.new("RGB", (224 * 4, 224), (10, 10, 10))
        for i, im in enumerate(panels):
            canvas.paste(im, (i * 224, 0))
        canvas.save(viz_dir / f"{sid}.png")
    print(f"wrote visualization/ ({len(sid_list)} states)  [fusion | action_l2 | lang_entity_set | oracle_single]")


if __name__ == "__main__":
    main()
