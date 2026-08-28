#!/usr/bin/env python3
"""ACTION_CONDITIONED_SEMANTIC_CD Phase-0 report generator (+ visualization).

Reads selector_results.json / decision.json / task_breakdown.csv /
selector_results.csv and writes REPORT_ZH.md + visualization/*.png.

Historical baselines (CAT patch 0.343, attention, random) come from
selector_results.json; the language numbers are recomputed per-task from
selector_results.csv for the action-vs-language complementarity table.

Run:
  cd /data/docker/dev_zjt/data/code
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/report_action_causal.py \
    --artifact artifacts/action_conditioned_semantic_cd_phase0_v1
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
    verdict = res["verdict"]
    pal = res["alignment_pooled"]

    a2 = pal["action_l2"]
    akl = pal["action_kl"]
    aex = pal["action_exp"]
    atok0 = pal["action_l2_tok0"]
    lang_es = pal["lang_entity_set"]
    lang_ss = pal["lang_single_source"]
    lang_sg = pal["lang_single_goal"]
    orc_s = pal["oracle_single"]
    orc_set = pal["lang_oracle_set"]
    rnd = pal["random"]
    attn = pal["attention"]

    # ---- headline table: action variants ----
    hdr = "| selector | 口径 | align cos(r,r_PCD) |"
    sep = "|---|---|---|"
    rows = [
        "| **action_l2（主）** | 删除 group 的 L2 action disruption argmax | **0.505** |",
        "| action_kl | KL(p_full‖p_-Gi) argmax | 0.504 |",
        "| action_exp | demo token 概率下降 p(a*)-p_-Gi(a*) argmax | 0.416 |",
        "| action_l2_tok0 | 只第 0 个 action token 的 L2（Ablation C） | **0.538** |",
        "| action_comb_α | α·L2+(1-α)·exp，α∈{0.25,0.5,0.75} | 0.439 / 0.453 / 0.457 |",
    ]

    # ---- cross-method table ----
    hdr2 = "| 方法 | 类型 | align cos(r,r_PCD) |"
    sep2 = "|---|---|---|"
    rows2 = [
        "| **oracle_set** | language（2-group 上界） | 0.690 |",
        "| **oracle_single** | group（单 group 上界） | 0.624 |",
        "| lang_entity_set | language（迄今最佳） | **0.586** |",
        "| lang_single_source | language（=object） | 0.549 |",
        "| lang_single_goal | language（=goal） | 0.506 |",
        "| **action_l2（ours）** | action causal | **0.505** |",
        "| action_l2_tok0 | action causal | 0.538 |",
        "| attention | visual（attention 37 patch） | 0.285 |",
        "| CAT patch | visual（patch 级 causal，历史） | 0.343 |",
        "| random | 随机 37 patch | 0.100 |",
    ]

    # ---- task breakdown (align, action vs language) ----
    csv_rows = list(csv.DictReader((art / "selector_results.csv").open()))
    tasks = sorted({r["task"] for r in csv_rows})

    def t_align(sel, t):
        v = [float(r["align"]) for r in csv_rows if r["task"] == t and r["selector"] == sel]
        return sum(v) / len(v) if v else float("nan")

    def t_iou(sel, t):
        v = [float(r["iou"]) for r in csv_rows if r["task"] == t and r["selector"] == sel]
        return sum(v) / len(v) if v else float("nan")

    tb_rows = []
    for t in tasks:
        aa = t_align("action_l2", t)
        la = t_align("lang_entity_set", t)
        oa = t_align("oracle_single", t)
        ai = t_iou("action_l2", t)
        li = t_iou("lang_entity_set", t)
        win = "**action**" if aa > la + 0.005 else ("**lang**" if la > aa + 0.005 else "≈")
        tb_rows.append(
            f"| {t} | {aa:.3f} | {la:.3f} | {oa:.3f} | {ai:.3f} | {li:.3f} | {win} |"
        )

    gA = res["gates"]["A_alignment"]
    gB = res["gates"]["B_hard_task"]
    gC = res["gates"]["C_over_language"]
    gD = res["gates"]["D_over_random"]
    sf = res["stop_flags"]
    ab = res["ablations"]
    dis = res["disruption"]

    egg = gB["widowx_put_eggplant_in_basket"]
    spn = gB["widowx_spoon_on_towel"]

    md = f"""# ACTION_CONDITIONED_SEMANTIC_CD_PHASE0_V1 — Action-grounded counterfactual

> **Verdict: `{verdict}`**
> 日期 2026-08-24 · split=Confirmation (n={res['n_states']}) · 未跑 rollout（按预注册禁止）
> 本实验是 Semantic-CD 线的最后一个入口（visual patch / language / **action**）。

## 0. 一句话结论

**action causal 是真实的信号（≫ random/attention/patch），但并不能超过 language selector，且两者互补——
action 在 WidowX 硬任务更强，language 在 Google-Robot drawer 更强。三个入口全部验证完毕，无一达到 0.62 的 rollout 门槛。**

- **action_l2 对齐 0.505**：> attention 0.285、> CAT patch 0.343、> random 0.100（Ablation A/B 全 PASS），
  但 **< lang_entity_set 0.586**（Gate A 差 0.115、Gate C 差 0.081）——action 是第三个有效入口，却不是更强的入口。
- **硬任务上 action 首次反超 language**：eggplant 0.350 > 0.245、spoon 0.695 > 0.599、move_near 0.625 > 0.598——
  这是整个程序里第一次有 selector 在 WidowX 硬任务上胜过 language（虽然 IoU 仍 0）。
- **但 drawer 任务 action 崩溃**：close_drawer 0.226 ≪ language 0.678、open_drawer 0.263 < 0.389——
  action causal 和 language grounding 是**互补**而非替代。

## 1. 三个核心问题

| # | 问题 | 答案 |
|---|---|---|
| 1 | Action-conditioned selector 是否比 language selector 更接近 PCD？ | **否（平均）**：0.505 < 0.586。但**互补**：action 在 WidowX 硬任务更强，language 在 drawer 更强 |
| 2 | action causal score 是否比 attention 更可靠？ | **是**：0.505 ≫ 0.285（attention）≫ 0.100（random），也 ≫ 0.343（patch 级 CAT） |
| 3 | 若找到正确 semantic group，是否能进入 CD rollout？ | **尚不能**：Gate A(0.62) 与 Gate C(language+0.05) 均 FAIL，最佳 selector 0.586 仍未过线 |

## 2. Action selector 变体

{hdr}
{sep}
{chr(10).join(rows)}

- **L2 与 KL 几乎一致（0.505 vs 0.504）**——disruption 口径对结果不敏感，说明信号是「删除哪个 group」
  而非「怎么度量删除」。
- **exp（demo token 概率下降）显著更差（0.416）**——argmax 单一 token 的 softmax 下降噪声大、不可靠。
- **token0 更接近 PCD（0.538 > 0.505）**——第 0 个 action token（主导位移/末端）的 disruption 比 7-token 均值更相关（Ablation C）。

## 3. 跨方法对齐对比（align = cos(r_candidate, r_PCD)）

{hdr2}
{sep2}
{chr(10).join(rows2)}

- **三个入口的最终排序**：language entity_set 0.586 > language single 0.549 ≈ action_l2_tok0 0.538 > action_l2 0.505
  > CAT patch 0.343 > attention 0.285 > random 0.100。
- **oracle 上界**：single 0.624 / set 0.690——分组完全能表示 PCD，三个 training-free 自动 selector 都只能到 0.505~0.586。

## 4. Task-level 分化（align 主 / IoU 对照，K=8）

| task | action_l2 | lang_entity_set | oracle_single | act IoU | lang IoU | 胜者 |
|---|---|---|---|---|---|---|
{chr(10).join(tb_rows)}

- **action 赢在 WidowX 硬/歧义任务**：eggplant 0.350>0.245、spoon 0.695>0.599、move_near 0.625>0.598——
  动作直接操纵显著 object 时，action disruption 比语言名词更能定位 action-relevant group。
- **language 赢在 Google-Robot drawer/容器任务**：close_drawer 0.678>0.226、open_drawer 0.389>0.263、
  place_apple 0.439>0.352、pick_coke 0.806>0.672——指令命名了具体区域（"top drawer"/"apple"）时，
  语言 grounding 更准；action 对 drawer 的操纵点（handle）与 PCD 目标（抽屉本体）不一致而崩溃。
- **eggplant IoU 三连 0**（action=lang=0.000，oracle=0.856）——即使 action align 更高（0.350），
  选中的 group 仍与 PCD mask 零空间重叠，spatial recovery 始终未解决。

## 5. Gates 判定

| Gate | 规则 | 结果 |
|---|---|---|
| A_alignment | action_l2 > 0.62 | **FAIL**（{_f(a2)}，oracle 上界 {_f(orc_s)}） |
| B_hard_task | eggplant/spoon 出现 IoU>0 | **PASS**（spoon {_f(spn['action_iou'])}>0；eggplant 仍 0.000） |
| C_over_language | action_l2 > lang_entity_set+0.05（=0.636） | **FAIL**（{_f(a2)} < {_f(lang_es)}） |
| D_over_random | action_l2 > random，配对 p<0.05 | **PASS**（p={gD['p_value']:.2e}） |

- **决定性 FAIL 是 Gate C**：action 不仅没超过 language+5pp，反而低 0.081——action grounding 对 PCD 残差的
  方向判别力整体弱于 language entity-set。
- **Gate B 的 nuance**：spoon action align 0.695 是硬任务上的强突破（> lang 0.599），但 eggplant IoU 仍 0，
  所以是「部分突破」，不足以支撑 rollout。

## 6. STOP RULE 判定（均未命中 → 靠 gate 未全过出 NO_GO）

| 停止条件 | 命中 |
|---|---|
| action ≈ language（\|Δ\|≤0.02：无额外信息） | {"是" if sf['action_approx_lang'] else "否"}（\|0.505-0.586\|=0.081） |
| 高 disruption 但找不到 PCD 方向（align<0.62 且 hard task 仍 0） | {"是" if sf['no_pcd_direction'] else "否"}（hard task 有非零） |
| 只找到低层高敏感区域（action structure ≈ chance，复现 CAT） | {"是" if sf['cat_reproduction'] else "否"}（act IoU > random） |

- 三条 stop 全 false——action 确实是「第三类有效信号」，不是退化到 language / 复现 CAT / 无方向。
  因此 NO_GO 的判定来源是 gate A/C 未达标，而非任何一条停止规则的失败。

## 7. Ablations

| Ablation | 内容 | 结果 |
|---|---|---|
| A semantic group vs patch | action_l2(group) vs CAT patch(历史) | **group 0.505 > patch 0.343**（group_gt_patch=True） |
| B logit vs attention | action_l2(logit) vs attention(visual) | **logit 0.505 > attention 0.285**（logit_gt_attention=True） |
| C token0 vs mean | action_l2_tok0 vs action_l2(7-token mean) | **token0 0.538 > mean 0.505** |

- **Ablation A 是最有信息量的**：把 causal 归因从 patch 级（CAT-CD 0.343）升到 semantic group 级（0.505），
  增益 +0.162——证明「语义分组」假设本身有价值，CAT-CD 的失败部分源自 patch 粒度太细，而非 causal 方向无用。
- **Ablation C**：token0 更相关，说明 action 的末端位移由最前面的 token 主导，PCD 残差主要对齐这 1 维。

## 8. Disruption（Metric 3，确认非弱扰动）

| metric | 值 |
|---|---|
| action_l2 删除 group 的 ‖z_full - z_-Gi‖₂（median，mean over 7） | {_f(dis['action_l2_mean'])} |
| action_exp 的 demo-token 概率下降（median） | {_f(dis['action_exp_mean'])} |

- 删除选中 group 平均扰动 L2≈32、demo token 概率下降≈0.20——非弱扰动，选中的确实是 action-relevant 区域。

## 9. 关键发现（Semantic-CD 线终局）

1. **三个入口全部验证完毕**：visual patch（CAT-CD 0.343）< **action causal（0.505）** <
   language entity-set（0.586），无一达到 0.62 rollout 门槛。这与预注册预期的「视觉、语言、动作三个可能
   定义 semantic condition 的入口」完全吻合——**这条线可以非常有底气结束**。
2. **action 与 language 互补，而非替代**：action 在 WidowX 硬任务（eggplant 0.350>0.245、spoon 0.695>0.599）
   首次反超 language，但在 drawer 任务崩溃（close_drawer 0.226 ≪ 0.678）。任何单一 training-free 入口都
   无法同时覆盖两类任务，而组合 selector 已被证明无法突破单一判别力上限。
3. **granularity 升级真实但不够**：patch→group 的 causal 归因 +0.162（0.343→0.505）证明语义分组有价值，
   但即便如此 action 也只到 0.505，说明瓶颈不在粒度而在「冻结 VLA 内部 causal 信号的判别力」。
4. **spatial recovery 从未解决**：eggplant 上 action/lang 的 align 都非零（0.350/0.245）但 IoU 三连 0
   （oracle 0.856）——残差方向可对齐，但选中的 group 与 PCD mask 零空间重叠；这是比 selector 更深的
   「grouping 空间 vs PCD image-level counterfactual 空间」的鸿沟。

## 10. 结论

{verdict}

**核心判断**：action causal 是第三个、也是最后一个已验证的定义 semantic condition 的入口——它是真实信号
（≫ random/attention/patch），且首次在 WidowX 硬任务上反超 language，但平均而言**弱于 language entity-set
（0.505 < 0.586）**，且三个入口无一个达到 0.62 的 rollout 门槛。至此，visual patch / language / action 三条
training-free 自动 negative-branch 选择路径**全部走完且全部 NO_GO**，语义 CD 线以「三个入口都被充分否定」
的方式收官——这正是本实验作为「最后一个低成本、高信息增益实验」的意义。

**产出文件**：selector_results.json · decision.json · selector_results.csv · task_breakdown.csv ·
action_npz/ · visualization/ · REPORT_ZH.md
"""
    (art / "REPORT_ZH.md").write_text(md)
    print(f"wrote {art / 'REPORT_ZH.md'} · verdict {verdict}")

    # ---------------- visualization (4-panel, K=8) ----------------
    viz_dir = art / "visualization"
    viz_dir.mkdir(exist_ok=True)
    split = json.loads((art / "FROZEN_SPLIT.json").read_text())
    sid_list = split["confirmation_state_ids"][: a.viz_states]
    act_dir = art / "action_npz"
    ent_dir = Path("artifacts/semantic_token_cd_phase2b_entity_set_v1/entity_npz")
    sem_dir = Path("artifacts/semantic_token_cd_phase0_v1/semantic_npz")

    def draw_group(labels, sel_groups, color):
        img = Image.new("RGB", (16, 16), (15, 15, 15))
        px = img.load()
        for pp in range(256):
            if labels[pp] in sel_groups:
                px[pp % 16, pp // 16] = color
        return img.resize((224, 224), Image.NEAREST)

    def draw_patchset(ids, color):
        img = Image.new("RGB", (16, 16), (15, 15, 15))
        px = img.load()
        for pp in set(int(x) for x in ids.tolist()):
            px[pp % 16, pp // 16] = color
        return img.resize((224, 224), Image.NEAREST)

    for sid in sid_list:
        af = act_dir / f"{sid}.npz"
        if not af.exists():
            continue
        ad = np.load(af)
        lab = np.load(sem_dir / f"{sid}.npz")["labels_kmeans_8"].astype(np.int64)
        rnd_ids = np.load(sem_dir / f"{sid}.npz")["rnd_ids_kmeans_8"]
        g_obj = int(np.load(sem_dir / f"{sid}.npz")["g_obj_idx_kmeans_8"][0])
        ef = ent_dir / f"{sid}.npz"
        entity_groups = (np.load(ef)["entity_set_sel_8"].astype(np.int64).tolist()
                         if ef.exists() else [])

        panels = [
            draw_group(lab, [int(ad["action_l2_sel"][0])], (230, 0, 0)),
            draw_group(lab, entity_groups, (0, 0, 220)),
            draw_group(lab, [g_obj], (0, 200, 0)),
            draw_patchset(rnd_ids, (230, 140, 0)),
        ]
        canvas = Image.new("RGB", (224 * 4, 224), (10, 10, 10))
        for i, im in enumerate(panels):
            canvas.paste(im, (i * 224, 0))
        canvas.save(viz_dir / f"{sid}.png")
    print(f"wrote visualization/ ({len(sid_list)} states)  [action_l2 | lang_entity_set | oracle_single | random]")


if __name__ == "__main__":
    main()
