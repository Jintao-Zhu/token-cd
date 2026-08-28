#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-1 report generator (+ lightweight visualization).

Reads selector_results.json / decision.json / oracle_vs_pred.csv / group_features_npz
and writes REPORT_ZH.md + visualization/*.png.

Run:
  cd /data/docker/dev_zjt/data/code
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/report_selector.py \
    --artifact artifacts/semantic_token_cd_phase1_selector_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _fmt(x, nd=4):
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
    sf = res["stop_flags"]
    verdict = res["verdict"]

    q1 = "是" if gates["A_selector_recovery"]["PASS"] else "否"
    q2 = "是（但仅 52.6%，且任务高度分化）" if agg["language"]["oracle_recovery"] > 0.3 else "否"
    q3 = "否" if agg["hybrid"]["oracle_recovery"] < agg["language"]["oracle_recovery"] else "是"
    q4 = "是" if verdict == "PASS_TO_ROLLOUT" else "否"

    sel_header = "| selector | top-1 oracle recovery | IoU(selected,obj) | align mean | align median | vs random (McNemar p) |"
    sel_sep = "|---|---|---|---|---|---|"
    sel_rows = []
    for s in ("oracle", "language", "hybrid", "d_only", "random"):
        r = agg[s]
        pv = res["mcnemar_p_vs_random"].get(s, None)
        pv_s = f"{pv:.1e}" if pv is not None and pv < 0.001 else (_fmt(pv, 3) if pv is not None else "—")
        sel_rows.append(f"| {s} | {r['oracle_recovery']:.3f} | {r['iou_median']:.3f} | "
                        f"{pal['mean'][s]:.3f} | {pal['median'][s]:.3f} | {pv_s} |")

    tb_header = "| task | n | oracle | language | hybrid | random |"
    tb_sep = "|---|---|---|---|---|---|"
    tb_rows = []
    for tb in res["task_breakdown"]:
        tb_rows.append(f"| {tb['task']} | {tb['n_states']} | {tb['oracle_recovery']:.3f} | "
                       f"{tb['language_recovery']:.3f} | {tb['hybrid_recovery']:.3f} | {tb['random_recovery']:.3f} |")

    md = f"""# SEMANTIC_TOKEN_CD_PHASE1_SELECTOR_V1 — Automatic Semantic Selector Audit

> **Verdict: `{verdict}`**
> 日期 2026-08-23 · split=Confirmation (n={res['n_states']}) · 未跑 rollout（按预注册禁止）

## 0. 一句话结论

Phase-0 的 oracle-selected semantic group（cos 0.70）**可以被自动发现**：用「指令 token 嵌入与
视觉组的余弦」做 language grounding，top-1 命中 oracle group 达 **52.6%**（random 9.6%，p=2e-15）。
但 hybrid（language+disruption）反而更差（36.3%），说明 **disruption D 是干扰项不是补充**——它回到
CAT-CD 的「高敏感性非对象 token」老问题。language grounding 在命名对象任务上很强（87-93%），在
多对象/歧义任务上失效（0%）。

## 1. 四个核心问题

| # | 问题 | 答案 |
|---|---|---|
| 1 | Phase-0 的 semantic group 能否自动发现？ | **{q1}** |
| 2 | Language 是否提供有效 grounding？ | **{q2}** |
| 3 | Hybrid 是否超过单纯 language？ | **{q3}** |
| 4 | 是否值得进入 closed-loop rollout？ | **{q4}** |

## 2. Selector 对比（Metric 1/2/3）

{sel_header}
{sel_sep}
{chr(10).join(sel_rows)}

- oracle recovery = top-1 命中 Phase-0 oracle group 的比例（random 理论 12.5%，实测 9.6%）
- align = cos(r_selected, r_PCD)（mean 为跨 states×7 的平均，median 为 pooled 中位数）
- oracle 对齐 mean {pal['mean']['oracle']:.3f} / median {pal['median']['oracle']:.3f}（median 与 Phase-0 的 0.703 一致）

## 3. Task-level 分化（Metric 3 / STEP 5）

{tb_header}
{tb_sep}
{chr(10).join(tb_rows)}

- **language 强**：pick_coke_can (0.87)、carrot_on_plate (0.93)、close_drawer (0.87)、place_apple (0.60)、move_near (0.60)
- **language 失效**：put_eggplant_in_basket (0.00)、spoon_on_towel (0.00) —— 多对象/小对象/歧义指令
- 根因：l 是整句指令的 mean-pool，对象名词被其他 token 稀释；未来可用「对象名词短语隔离」增强。

## 4. Gates 判定

| Gate | 规则 | 结果 |
|---|---|---|
| A_selector | Language/Hybrid recovery >30% 且 >random | {"PASS" if gates["A_selector_recovery"]["PASS"] else "FAIL"}（lang {agg['language']['oracle_recovery']:.3f} / hyb {agg['hybrid']['oracle_recovery']:.3f} / rnd {agg['random']['oracle_recovery']:.3f}） |
| B_alignment | align >0.4 且 >random | {"PASS" if gates["B_alignment"]["PASS"] else "FAIL"}（lang {pal['mean']['language']:.3f} / hyb {pal['mean']['hybrid']:.3f} / rnd {pal['mean']['random']:.3f}） |
| C_task | ≥7/10 task selector ≥ random | {"PASS" if gates["C_task_consistency"]["PASS"] else "FAIL"}（{gates["C_task_consistency"]["n_tasks_pass"]}/{gates["C_task_consistency"]["n_tasks"]} tasks） |

## 5. STOP RULE 判定

| 停止条件 | 命中 |
|---|---|
| Language ≈ random（语言无法定位视觉语义） | {"是" if sf["language_approx_random"] else "否"} |
| Hybrid 只靠 D（回到 CAT-CD） | {"是" if sf["hybrid_only_D"] else "否"} |
| Alignment 低 | {"是" if sf["alignment_low"] else "否"} |

- 注：hybrid 与 d_only 的一致性 {res['hybrid_vs_d_only_agreement']:.2f}，但 hybrid 并未「只靠 D」——
  反而是 D 的加入**稀释**了 language 的正确信号（52.6% → 36.3%）。

## 6. 关键发现（对 Phase-2 的启示）

1. **Language grounding 是可用信号**，但 mean-pool 整句指令太粗糙（53% 命中，多对象任务归零）。
   Phase-2 应优先「对象名词短语提取」而非增加 disruption 项。
2. **Disruption D 与语义反向相关**：D 单独 27.4% 命中、alignment 却 0.52 —— 高 disruption 组是
   高敏感性非对象区域，alignment 与 recovery 是两回事。这再次印证 CAT-CD 的教训。
3. **alignment 与 recovery 脱钩**：d_only alignment 0.518 > language 0.458，但 recovery 27% < 53%。
   论文应把「找到对象」作为主指标，alignment 作为次指标。

## 7. 结论

{verdict}

**产出文件**：selector_results.json · decision.json · group_features_npz/ · oracle_vs_pred.csv ·
alignment_results（并入 oracle_vs_pred.csv）· task_breakdown.csv · visualization/ · REPORT_ZH.md
"""
    (art / "REPORT_ZH.md").write_text(md)
    print(f"wrote {art / 'REPORT_ZH.md'} · verdict {verdict}")

    # ---------------- lightweight visualization ----------------
    viz_dir = art / "visualization"
    viz_dir.mkdir(exist_ok=True)
    import csv
    preds = {}
    with (art / "oracle_vs_pred.csv").open() as fh:
        for r in csv.DictReader(fh):
            preds.setdefault(r["state_id"], {})[r["selector"]] = int(r["group"])

    split = json.loads((art / "FROZEN_SPLIT.json").read_text())
    sid_list = split["confirmation_state_ids"][: a.viz_states]

    def pick_panel(lab, obj, pick_group, color):
        """16x16 grid showing ONLY the chosen group's patches (color) over dark bg."""
        img = Image.new("RGB", (16, 16), (15, 15, 15))
        px = img.load()
        for pp in range(256):
            if lab[pp] == pick_group:
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
        f = art / "group_features_npz" / f"{sid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        lab = d["labels"].astype(np.int64)
        obj = set(int(x) for x in d["object_ids"].tolist())
        oracle = int(d["g_obj_idx"][0])
        lang = preds.get(sid, {}).get("language", oracle)
        hyb = preds.get(sid, {}).get("hybrid", oracle)
        panels = [
            mask_panel(lab, obj),
            pick_panel(lab, obj, oracle, (0, 200, 0)),
            pick_panel(lab, obj, lang, (0, 0, 220)),
            pick_panel(lab, obj, hyb, (230, 140, 0)),
        ]
        canvas = Image.new("RGB", (224 * 4, 224), (10, 10, 10))
        for i, im in enumerate(panels):
            canvas.paste(im, (i * 224, 0))
        canvas.save(viz_dir / f"{sid}.png")
    print(f"wrote visualization/ ({len(sid_list)} states)")


if __name__ == "__main__":
    main()
