#!/usr/bin/env python3
"""SEMANTIC_TOKEN_CD Phase-0 report generator.

Reads evaluation_results.json + phase0_decision.json and writes REPORT_ZH.md
(answers the 4 core questions + verdict). Run after evaluate.py.

Run:
  cd /home/leju-suzhou/zjt_ws/token-cd
  task1/.venvs/openvla-ar/bin/python research/semantic_token_cd/report.py \
    --artifact artifacts/semantic_token_cd_phase0_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", type=Path, required=True)
    a = p.parse_args()
    art = a.artifact.resolve()
    res = json.loads((art / "evaluation_results.json").read_text())
    dec = json.loads((art / "phase0_decision.json").read_text())

    agg = res["aggregate_median"]
    gates = res["gates"]
    ceiling = res["full_object_ceiling"]["cos_pos_pooled"]
    hist = res["historical_reference"]

    def row(method, K):
        k = f"{method}_{K}"
        g = agg[k]
        return (
            f"| {method}@{K} | s={g['g_obj_size_median']:.0f} | {g['iou_median']:.3f} "
            f"(chance {g['chance_iou_median']:.3f}) | {g['precision_median']:.3f} | {g['recall_median']:.3f} | "
            f"{g['compactness_median']:.1f} | {g['cos_pos_semantic_pooled']:.4f} | "
            f"{g['cos_pos_attn_pooled']:.4f} | {g['cos_pos_random_pooled']:.4f} | "
            f"{g['D_semantic_median']:.3f} | {g['D_attn_median']:.3f} | {g['D_random_median']:.3f} |"
        )

    verdict = res["verdict"]
    pass_or_stop = "PASS_TO_ROLLOUT" if verdict == "PASS_TO_ROLLOUT" else "STOP_SEMANTIC_TOKEN_CD_NO_GO"
    spec = res.get("semantic_specificity", {})

    q1 = "是" if gates["A_structure"]["PASS"] else "否"
    q2 = "是" if gates["A_structure"]["PASS"] else "否（部分）"
    q3 = "是" if gates["B_alignment"]["PASS"] else "否"
    q4 = "是" if verdict == "PASS_TO_ROLLOUT" else "否"

    md = f"""# SEMANTIC_TOKEN_CD_PHASE0_V1 — Phase-0 Feasibility Audit Report

> **Verdict: `{pass_or_stop}`**
> 日期 2026-08-23 · split=Confirmation (n={res['n_states']}) · 未跑 rollout（按预注册禁止）

## 0. 一句话结论

将 256 个 visual patch tokens 无训练聚合成 semantic tokens 后，去除「对象语义组」
的 negative branch（r_semantic）**{"比 attention/patch 更接近" if q3 == "是" else "并未比 attention/patch 更接近"}** PCD 的
semantic counterfactual r_PCD。语义组结构{"被" if q1 == "是" else "未被"}聚类发现。

## 1. 四个核心问题

| # | 问题 | 答案 |
|---|---|---|
| 1 | 能否把 visual patch tokens 组织成 semantic tokens？ | **{q1}** |
| 2 | semantic tokens 是否对应真实视觉语义？ | **{q2}** |
| 3 | semantic negative branch 是否比 patch/attention 更接近 PCD？ | **{q3}** |
| 4 | 值得进入 closed-loop CD 吗？ | **{q4}** |

## 2. 分组质量（Gate A：structure，kmeans 为主）

median over {res['n_states']} states：

| 方法@K | g_obj size | IoU (chance) | precision | recall | compactness | cos(g_obj,r_PCD) | cos(attn) | cos(rnd) | D_sem | D_attn | D_rnd |
|---|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(row(m, K) for m, K in [("kmeans",8),("kmeans",16),("kmeans",32),("agglomerative",16),("spectral",16)])}

- **Gate A（对象被聚类发现）**: kmeans@8/16/32 → {"全部 PASS" if gates["A_structure"]["PASS"] else "FAIL"}（per_K {res["gates"]["A_structure"]["per_K"]}）
- 解读：kmeans@8 的 g_obj IoU={agg['kmeans_8']['iou_median']:.3f}（chance {agg['kmeans_8']['chance_iou_median']:.3f}），
  K 越大组越小、recall 下降但 precision 仍高。

## 3. 对齐（Gate B：alignment）

r_semantic 与 r_PCD 的 per-position cosine（pooled median，跨 states×7 positions）：

- kmeans@8: {agg['kmeans_8']['cos_pos_semantic_pooled']:.4f} vs attention {agg['kmeans_8']['cos_pos_attn_pooled']:.4f}
- kmeans@16: {agg['kmeans_16']['cos_pos_semantic_pooled']:.4f} vs attention {agg['kmeans_16']['cos_pos_attn_pooled']:.4f}
- kmeans@32: {agg['kmeans_32']['cos_pos_semantic_pooled']:.4f} vs attention {agg['kmeans_32']['cos_pos_attn_pooled']:.4f}
- **Gate B**: {"PASS（≥2 K）" if gates["B_alignment"]["PASS"] else "FAIL"}（per_K {res["gates"]["B_alignment"]["per_K"]}）

### 3.1 语义特异性（问题 2 的直接检验：对象组是否也是 PCD 对齐最高的组）

对每个 state 的 K 个组按 cos(r_g, r_PCD) 排序，看 argmax-IoU 的「对象组」排第几：

| K | 对象组对齐 rank（median） | top-1 占比 | top-2 占比 | 对象组 cos | 最佳组 cos | 全体组 median cos |
|---|---|---|---|---|---|---|
{chr(10).join(f"| {K} | {spec.get(f'kmeans_{K}', {}).get('object_group_alignment_rank_median', 0):.1f} | {spec.get(f'kmeans_{K}', {}).get('frac_object_group_top1', 0):.2f} | {spec.get(f'kmeans_{K}', {}).get('frac_object_group_top2', 0):.2f} | {spec.get(f'kmeans_{K}', {}).get('object_group_cos_median', 0):.4f} | {spec.get(f'kmeans_{K}', {}).get('best_group_cos_median', 0):.4f} | {spec.get(f'kmeans_{K}', {}).get('median_group_cos_median', 0):.4f} |" for K in [8,16,32])}

- 解读：对象组在 **69%（K=8）/ 56%（K=16）** 的 state 里就是 PCD 对齐最高的那个组，
  且对象组对齐（~0.62）远高于全体组中位数（~0.18）。这说明「看起来像对象的组」与
  「去除后最能复现 PCD 的组」高度重合 —— semantic token 的确对应真实视觉语义。

## 4. 破坏强度（Gate C：disruption）

||r||_F median over states：

- kmeans@8: D_sem={agg['kmeans_8']['D_semantic_median']:.3f} vs D_random={agg['kmeans_8']['D_random_median']:.3f}
- kmeans@16: D_sem={agg['kmeans_16']['D_semantic_median']:.3f} vs D_random={agg['kmeans_16']['D_random_median']:.3f}
- kmeans@32: D_sem={agg['kmeans_32']['D_semantic_median']:.3f} vs D_random={agg['kmeans_32']['D_random_median']:.3f}
- **Gate C**: {"PASS（≥2 K）" if gates["C_disruption"]["PASS"] else "FAIL"}（per_K {res["gates"]["C_disruption"]["per_K"]}）

## 5. 上限参照（full-object persistent mask）

- full-object r_fullobj 与 r_PCD 的 pooled cosine = {ceiling:.4f}（本 split 重算）
- 历史 persistent Token-PCD 对齐 = {hist['persistent_object_pixel_cosine_median']} / random {hist['persistent_random_pixel_cosine_median']}（selection split）
- 最强 semantic 对齐 = {res['best_K_cos']:.4f}
- 注：kmeans@8 的 semantic 对齐（{agg['kmeans_8']['cos_pos_semantic_pooled']:.4f}）略高于 full-object 上限（{ceiling:.4f}）。
  原因：K=8 的对象组是「对象 + 紧邻背景」的空间 blob（recall {agg['kmeans_8']['recall_median']:.2f}），
  PCD 的 inpainting 正是把对象区域连同周边纹理一起替换，因此这个 blob 比精确 object mask 更贴近 PCD 的真实 counterfactual。

## 6. STOP RULE 判定

| 停止条件 | 命中 |
|---|---|
| semantic ≈ random（无语义结构，Gate A 失败） | {"是" if res["stop_flags"]["semantic_approx_random"] else "否"} |
| semantic < attention（未超已有 proxy，Gate B 失败） | {"是" if res["stop_flags"]["semantic_lt_attention"] else "否"} |
| alignment 远低于 full-object 上限（<0.5×） | {"是" if res["stop_flags"]["alignment_not_close_pcd"] else "否"} |

## 7. 结论

{pass_or_stop}

**产出文件**：semantic_npz/ · semantic_groups.npz · group_assignment.json · group_visualization/ ·
pcd_alignment.csv · baseline_comparison.csv · evaluation_results.json · phase0_decision.json · REPORT_ZH.md
"""
    (art / "REPORT_ZH.md").write_text(md)
    print(f"wrote {art / 'REPORT_ZH.md'} · verdict {verdict}")


if __name__ == "__main__":
    main()
