# matched 实验总目录（全部数据结果汇总）

生成时间 2026-10-16 15:36:34+0800 · 生成脚本 `scripts/build_matched_experiment_index.py` · 直接扫描 `artifacts/` 得到

## 0. 概览

- 含 matched（或 matched 变体）臂的 artifact：**59 个**
- matched 干预总集数：**62,431 集**
- 覆盖模型：OpenVLA, OpenVLA (7-card), SpatialVLA, pi0-FAST, unknown
- 单实验规模：1 – 16,200 集
- 跑满稠密网格的 artifact：49 / 59（其余为部分覆盖或已提前停止）

## 1. 已核实的关键结果（数字均在本会话中从 episode summary 重算）

### V1. OpenVLA，9 任务，λ=1.0：matched 与 12 种预算配置

来源：`artifacts/openvla_lambda1_extension_200_299_v1` + `artifacts/openvla_budget_rule_9task_200_299_v1`

| 条件 | 成功/总数 | 比例或效应 | 备注 |
|---|---|---|---|
| matched (K8) | 281/900 | 31.2% | 基线 |
| fixed 16 / 24 / 32 / 40 | 289 / 267 / 267 / 275 of 900 | 32.1 / 29.7 / 29.7 / 30.6% | 全部 Holm 不显著 |
| fixed 48 / 56 / 64 | 285 / 296 / 279 of 900 | 31.7 / 32.9 / 31.0% | 全部 Holm 不显著 |
| TopP 0.75 / 0.80 / 0.85 / 0.90 | 266 / 264 / 280 / 293 of 900 | 29.6 / 29.3 / 31.1 / 32.6% | 全部 Holm 不显著 |

**解读与限制**：不显著 ≠ 等效；九任务上没有任何预算规则被区分出来

### V2. λ × 预算规则交互（4 Google 任务 × 100 seeds × 2 λ）

来源：`artifacts/openvla_lambda_budget_grid_200_299_v1`，报告 `docs/OPENVLA_BUDGET_LAMBDA_TASKMEAN_FINAL.md`

| 条件 | 成功/总数 | 比例或效应 | 备注 |
|---|---|---|---|
| matched λ=0.5 / λ=1.0 | 205/400 / 201/400 | 51.25% / 50.25% | 自身对 λ 无显著变化（净 −4, p=0.79） |
| fixed16 λ=0.5 / λ=1.0 | 165/400 / 206/400 | 41.25% / 51.50% | λ 效应净 +41，Holm 0.001 |
| fixed48 λ=0.5 / λ=1.0 | 199/400 / 202/400 | 49.75% / 50.50% | λ 效应净 +3，Holm 1.0 |
| random48 λ=0.5 / λ=1.0 | 157/400 / 189/400 | 39.25% / 47.25% | λ 效应净 +32，Holm 0.014 |
| TopP0.90 λ=0.5 / λ=1.0 | 201/400 / 226/400 | 50.25% / 56.50% | λ 效应净 +25，Holm 0.086 |
| 交互量 I（vs matched） | fixed16 +11.25pp、random48 +9.00、TopP90 +7.25、fixed48 +1.75 | CI [3.75,18.75] / [1.50,16.25] / [−0.50,15.00] / [−5.50,9.00] | Holm 0.021 / 0.058 / 0.130 / 0.661 |

**解读与限制**：唯一 Holm 显著的交互是「预算过小」的代价随 λ 缩小；这不能推广成「数量维度失效」

### V3. 去掉逐步自适应（任务均值常数）对 matched 的消融

来源：`artifacts/openvla_taskmean_2lambda_200_299_v1`（新，800 集）+ `artifacts/l11_taskmean_budget_100_199_v1`（旧，400 集）

| 条件 | 成功/总数 | 比例或效应 | 备注 |
|---|---|---|---|
| taskmean λ=0.5 vs matched | 182/400 vs 205/400 | −5.75pp | p=0.054，Holm 0.108（不显著） |
| taskmean λ=1.0 vs matched | 199/400 vs 201/400 | −0.50pp | p=0.93 |
| 复现：λ=0.5，seeds 100–199 | 185/400 vs 213/400 | −7.00pp | p=0.012（另一 seed 区间显著） |

**解读与限制**：常数来自 λ=1 参考且用了同一批评估 seed，属于事后均值匹配消融；同一效应在两个 seed 区间一个显著一个不显著

### V4. λ=0.5、seeds 100–199 的固定预算阶梯 vs matched（旧实验）

来源：`artifacts/prompt_attn_l11_token_count_v1`（matched 与 5 个固定档）

| 条件 | 成功/总数 | 比例或效应 | 备注 |
|---|---|---|---|
| matched | 213/400 | 53.25% | 同一批 task/seed |
| fixed 16 / 24 / 32 / 48 / 64 | 151 / 173 / 196 / 207 / 198 of 400 | 37.75 / 43.25 / 49.00 / 51.75 / 49.50% | 预算过小时明显吃亏，48 档接近 matched |

**解读与限制**：这条阶梯是后来解释「K 调大 ⇒ 预算变小 ⇒ 变差」的参照系

### V5. KMeans 粒度 K=16/32（漂移检出后停止，仅冒烟证据）

来源：`artifacts/openvla_kmeans_budget_smoke_v1`（7 集）

| 条件 | 成功/总数 | 比例或效应 | 备注 |
|---|---|---|---|
| close_drawer/seed200 平均 m_t | K8 43.5 / K16 8.2 / K32 6.5 | — | K8 冒烟与历史逐 step 完全相同（113/113） |
| pick_coke_can/seed200 平均 m_t | K8 39.8 / K16 16.1 / K32 4.1 | — | 下降比 1/K 更陡 |
| 主实验 1,800 集 | 0/1800 | — | 启动前叫停，未产生主结果 |

**解读与限制**：3 任务 × 1 seed，只能证明「K 改变的是预算规模」，不能证明效能

### V6. 同状态反事实探针：m_t 随 K 的曲线（提前停止）

来源：`artifacts/openvla_k_probe_200_219_v1`（13 集 + 1 集 preflight）

| 条件 | 成功/总数 | 比例或效应 | 备注 |
|---|---|---|---|
| close_drawer | K8 39.2 → K9 30.1 → K10 24.3 → K12 15.8 → K16 9.8 | K10/K8 = 0.62 | 4 集 |
| open_drawer | K8 36.2 → K9 42.3 → K10 37.1 → K12 16.8 → K16 7.4 | K10/K8 = 1.02 | 3 集 |
| eggplant | K8 29.5 → K9 29.2 → K10 27.7 → K12 26.6 → K16 23.6 | K10/K8 = 0.94 | 3 集 |

**解读与限制**：K=10 对预算的压缩幅度**依任务而异**；仅 3–4 集/任务，属初步

### V7. 真跑 K=10（进行中，λ=0.5 与 λ=1.0）

来源：`artifacts/openvla_k10_2lambda_200_299_v1`，与网格 matched 同 task/seed 配对

| 条件 | 成功/总数 | 比例或效应 | 备注 |
|---|---|---|---|
| λ=1.0（396 对） | K10 184/396 vs matched 199/396 | −3.79pp | p=0.199，区间约 [−9.2, +1.6] 含 0 |
| λ=0.5（286 对，缺 pick） | K10 160/286 vs matched 165/286 | −1.75pp | p=0.63 |
| 实测预算 | m_t 24.8–26.2 vs matched 33.6–35.5 | 约 −27% | 预算确实被压小 |

**解读与限制**：两档 λ 都略负但都不显著；move_near 在两个 λ 上都最负（−9 / −8）。**仍在跑**

### V8. 复用对照的完整性审计（10,800 集）

来源：`artifacts/openvla_kmeans_budget_9task_200_299_v1/CONTROL_REUSE_AUDIT.json`

| 条件 | 成功/总数 | 比例或效应 | 备注 |
|---|---|---|---|
| K8 参考 | 900/900 | technical_pass 全真 | — |
| 11 个对照臂 | 9,900/9,900 | technical_pass 全真 | 排除 20 集诊断臂 |
| 配对身份 | 900/900 个 (task,seed) | 初始 SHA + 指令一致 | 0 失败 |

**解读与限制**：只证明数据完整可复用，不是效能结论

## 2. 完整目录（按主题分组）

### G. 冒烟 / 预检 / 审计（不构成结论）

| artifact | 模型 | 任务 | seeds | matched 集数 | matched 臂 | 对照臂数 | 稠密 | 状态 |
|---|---|---|---|---|---|---|---|---|
| `l11_full_rt_open_drawer_reproduction_100_129_v1` | OpenVLA | 1 | 100–129 | 30 | l11_matched | 1 | 是 | 完成 |
| `openvla_budget_rule_smoke3_v1` | OpenVLA | 1 | 200–219 | 20 | lambda_100 | 0 | 是 | 完成 |
| `openvla_budget_rule_smoke_v1` | OpenVLA | 1 | 200–219 | 20 | lambda_100 | 2 | 是 | 完成 |
| `spatialvla_widowx_lambda_0_299_preflight` | SpatialVLA | 4 | 0–0 | 12 | lambda_025, lambda_075, lambda_100 | 0 | 是 | 完成 |
| `openvla_budget_rule_smoke2_v1` | OpenVLA | 1 | 211–220 | 10 | lambda_100 | 0 | 是 | 完成 |
| `spatialvla_widowx_lambda_high_0_299_preflight` | SpatialVLA | 4 | 0–0 | 8 | lambda_125, lambda_150 | 0 | 是 | 完成 |
| `openvla_kmeans_budget_smoke_v1` | OpenVLA | 3 | 200–200 | 7 | matched_k08_l100, matched_k16_l100, matched… | 0 | 是 | 已停止（冒烟 7/9 后叫停） |
| `spatialvla_l10_lambda_rtfull9_preflight_20261008` | SpatialVLA | 2 | 100–100 | 6 | lambda_025, lambda_075, lambda_100 | 0 | 是 | 完成 |
| `widowx_stack_cube_video_audit_more_20260927` | unknown | 1 | 16–251 | 5 | l11_matched | 1 | 否 | 部分覆盖 |
| `openvla_lambda_budget_grid_smoke_v1` | OpenVLA | 1 | 201–203 | 3 | matched_l050 | 3 | 是 | 完成 |
| `simpler_rt_depth0_other_tasks_video_audit_20260927` | OpenVLA | 2 | 64–249 | 3 | l11_matched | 1 | 否 | 部分覆盖 |
| `widowx_stack_cube_video_audit_20260927` | unknown | 1 | 10–247 | 3 | l11_matched | 1 | 否 | 部分覆盖 |
| `spatialvla_l10_remaining5_preflight` | SpatialVLA | 2 | 100–100 | 2 | l10_matched | 0 | 是 | 完成 |
| `spatialvla_l10_vs_l11_4task_n100_preflight` | SpatialVLA | 1 | 100–100 | 2 | l10_matched, l11_matched | 0 | 是 | 完成 |
| `spatialvla_simpler_remaining5_preflight` | SpatialVLA | 2 | 100–100 | 2 | l11_matched | 1 | 是 | 完成 |
| `pi0_fractal_google_preflight24_v1` | pi0-FAST | 1 | 0–0 | 1 | l2_matched_pcd | 1 | 是 | 完成 |
| `pi0_fractal_google_preflight_v1` | pi0-FAST | 1 | 0–0 | 1 | l2_matched_pcd | 1 | 是 | 完成 |
| `spatialvla_l10_vanilla_rtfull7_preflight_20260930` | SpatialVLA | 1 | 0–0 | 1 | l10_matched | 1 | 是 | 完成 |

### A. 数量规则之争：matched vs 固定预算 / TopP / 随机 / 任务均值 / K 粒度

| artifact | 模型 | 任务 | seeds | matched 集数 | matched 臂 | 对照臂数 | 稠密 | 状态 |
|---|---|---|---|---|---|---|---|---|
| `openvla_lambda_budget_grid_200_299_v1` | OpenVLA | 4 | 200–299 | 800 | matched_l050, matched_l100 | 8 | 是 | 完成 |
| `openvla_k10_2lambda_200_299_v1` | OpenVLA | 4 | 200–299 | 776 | matched_k10_l050, matched_k10_l100 | 0 | 否 | 进行中（真跑 K=10） |
| `prompt_attn_l11_token_count_v1` | OpenVLA | 4 | 100–199 | 400 | l11_matched | 5 | 是 | 完成 |
| `openvla_budget_rule_9task_200_299_v1` | OpenVLA | 2 | 201–210 | 20 | lambda_100 | 11 | 是 | 完成 |
| `openvla_k_probe_200_219_v1` | OpenVLA | 4 | 200–215 | 13 | lambda_100 | 0 | 否 | 已停止（13/180 后叫停，数据保留） |

### B. 数量维度的因果消融（把 matched 的预算改大改小、或打乱它与状态的对应）

| artifact | 模型 | 任务 | seeds | matched 集数 | matched 臂 | 对照臂数 | 稠密 | 状态 |
|---|---|---|---|---|---|---|---|---|
| `seven_card_matched_dose_response_v1` | OpenVLA (7-card) | 4 | 100–199 | 1,600 | matched_scale_050, matched_scale_075, match… | 0 | 是 | 完成 |
| `l11_matched_budget_smallscale_v1` | OpenVLA | 4 | 100–199 | 1,200 | matched_small16_x150, matched_small16_x200,… | 0 | 是 | 完成 |
| `l11_matched_plus_uniform16_v1` | OpenVLA | 9 | 100–199 | 900 | matched_uniform16 | 0 | 是 | 完成 |
| `l11_matched_budget_causal_pilot_v1` | OpenVLA | 4 | 100–124 | 400 | matched_scale_050, matched_scale_075, match… | 3 | 是 | 完成 |
| `l11_matched_state_coupling_v1` | OpenVLA | 4 | 100–199 | 400 | matched_shuffle | 0 | 是 | 完成 |

### E. 跨模型：同一 matched 规则换架构

| artifact | 模型 | 任务 | seeds | matched 集数 | matched 臂 | 对照臂数 | 稠密 | 状态 |
|---|---|---|---|---|---|---|---|---|
| `spatialvla_widowx_lambda_0_299_v1` | SpatialVLA | 4 | 0–299 | 3,600 | lambda_025, lambda_075, lambda_100 | 0 | 是 | 完成 |
| `spatialvla_l10_lambda_rtfull9_100_299_v1` | SpatialVLA | 9 | 100–299 | 2,786 | lambda_025, lambda_075, lambda_100 | 0 | 否 | 部分覆盖 |
| `spatialvla_l10_vanilla_expand9_to300_v1` | SpatialVLA | 9 | 0–299 | 1,800 | l10_matched | 1 | 否 | 部分覆盖 |
| `spatialvla_widowx_lambda_high_0_299_v1` | SpatialVLA | 4 | 0–299 | 1,616 | lambda_125, lambda_150 | 0 | 否 | 部分覆盖 |
| `spatialvla_l10_vanilla_rtfull7_to300_v1` | SpatialVLA | 7 | 0–299 | 1,400 | l10_matched | 1 | 否 | 部分覆盖 |
| `spatialvla_l10_vs_l11_4task_n100` | SpatialVLA | 4 | 100–199 | 800 | l10_matched, l11_matched | 0 | 是 | 完成 |
| `spatialvla_l10_matched_rtfull7_n100_199` | SpatialVLA | 7 | 100–199 | 700 | l10_matched | 0 | 是 | 完成 |
| `spatialvla_l10_matched_remaining5_n100` | SpatialVLA | 5 | 100–199 | 500 | l10_matched | 0 | 是 | 完成 |
| `spatialvla_simpler_l11matched_remaining5_n100` | SpatialVLA | 5 | 100–199 | 500 | l11_matched | 1 | 是 | 完成 |
| `pi0_fractal_l2_matched_google800_v1` | pi0-FAST | 4 | 0–99 | 400 | l2_matched_pcd | 1 | 是 | 完成 |
| `seven_card_spatialvla_factorization_v1` | SpatialVLA | 4 | 100–199 | 400 | kmeans_matched | 2 | 是 | 完成 |
| `spatialvla_simpler_l11matched_4task_n100` | SpatialVLA | 4 | 100–199 | 400 | l11_matched | 1 | 是 | 完成 |
| `pi0_fractal_l10_matched_pcd_google800_v1` | pi0-FAST | 4 | 0–99 | 344 | l10_matched_pcd | 1 | 否 | 部分覆盖 |
| `spatialvla_drawers_rt_depth0_vanilla_l10_100_199_v1` | SpatialVLA | 2 | 100–199 | 200 | l10_matched | 1 | 是 | 完成 |

### D. 内容/空间选择：选哪些 token（数量保持不变）

| artifact | 模型 | 任务 | seeds | matched 集数 | matched 臂 | 对照臂数 | 稠密 | 状态 |
|---|---|---|---|---|---|---|---|---|
| `l11_matched_spatial_v1` | OpenVLA | 4 | 100–199 | 1,200 | matched_spatial_both, matched_spatial_corne… | 0 | 是 | 完成 |
| `simpler_geometry_selector_4arm_v2_deterministic` | OpenVLA | 4 | 100–199 | 800 | l11_matched, random_matched | 2 | 是 | 完成 |
| `simpler_geometry_selector_4arm_v2` | OpenVLA | 4 | 100–104 | 40 | l11_matched, random_matched | 2 | 是 | 完成 |
| `simpler_geometry_selector_4arm_v2_auditfix1` | OpenVLA | 4 | 100–104 | 40 | l11_matched, random_matched | 2 | 是 | 完成 |

### F. 跨设置泛化：7 任务 × 300 seeds、两种渲染

| artifact | 模型 | 任务 | seeds | matched 集数 | matched 臂 | 对照臂数 | 稠密 | 状态 |
|---|---|---|---|---|---|---|---|---|
| `simpler_rt_depth0_7task_4200_v1` | OpenVLA | 7 | 0–299 | 2,100 | l11_matched | 1 | 是 | 完成 |
| `simpler_rt_full_7task_4200_v1` | OpenVLA | 7 | 0–299 | 2,100 | l11_matched | 1 | 是 | 完成 |
| `simpler_rt_depth0_drawers_0_299_v1` | OpenVLA | 2 | 0–299 | 600 | l11_matched | 1 | 是 | 完成 |

### C. 引导强度 λ：matched 随 λ 的行为

| artifact | 模型 | 任务 | seeds | matched 集数 | matched 臂 | 对照臂数 | 稠密 | 状态 |
|---|---|---|---|---|---|---|---|---|
| `openvla_global_lambda_grid_0_199_v1` | OpenVLA | 9 | 0–199 | 16,200 | lambda_000, lambda_025, lambda_050, lambda_… | 0 | 是 | 完成 |
| `prompt_attn_l11_matched_dense_lambda_remaining5_0_99_v1` | OpenVLA | 5 | 0–99 | 5,500 | l11_matched_lambda_010, l11_matched_lambda_… | 0 | 是 | 完成 |
| `openvla_global_lambda_grid_225_250_0_199_v1` | OpenVLA | 9 | 0–199 | 3,600 | lambda_225, lambda_250 | 0 | 是 | 完成 |
| `prompt_attn_l11_matched_dense_lambda_0_99_v1` | OpenVLA | 4 | 0–99 | 3,200 | l11_matched_lambda_010, l11_matched_lambda_… | 0 | 是 | 完成 |
| `openvla_guidance_strength_scan_100_199_v1` | OpenVLA | 9 | 100–199 | 2,700 | lambda_025, lambda_075, lambda_100 | 0 | 是 | 完成 |
| `openvla_lambda1_extension_200_299_v1` | OpenVLA | 9 | 200–299 | 900 | lambda_100 | 0 | 是 | 完成 |
| `seven_card_lambda_sensitivity_v1` | OpenVLA (7-card) | 4 | 100–149 | 600 | lambda_025, lambda_075, lambda_100 | 0 | 是 | 完成 |
| `attn_semantic_lambda_sweep_v1` | unknown | 1 | 100–199 | 500 | lambda_010, lambda_025, lambda_050, lambda_… | 1 | 是 | 完成 |
| `seven_card_c6_lambda1_heldout_150_199_v1` | OpenVLA (7-card) | 4 | 150–199 | 200 | lambda_100 | 0 | 是 | 完成 |

### H. 未归类

| artifact | 模型 | 任务 | seeds | matched 集数 | matched 臂 | 对照臂数 | 稠密 | 状态 |
|---|---|---|---|---|---|---|---|---|
| `l11_rt_depth0_pick_move_full_100_129_v1` | OpenVLA | 2 | 100–129 | 60 | l11_matched | 1 | 是 | 完成 |

## 3. 已停止 / 未跑满的实验（不要当结论引用）

- `openvla_k10_2lambda_200_299_v1`：进行中（真跑 K=10）（matched 776 集）
- `openvla_k_probe_200_219_v1`：已停止（13/180 后叫停，数据保留）（matched 13 集）
- `openvla_kmeans_budget_smoke_v1`：已停止（冒烟 7/9 后叫停）（matched 7 集）
- `pi0_fractal_l10_matched_pcd_google800_v1`：网格未填满（matched 344 集）
- `simpler_rt_depth0_other_tasks_video_audit_20260927`：网格未填满（matched 3 集）
- `spatialvla_l10_lambda_rtfull9_100_299_v1`：网格未填满（matched 2,786 集）
- `spatialvla_l10_vanilla_expand9_to300_v1`：网格未填满（matched 1,800 集）
- `spatialvla_l10_vanilla_rtfull7_to300_v1`：网格未填满（matched 1,400 集）
- `spatialvla_widowx_lambda_high_0_299_v1`：网格未填满（matched 1,616 集）
- `widowx_stack_cube_video_audit_20260927`：网格未填满（matched 3 集）
- `widowx_stack_cube_video_audit_more_20260927`：网格未填满（matched 5 集）

## 4. 数据质量与注意事项（引用本目录数字前必读）

- **早期任务均值被污染**：某张任务表的 `ref_mt` 曾把 fixed64 的值混进 matched 均值，报出 45–57；正确值约为 close 35.2 / open 37.0 / pick 29.3 / move 33.3。未使用的其他任务常数（apple 57 等）仍是污染值，不得引用。
- **进程历史影响轨迹**：seed 分块（`[200]`、`[201–210]`…）不同会导致同一初始状态走出不同轨迹；比较必须用相同分块，且不能用「初始 SHA 相同」推断轨迹相同。
- **渲染器差异**：旧种子段（100–199）的 pick/move 与 200–299 不是同一渲染设置，跨种子段比较只能当方向性复现。
- **不显著 ≠ 等效**：多处曾用「打平」「差异消失」「不敏感」描述非显著结果，已作废；现在的表述一律给出观测差值与区间。
- **λ 只有离散测点**：「matched 对 λ 不敏感」这类说法在本仓库中没有被检验过（只测了若干 λ 点）。
- **K 不是粒度旋钮**：簇大小约 256/K，改 K 同时改预算；K=16/32 的 1,800 集因此没有跑（漂移在冒烟阶段被检出）。
- **随机内容对照的正确用法**：内容维度的干净对照是 fixed48 vs random48（同数量、不同内容）；matched vs random48 同时改变数量规则与内容，不能单独归因。

## 5. 相关详细报告

- [`docs/OPENVLA_KMEANS_BUDGET_9TASK_PLAN.md`](<docs/OPENVLA_KMEANS_BUDGET_9TASK_PLAN.md>) — 存在
- [`docs/OPENVLA_KMEANS_BUDGET_9TASK_TABLE.md`](<docs/OPENVLA_KMEANS_BUDGET_9TASK_TABLE.md>) — 尚未生成
- [`docs/OPENVLA_KMEANS_BUDGET_9TASK_FINAL.md`](<docs/OPENVLA_KMEANS_BUDGET_9TASK_FINAL.md>) — 尚未生成
- [`docs/OPENVLA_BUDGET_LAMBDA_TASKMEAN_FINAL.md`](<docs/OPENVLA_BUDGET_LAMBDA_TASKMEAN_FINAL.md>) — 存在
- [`docs/OPENVLA_K10_2LAMBDA_FINAL.md`](<docs/OPENVLA_K10_2LAMBDA_FINAL.md>) — 尚未生成
- [`docs/OPENVLA_K_PROBE_BUDGET_CURVE.md`](<docs/OPENVLA_K_PROBE_BUDGET_CURVE.md>) — 尚未生成
- [`docs/OPENVLA_BUDGET_RULE_9TASK_FINAL.md`](<docs/OPENVLA_BUDGET_RULE_9TASK_FINAL.md>) — 存在
- [`docs/OPENVLA_LAMBDA_BUDGET_GRID_FINAL.md`](<docs/OPENVLA_LAMBDA_BUDGET_GRID_FINAL.md>) — 存在
- [`docs/OPENVLA_BUDGET_RULE_REPRODUCIBILITY_FINDING.md`](<docs/OPENVLA_BUDGET_RULE_REPRODUCIBILITY_FINDING.md>) — 存在

## 6. 方法与口径说明

- 「matched 集数」= 该 artifact 中所有 matched 类臂（名字含 `matched` 或以 `lambda_` 开头）的 episode summary 文件数，直接数文件得到。
- 「稠密」= 每个臂的集数等于（任务数 × seed 跨度），用来发现缺块；它不判断该实验是否本来就不需要跑满。
- 只有第 1 节列出的结果是本会话中重算并核对过的；其他 artifact 我只给出规模与设置，**不替它们填结论**。

