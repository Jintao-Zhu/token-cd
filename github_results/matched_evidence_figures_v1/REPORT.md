# Matched 证据图表

## 1. Matched vs 20 个对照的总体 paired net

![overall controls](fig01_overall_controls_net.png)

数据口径：Matched 成功数减对照成功数；正数表示 Matched 更好。

## 2. 关键 Arm 的逐任务成功率

![key arms by task](fig02_key_arms_by_task.png)

包含：Matched、K32、Shuffle、Entity-TopP、TopP80、Random mask。

## 3. Fixed-m 响应曲线

![fixed count curve](fig03_fixed_count_curve.png)

m = 16, 24, 32, 48, 64；红色虚线为 L11+Matched 总体 53.2%。

## 4. 位置 × 数量 factorial

![position count factorial](fig07_position_count_factorial.png)

均为 seeds 100–199、400 paired episodes。

## 5. Matched m 与哪些状态量相关

![mechanism heatmap](fig04_budget_mechanism_heatmap.png)

数值为每个任务内的 Spearman rho。

## 6. 相对剂量响应：Small / Middle / Large

![dose response](fig05_dose_response_by_size.png)

横轴 r = m / m_matched；纵轴分别为 feature shift、action JS、clean-action support drop。

## 7. Shuffle 预算错配与 Matched 优势

![shuffle mismatch](fig06_shuffle_mismatch_bins.png)

按 episode 平均错配 D 分组；Matched-only 与 Shuffle-only 为 discordant episode 数。

## 8. Matched m 与 effective knee

![phase A knee](fig08_phaseA_knee.png)

逐任务 Spearman(matched m, effective knee)。

## 原始汇总

- `artifacts/CONSOLIDATED_DATA_2026-09-20.md`
- `artifacts/l11_matched_state_coupling_v1/STATE_COUPLING_REPORT.md`
- `artifacts/l11_budget_response_curve_v1/PHASE_A_REPORT.md`
- `artifacts/l11_entity_budget_calibration_v1/CALIBRATION_REPORT.md`
- `artifacts/l11_entity_budget_closed_loop_100_199_v1/CLOSED_LOOP_REPORT.md`
- `artifacts/l11_random_mask_matched_count_100_199_v1/CLOSED_LOOP_REPORT.md`
- `artifacts/matched_dynamic_budget_analysis_v1/ANALYSIS_REPORT.md`
- `artifacts/intervention_dose_response_v1/DOSE_REPORT.md`
