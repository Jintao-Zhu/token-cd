# DTP OpenVLA 校准协议

状态更新：协议已执行完毕——60状态离线校准 → 3臂闭环校准(66 episode) → 冻结 l11_k64_t05 → 正式400(0–99×4任务, 167 vs vanilla 168)。实现差距与v2候选见 REPORT.md、IMPLEMENTATION_GAP.md。

## 来源审计

- 论文：https://arxiv.org/html/2601.16065v1 ，§3、附录A.2。
- 官方源码链接：https://anonymous.4open.science/r/CBD3 。本次查询 /api/repo/CBD3/files 返回 HTTP 410，正文 {"error":"repository_expired"}。
- 本地文件名与 research/docs 文本搜索未找到可核实的 DTP 源码副本；现有 Prompt/Action 实验不能当作原作者实现。
- 论文确认 Action attention 按全层视觉注意力比例加权；保护集之外满足 A[v] > tau * max(A[G]) 的位置为剪枝候选。
- 论文附录明确：一般流程逐 action token 检测再以 masked attention 重新生成；SpatialVLA 特别简化为只分析第一个 action token，整步共用 mask。OpenVLA 不自动沿用该简化。
- Gaussian 核/边界处理、四角抑制数值及顺序、层权重的精确归一化、query 与预测位置对应、cache/prefix 与重生成过程均待源码审计。不能把 harmonic reconstruction 或 SHR logit CD 当作 DTP 剪枝实现。

## 离线阶段

四任务 open_drawer、close_drawer、pick_coke_can、move_near，各5 episode × 早中晚3状态，共60状态。复用入口：artifacts/prompt_attn_l11_budget_diagnostic_v1/selection_manifest.json；正式使用前逐项验证文件、图像和物理初态身份。

先诊断语言模型0–31层的完整指令 attention，核对实际 query/key 位置、clean action 一致性。同图目标切换保持保护数量一致，区分换目标和同义改写。跨层排名一致性不单独充当语义正确性证据。

相关性层候选缩减后，才检查保护预算 k=64、109、154（25%、42.58%、60.16%）与 tau=0.5、1.0、1.5。k 指保护数量，不是剪掉数量。候选值不构成推荐配置。逐动作维度保存剪枝数、空剪枝率、保护集/剪枝集、空间处理前后排名和集合差异。保存原始全层评分，使空间处理可离线重算。

源码核实后检查四角抑制和 Gaussian 的原始处理及单项消融。不得根据热图、少量动作变化或最终前100seed成功率挑参数。

## 场景隔离与闭环

最多2–3个候选进入小规模校准，建议每任务10个去重物理场景。从100–299按固定顺序筛选，排除与正式0–99或其他校准场景物理初态重复者。不能只用包含seed或RNG元数据的snapshot文件哈希判定物理去重；应记录环境配置、指令、sim/agent状态及RGB身份。

统一配置冻结后，正式四任务各100seed（0–99），共400 episode；保留前100seed内部重复关系并按物理场景聚类报告。它们是固定基准验证，不能称为未见测试。

技术失败重试与正常策略失败分开。所有闭环使用tmux持久运行，记录版本、配置、场景hash、视频、完整终止原因。运行前核实可用GPU及显存。

## 交付

离线逐状态图表、保护/剪枝数量统计、源码审计记录、校准配对视频及结果、跨任务统一配置锁定文件、400episode固定基准报告。

原源码阻碍已由用户授权论文重实现解决，不再等待源码。当前限制是两个drawer任务在现有100–299seed中各只剩1个与正式集不重叠的物理初态；见CALIBRATION_SCENE_AUDIT.json。

## v2 机制审计（已执行，2026-09-09）

正式400封口后按外部审查新增阶段：先跑两项 sanity（当前harness vanilla 与 control 逐 executed action 比对、hook 与 outputs.attentions 逐元素比对），再在22个锁定校准场景跑 v2 固定mask臂（v2_fix_k64_t05、v2_fix_k109_t05，L11/τ0.5）共44 episode。预注册判定：dynamic-k64 vs fixed-k64 回答检测时序/cache反馈问题；fixed-k64 vs fixed-k109 回答保护区宽度问题；保持超参冻结，不把本轮当成功率调参。结果与判定见 REPORT.md、closed_loop/V2_AUDIT.json。
