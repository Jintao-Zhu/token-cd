# VTCG

本仓库研究视觉语言动作模型中的视觉 token 反事实引导。当前方法以 OpenVLA-7B 为基础，在推理时构造局部视觉特征负分支，并通过对比解码调整连续动作。仿真实验与真机迁移入口复用同一方法类；真机设备的动作映射由独立适配器承担。

实现位于 [`real-robot-transfer` 分支](https://github.com/Jintao-Zhu/token-cd/tree/real-robot-transfer)。[方法细节](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/METHOD.md)、[迁移说明](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/README.md)和[硬件接口约定](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/HARDWARE_ADAPTER.md)分别记录算法定义、代码边界与设备侧约束。

## 方法

对于图像 `I_t` 和指令 `c`，OpenVLA 的 projector 输出 16×16 网格上的 256 个视觉特征。VTCG 在同一个控制步中完成以下计算：

| 环节 | 固定设计 |
|---|---|
| 动态预算 | 将 projector 特征聚成 8 组；指令中的源、目标实体各匹配一个语义组，其并集大小给出 `m_t`。 |
| 视觉位置 | 对 decoder 第 11 个零基层中“非特殊指令 token → 视觉 token”的 softmax 后注意力按 head 与查询求均值，取分数最高的 `m_t` 个位置。KMeans 给预算，L11 给位置。 |
| 负分支 | 在 projector 输出处，对选中位置求 16×16 四邻接网格的调和延拓（`β=0`）；其余特征逐位保留。 |
| 动作引导 | 负分支沿干净分支动作 token 前缀 teacher-force。动作维度 0–5 使用 `z* = 1.5 z_clean − 0.5 z_negative`；夹爪维度 6 沿用干净分支，随后 greedy 解码。 |

这里的负分支是特征空间干预，作用是测量局部视觉信息变化对同一动作前缀的 logit 影响。所选区域由每步图像和指令共同决定；动态预算与注意力排序是两个独立环节。[`prompt_attn_shr_policy.py`](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/research/semantic_token_cd/prompt_attn_shr_policy.py)包含完整推理与逐步审计，[`st_shr_policy.py`](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/research/semantic_token_cd/st_shr_policy.py)实现调和重建，[`global_merge_policy.py`](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/research/semantic_token_cd/global_merge_policy.py)实现 projector 干预和同前缀负分支。

## 真机迁移边界

迁移包提供不依赖 SIMPLER 的 OpenVLA 推理包装、VTCG 固定配置、单帧／流式推理及逐步日志。模型输出是按所选 `unnorm_key` 反归一化后的七维动作；实际控制仍需明确相机视角、动作统计、坐标系、单位、夹爪语义与控制时序。相关接口定义在 [`adapter.py`](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/adapter.py)，checkpoint 内容与运行版本见[迁移清单](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/TRANSFER_MANIFEST.md)。权重不在 Git 仓库内。

目前结果来自仿真，真实机械臂、相机和驱动尚未指定。SIMPLER 与 LIBERO 的观察并不一致，实验结论和限制汇总在 [KNOWN_EVIDENCE.md](KNOWN_EVIDENCE.md)。历史目录与日志键中的 `l11_matched` 保留为结果索引；方法名称统一为 **VTCG**。

## 其他研究

`research/semantic_token_cd/` 包含主要的 token 级对比解码实验；`research/ar_*`、`research/coreact_*`、`research/token_pcd_*` 和 `research/causal_failure_gate/` 保存相关机制研究。[SHR 方法与结果报告](docs/SHR_COMPREHENSIVE_REPORT_2026-09-05.md)和[精简实验数据](github_results/SHR_FOLLOWUP_RELEASE.md)保留了较早的实验脉络。
