# VTCG 真机迁移

本目录将仿真中的 VTCG 推理方法接到一个与具体机械臂无关的 OpenVLA 运行入口。方法主体仍是 `research/semantic_token_cd/prompt_attn_shr_policy.py`；`policy.py` 固定实验配置并检查每步 trace，`openvla_runtime.py` 只替代原先依赖 SIMPLER 的 OpenVLA 包装。这里没有另写一套 selector 或负分支算法。

## 固定方法配置

| 项目 | 配置 |
|---|---|
| 基础模型 | OpenVLA-7B，checkpoint 自带 tokenizer、processor 与动作统计 |
| 预算 | KMeans `K=8`、`random_state=0`、`n_init=10`；实体匹配簇的并集大小为逐步预算 `m_t` |
| 选择器 | zero-based decoder layer 11 的指令到视觉注意力；按分数取 Top-`m_t`，视觉注意力不重新归一化 |
| 反事实 | projector 输出的所选 token 做 16×16 四邻接调和重建，`β=0` |
| 对比 | 负分支使用干净动作前缀；`λ=0.5` 作用于动作维度 0–5，夹爪维度 6 保持干净分支 |
| 解码 | 抑制 EOS，固定七个动作 token，greedy 解码；使用 `unnorm_key` 反归一化 |

[METHOD.md](METHOD.md)给出注意力索引、重建方程、同前缀计算与审计字段。`st_shr_policy.py` 和 `global_merge_policy.py` 分别负责调和重建与 projector 干预；迁移入口沿用研究代码中的 `PromptAttentionSHRInference`。

## 代码与设备边界

`runner.py` 接受保存的 RGB 图像或由 `adapter.py` 定义的流式观测接口。输入被缩放到 224×224 并交给 checkpoint processor；输出包含七维反归一化模型动作、`m_t`、所选位置、各分支 token、图像哈希和干预审计。运行入口默认只记录；显式 `--execute` 才调用设备适配器的 `send_command()`。

模型动作与机器人控制命令之间的坐标系、单位、旋转和夹爪映射由设备适配器定义。`HARDWARE_ADAPTER.md` 记录该边界所需的标定和控制约定。checkpoint 完整目录、校验方式及源运行环境见 [TRANSFER_MANIFEST.md](TRANSFER_MANIFEST.md)和 [SOURCE_RUNTIME.md](SOURCE_RUNTIME.md)。权重和机器人驱动均不包含在 Git 中。

当前未锁定机械臂、相机、控制接口或真机动作统计，因此这里交付的是方法推理与可审计的接入边界，不包含真机性能结论。已有实验在仿真中完成；真实实验应保留原始图像、机器人状态、映射前后的动作及任务结果，供同场景对照分析。
