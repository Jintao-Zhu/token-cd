# VTCG 方法设计与计算细节

## 目标

给定图像 `I_t` 和指令 `c`，OpenVLA 形成 `H_t ∈ R^(256×d)` 的视觉 projector 特征，位于 16×16 patch 网格。方法在**不微调模型**的推理阶段构造局部视觉信息减弱的反事实负分支，以同一动作前缀下的 logits 差值调整动作选择。它不是像素物体删除，也不声称重建真实遮挡图像。

## 1. 动态预算和位置选择

规则解析指令的源/目标实体，去重得到 `E(c)`。对 256 个 projector 向量做 KMeans：`K=8, random_state=0, n_init=10`。每个实体短语的 token embedding 求均值，再与各簇的均值向量算余弦相似度，匹配一个簇。匹配簇的视觉 token 并集大小为 `m_t`。KMeans 提供随图像变化的 **Matched 预算**，不直接提供最终扰动位置。

读取 zero-based decoder layer 11 的 post-softmax 注意力。查询是指令的非特殊文本 token，key 是多模态序列中的 256 个视觉位置 `[1,256]`。对 head 和文本查询均值，得到 `a_i`；不对视觉注意力重新归一化。按 `a_i` 降序、token 索引升序打破平局，取前 `m_t` 个位置为 `G_t`。因此每步 `|G_t|=m_t`，但 `G_t` 不必等于 KMeans 区域。

## 2. projector 特征的调和重建

在 16×16 四邻接网格，将 `G_t` 中的 projector 特征重建为未选中邻域的调和延拓。代码求解：

```text
(L_GG + βI) X = −L_GC H_C + βT,    β = 0
```

`L` 为网格图拉普拉斯矩阵，`C` 是 `G_t` 的补集。本设置 `β=0`，没有时间先验；所有未选中向量逐位保留。forward hook 在 projector 输出处替换所选位置，形成负分支。代码核查重建有限值、未选中位置 bit-identical，以及干净/负分支 hook 前的特征相同。

## 3. 同前缀对比解码

干净分支 greedy 生成七个 OpenVLA 动作 token，得 `z_q+`。负分支对相同 task prompt 和**干净分支的动作 token 前缀**做 teacher-forced forward，得 `z_q−`。两分支不各自探索不同动作前缀。

```text
z_q* = 1.5 z_q+ − 0.5 z_q−,   q = 0…5
z_6* = z_6+                     gripper 维度保持干净分支
a_q* = argmax(z_q*)             greedy 解码
```

维度 0…2 为平移，3…5 为旋转，6 为夹爪。生成时抑制 EOS，要求完整七维动作。模型 action bins 和所选 `unnorm_key` 的 q01/q99/mask 用于反归一化。**该输出仍处于模型动作空间**：坐标框架、单位、旋转约定、夹爪映射和执行时长由真机适配器确定。

## 4. 审计和适用边界

逐步 trace 记录预算、所选位置、attention 层、clean/negative/final token、动作变化和 hook 等价性。`policy.infer()` 对关键检查失败即停止。runner 日志中保留图像 SHA256；采集系统应另存原始图像或视频供回查。

本实现依赖 OpenVLA/Prismatic 的 224×224 输入、256 个视觉 token、32 层语言 decoder、七维动作 token，以及 checkpoint 的 processor/tokenizer。改变模型结构或动作 head 时，必须重新核查视觉索引、L11、action bins 和解码规则。真实机器人数据分布、控制延迟与标定不在算法保证内。
