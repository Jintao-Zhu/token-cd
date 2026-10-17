# token-cd：OpenVLA L11-Matched 对比解码

本仓库研究视觉语言动作模型的视觉 token 反事实引导。目前提供的**真机迁移包**复用我们在仿真实验中的 OpenVLA L11-Matched 方法代码，包含从相机图像到模型动作的推理入口，以及连接具体机器人的适配接口。真机实验尚未完成；现有实验结果均来自仿真。

**阅读入口：**[真机迁移指南](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/README.md) · [方法设计与公式](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/METHOD.md) · [硬件适配接口](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/HARDWARE_ADAPTER.md) · [完整实现分支](https://github.com/Jintao-Zhu/token-cd/tree/real-robot-transfer)

## 方法：一个控制步内发生什么

输入一帧 RGB 图像和任务指令后，OpenVLA-7B 产生 256 个视觉 projector token，并预测一个由七个 token 表示的动作。L11-Matched 在推理时执行以下四步：

1. **确定动态扰动预算。** 从指令中解析源实体和目标实体。对 256 个 projector 特征做 KMeans 分组（`K=8`），按实体短语 embedding 与簇特征的余弦相似度匹配簇。匹配簇的 token 并集大小为当前控制步预算 `m_t`。
2. **定位与任务相关的视觉位置。** 读取第 11 个零基索引 decoder 层（L11）中，指令文本 token 指向视觉 token 的 softmax 后注意力。按注意力排序选出前 `m_t` 个位置。KMeans 决定扰动**数量**，L11 注意力决定扰动**位置**。
3. **构造视觉负分支。** 在 16×16 视觉网格上，根据未选中邻域对所选 projector 特征做调和重建；未选中位置保持逐位相同。由此得到局部视觉信息减弱的反事实分支。
4. **使用相同动作前缀对比解码。** 负分支以干净分支的动作 token 前缀做 teacher-forced 计算。前六维动作使用 `z* = 1.5 z_clean − 0.5 z_negative`；第七维夹爪保留干净分支。最后用 greedy 解码及 checkpoint 中的动作统计得到七维模型动作。

[方法文档](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/METHOD.md)进一步说明注意力索引、调和重建、动作解码和逐步审计字段。核心代码位于 [`prompt_attn_shr_policy.py`](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/research/semantic_token_cd/prompt_attn_shr_policy.py)；真机迁移入口直接复用该实现，避免维护另一份算法副本。

## 从离线图像开始复现

请使用 [`real-robot-transfer` 分支](https://github.com/Jintao-Zhu/token-cd/tree/real-robot-transfer)。Git 仓库不包含 OpenVLA 权重；先准备完整的本地 checkpoint 和兼容 CUDA 的运行环境。文件范围见 [checkpoint 清单](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/TRANSFER_MANIFEST.md)，原实验机版本见 [运行环境记录](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/SOURCE_RUNTIME.md)。真机推理入口不依赖 SIMPLER、SAPIEN 或仿真渲染器。

先在仓库根目录用一张保存的相机图片运行离线推理。下面的命令只记录预测，不向机器人发送指令：

```bash
python -m deployment.real_robot_transfer.runner \
  --checkpoint /models/openvla-7b \
  --unnorm-key YOUR_ROBOT_STATS_KEY \
  --instruction "put the carrot on the plate" \
  --image /data/camera_frame.png \
  --log /data/l11_matched_offline.jsonl
```

`--unnorm-key` 必须对应实际机器人的动作数据分布。原 checkpoint 含 `bridge_orig`、`fractal20220817_data` 等统计，但不能据此推断它们适合另一台机械臂。JSONL 记录图像哈希、干净与引导后的动作 token、选中的视觉位置、`m_t`、解码动作及干预审计结果。

接入相机和机器人时，按[适配接口](https://github.com/Jintao-Zhu/token-cd/blob/real-robot-transfer/deployment/real_robot_transfer/adapter.py)将模型动作映射到控制器的坐标系、单位、夹爪定义及限幅规则。使用 `--adapter your_package.robot_adapter:RobotAdapter` 时默认只记录；只有显式加上 `--execute` 才会调用适配器的 `send_command()`。

## 真机复现还需要什么

该分支固定了 L11-Matched 算法配置、图像预处理路径、动作解码和 trace 核查。真实轨迹还需要与机器人匹配的 checkpoint／动作统计、相机与机械臂标定、设备驱动适配、控制频率、安全限幅，以及真机任务结果记录。计划使用的机械臂、相机和控制接口目前尚未确定，因此这里不报告真机成功率，也不把模型的七维动作直接当作硬件控制命令。

## 仓库中的其他研究

`research/semantic_token_cd/` 保存主要的 token 级对比解码实验；`research/ar_*`、`research/coreact_*`、`research/token_pcd_*` 和 `research/causal_failure_gate/` 保存相关的动作 token、激活、PCD 和失败门控研究。此前的[SHR 方法与结果报告](docs/SHR_COMPREHENSIVE_REPORT_2026-09-05.md)及[精简实验数据](github_results/SHR_FOLLOWUP_RELEASE.md)仍可查阅。大型 checkpoint 和原始 rollout 数据未放入 Git。
