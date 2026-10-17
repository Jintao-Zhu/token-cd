# OpenVLA VTCG：真机迁移交接

本目录提供 **相机图像 → OpenVLA → VTCG → 真机适配器** 的运行入口。算法复用 `research/semantic_token_cd/prompt_attn_shr_policy.py`，与仿真实验是同一实现。现有结果来自仿真；这里尚无特定机械臂的驱动、标定文件或真机结果。接手者可先在真实相机图像上离线推理，再实现其设备的观察和执行接口。

## 方法概览

每个控制步，OpenVLA 产生七维动作的干净分支及 256 个视觉 projector 特征。规则实体解析和 KMeans K=8 确定当前图像的动态预算 `m_t`。从第 11 个零基层的任务文字到视觉 token 注意力中选出前 `m_t` 个位置。对这些位置在 16×16 网格上做调和重建，构成局部视觉信息减弱的负分支。负分支使用干净动作的相同 token 前缀。前六维动作 logits 执行 `1.5 × clean − 0.5 × negative`；夹爪维度保留干净分支，最后 greedy 解码。完整定义与审计见 [METHOD.md](METHOD.md)。

## 关键文件

| 路径 | 作用 |
|---|---|
| `research/semantic_token_cd/prompt_attn_shr_policy.py` | L11、Matched 预算、负分支、logit 对比 |
| `research/semantic_token_cd/st_shr_policy.py` | 调和重建 |
| `research/semantic_token_cd/global_merge_policy.py` | projector 替换及同前缀负分支 |
| `deployment/real_robot_transfer/openvla_runtime.py` | 不依赖 SIMPLER 的模型加载、图像预处理、动作反归一化 |
| `deployment/real_robot_transfer/policy.py` | 固定实验配置与逐步审计 |
| `deployment/real_robot_transfer/runner.py` | 单图像推理、真机 dry-run、显式执行 |
| `deployment/real_robot_transfer/adapter.py` | 真机输入输出接口 |

## 使用顺序

1. 克隆 `real-robot-transfer` 分支，准备完整的 OpenVLA-7B checkpoint，安装 Python 3.10、适合本机 CUDA 的 PyTorch/torchvision，以及 [requirements.txt](requirements.txt) 中的其余包。checkpoint 不在 Git；复制与校验方法见 [TRANSFER_MANIFEST.md](TRANSFER_MANIFEST.md)。运行命令均在仓库根目录执行。推理入口不需要 SAPIEN、SIMPLER 或渲染器。
2. 选择对应实际机器人 embodiment 的 `--unnorm-key`。checkpoint 内含 `bridge_orig` 与 `fractal20220817_data`，但这些统计不自动适配陌生机械臂。新微调 checkpoint 需提供七维 q01/q99/mask 统计。
3. 先保存一帧真实相机图像，运行单图像入口；它只输出动作与诊断，不发送控制命令。

```bash
python -m deployment.real_robot_transfer.runner \
  --checkpoint /models/openvla-7b \
  --unnorm-key bridge_orig \
  --instruction "put the carrot on the plate" \
  --image /data/camera_frame.png \
  --log /data/vtcg_offline.jsonl
```

这里 `bridge_orig` 仅说明参数格式。实际实验须核对 embodiment。输入从文件解码成 RGB，再用 OpenCV `INTER_AREA` 缩放到 224×224，并交由 checkpoint 自带 processor。英语指令需采用规则实体提取器可识别的命令形式。JSONL 包含图像哈希、七维反归一化动作、预算、所选 token、各分支 token 及审计字段。

4. 按 [HARDWARE_ADAPTER.md](HARDWARE_ADAPTER.md) 实现相机采集和动作映射，可从 [example_adapter.py](example_adapter.py) 起步。先做只记录不执行的 dry-run：

```bash
python -m deployment.real_robot_transfer.runner \
  --checkpoint /models/openvla-7b \
  --unnorm-key YOUR_ROBOT_STATS_KEY \
  --instruction "put the carrot on the plate" \
  --adapter your_package.robot_adapter:RobotAdapter \
  --steps 20 --interval-s 1.0 \
  --log /data/vtcg_dry_run.jsonl
```

5. 核对图像、状态、七维动作和经限幅的命令，并确认坐标系、单位、夹爪方向、延迟和急停。只有在设备侧完成这些准备后，增加 `--execute` 才会调用适配器的 `send_command()`。每次发送前先记录命令提议，发送后记录发送事件。现场应额外保存原始视频、机器人反馈、任务结果和失败原因。

## 可复现范围

同一 checkpoint、processor、运行库、指令和图像下，本入口固定仿真中的算法设置并输出可核对的 trace。真实轨迹还取决于相机标定、动作统计、机械臂接口、频率、场景与安全流程，这些必须由真机合作者另行锁定。建议做 Vanilla OpenVLA 与 VTCG 同场景对照，记录逐步日志和成功率。现有仿真成功率不能作为真机效果。
