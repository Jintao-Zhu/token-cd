# WidowX + pi0-FAST：输入约定才是病根（2026-10-13）

## 结论

`checkpoints/openpi_fast_finetuned/bridge`（BridgeData V2 微调的 pi0-FAST）**可以驱动 SIMPLER
WidowX 四个任务**。此前"163 集零交互、`moved_wrong_obj=False` 100%"（`KNOWN_LIMITATIONS.md` §5）
不是该 checkpoint 的能力上限，而是**喂给它的 state 语义错了 + 动作没有裁剪**。

## 证据

同一 checkpoint、同一 server、同一环境，只改客户端输入约定：

| 条件 | state 布局 | 动作裁剪 | 集数 | 成功 | 有交互（`moved_correct_obj=True`） |
|---|---|---:|---:|---:|---:|
| 条件 1（`runs/widowx_bridge_normfix`） | `[pos3, quat4, 1−gripper]`（Fractal/LIBERO 约定） | 无 | 24 | 0 | **0/24** |
| 条件 2（`runs/widowx_bridge_layout2`） | `[pos3, euler_rpy(R_topdown 共轭), PAD(0), gripper]` | q01/q99 裁剪 | 7 | 2 | **4/7** |
| 参照：INTACT π0 flow-matching（已知可用） | 作者参考实现 | 有 | 16 | 8 | 9/16 |

条件 2 的交互率（57%）已与 INTACT 参照（56%）持平。

## 布局是怎么定出来的

不靠猜，直接读 checkpoint 自带的 `assets/local/bridge_lerobot/norm_stats.json`：

| dim | ckpt q01/q99 | 对照（INTACT Bridge 官方统计） | 判定 |
|---|---|---|---|
| 0–2 | 0.1710/0.4532 … | 0.1702/0.4527 … | 位置 xyz |
| 3–5 | −0.3649/0.3780、−0.5419/0.2757、−1.3543/1.8501 | −0.3656/0.3788、−0.5435/0.2772、−1.3501/1.8378 | **euler rpy**（`R_topdown` 共轭后） |
| **6** | **std=0，q01=q99=0** | — | **常量 padding** |
| 7 | 0.0522/1.0106 | Bridge gripper 0.0522/1.0106 | gripper |

→ 训练布局是 `[x, y, z, roll, pitch, yaw, 0, gripper]`；旧代码喂的是 `[pos, quat_xyzw, 1−gripper]`：
**维度巧合都是 8，语义全错**。

## 代码

`token-cd/research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py`：

* `PI0FAST_WIDOWX_LAYOUT=bridge` 时，widowx 任务走 `_bridge_state()`：
  `mat2euler(quat2mat(eef[3:7]) @ R_topdown.T)` + `pad=0` + 原始 gripper（`R_topdown = [[0,0,1],[0,1,0],[-1,0,0]]`，
  与 INTACT `BridgeSimplerAdapter` 一致；用 `transforms3d` 保证 `sxyz` 约定一致）
* 动作前 6 维按训练 `q01/q99` 裁剪（INTACT 在归一化空间裁 ±1 的等价形式）
* 默认（不设环境变量）= 旧行为，保证历史结果可复现
* 只对 widowx 生效，google_robot 路径不变

运行：`bash scripts/run_widowx_bridge_layout2.sh 1 8121 0-3`

## 另外两个环境坑（都会让 widowx 复现失败）

1. **PYTHONPATH 覆盖 `mani_skill2_real2sim`**：把 `SimplerEnv` / `SpatialVLA-SimplerEnv` 加进
   PYTHONPATH 会换成另一个 fork，其 WidowX `obs["agent"]` **没有 `eef_pos`**
   （实测：仅 `token-cd` → 有；加那两个 → 没有）。INTACT 参考实现因此报 `KeyError: 'eef_pos'`。
   **跑 widowx 的 INTACT/参考路径时，PYTHONPATH 保持最小。**
2. **INTACT 运行时的 cuDNN SDPA 崩溃**：首次 SigLIP 注意力报
   `cuDNN Frontend error: No execution plans support the graph`；
   需 `torch.backends.cuda.enable_cudnn_sdp(False)`。

## 仍未做的（可继续）

* 条件 2 样本仍小（8/16），跑满后应有更稳的估计；
* **pi0-FAST 的 WidowX λ 扫描从未跑过**（9 任务主实验里 WidowX 被排除）——现在输入约定修好了，
  可以真正开跑；
* 注意：已在跑的 SpatialVLA WidowX λ 扫描（`spatialvla_widowx_lambda_*`）**不受本问题影响**，
  它走的是 SpatialVLA 自己的适配器（已被验证能驱动这四个任务）；
* `R_topdown` 共轭与动作裁剪对最终数值的贡献尚未分开消融（可做 2×2 消融）。
