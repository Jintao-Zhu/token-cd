# Prompt–Action Complement v1：失败/收益机制分析

## 结论

四组 selector 确实换入了不同信息；实验差异不是名义分组或覆盖量不一致造成的。

当前最受数据支持的解释是：**换入 token 改变了负分支所提供的动作证据方向；固定正向 contrastive guidance 会在低 margin 动作上频繁翻转，并且这种修正在不同任务阶段不总是有利。** 谐波重建是变化传递的中介，但“重建误差越大，效果越差”不是主要规律。

具体而言：

- `move_near` 中，HH 的主要收益发生在接近、抓取/推动正确物体以及向目标运输阶段。
- `close_drawer` 中，LH 的主要伤害发生在接触后的持续关闭阶段；它常能让抽屉开始移动，但无法把操作持续到成功阈值。
- “Action-high”不能脱离 Prompt 条件理解：HH 补选 token 的平均 Action 全局排名是 14.4，而 LH 是 93.0；两者不是同等强度的 Action-high。

## 1. 闭环结果

每任务 100 个相同 canonical seeds，单位为成功数：

| Task | Original | HH | HL | LH |
|---|---:|---:|---:|---:|
| open_drawer | 50 | 38 | 34 | 39 |
| close_drawer | 82 | 81 | 69 | 56 |
| pick_coke_can | 44 | 39 | 35 | 40 |
| move_near | 62 | 73 | 64 | 62 |
| **Overall** | **238/400** | **231/400** | **202/400** | **197/400** |

两个关键配对对照：

- `move_near`，HH vs Original：Rescue 19，Harm 8，Net +11。
- `close_drawer`，LH vs HH：Rescue 6，Harm 31，Net -25。

## 2. Stage A：四组实际换入了什么

918 个同状态样本；四组使用相同的 `m` 和相同核心，只改变补选 token。

| Arm | 实际替换数 | 与 Original mask Jaccard | 补选 Prompt 全局排名 | 补选 Action 全局排名 | Prompt 分数 | Action 分数 |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0.00 | 1.000 | 30.8 | 46.8 | 2.474e-3 | 1.491e-3 |
| HH | 5.76 | 0.706 | 49.3 | **14.4** | 1.372e-3 | **3.838e-3** |
| HL | 8.15 | 0.622 | 116.6 | 198.8 | 1.675e-4 | 2.684e-6 |
| LH | 8.22 | 0.620 | 163.5 | 93.0 | 4.572e-5 | 8.775e-5 |

说明：

1. 分组确实生效。HH 换入的是全局 Action 排名很高的 token；HL 两个排名都低；LH 的 Prompt 排名最低。
2. HH 与 LH 虽都叫 Action-high，但实际 Action 强度相差很大：平均分约相差 44 倍，平均排名 14.4 vs 93.0。
3. 因而 `close_drawer` 的 HH–LH 差异不能解释成“Prompt 高低是唯一变量”；它是候选池联合约束后的真实组合效应。

## 3. token 与核心的关系及谐波重建

240 个锁定同状态样本，重新计算 projector feature 与相同 harmonic reconstruction。

| Arm | 补选–核心 cosine | 补选内部 cosine | 补选重建误差 | 全选区重建误差 norm |
|---|---:|---:|---:|---:|
| Original | 0.248 | 0.263 | 21.63 | 148.23 |
| HH | 0.165 | **0.234** | **22.92** | 145.31 |
| HL | 0.139 | 0.445 | 18.57 | 138.57 |
| LH | **0.094** | 0.442 | 19.35 | 140.15 |

解释：

- HH 的补选内部最不冗余、且在三个新组中最难由周围 token 重建，说明它确实引入了不同且有影响力的视觉特征。
- HL/LH 的补选内部更相似，重建误差反而更低。
- 但 HH 是三个修改组中闭环最好的一组，而 LH/HL 更差。因此数据不支持“重建误差大就是 Harm”的单调解释。
- cosine 只描述 feature 几何关系，不能直接当作物体语义真值。

## 4. Stage B：负分支如何改变动作

240 个同状态样本，相对 Original：

| Arm | residual norm 比 | residual 方向 cosine | 至少一个动作维度变化 | 平均变化维数 | 平均动作差 |
|---|---:|---:|---:|---:|---:|
| HH | 0.977 | 0.860 | **63.3%** | 1.171 | 0.00938 |
| HL | 0.950 | 0.873 | 59.6% | 1.063 | 0.00755 |
| LH | 0.950 | 0.872 | 60.8% | 1.071 | 0.00772 |

动作被翻转时，clean policy 的前两名 logit margin 平均约 1.49；未翻转时约 7.15。HH 的逐维翻转率为：

`[27.1%, 19.2%, 18.3%, 22.9%, 14.6%, 15.0%]`。

因此主要现象不是 residual 整体变强，而是：

1. residual 方向发生明显变化（cosine 约 0.86–0.87）；
2. clean 本来犹豫的动作维度最容易被翻转；
3. 第 0、3 维最敏感，早期维度又会通过 autoregressive prefix 影响后续维度。

这更支持“引导方向/低 margin 翻转”机制，而不是单一干预强度机制。

## 5. 轨迹阶段归因

轨迹分析从相同 canonical snapshot 回放当时保存的 `executed_actions`，不重新调用模型。

### move_near：HH 相对 Original 的 +11

19 个 Rescue：

| 指标 | Original | HH |
|---|---:|---:|
| 最终推动正确物体 | 63.2% | **100%** |
| 推动错误物体 | 36.8% | **5.3%** |
| 首次正确移动步数（能移动者） | 35.6 | **30.3** |
| TCP 到 source 的最小距离 | 8.9 cm | **3.1 cm** |
| source–target 最小 XY 距离 | 21.6 cm | **15.5 cm** |

8 个 Harm：Original 全部到达 near 条件；HH 中 7/8 曾推动正确物体，但 0/8 到达 near 条件，source–target 最小 XY 距离由 11.3 cm 变为 26.4 cm。

判断：HH 的收益与伤害都不是单纯“是否识别 source”。它在 Rescue 中显著改善接近、正确物体操作和运输；在 Harm 中则给出错误的运输方向或未完成关系约束。

### close_drawer：LH 相对 HH 的 -25

31 个 Harm：

| 指标 | HH | LH |
|---|---:|---:|
| 抽屉推进至少 1 cm | **100%** | 83.9% |
| 推进至少 5 cm | **100%** | 80.6% |
| 推进至少 10 cm | **100%** | 25.8% |
| 曾达到成功阈值 qpos≤0.05 | **100%** | **0%** |
| 最终 qpos（越小越闭合） | **0.0099** | 0.1337 |
| late 阶段被 guidance 改变的维数 | **0.40** | **1.45** |

LH 的第一次 1 cm 推进平均比 HH 晚约 4.6 步；到中后段差距迅速扩大。所有失败轨迹都没有出现“已经关好又重新打开”。

判断：主要伤害不是完成后反弹，也不只是最初目标选择错误；LH 通常在接触/操作阶段无法持续产生有效关闭动作，而 guidance 在后段仍持续修改更多动作维度。

## 6. 三种机制的证据判定

| 假说 | 判定 | 依据 |
|---|---|---|
| 重建问题 | **不是主要单一原因，但参与传递** | HH 的补选重建误差最高却是修改组中闭环最好；误差大小与成功率不单调。 |
| 引导问题 | **强支持** | residual 范数相近而方向改变；约 60% 状态动作变化；翻转集中在低 clean margin；固定正向引导在不同任务可 Rescue 也可 Harm。 |
| 阶段问题 | **强支持** | HH 在 move 的接近/运输阶段获益；LH 在 drawer 的持续关闭阶段受损，且 late guidance 不收敛。 |

综合起来，最合理的机制是：

> Action-aware token 能提供有效的控制相关证据，但必须受 Prompt 相关性与任务阶段约束。固定的 selector 组合加固定正向 λ，会把同一种证据在全过程持续放大；当 clean 动作 margin 很小时，它容易改变 residual 方向并翻转动作，所以同一规则会在一个任务产生 Rescue、在另一个任务产生 Harm。

## 7. 结论边界

- Stage A/B 是同状态诊断，不能把某个失败 episode 中的每次动作都标成有害。
- 闭环轨迹在分叉后不再是同状态比较；这里使用同初态与配对 outcome 做阶段归因，不把后期逐帧差异当作单步因果效应。
- 轨迹回放验证了保存动作造成的物理过程，但未做“只替换某一步动作后统一 continuation”的介入实验。因此阶段结论是强机制证据，不是单步充分因果证明。

## 8. 结果文件

- `FINAL_RESULTS.json`：本次机制分析的机器可读摘要。
- `stage_a_selection_per_state.csv` / `stage_a_selection_summary.json`：逐状态选区、排名、分数与重叠。
- `feature_reconstruction_per_state.csv` / `feature_reconstruction_summary.json`：feature 相似度与谐波重建误差。
- `stage_b_action_per_state.csv` / `stage_b_action_summary.json`：扰动、residual、margin 与动作差异。
- `closed_loop_key_pair_categories.csv` / `closed_loop_key_pair_summary.json`：关键配对 outcome 与全过程诊断汇总。
- `trajectory_phase_per_episode.csv` / `trajectory_phase_summary.json`：关键翻转案例的物理阶段回放统计。

