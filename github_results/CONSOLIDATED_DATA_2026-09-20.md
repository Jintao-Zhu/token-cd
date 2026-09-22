# 现有实验数据汇总

## SOURCE: docs/2026-09-18_EXPERIMENT_LOG.md

# 9/17–9/18 实验记录

**主题**：L11-Matched 的机制探因与改进尝试
**基准**：matched = 213/400 (53.2%)（4 任务 × 100 seeds，seeds 100–199）
**备注**：以下所有 p 值均为同-seed 配对 McNemar 精确检验

---

## 实验 1：去掉 Top-p 的 [16,64] 数量限制

**初衷**：matched 没有数量上下限，top-p 被夹在 [16,64]。怀疑这个"夹子"让 top-p 吃了亏，想去掉后重比。

**结果**（4 任务 × 100 seeds）

| 任务 | Matched | 夹后 TopP85 | 无夹 TopP85 | 无夹 vs 夹后 |
|---|---|---|---|---|
| open_drawer | 44 | 38 | 38 | 0 |
| close_drawer | 76 | 73 | 73 | 0 |
| pick_coke_can | 33 | 35 | 35 | 0 |
| move_near | 60 | 60 | **40** | **−20** |
| **总体** | 213/400 | 206/400 | **186/400** | **−20 (p=0.002)** |

**关键反证**：在"上下限都从未触发"的 314 个 seed 上（clip 完全无关），两臂本应完全相同，实际却有 **−19** 的差异（3 胜 / 22 负）。

**结论**：❌ **实验无效，clip 与结果无关。**
差异来自数值敏感性 —— `m` 由"累计注意力过阈值"取整得到，**浮点差 1e−7 即可让 m 差 1**，闭环放大后改变结局。
副作用：发现 move_near 是数值最敏感的任务（95/100 的 seed 会分叉），pick_coke_can 最稳（0/100）。

---

## 实验 2：条件缩放（m<16 时乘 1.5 / 2.0 / 2.5）

**初衷**：matched 有 15% 的步 `m<16`，而固定值扫描显示 `m=16` 只有 37.8%。**怀疑"m 太小"是 matched 的短板，抬起来能修。**

**结果**（4 任务 × 100 seeds；只改动 <16 的步，实测平均 +4.4 / +8.6 / +13.1 个 token）

| 臂 | Matched | 该臂 | 净 | 救/害 | p |
|---|---|---|---|---|---|
| ×1.5 | 213/400 | 209/400 | −4 | 38/42 | 0.738 |
| ×2.0 | 213/400 | 204/400 | −9 | 38/47 | 0.386 |
| ×2.5 | 213/400 | 196/400 | −17 | 38/55 | 0.097 |

**逐任务净值**

| 任务 | ×1.5 | ×2.0 | ×2.5 |
|---|---|---|---|
| open_drawer | −2 | −9 | −2 |
| close_drawer | −6 | −6 | −1 |
| **pick_coke_can**（m 最小的任务） | **+4** | **+3** | −4 |
| move_near | 0 | +3 | **−10** |

**结论**：❌ **假设不成立。** 随倍数单调恶化；唯一正收益在 m 最小的任务上且不显著。
**matched 的小值不是毛病，是它本来就该那么小。**

---

## 实验 3：L14 单层闭环

**初衷**：项目此前通过离线诊断把 L11 定为最优单层，但**L14 单独从未跑过闭环**。要独立复核选层结论。

**结果**（4 任务 × seeds 0–99，与已有的 L11 / L11+L14 同 seed）

| 任务 | L11（基准） | L14 | L14 净 | p | L11+L14 | 净值 |
|---|---|---|---|---|---|---|
| open_drawer | 50 | 45 | −5 | 0.511 | 36 | −14 |
| close_drawer | 82 | 74 | −8 | 0.115 | 72 | −10 |
| pick_coke_can | 44 | 38 | −6 | 0.418 | 38 | −6 |
| move_near | 62 | 57 | −5 | 0.405 | 62 | 0 |
| **总体** | **238/400** | **214/400** | **−24** | | **208/400** | **−30** |

| 总体配对 | 净 | p |
|---|---|---|
| L14 vs L11 | **−24** | **0.034** |
| L11+L14 vs L11 | **−30** | **0.006** |

**对照公平性**：三臂的平均 mask 数量差 < 1.6 个 token（差异来自闭环分叉，非规则不同）。

**结论**：✅ **选层结论被独立确认。** L11 确实是最优单层，L14 显著更差。

---

## 实验 4：matched ∪ 空间均匀 16 token

**初衷**：matched 的 mask 与实体簇只重叠 47%，说明它"没覆盖物体"。**若在 matched 之外再空间均匀补 16 个 token，能否补上遗漏？**
（16 个 token 取自全局 16×16 网格的 4×4 步长，**可能与 matched 已选的重叠**，不是额外 16 个）

**结果**（9 任务 × 100 seeds）

| 任务 | Matched | +uniform16 | 净 | p |
|---|---|---|---|---|
| open_drawer | 44 | 39 | −5 | 0.442 |
| **close_drawer** | 76 | 66 | **−10** | **0.041** |
| pick_coke_can | 33 | 35 | +2 | 0.832 |
| move_near | 62 | 58 | −4 | 0.557 |
| place_apple | 0 | 0 | 0 | 1.000 |
| carrot_on_plate | 0 | 3 | +3 | 0.250 |
| put_eggplant | 3 | 1 | −2 | 0.500 |
| spoon_on_towel | 2 | 1 | −1 | 1.000 |
| **stack_cube** | 5 | 0 | **−5** | 0.062 |
| **总体** | **225/900** | **203/900** | **−22** | **0.043** |

mask 从平均 29.5 → 44.4 个 token。

**结论**：❌ **有害。** 额外加的 token 不是 attention 选出来的 → 帮不上忙。
**说明"擦哪些"很重要，不能随便加。**

---

## 实验 5：空间后处理（四角置零 / 高斯平滑 / 两者）

**初衷**：matched 的 mask 平均有 5.75 个连通块（随机 24.9）。**若让 mask 更连片（更"像一个物体"），能否改善？**
（项目此前离线诊断过：Gaussian 让连通块 6.92→3.44，但 target response 掉 23%，判定 No-Go。本实验用闭环验证该判断。）

**结果**（4 任务 × 100 seeds；**数量 m 完全不变**，只改排序）

| 任务 | 臂 | Matched | 该臂 | 净 | p |
|---|---|---|---|---|---|
| open_drawer | corners | 44 | 38 | −6 | 0.362 |
| open_drawer | gaussian | 44 | 36 | −8 | 0.185 |
| **open_drawer** | **both** | 44 | **26** | **−18** | **0.001** |
| close_drawer | corners | 76 | 69 | −7 | 0.092 |
| close_drawer | gaussian | 76 | 66 | −10 | 0.087 |
| close_drawer | both | 76 | 68 | −8 | 0.134 |
| pick_coke_can | corners | 33 | 34 | +1 | 1.000 |
| pick_coke_can | gaussian | 33 | 37 | +4 | 0.541 |
| pick_coke_can | both | 33 | 35 | +2 | 0.824 |
| **move_near** | **corners** | 60 | **73** | **+13** | **0.015** |
| move_near | gaussian | 60 | 52 | −8 | 0.185 |
| move_near | both | 60 | 68 | +8 | 0.134 |

| 总体 | Matched | 该臂 | 净 | p |
|---|---|---|---|---|
| corners | 213/400 | 214/400 | +1 | 1.000 |
| **gaussian** | 213/400 | 191/400 | **−22** | **0.043** |
| both | 213/400 | 197/400 | −16 | 0.117 |

连通块：5.76 → 3.00。闭环内平均替换 8.35（corners）/ 14.08（gaussian）个 token。

**结论**：❌ **空间平滑有害，离线 No-Go 判断正确。**
**"让 mask 更连片" ≠ "让 mask 更好"。L11 排序的逐-patch 锐度本身就是信号。**
corners 总体中性，但 move_near +13（该任务数值最敏感，需谨慎）。

---

## 实验 6：指定 patch 位置消融（进行中 156/1200）

**初衷**：survey 发现 patch (0,1) 和 (0,13) 在**四个任务里注意力都异常高**（29~50× 中位数，被擦 29~83%）。

| patch | 位置 | open_drawer | close_drawer | pick_coke_can | move_near |
|---|---|---|---|---|---|
| patch 0 | (0,0) 左上角 | 14.3x / 37% | 7.2x / 40% | 7.6x / 6% | 1.7x / 5% |
| patch 1 | (0,1) | 9.6x / 47% | 43.5x / 47% | 50.4x / 69% | 29.4x / 77% |
| patch 13 | (0,13) | 38.1x / 83% | 45.0x / 75% | 21.5x / 72% | 20.7x / 56% |
| patch 15 | (0,15) | 0.7x / 0% | 0.2x / 0% | 3.2x / 7% | 2.3x / 3% |

**假说**：这些位置是"注意力陷阱"（模型走神时的默认去处），擦不擦无所谓 → 若成立，说明 matched 有 30~80% 的步把名额浪费了。

**中期结果**（open_drawer，47 个三臂齐全的 seed）

| 臂 | Matched | 该臂 | 净 | 救/害 | p |
|---|---|---|---|---|---|
| off_p0 | 22/47 | 15/47 | −7 | 5/12 | 0.143 |
| off_p1_p13 | 22/47 | 18/47 | −4 | 5/9 | 0.424 |
| off_p0_p1_p13 | 22/47 | 14/47 | −8 | 2/10 | **0.039** |

**初步结论**：❌ **不是陷阱，是真信号。** 去掉它们反而变差。
（与 corners 实验一致：三个独立实验都指向"把高注意力位置拿掉 = 变差"。）

---

## 汇总：六个实验的结论

| # | 实验 | 改动 | 结果 | 结论 |
|---|---|---|---|---|
| 1 | 无夹 TopP85 | 去掉数量上下限 | −20（但无效） | clip 无关；数值敏感性才是原因 |
| 2 | 条件缩放 | m<16 时 ×1.5/2/2.5 | −4 / −9 / −17 | ❌ 小值不是毛病 |
| 3 | L14 单层 | 换注意力层 | −24 (p=0.034) | ✅ L11 确实最优 |
| 4 | +均匀16 | 额外加非 attention token | −22 (p=0.043) | ❌ 擦哪些很重要 |
| 5 | 空间后处理 | 平滑 / 去角 | −22 (gaussian) | ❌ 连片 ≠ 更好 |
| 6 | patch 消融 | 去掉热点 patch | −8（初步） | ❌ 热点是真信号 |

**统一规律**：

> **matched（L11 + attention 排序 + 自身 KMeans 预算）已经在局部最优。**
> **换层、改数量、加 token、平滑排序、去掉热点 —— 五个方向全部变差或打平。**
> **唯一有效的方向是"不要动它"。**

## SOURCE: artifacts/l11_budget_response_curve_v1/PHASE_A_REPORT.md

# Phase A: Matched 预算 vs 有效反事实预算

状态总数: **800**，其中有明确 knee 的: **695**

## 总相关（pooled）

- Spearman(ρ) = **0.042** (p = 0.2651), n = 695

## episode 内中心化（真正的机制检验）

- Spearman(Δmatched, Δknee) = **0.081** (p = 0.0336)
- episode-cluster bootstrap(5000): mean = 0.082, 95% CI = [-0.026, 0.188]

## 逐任务

| 任务 | n | Spearman(matched, knee) | mean |matched−knee| |
|---|---:|---:|---:|
| google_robot_open_drawer | 166 | 0.267 | 16.21 |
| google_robot_close_drawer | 171 | -0.225 | 19.50 |
| google_robot_pick_coke_can | 175 | 0.128 | 12.78 |
| google_robot_move_near | 183 | 0.002 | 19.90 |

## 方法对比：|m_method − m_knee|

| 方法 | mean | median | Spearman | |
|---|---:|---:|---:|---|
| matched | 17.13 | 16.00 | 0.042 | |
| TopP80 | 13.68 | 12.00 | 0.111 | |
| TopP85 | 18.31 | 17.00 | 0.088 | |
| fixed32 | 11.89 | 15.00 | nan | |
| m80(robust) | 20.90 | 16.00 | -0.067 | |

## SOURCE: artifacts/l11_matched_state_coupling_v1/STATE_COUPLING_REPORT.md

# Matched 状态耦合消融（episode 内打乱预算 tape）

可比配对 episode 数: **400**

| 任务 | n | Matched | Shuffle | Harm | Rescue | 净 | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| open_drawer | 100 | 44/100 (44.0%) | 35/100 (35.0%) | 22 | 13 | +9 | 0.1755 |
| close_drawer | 100 | 76/100 (76.0%) | 70/100 (70.0%) | 19 | 13 | +6 | 0.3771 |
| pick_coke_can | 100 | 33/100 (33.0%) | 35/100 (35.0%) | 16 | 18 | -2 | 0.8642 |
| move_near | 100 | 60/100 (60.0%) | 58/100 (58.0%) | 16 | 14 | +2 | 0.8555 |
| **总体** | **400** | **213/400 (53.2%)** | **198/400 (49.5%)** | **73** | **58** | **+15** | **0.2211** |

## 操纵检查

- 每条 episode 的 |tape − 当前状态| 平均 = **16.02**，范围 [2.2, 37.1]
- tape 的 m 均值 = 33.84；当前状态自己算的 m 均值 = 34.43

- 超长步数（需扩展池）: 0 条 episode

## SOURCE: artifacts/l11_entity_budget_calibration_v1/CALIBRATION_REPORT.md

# Entity-Conditioned Budget Calibration

States: **800** (4 tasks × 50 seeds × 4 progress states)

Calibration target: mean matched m = **34.024**

| Estimator | tau | mean m | std | median | P10 | P90 | rho vs matched |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.7855 | 34.03 | 9.21 | 33.0 | 23 | 47 | 0.009 |
| entity | 0.7830 | 34.05 | 8.67 | 34.0 | 23 | 45 | -0.041 |
| generic | 0.7240 | 34.02 | 10.37 | 34.0 | 20 | 47 | -0.012 |

## Per-task Spearman

| Task | full | entity | generic |
|---|---:|---:|---:|
| open_drawer | -0.044 | 0.003 | 0.043 |
| close_drawer | -0.172 | -0.193 | -0.174 |
| pick_coke_can | 0.049 | 0.044 | -0.036 |
| move_near | -0.114 | -0.121 | -0.046 |

## Within-episode centered Spearman

| Estimator | rho | p |
|---|---:|---:|
| full | -0.090 | 0.0105 |
| entity | -0.094 | 0.0075 |
| generic | -0.080 | 0.0236 |

## SOURCE: artifacts/l11_entity_budget_closed_loop_100_199_v1/CLOSED_LOOP_REPORT.md

# Entity / Generic Top-P budget closed loop

Matched is the shared paired baseline. Lower p values are exact McNemar tests.

## Entity_top_p

| Task | Matched | Arm | Rescue | Harm | Net | McNemar p | Matched m | Arm m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **OVERALL** | 213/400 | 198/400 | 50 | 65 | -15 | 0.1915 | 33.84 | 33.05 |
| open_drawer | 44/100 | 31/100 | 11 | 24 | -13 | 0.0410 | 35.19 | 31.06 |
| close_drawer | 76/100 | 74/100 | 9 | 11 | -2 | 0.8238 | 32.66 | 33.10 |
| pick_coke_can | 33/100 | 38/100 | 18 | 13 | +5 | 0.4731 | 25.54 | 33.71 |
| move_near | 60/100 | 55/100 | 12 | 17 | -5 | 0.4583 | 41.98 | 34.32 |

## SOURCE: artifacts/l11_random_mask_matched_count_100_199_v1/CLOSED_LOOP_REPORT.md

# Random-mask / Matched-count closed loop

Matched is the shared paired baseline. Lower p values are exact McNemar tests.

## Random_mask

| Task | Matched | Arm | Rescue | Harm | Net | McNemar p | Matched m | Arm m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **OVERALL** | 213/400 | 141/400 | 28 | 100 | -72 | 0.0000 | 33.84 | 33.54 |
| open_drawer | 44/100 | 19/100 | 4 | 29 | -25 | 0.0000 | 35.19 | 36.45 |
| close_drawer | 76/100 | 43/100 | 5 | 38 | -33 | 0.0000 | 32.66 | 31.15 |
| pick_coke_can | 33/100 | 28/100 | 11 | 16 | -5 | 0.4421 | 25.54 | 24.47 |
| move_near | 60/100 | 51/100 | 8 | 17 | -9 | 0.1078 | 41.98 | 42.09 |

## SOURCE: artifacts/matched_dynamic_budget_analysis_v1/ANALYSIS_REPORT.md

# Matched dynamic-budget mechanism analysis

All analyses reuse existing states/episodes. No new rollout episodes.

## Phase 1: What distinguishes small-m and large-m states?

Spearman rho between matched m and each state metric. Computed per task.

| Metric | open | close | pick | move |
|---|---:|---:|---:|---:|
| bbox_area | 0.831 | 0.899 | 0.804 | 0.871 |
| mean_spatial_distance | 0.856 | 0.861 | 0.677 | 0.780 |
| components | -0.360 | -0.251 | 0.457 | 0.691 |
| density | 0.321 | 0.205 | -0.308 | -0.466 |
| cluster_intra_cosine_distance | 0.551 | 0.775 | 0.398 | 0.183 |
| cluster_separation_margin | 0.025 | 0.368 | -0.095 | 0.213 |
| semantic_centroid_cosine | -0.499 | -0.493 | -0.328 | -0.103 |
| semantic_margin | 0.192 | 0.341 | -0.210 | -0.308 |
| cluster_size_entropy | 0.249 | 0.249 | 0.461 | 0.108 |
| matched_cluster_size_rank_mean | -0.898 | -0.940 | -0.924 | -0.705 |

### Small / Middle / Large states: metric means

Within each task, Small = lowest 25% m, Large = highest 25% m.

#### bbox_area

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 26.180 | 67.130 | 88.600 |
| close_drawer | 26.460 | 59.120 | 83.680 |
| pick_coke_can | 38.440 | 92.120 | 211.800 |
| move_near | 53.580 | 122.870 | 216.160 |

#### mean_spatial_distance

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 1.847 | 2.618 | 2.886 |
| close_drawer | 1.932 | 2.494 | 2.905 |
| pick_coke_can | 2.172 | 3.474 | 5.496 |
| move_near | 2.346 | 3.872 | 5.098 |

#### components

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 2.520 | 1.500 | 1.520 |
| close_drawer | 1.880 | 1.690 | 1.420 |
| pick_coke_can | 2.020 | 3.440 | 7.140 |
| move_near | 1.820 | 3.180 | 4.960 |

#### density

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 0.476 | 0.627 | 0.604 |
| close_drawer | 0.532 | 0.625 | 0.632 |
| pick_coke_can | 0.463 | 0.399 | 0.276 |
| move_near | 0.477 | 0.371 | 0.316 |

#### cluster_intra_cosine_distance

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 0.166 | 0.329 | 0.335 |
| close_drawer | 0.178 | 0.270 | 0.328 |
| pick_coke_can | 0.214 | 0.302 | 0.296 |
| move_near | 0.260 | 0.295 | 0.297 |

#### cluster_separation_margin

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 0.117 | 0.125 | 0.124 |
| close_drawer | 0.086 | 0.116 | 0.159 |
| pick_coke_can | 0.118 | 0.096 | 0.088 |
| move_near | 0.072 | 0.124 | 0.103 |

#### semantic_centroid_cosine

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 0.079 | 0.070 | 0.066 |
| close_drawer | 0.078 | 0.073 | 0.066 |
| pick_coke_can | 0.038 | 0.037 | 0.032 |
| move_near | 0.050 | 0.055 | 0.046 |

#### semantic_margin

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 0.016 | 0.023 | 0.021 |
| close_drawer | 0.015 | 0.020 | 0.023 |
| pick_coke_can | 0.007 | 0.007 | 0.004 |
| move_near | 0.023 | 0.021 | 0.015 |

#### cluster_size_entropy

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 1.948 | 2.011 | 1.999 |
| close_drawer | 1.956 | 2.001 | 1.990 |
| pick_coke_can | 1.912 | 1.952 | 1.976 |
| move_near | 1.955 | 1.972 | 1.975 |

#### matched_cluster_size_rank_mean

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 7.480 | 2.650 | 1.440 |
| close_drawer | 7.340 | 3.800 | 1.580 |
| pick_coke_can | 7.680 | 5.530 | 2.000 |
| move_near | 7.240 | 5.910 | 4.340 |

## Phase 2: Does matched m cover the KMeans task-related support?

For each state, `G` is the matched KMeans token set and `S_k` is full-L11 Top-k.

| Method | Recall | Precision |
|---|---:|---:|
| matched | 0.378 | 0.378 |
| fixed34 | 0.415 | 0.381 |
| entity_top_p | 0.414 | 0.378 |
| shuffle | 0.391 | 0.370 |

### By matched-m quartile

| Task | Size | Matched recall | Fixed34 recall | Entity recall | Shuffle recall |
|---|---:|---:|---:|---:|---:|
| open_drawer | small | 0.315 | 0.605 | 0.570 | 0.539 |
| open_drawer | middle | 0.585 | 0.501 | 0.489 | 0.501 |
| open_drawer | large | 0.595 | 0.435 | 0.404 | 0.504 |
| close_drawer | small | 0.356 | 0.661 | 0.657 | 0.514 |
| close_drawer | middle | 0.396 | 0.405 | 0.474 | 0.378 |
| close_drawer | large | 0.509 | 0.346 | 0.329 | 0.422 |
| pick_coke_can | small | 0.191 | 0.363 | 0.357 | 0.239 |
| pick_coke_can | middle | 0.178 | 0.237 | 0.229 | 0.170 |
| pick_coke_can | large | 0.077 | 0.067 | 0.062 | 0.057 |
| move_near | small | 0.440 | 0.615 | 0.643 | 0.560 |
| move_near | middle | 0.522 | 0.496 | 0.506 | 0.509 |
| move_near | large | 0.368 | 0.258 | 0.262 | 0.304 |

## Phase 3: Does shuffle harm grow with budget mismatch?

`D = mean |m_shuffle - m_matched| / m_matched`. Episode-level matched-vs-shuffle outcomes.

| D bin | n | Matched | Shuffle | Matched-only | Shuffle-only | Net (matched-only − shuffle-only) |
|---|---:|---:|---:|---:|---:|---:|
| D < 20% | 11 | 4/11 | 3/11 | 2 | 1 | +1 |
| 20–40% | 126 | 74/126 | 57/126 | 25 | 8 | +17 |
| D ≥ 40% | 263 | 135/263 | 138/263 | 46 | 49 | -3 |

### Under- vs over-budget direction

| Direction | n | Matched-only | Shuffle-only | Net |
|---|---:|---:|---:|---:|
| under (< −10%) | 122 | 24 | 18 | +6 |
| balanced | 186 | 29 | 20 | +9 |
| over (> +10%) | 92 | 20 | 20 | +0 |

## SOURCE: artifacts/intervention_dose_response_v1/DOSE_REPORT.md

# Intervention-dose response analysis

800 states, each scanned with 5 relative budgets plus Fixed34 and Entity-TopP.

Metrics: `D_feat` visual feature change, `D_action` action JS divergence, `D_support` clean-action log-probability drop.

## Matched-relative response

| r = m/m0 | Small | Middle | Large |
|---|---:|---:|---:|
### D_feat

| r | Small | Middle | Large |
|---|---:|---:|---:|
| 0.5 | 0.1483 | 0.2190 | 0.2650 |
| 0.75 | 0.1821 | 0.2700 | 0.3301 |
| 1 | 0.2116 | 0.3157 | 0.3857 |
| 1.25 | 0.2374 | 0.3572 | 0.4352 |
| 1.5 | 0.2616 | 0.3946 | 0.4803 |

### D_action

| r | Small | Middle | Large |
|---|---:|---:|---:|
| 0.5 | 0.2531 | 0.3195 | 0.3979 |
| 0.75 | 0.2946 | 0.3492 | 0.4134 |
| 1 | 0.3212 | 0.3715 | 0.4219 |
| 1.25 | 0.3419 | 0.3770 | 0.4251 |
| 1.5 | 0.3522 | 0.3798 | 0.4309 |

### D_support

| r | Small | Middle | Large |
|---|---:|---:|---:|
| 0.5 | 2.6250 | 3.8503 | 5.7227 |
| 0.75 | 3.2826 | 4.6039 | 6.4919 |
| 1 | 3.8267 | 5.1860 | 6.7769 |
| 1.25 | 4.3059 | 5.5755 | 6.8940 |
| 1.5 | 4.6461 | 5.8166 | 7.0887 |

## Fixed34 / Entity vs Matched scale

| Budget | Metric | Overall mean | Overall std | Small mean | Middle mean | Large mean |
|---|---:|---:|---:|---:|---:|---:|
| Matched r=1 | D_feat | 0.3044 | 0.0857 | 0.2116 | 0.3157 | 0.3857 |
| Matched r=1 | D_action | 0.3716 | 0.2693 | 0.3212 | 0.3715 | 0.4219 |
| Matched r=1 | D_support | 5.2650 | 4.5949 | 3.8267 | 5.1860 | 6.7769 |
| Fixed34 | D_feat | 0.3152 | 0.0270 | 0.3234 | 0.3164 | 0.3058 |
| Fixed34 | D_action | 0.3851 | 0.2732 | 0.3774 | 0.3712 | 0.4067 |
| Fixed34 | D_support | 5.6609 | 4.6715 | 5.4782 | 5.2762 | 6.2277 |
| Entity-TopP | D_feat | 0.3125 | 0.0472 | 0.3216 | 0.3202 | 0.2959 |
| Entity-TopP | D_action | 0.3858 | 0.2743 | 0.3829 | 0.3726 | 0.4018 |
| Entity-TopP | D_support | 5.6512 | 4.6248 | 5.5252 | 5.3755 | 6.0524 |

## r=1 by task and matched-m tercile

| Task | Size | m0 mean | D_feat | D_action | D_support |
|---|---:|---:|---:|---:|---:|
| open_drawer | small | 16.8 | 0.1966 | 0.2899 | 3.3885 |
| open_drawer | middle | 40.6 | 0.3315 | 0.4032 | 5.7191 |
| open_drawer | large | 49.3 | 0.3625 | 0.5445 | 8.6232 |
| close_drawer | small | 16.8 | 0.2135 | 0.3997 | 4.5551 |
| close_drawer | middle | 34.1 | 0.3113 | 0.2554 | 3.9476 |
| close_drawer | large | 49.1 | 0.3742 | 0.2765 | 5.1511 |
| pick_coke_can | small | 12.0 | 0.1955 | 0.2097 | 2.3870 |
| pick_coke_can | middle | 24.0 | 0.2822 | 0.2914 | 4.2750 |
| pick_coke_can | large | 43.4 | 0.3631 | 0.2864 | 4.5394 |
| move_near | small | 21.4 | 0.2460 | 0.4626 | 5.5843 |
| move_near | middle | 37.6 | 0.3402 | 0.5009 | 6.7171 |
| move_near | large | 62.1 | 0.4313 | 0.5370 | 8.2158 |

## Budget-vs-state trend

| Budget | Metric | Spearman rho vs m0 |
|---|---:|---:|
| Fixed34 | D_feat | -0.2373 |
| Fixed34 | D_action | 0.0665 |
| Fixed34 | D_support | 0.0958 |
| Entity-TopP | D_feat | -0.1810 |
| Entity-TopP | D_action | 0.0449 |
| Entity-TopP | D_support | 0.0736 |

## SOURCE: generated factorial/current tables

### Main closed-loop arms (seeds 100-199 unless noted)

| arm | open | close | pick | move | total | n |
|---|---:|---:|---:|---:|---:|---:|
| L11+Matched | 44 | 76 | 33 | 60 | 213 | 400 |
| L11+K32(Fixed32) | 40 | 67 | 35 | 54 | 196 | 400 |
| L11+K48 | 37 | 77 | 33 | 60 | 207 | 400 |
| L11+K64 | 42 | 67 | 32 | 57 | 198 | 400 |
| L11+Shuffle | 35 | 70 | 35 | 58 | 198 | 400 |
| L11+Entity-TopP | 31 | 74 | 38 | 55 | 198 | 400 |
| L11+TopP80 | 32 | 68 | 34 | 60 | 194 | 400 |
| L11+TopP85 | 38 | 73 | 35 | 60 | 206 | 400 |
| KMeans cluster+Matched | 40 | 62 | 24 | 67 | 193 | 400 |
| Random+Matched | 19 | 43 | 28 | 51 | 141 | 400 |

### Fixed34 current partial paired data

| arm | task | n | Matched | Arm | Harm | Rescue | Net | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| l11_fixed34 | google_robot_open_drawer | 100 | 44 | 37 | 28 | 21 | +7 | 0.391603 |
| l11_fixed34 | google_robot_close_drawer | 60 | 43 | 41 | 8 | 6 | +2 | 0.790527 |
| l11_fixed34 | google_robot_pick_coke_can | 42 | 12 | 16 | 2 | 6 | -4 | 0.289062 |
| l11_fixed34 | google_robot_move_near | 44 | 27 | 23 | 5 | 1 | +4 | 0.218750 |
| random_fixed34 | google_robot_open_drawer | 100 | 44 | 28 | 24 | 8 | +16 | 0.007000 |
| random_fixed34 | google_robot_close_drawer | 58 | 42 | 24 | 19 | 1 | +18 | 0.000040 |
| random_fixed34 | google_robot_pick_coke_can | 42 | 12 | 10 | 5 | 3 | +2 | 0.726562 |
| random_fixed34 | google_robot_move_near | 42 | 26 | 23 | 8 | 5 | +3 | 0.581055 |
