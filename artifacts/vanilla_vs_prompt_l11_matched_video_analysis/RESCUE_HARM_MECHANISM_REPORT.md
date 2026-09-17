# Prompt-L11-Matched 的 Rescue/Harm 机制诊断

## 结论

当前证据不支持一个统一的 Harm 原因，而支持两个不同机制：

1. **Drawer：后期持续/过强的动作修正。** 成功与 Rescue 轨迹中的 guidance 通常随过程推进而减弱；Harm 中 guidance 在中后期仍然频繁改变动作，甚至继续增强，导致不能形成或维持有效拉/推动作。
2. **Pick Coke：错误方向或错误阶段动作。** 多数 Harm 不是 guidance 更强，而是接近方向错误、转向干扰物或接近后失去目标；另有少数已经抓住但没有完成抬升。因此单纯降低统一 lambda 不能解决这类 Harm。

这说明 Prompt-L11-Matched 的主要问题不是“始终选错 mask”或“始终干预太强”，而是：**固定的正向 contrastive guidance 没有判断当前修正方向是否有利，也没有区分接近、接触和持续操作阶段。**

## 全部 0–99 配对数据

下表基于三个任务各100个同 seed 配对 episode，而非只基于视频案例。`flip` 是前6个动作维度被 guidance 改变的比例；`action L2` 是 guided 与 clean 动作差；`delta jerk` 是连续两步 guidance 修正量的变化。

| Task | 类别 | Episode | flip | action L2 | delta jerk |
|---|---|---:|---:|---:|---:|
| open_drawer | Rescue | 28 | 27.3% | 0.052 | 0.083 |
| open_drawer | Harm | 5 | **36.6%** | **0.080** | **0.122** |
| close_drawer | Rescue | 29 | 24.5% | 0.049 | 0.070 |
| close_drawer | Harm | 3 | **46.0%** | **0.113** | **0.129** |
| pick_coke_can | Rescue | 31 | **24.3%** | **0.041** | **0.064** |
| pick_coke_can | Harm | 10 | 16.3% | 0.035 | 0.055 |

Drawer Harm 的干预明显更强、更频繁、更不稳定；Pick Coke 的方向恰好相反，所以强度不是 Pick Harm 的主要解释。

## 最关键的阶段证据

比较轨迹最后三分之一：

| Task | 类别 | late flip | late action L2 |
|---|---|---:|---:|
| open_drawer | Rescue | 16.9% | 0.037 |
| open_drawer | Harm | **33.2%** | **0.075** |
| close_drawer | Rescue | 12.1% | 0.025 |
| close_drawer | Harm | **46.3%** | **0.149** |
| pick_coke_can | Rescue | 14.7% | 0.021 |
| pick_coke_can | Harm | 12.4% | 0.025 |

Drawer 成功轨迹的 guidance 会明显“收手”；Harm 轨迹没有。这是当前最有价值的风险信号。但它仍是结果相关性：轨迹卡住也可能反过来使模型持续修正，必须用短时多步干预验证因果。

## 27对可复现视频中的行为模式

- `open_drawer` 的4个 Harm：Prompt 有3个完全没有产生1 cm有效位移，另1个很晚才产生轻微位移；Vanilla 全部完成。
- `close_drawer` 的3个 Harm：Prompt 都能开始推动，但不能持续到成功阈值；Vanilla 全部完成。
- `pick_coke_can` 的5个 Harm：4个没有形成稳定抓取，其中包含明显转向苹果、海绵、Pepsi或空白区域；1个抓住 Coke 但没有完成抬升。
- 5个 Coke Rescue 中，Prompt 全部完成抓取和抬升；Vanilla 要么没有正确接近，要么接近/抓住后没有完成抬升。

因此 Rescue 与 Harm 改变的是同一条链路：**目标接近 → 接触/抓取 → 持续操作**。Prompt 有时把系统推过原策略的决策边界，有时则推向错误一侧。

## 已排除的简单解释

1. **不是只由 mask 数量决定。** open Harm 的平均 m 更大，但 close Harm 的平均 m 反而更小；不存在统一的“大 mask 有害”。
2. **不是 Prompt/SHR mask 重叠率可直接识别。** Pick 的 Rescue 与 Harm 平均重叠率分别为0.184和0.178，几乎一样。
3. **不是长尾 action token 被大量错误抬高。** Harm 中 guidance 改出的 winner 落在 clean Top-10 之外的比例只有约2%–3%；PSC Top-K 很难解决主要问题。
4. **不是统一减小 lambda 就能解决。** 以前 Adaptive-SHR 把高 action-shift 步骤从0.5降到0.25，2700 episode 上反而比 SHR 净少56次成功，而且几乎每个 episode 都触发。粗粒度 shift threshold 不够。

## 如何保留 Rescue、减少 Harm

不要一次混合很多规则。应分两个单变量实验。

### A. Drawer：阶段化强度控制

- 接近阶段保持 `lambda=0.5`，保留帮助对准把手和形成接触的 Rescue。
- 一旦检测到抽屉已经持续开始运动/已经接触，比较 `lambda=0.25`、`lambda=0.1` 和 clean fallback。
- 增加一个连续窗口安全条件：若 guidance 在2–3个 replan 中持续改变多个维度，或修正方向反复变化，则临时降 lambda，而不是根据单步 action-shift 全局降权。
- 第一轮只在 drawer 的配对场景中做短时多步分叉；单步替换不足以代表累积效应。

### B. Pick Coke：目标进展/方向门控

- 接近阶段保留 Prompt guidance，但只有当它连续2–3步让 TCP–目标距离下降时才继续；若距离持续上升，回退 clean。
- 抓取后单独切换到“抬升/保持”阶段，降低横向与旋转修正，只保留有利的竖直提升分量。
- 先用 simulator 的目标距离和 grasp 状态做 oracle 机制验证；若有效，再换成可部署的目标跟踪/接触估计。不要一开始就把目标检测误差混入实验。

### 最小闭环验证顺序

1. Drawer：`Original Prompt`、`contact 后 lambda=0.25`、`contact 后 clean`，每任务100个同 seed。
2. Pick Coke：`Original Prompt`、`progress gate`、`grasp 后 phase gate`，100个同 seed。
3. 只有两项分别有效后，再合并成一个统一方法。

## 结论边界

目前可以较有把握地说：**Drawer Harm 与中后期持续的大幅 guidance 强相关；Pick Coke Harm 主要与修正方向和阶段错误有关。** 还不能声称某个具体阈值已经因果地消除了 Harm。最终规则必须通过同 snapshot、同 seed 的多步分叉和100-episode闭环实验验证。
