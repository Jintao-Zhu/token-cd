# Drawer 顶层 vs 中层 · 交叉干预机制实验报告

- 协议：`PROMPT_SIMREGION_OPEN_TOP_MID_V1`
- 日期：2026-09-08
- 类型：机制研究（simulator 区域标注代替 selector，**不是** 原版 SHR 性能评估）
- 关键问题：SHR 的收益来自“增强当前指令需要的视觉证据”，还是“扰动任意视觉区域碰巧改善动作”？
- 判定标准：同一个物理帧上，把同一抽屉区域在“它是目标 / 它不是目标”两种角色下分别干预，观察作用是否随任务角色发生符合指令的变化。

## 1. 实验设计回顾

- 同画面构造：同一 seed 下分别建 `OpenTopDrawerCustomInScene-v0` 与 `OpenMiddleDrawerCustomInScene-v0`，两条指令都能执行；物理初始状态相同、成功判定随指令改变。
- 区域构造：当前帧 overhead 相机外参 + URDF 抽屉前板几何投影到 16×16 token，再按当前帧 Segmentation 剔除邻抽屉占优 cell。规则与指令无关，因此同帧下 top/middle 两区域 mask 完全固定。
- 干预强度配置（沿用现有 SHR 配置，锁定）：`lambda=0.5`、`beta=0`、harmonic 16×16 四邻 Dirichlet、prefix=共享 clean greedy、sampling=False、仅动作维度 0–5 受引导、gripper 用 clean 正维度。
- 三组策略：`vanilla` / `shr_target`（mask=指令目标抽屉）/ `shr_other`（mask=另一抽屉），闭环每步按当前观测重新生成区域，不固定初始 mask。

## 2. 有效性校验（全部通过）

| 检查 | 结果 |
| --- | --- |
| Snapshot 配对 manifest | 20/20 seed 的 top/middle `sim_state`、`agent_state`、`rng_state`、RGB 全部逐位相等（`all_pairs_exact: true`） |
| 闭环 pairing 抽查 | 120 对全部一致，state/rgb mismatch = 0 |
| SHR 技术审计（shr 双臂 × 60 ep） | 80/80 通过，0 失败（feature 替换一致、prefix 一致、recon 有限、λ=0.5、β=0、mask 与 region 一致） |
| GPU 故障 | 全程未使用故障 GPU0（Vulkan 初始化错误）；top 修复片与全量在 GPU3/4 完成 |

区域覆盖量（离线样本 60+60 帧上的统计）：

- top 区域 token 数：均值 24.1，范围 11–33
- middle 区域 token 数：均值 17.2，范围 1–24
- 两区域重叠 token：均值 3.4，范围 0–24（个别帧两抽屉投影发生重叠，已记录并在解释中作为混杂考量）
- 对应特征扰动幅度：top 区域均值 117.2，middle 区域 94.1（区域越大扰动越强；该差异在交叉比较中天然被“同 mask × 双指令”设计控制）

### 样本有效性声明（重要）

环境中 reset 的初始 `sim_state` 在 20 个 nominal seed 上只有 **9 种不同取值**（同一 sim_state 桶内的 seed，其 agent/RGB/行为仍可能不同，因此不是简单重复）。同一 seed 的 top/middle 配对始终严格成立；但“20 个初始场景”不构成 20 个独立场景。离线方向对齐与闭环结论应视为 **有限样本的机制证据**，不宜外推成功率。

## 3. 离线四分支解码（同帧 × 双指令 × target/other 区域）

- 帧来源：20 seed × 每条轨迹 {step 0, 56, 112}，top/middle 各 60 帧；step0 由双 origin 各解码一次并验证逐位一致。
- 每帧四分支共享同一图像、同一区域 mask、同一 clean 前缀；`effect = ||guided_action[0:6] − clean_action[0:6]||`。

### 3.1 角色效应（同帧内 target 区域 vs other 区域的干预强度）

| 指令 | n | effect_target 均值 | effect_other 均值 | Δ(target−other) | target 更大的比例 |
| --- | --- | --- | --- | --- | --- |
| open top drawer | 120 | 0.0328 | 0.0292 | +0.0035 | 35.0% |
| open middle drawer | 120 | 0.0341 | 0.0408 | −0.0067 | 27.5% |
| 合计 | 240 | 0.0334 | 0.0350 | −0.0016 | 31.3% |

配对检验：`Δ = −0.0016, SE = 0.0045, t = −0.35`。**不存在“目标区域干预更强”的系统性效应**，且两指令下方向相反（top +、middle −），说明这主要是区域身份/大小差异而非角色一致性效应。

### 3.2 同 mask × 跨指令敏感性（固定区域，看指令切换后作用是否改变）

| mask | n | 跨指令 effect 差的均值 | 中位数 | >0.01 的比例 |
| --- | --- | --- | --- | --- | --- |
| top 区域 | 120 | 0.0447 | 0.0257 | 64.2% |
| middle 区域 | 120 | 0.0416 | 0.0228 | 61.7% |

同一 mask 在“当目标”与“当非目标”时，effect 大小经常变化（约 62–64% 帧超过 0.01，相对变化幅度可观）。但变化的**方向**没有一致性：

- 作为目标时 effect 更大的帧：mask=top 仅 35.0%，mask=middle 40.8%，合并 37.9%（`t≈−0.33`）。

结论：指令确实会调制同一区域的解码结果（说明指令信息能影响被扰动特征的下游作用），但这种调制**不系统地向“目标角色更有效”倾斜**。

### 3.3 step0 严格同状态下的方向对齐（辅助指标）

16 个唯一 step0 帧 × 双指令，将 guided 平移（delta 前三维）投影到 eef→各抽屉 handle 方向：

| 指令 | n | 目标分支 proj→目标 handle 均值 | 目标分支 proj→other handle 均值 | “更指向目标”的比例 |
| --- | --- | --- | --- | --- |
| open top | 16 | −0.0067 | −0.0056 | 12.5% |
| open middle | 16 | +0.0046 | +0.0060 | 18.8% |

引导后的位移没有可靠地指向目标 handle（两比例都在或低于随机水平），方向证据不成立。此指标是辅助的：动作平移的坐标/参考系与 handle 世界坐标的对齐是近似处理，且样本量小。

## 4. 闭环机制实验（20 seed × 2 指令 × 3 策略）

所有条目为 nominal 计数（每格 20 ep；样本有效性见 2 节声明）。目标抽屉位移用 `final_qpos`，`opened` 以 `qpos ≥ 0.15` 计。

### 4.1 成功率与“最终打开目标抽屉”

| 任务 | vanilla | shr_target | shr_other |
| --- | --- | --- | --- |
| open top：success | 8/20 (40%) | 9/20 (45%) | 11/20 (55%) |
| open top：final 打开目标 | 8/20 | 8/20 | 10/20 |
| open middle：success | 6/20 (30%) | 4/20 (20%) | 7/20 (35%) |
| open middle：final 打开目标 | 6/20 | 3/20 | 7/20 |

- 目标抽屉最终位移均值：top：vanilla 0.119 / target 0.124 / other 0.113；middle：vanilla 0.087 / target 0.060 / other 0.084。非目标抽屉终位均值≈0（三种策略都不会误开“另一个抽屉”）。

### 4.2 paired Rescue/Harm（相对同 seed 的 vanilla）

| 任务 | shr_target | shr_other |
| --- | --- | --- |
| open top | rescue 4 / harm 3 | rescue 5 / harm 2 |
| open middle | rescue 4 / harm 6 | rescue 5 / harm 4 |

### 4.3 同 seed 下 target 臂 vs other 臂的成功模式

| 任务 | same | neither | target_only | other_only |
| --- | --- | --- | --- | --- |
| open top | 6 | 6 | 3 | 5 |
| open middle | 3 | 12 | 1 | 4 |

### 4.4 接触行为（辅助，有测量口径限制）

首次接触判定 = TCP 中心进入 handle 中心 0.06 m 窗口内。计数（首次接触目标抽屉）：

| 任务 | vanilla | shr_target | shr_other |
| --- | --- | --- | --- |
| open top | 5/20 | 0/20 | 6/20 |
| open middle | 3/20 | 5/20 | 2/20 |

注意：top 任务 shr_target 有 8–9 ep 成功打开目标抽屉、但 0 次进入 TCP≈handle 中心窗口，说明该阈值会漏检“沿把手条侧面/偏离中心接触”的方式，此列只能当粗粒度参考，不能据此说 shr_target 没接触目标。

## 5. 解释（结论性陈述，保持克制）

1. **没有出现支持“任务条件化视觉证据增强”的证据。** 闭环里 target 区域引导从未系统性优于 other 区域引导：top 上 other 11/20 > target 9/20 > vanilla 8/20；middle 上 target 是三者中最低（4/20，且 paired harm 6 > rescue 4）。middle 任务里把目标（middle）区域做谐波重建反而伴随成功率名义下降、目标位移均值下降（0.087 → 0.060）。
2. **离线结果支持“区域作用会被指令调制，但方向无益”。** 同 mask 在目标/非目标角色间 effect 大小改变频率高（62–64%），但角色方向随机（37.9% 目标更大）；step0 方向对齐 ~ chance。因此若存在“指令相关”作用，它表现为对同一视觉区域扰动的非特异响应，而不是把动作推向当前目标。
3. **整轮更接近“区域固定扰动 / 通用视觉效应 + 指令导致的次优扰动”的图景，而非“需要更好的 selector”。** 干预非目标区域（other）在闭环中从未更差、有时更好，说明被干扰的区域是否属于任务目标，并非收益来源；任务条件化解释无法由本数据支撑。
4. **负结果开放解读。** 这不能证明原版 SHR（带真实 selector、其他 λ/层配置）无效，只能说明：在本场景与当前对比解码配置下，simulator 标注的“目标区域优先增强”假设没有得到机制支持；瓶颈更可能出在**负分支/对比方向或强度（λ/差分方向）**，而不是区域选取本身。

## 6. 局限

- 20 nominal seed 只有 9 种初始 sim_state；同一 seed 的 top/middle 严格配对成立，但样本独立性弱于名义值，所有成功率差都在噪声区间内，未做显著性外推。
- 区域是 simulator 标注（研究工具），不涉及原版 selector 的注意/聚类质量；本实验只回答“区域角色×干预”的机制问题。
- 接触/方向指标均为辅助近似；闭环期间每步重新选区域，后续帧不再受跨轨迹 mask 一致约束（按协议设计如此）。
- 报告未对 top/middle 结果做合并统计检验；上面只有描述性统计与离线配对 t。

## 7. 建议的下一步（不急于提新算法）

- 在**同一批 step0 配对帧**上做“差分方向翻转/负分支变体”的离线扫描（保持同 mask、同 prefix），看目标角色收益是否对对比方向敏感——这直接检验负分支是否为瓶颈。
- 若仍无角色效应，则回到“视觉扰动的一般效应”研究：量化哪种区域（如 target/other/空白/整帧）的谐波重建在多大程度上只是引入随机化扰动，而不携带任务证据。
- selector 路线（注意力聚类等）的优先级应下调：本实验表明“选得对但方向差”的可能性大于“方向对但选不对”。

## 8. 产物位置

- 快照与配对 manifest：`artifacts/xdrawer_topmid_v1/snapshots/`
- 闭环 episode summaries：`artifacts/xdrawer_topmid_v1/runs/full/episodes/`
- 闭环聚合：`artifacts/xdrawer_topmid_v1/runs/full/statistics/closed_loop_summary.json`
- 离线四分支解码：`artifacts/xdrawer_topmid_v1/runs/full/offline/offline_branches_{top,middle}.json`
- 离线分析：`artifacts/xdrawer_topmid_v1/runs/full/offline/offline_analysis.json`
- 配置锁：`artifacts/xdrawer_topmid_v1/runs/full/CONFIG_LOCK.json`
