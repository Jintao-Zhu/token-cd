# SHR / Token-CD 实验完整报告

**整理日期：** 2026-09-05

**代码仓库：** `token-cd`

**主要模型：** OpenVLA-7B；迁移验证模型为 open-pi-zero（Pi0）

**主要环境：** SIMPLER 9 tasks
**统计单位：** 闭环 episode；所有“Rescue/Harm”均为同场景、同 seed 的成对结果

---

## 1. 执行摘要

本实验线研究一个 training-free 问题：能否从视觉—语言—动作模型内部构造一个“缺少任务相关视觉证据”的反事实负分支，并通过正负分支的动作输出差异增强策略对任务相关证据的利用。

当前最强、最可靠的结果来自 **SHR（Spatial Harmonic Reconstruction）**：

- 在 OpenVLA 的 9 tasks × 300 seeds、共 2700 个严格配对场景上：
  - Vanilla：507/2700，18.78%；
  - Recon：614/2700，22.74%；
  - SHR：646/2700，23.93%。
- SHR 相对 Vanilla：Rescue 270、Harm 131、净增 139 次成功，提升 **+5.15 percentage points**，精确 McNemar $p=3.38\times10^{-12}$。
- Recon 相对 Vanilla：Rescue 243、Harm 136、净增 107，提升 **+3.96 pp**，$p=4.25\times10^{-8}$。
- SHR 相对 Recon：提升 +1.19 pp，但 $p=0.101$，因此只能说 SHR 数值更高，尚不能声称显著优于 Recon。

任务层面，收益主要来自 Google Robot：

- close_drawer：SHR 69.67% vs Vanilla 51.33%，+18.33 pp；
- open_drawer：46.00% vs 23.67%，+22.33 pp；
- move_near：65.67% vs 61.33%，+4.33 pp；
- pick_coke_can：32.33% vs 27.67%，+4.67 pp。

后续变体给出了同样重要的负面证据：

1. **简单减小 guidance 不够。** Adaptive-SHR 几乎每个 episode 都触发降 λ，最终从 SHR 的 23.93% 降到 21.85%。action-shift 阈值没有区分“有益的大改动”和“有害的大改动”。
2. **把 semantic group 空间裁干净通常会更差。** SC-SHR、IC-SHR、Boundary-SHR 和 Partial-SHR 多数下降，说明 SHR 的有效证据不是等同于精确物体分割；分散的、边界的、上下文性的 token 可能属于有用的因果方向。
3. **简单时间平滑没有解决问题。** ST-SHR 最佳 β=1 仍比 SHR 少2次成功；时间先验仅约27%的 entity-step 真正使用，约73%因对齐失败回退，而且它只平滑负分支重建，不稳定 selector 和最终 logits。
4. **SHR残差中与正分支同向的分量不能简单删掉。** Projected-SHR 的正交化版本总体低于原SHR，尤其损害 open_drawer。
5. **方法不能无修改地跨策略架构迁移。** Pi0-SHR 在5任务上总体 70.80%，低于Pi0 Vanilla的72.93%；只有 open_drawer 提升。OpenVLA 的 logit CD 与 Pi0 的 flow-velocity guidance 不是同一种决策几何。
6. **Positive-support 长尾翻转现象确实存在，但尚无闭环结论。** PSC-SHR 的离线重放显示 Top-10 约束会改变大量 episode，Top-20很弱，Top-50完全无效；根据用户决定，排队的PSC闭环 rollout 已取消，因此不能报告PSC的成功率或Rescue/Harm。

总体判断：

> SHR 已经证明“语义选择的反事实视觉重建 + 动作输出空间的对比解码”有效；当前瓶颈不是单一的 reconstruction 平滑度，也不能简化为“mask必须像物体分割一样准确”。下一阶段最应系统研究的是 mask覆盖量、负分支退化强度和logit guidance强度的联合关系。

---

## 2. 核心研究思想

### 2.1 正负分支不是训练样本

这里的“positive/negative”是同一次推理中的两个分支，而非对比学习的数据样本：

\[
z^+=F(I,x),\qquad z^-=F(I^{-},x)
\]

其中：

- $I$ 或 $V^+$：原始视觉输入/视觉表示；
- $I^{-}$ 或 $V^-$：删除任务相关视觉证据后构造的反事实表示；
- $x$：同一条任务指令；
- $z^+,z^-$：两个分支的 action-vocabulary logits。

最终使用：

\[
z^*=z^+ + \lambda(z^+-z^-)
     =(1+\lambda)z^+-\lambda z^-.
\]

直觉是：如果删除某组视觉证据后某个动作变得不可能，那么 $z^+-z^-$ 表示该视觉证据对动作的支持方向，CD将其放大。

### 2.2 SHR的准确定位

SHR不是“在视觉feature上做线性插值后只forward一次”。它包含两个明确阶段：

1. **visual-token空间构造counterfactual negative branch**；
2. **action-logit空间进行contrastive decoding**。

因此最准确的定义是：

> Token-level counterfactual construction + logit-level contrastive decoding.

### 2.3 视觉单位

OpenVLA输入被处理为224×224，视觉骨干patch size为14，因此得到：

\[
224/14=16,\qquad 16\times16=256.
\]

论文中建议称为：

> 256 visual patch tokens arranged on a 16×16 spatial grid.

SHR修改的是patch经过视觉编码器和projector后的token特征，不是直接修改原始像素块。每个token虽对应约14×14的输入区域，但经过ViT后已包含全局上下文。

---

## 3. 标准SHR的详细设计

### 3.1 Positive branch

对原始图像和原始指令执行OpenVLA：

\[
z^+=F(V^+,x).
\]

同时在projector处取得视觉特征：

\[
V^+\in\mathbb{R}^{256\times4096}.
\]

解码为greedy、`do_sample=False`。

### 3.2 Semantic entity selector

1. 对256个projector token执行KMeans：$K=8$、seed=0、`n_init=10`；
2. 从instruction规则解析source/target entity；
3. 将entity的语言token embedding平均池化为 (e_{entity})；
4. 每个cluster用其视觉token均值表示；
5. 对每个entity选择cosine similarity最高的Top-1 cluster；
6. 对source/target选中的cluster去重并取并集，得到 $G$。

这里的selector不是目标检测器，也不保证 $G$ 是一个连通物体区域。它选择的是模型内部与指令实体相关的视觉表示集合。

### 3.3 Harmonic negative reconstruction

在16×16网格上建立四邻接图。对被选择区域 $G$ 的特征求离散Dirichlet谐波解：

\[
L_{GG}\hat V_G=-L_{GC}V_C,
\]

其中 $C=V\setminus G$。等价地，每个待重建token最终满足其上下左右邻居特征的离散调和关系。

重要澄清：当前实现不是按原token与四邻居的feature similarity计算softmax权重，也不是简单逐轮平均；它是对整个区域联立求解线性方程。非选中token保持bit-identical，token总数和位置保持不变。

构造：

\[
V^-_i=
\begin{cases}
\hat V_i,&i\in G,\\
V^+_i,&i\notin G.
\end{cases}
\]

### 3.4 Negative forward与自回归prefix

负分支使用相同图像输入和相同instruction，但在projector hook处把视觉token替换为 $V^-$。

7个action dimension具有自回归依赖。为了让正负logit差异只反映视觉counterfactual，而不是早期动作token不同导致的prefix漂移，负分支使用正分支clean greedy action作为teacher-forced prefix：

\[
z_q^-=F(V^-,x,a^+_{<q}).
\]

因此每个dimension依赖前面的dimension，但正负分支在同一dimension看到相同的clean prefix。

### 3.5 Logit fusion与动作输出

前6个连续动作维度使用：

\[
z_q^*=1.5z_q^+-0.5z_q^-,\quad q=0,\ldots,5.
\]

第7维gripper/terminate保持positive，不进行guidance。随后每一维在256个可执行action bins上取argmax并反量化为机器人动作。

---

## 4. Baseline Recon与SHR的区别

Recon和SHR共享：

- KMeans K=8 semantic selector；
- 相同 $G$；
- 同样的正负branch；
- 相同teacher-forced clean prefix；
- 相同 λ=0.5 logit CD。

区别仅在negative reconstruction：

- **Recon：** 从 $C=V\setminus G$ 中用cosine FPS选择 $M=10$ 个basis token，再用ridge回归重建 $G$，
  \(\rho=10^{-3}\operatorname{tr}(B_cB_c^T)/M\)。
- **SHR：** 利用16×16空间图上的四邻接Dirichlet谐波方程，从区域边界传播上下文。

Recon偏向低秩feature解释；SHR偏向空间连续的上下文补全。

---

## 5. 严格配对协议

主实验采用 `vanilla_recon_shr_canonical_0_299_v2`：

- 9 tasks；
- 每任务seeds 0–299，共2700个场景；
- 每个场景运行Vanilla、Recon、SHR，共8100个主arm episode；
- 保存2700个canonical simulator snapshots；
- 每个arm恢复同一个pickle snapshot，而不是重新随机reset；
- 校验canonical snapshot、initial state、initial RGB的SHA256；
- 主实验paired hash mismatch为0。

指标：

- Success Rate：成功episode比例；
- Rescue：baseline失败、方法成功；
- Harm：baseline成功、方法失败；
- Net：Rescue−Harm；
- 显著性：discordant pair上的exact two-sided McNemar/binomial test。

这种协议十分关键。早期跨进程仅凭seed reset的实验出现过SAPIEN浮点/物理状态不完全一致；后续canonical snapshot v2消除了这一混杂因素。

---

## 6. 主实验结果：Vanilla、Recon、SHR

### 6.1 总体结果

| Arm | Success | Success Rate | 相对Vanilla Δ |
|---|---:|---:|---:|
| Vanilla | 507/2700 | 18.78% | — |
| Recon | 614/2700 | 22.74% | +3.96 pp |
| SHR | 646/2700 | **23.93%** | **+5.15 pp** |

### 6.2 Paired结果

| 比较 | Rescue | Harm | Net | ΔSR | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Recon vs Vanilla | 243 | 136 | +107 | +3.96 pp | $4.25\times10^{-8}$ |
| SHR vs Vanilla | 270 | 131 | **+139** | **+5.15 pp** | $3.38\times10^{-12}$ |
| SHR vs Recon | 195 | 163 | +32 | +1.19 pp | 0.101 |

### 6.3 逐任务结果

| Task | Vanilla | Recon | SHR | SHR−Vanilla |
|---|---:|---:|---:|---:|
| close_drawer | 51.33% | 68.00% | **69.67%** | **+18.33 pp** |
| open_drawer | 23.67% | 37.67% | **46.00%** | **+22.33 pp** |
| pick_coke_can | 27.67% | **33.33%** | 32.33% | +4.67 pp |
| move_near | 61.33% | 59.67% | **65.67%** | +4.33 pp |
| place_apple_in_closed_top_drawer | 0.00% | 0.00% | 0.00% | 0.00 pp |
| carrot_on_plate | **4.67%** | 5.33% | 1.33% | −3.33 pp |
| put_eggplant_in_basket | 0.33% | 0.33% | 0.00% | −0.33 pp |
| spoon_on_towel | 0.00% | 0.33% | 0.33% | +0.33 pp |
| stack_cube | 0.00% | 0.00% | 0.00% | 0.00 pp |

逐任务paired变化：

| Task | Recon vs Vanilla R/H | Recon Net | SHR vs Vanilla R/H | SHR Net |
|---|---:|---:|---:|---:|
| close_drawer | 69/19 | +50 | 76/21 | **+55** |
| open_drawer | 70/28 | +42 | 88/21 | **+67** |
| pick_coke_can | 53/36 | +17 | 55/41 | +14 |
| move_near | 41/46 | −5 | 46/33 | +13 |
| place_apple | 0/0 | 0 | 0/0 | 0 |
| carrot_on_plate | 8/6 | +2 | 4/14 | −10 |
| put_eggplant | 1/1 | 0 | 0/1 | −1 |
| spoon_on_towel | 1/0 | +1 | 1/0 | +1 |
| stack_cube | 0/0 | 0 | 0/0 | 0 |

### 6.4 主结果解释

1. SHR对drawer任务的收益最大，说明空间上下文重建特别适合“局部操作结构”——目标区域被反事实删除后，模型仍看到场景布局，但缺少把手/抽屉相关动作证据。
2. pick_coke上Recon略高于SHR，表明小物体任务未必偏好纯空间平滑，低秩上下文重建可能更稳。
3. move_near上SHR只有温和提升，双实体选择、source-target关系和大范围mask使方向更复杂。
4. 五个WidowX/长程任务接近0% floor。它们不能用于可靠比较变体；这些失败首先是base policy能力不足，而不是SHR特有失败。

---

## 7. 失败机制与案例观察

### 7.1 “mask看起来像目标”不保证成功

已观察到：有的harm case同时mask了Pepsi罐与Coke罐；有的rescue case在RGB overlay上mask似乎与Coke无关。对此不能简单推出selector完全错误，原因包括：

- visual token经过ViT后带有全局上下文，不是独立的14×14像素块；
- cluster表示的是模型内部特征相似性，而不是传统实例分割；
- 背景、桌面、机械臂或干扰物token可能编码目标位置、可达性和动作参照；
- CD使用的是“删除前后动作logits差异”，不是mask本身的视觉可解释性；
- 同一semantic group可能同时覆盖目标、同类干扰物和与任务相关的上下文。

因此，overlay是必要诊断，但不能单独作为因果正确性的判据。

### 7.2 Harm不只来自selector错误

当前证据支持至少三种Harm来源：

1. selector确实选错或把多个实例混在同一cluster；
2. selector方向基本合理，但负分支破坏范围/强度过大；
3. negative residual方向合理，但固定 λ=0.5 在某些状态把动作推过决策边界。

这也是后续Adaptive、Component、Temporal、Support和Projection实验的出发点。

---

## 8. Adaptive-SHR v1

### 8.1 假设与设计

先计算标准SHR λ=0.5产生的动作token，并与Vanilla比较前6维：

\[
S=\frac{\#(a_q^{SHR}\neq a_q^{Vanilla})}{6}.
\]

- $S\le0.33$：保持 λ=0.5；
- $S>0.33$：改为 λ=0.25。

除此之外，selector、harmonic reconstruction、prefix和snapshot均不变。

### 8.2 结果

| Task | SHR | Adaptive | Δ | Rescue vs SHR | Harm vs SHR |
|---|---:|---:|---:|---:|---:|
| close_drawer | 69.67% | 65.33% | −4.33 pp | 35 | 48 |
| open_drawer | 46.00% | 28.00% | **−18.00 pp** | 18 | 72 |
| pick_coke_can | 32.33% | 33.33% | +1.00 pp | 45 | 42 |
| move_near | 65.67% | 61.67% | −4.00 pp | 30 | 42 |
| place_apple | 0.00% | 0.33% | +0.33 pp | 1 | 0 |
| carrot_on_plate | 1.33% | 2.67% | +1.33 pp | 4 | 0 |
| put_eggplant | 0.00% | 0.33% | +0.33 pp | 1 | 0 |
| spoon_on_towel | 0.33% | 0.67% | +0.33 pp | 2 | 1 |
| stack_cube | 0.00% | 4.33% | +4.33 pp | 13 | 0 |
| **Overall** | **646/2700** | **590/2700** | **−2.07 pp** | **149** | **205** |

触发统计：

- 2693/2700个episode至少一次降至λ=0.25，episode触发率99.74%；
- 265800个control steps中115917次触发，step触发率43.61%。

### 8.3 Insight

Adaptive规则失败的首要原因不是“降低λ一定错”，而是指标失去选择性：几乎所有episode都触发。动作token变化数量同时包含有益Rescue和有害Harm，无法估计方向是否正确。open_drawer的大幅下降进一步说明该任务依赖足够强的guidance。

结论：后续strength control需要用连续的margin、positive support、residual norm或不确定性校准，不能只统计argmax翻转数。

---

## 9. SC-SHR与IC-SHR：空间连通区域筛选

### 9.1 SC-SHR

SC-SHR在semantic union mask上做4-neighbor component decomposition：单实体保留最大component，双实体保留最大两个component，其他SHR设置不变。

局部seeds 300–399、386个有效配对：

| Task | Vanilla | SHR | SC-SHR |
|---|---:|---:|---:|
| close_drawer | 49.5% | 67.0% | 61.9% |
| open_drawer | 18.6% | 32.0% | 34.0% |
| pick_coke_can | 25.0% | 30.0% | 25.0% |
| move_near | 71.7% | 73.9% | 55.4% |

SHR→SC-SHR：Rescue 46、Harm 71、Net −25。最明显退化在move_near。该早期实验存在部分环境复现噪声，证据等级低于canonical v2，但方向与后续IC/SP实验一致。

### 9.2 IC-SHR

IC-SHR不再简单取最大块，而对component打分：

\[
Score(C_i)=0.5S_{sem}+0.3S_{center}+0.2S_{size}.
\]

单目标Top-1，双目标Top-2；component内部不拆分。4核心任务各300个canonical snapshot，共1200 episode。

| Task | Vanilla | SHR | IC-SHR | Δ vs SHR | Rescue | Harm |
|---|---:|---:|---:|---:|---:|---:|
| close_drawer | 51.33% | 69.67% | 69.67% | 0.00 pp | 29 | 29 |
| open_drawer | 23.67% | 46.00% | 32.67% | **−13.33 pp** | 28 | 68 |
| pick_coke_can | 27.67% | 32.33% | 31.00% | −1.33 pp | 42 | 46 |
| move_near | 61.33% | 65.67% | 63.67% | −2.00 pp | 28 | 34 |
| **Overall** | 41.00% | **53.42%** | 49.25% | **−4.17 pp** | **127** | **177** |

Component诊断：

| Task | 平均components | 平均选中 | token删除比例 |
|---|---:|---:|---:|
| close_drawer | 1.74 | 1.00 | 5.33% |
| open_drawer | 1.86 | 1.00 | 4.75% |
| pick_coke_can | 3.49 | 1.00 | 33.15% |
| move_near | 3.77 | 1.79 | 17.01% |

### 9.3 Insight

IC-SHR只从open_drawer mask中平均再删4.75%的token，却损失13.33 pp，说明被删除的小component并非无关噪声。可能解释为：

- 小component携带把手、夹爪接触区或任务状态证据；
- KMeans语义组是分布式表示，不能按传统图像连通性裁剪；
- center/size prior与机器人任务中的真实目标位置没有稳定对应关系。

所以“semantic group包含多个空间区域”这一观察成立，但“只保留最像目标的区域会更好”没有得到支持。

---

## 10. Structure-Preserving SHR（Boundary / Partial）

为了进一步隔离“是否应保留semantic region的一部分”，在3个核心任务、seeds 0–99上运行：

- Boundary-SHR：只重建mask腐蚀后的内部，保留一圈4-neighbor边界；
- Partial-SHR-50：每个entity region确定性随机重建约50% token。

| Task | SHR | Boundary-SHR | Partial-SHR-50 |
|---|---:|---:|---:|
| close_drawer | 78% | 48% | 60% |
| move_near | 67% | 49% | 63% |
| pick_coke_can | 41% | 37% | 38% |
| **Pooled** | **186/300 (62.00%)** | **134/300 (44.67%)** | **161/300 (53.67%)** |

相对SHR：

- Boundary-SHR：Rescue 21、Harm 73、Net −52；
- Partial-SHR-50：Rescue 33、Harm 58、Net −25。

Insight：完整删除semantic group往往比“保护边界”或随机减半更有效。SHR需要足够完整的counterfactual intervention，过于保守会使negative与positive差异失去任务方向。

---

## 11. ST-SHR：时间连续性

### 11.1 设计

在replan (t>0) 时，将上一帧同entity的重建区域按质心位移做整数平移，作为：

\[
(L_{GG}+\beta I)\hat V_G=-L_{GC}V_C+\beta T.
\]

测试 β=0.25、1、4；其余与标准SHR一致。时间对齐要求每个current token平移后都能在previous reconstructed region中精确找到对应token，位移超过4格、越界或不完整覆盖则回退β=0。

### 11.2 结果（9 tasks × seeds 0–99）

| Arm | N | Success | 同seed SHR | Rescue vs SHR | Harm vs SHR | Net |
|---|---:|---:|---:|---:|---:|---:|
| ST-SHR β=0.25 | 899 | 228 | 235 | 35 | 42 | −7 |
| ST-SHR β=1 | 900 | 233 | 235 | 40 | 42 | −2 |
| ST-SHR β=4 | 900 | 223 | 235 | 34 | 46 | −12 |

β=1逐任务：

| Task | SHR | ST-SHR β=1 | Δ |
|---|---:|---:|---:|
| close_drawer | 78% | 79% | +1 pp |
| open_drawer | 46% | 39% | −7 pp |
| pick_coke_can | 41% | 41% | 0 pp |
| move_near | 67% | 67% | 0 pp |
| place_apple | 0% | 1% | +1 pp |
| carrot_on_plate | 3% | 2% | −1 pp |
| put_eggplant | 0% | 1% | +1 pp |
| spoon_on_towel | 0% | 3% | +3 pp |
| stack_cube | 0% | 0% | 0 pp |

### 11.3 机制诊断

- 时间prior实际使用率约27%；约73%的entity-step回退。
- β=1主要fallback：`incomplete_overlap` 59051次、`dedup_empty` 20860次、`displacement` 11552次、`warp_oob` 3473次；成功使用36936次。
- β=1 residual temporal cosine约0.397，residual jerk约0.537。
- β=1 action jitter约0.157，高于相同900场景标准SHR约0.143。

因此ST-SHR没有稳定最终动作。原因是：

1. 16×16网格上的整数质心平移过粗；
2. exact whole-region overlap条件过严，KMeans mask轻微变形就失败；
3. prior只约束negative reconstruction，不稳定每帧重新产生的KMeans mask；
4. 真实抓取、遮挡、旋转和抽屉运动需要快速适应，强β会保留过期证据；
5. feature平滑经过非线性decoder和argmax后不保证action平滑。

结论：时间连续性思想尚未被否定，但v1对齐方式和作用层级不合适。

---

## 12. PSC-SHR：Positive-support约束（仅离线）

### 12.1 假设

若一个action token在positive中很不可能，但在negative中更不可能，则 $z_i^+-z_i^->0$ 仍可能把它推成最终winner。PSC只允许positive认为合理的candidate参与CD。

测试：Top-K，K=10/20/50；以及 (p_i^+>0.1\max p^+)。

### 12.2 离线结果

| Task | Top-10 changed episode | Top-20 | Top-50 | Relative-0.1 |
|---|---:|---:|---:|---:|
| close_drawer | 74.00% | 15.67% | 0% | 100% |
| open_drawer | 81.67% | 21.33% | 0% | 100% |
| pick_coke_can | 42.00% | 2.33% | 0% | 99.33% |
| move_near | 86.00% | 7.33% | 0% | 100% |

离线只能回答“过滤是否会改变旧轨迹上的action token”，不能把改变后的动作重新作用于环境，因此不能计算PSC成功率，也不能将旧SHR outcome重新标成PSC Rescue/Harm。

根据用户决定，PSC排队rollout已取消。当前可得结论只有：Top-10确实能命中潜在长尾翻转；Top-20作用很弱；Top-50是完全no-op；relative-0.1过强。

---

## 13. Projected-SHR：action residual几何

### 13.1 假设与设计

在每个256-way action dimension内，将centered SHR residual (r) 分解为相对positive centered logits (p_c) 的平行与正交部分：

\[
r=r_{\parallel}+r_{\perp},\qquad
g=r_{\perp}+\eta r_{\parallel}.
\]

- η=0：仅保留正交分量；
- η=0.25：保留25%平行分量；
- λ仍为0.5；negative branch完全保持标准SHR。

### 13.2 结果（5 tasks × 300）

| Task | 原SHR | Projected η=0 | Projected η=0.25 |
|---|---:|---:|---:|
| close_drawer | 69.67% | 70.33% | 57.00% |
| move_near | 65.67% | 67.33% | 53.67% |
| open_drawer | 46.00% | 35.33% | 36.33% |
| pick_coke_can | 32.33% | 31.00% | 30.33% |
| carrot_on_plate | 1.33% | 0.00% | 0.00% |
| **Pooled** | **645/1500 (43.00%)** | **612/1500 (40.80%)** | **532/1500 (35.47%)** |

相对原SHR：

- η=0：Rescue 151、Harm 184、Net −33；
- η=0.25：Rescue 102、Harm 215、Net −113。

Insight：与positive logits同向的residual不是单纯的无效缩放，它包含任务有用信息；将其投影掉会明显损害open_drawer。η=0在close/move略升，说明不同任务可能需要不同几何控制，但当前统一投影不成立。

---

## 14. Pi0-SHR迁移实验

### 14.1 设计

Pi0不是离散action-token自回归模型，而是flow-matching policy。实验保持同样的KMeans-K8 semantic region和四邻接harmonic reconstruction，构造positive/negative condition caches。在10个Euler flow steps的每一步，两个分支共享同一个当前noisy action state：

\[
v^*=v^+ +0.5(v^+-v^-).
\]

前6个动作维度guidance，gripper velocity保持positive。每次replan两臂共享确定性initial action noise；每个seed恢复同一个simulator snapshot。

### 14.2 结果（5 tasks × 300、双臂3000 episodes）

| Task | Pi0 Vanilla | Pi0-SHR | Δ | Rescue | Harm |
|---|---:|---:|---:|---:|---:|
| close_drawer | 225/300 (75.00%) | 215/300 (71.67%) | −3.33 pp | 22 | 32 |
| open_drawer | 142/300 (47.33%) | 150/300 (50.00%) | **+2.67 pp** | 46 | 38 |
| move_near | 254/300 (84.67%) | 243/300 (81.00%) | −3.67 pp | 21 | 32 |
| pick_coke_can | 296/300 (98.67%) | 292/300 (97.33%) | −1.33 pp | 4 | 8 |
| carrot_on_plate | 177/300 (59.00%) | 162/300 (54.00%) | −5.00 pp | 39 | 54 |
| **Overall** | **1094/1500 (72.93%)** | **1062/1500 (70.80%)** | **−2.13 pp** | **132** | **164** |

Insight：SHR在open_drawer上的方向可能跨架构保留，但把OpenVLA的logit CD直接映射为Pi0 flow velocity guidance并不普遍有效。Pi0每一步的velocity会连续积分，误差会在10步flow中累积；其guidance强度、时间调度和action维度尺度需要单独校准。

---

## 15. 当前实验完成状态

| 实验 | 计划规模 | 本地状态 | 可否形成闭环结论 |
|---|---:|---|---|
| Canonical Vanilla/Recon/SHR | 9×300×3 = 8100 | 完成；2700 snapshots；0 hash mismatch | 可以，主结果 |
| Adaptive-SHR | 9×300 = 2700 | 完成 | 可以 |
| IC-SHR | 4×300 = 1200 | 完成；technical pass 1200 | 可以 |
| ST-SHR β=.25/1/4 | 2700 | 2699 summaries；β=.25缺1例 | 可以，缺失已在N中注明 |
| SC-SHR | 约4×100 | 386 valid pairs | 可以作早期/辅助证据 |
| Boundary/Partial-SHR | 3×100×2 = 600 | 完成 | 可以，筛查级证据 |
| Projected-SHR | 5×300×2 = 3000 | 完成 | 可以 |
| Pi0 Vanilla/Pi0-SHR | 5×300×2 = 3000 | 完成 | 可以 |
| PSC-SHR offline | 4×300 stored logits | 完成 | 仅动作变化诊断 |
| PSC-SHR rollout | 原计划4×100 | 已按用户要求取消/未运行 | 不可以 |
| ST-Recon β=1 | 9×100 = 900 | 804 summaries，无活跃进程 | 不完整，不纳入主结论 |

截至本报告检查时，本机没有正在运行的上述SHR/Pi0 rollout进程。

---

## 16. 跨实验统一Insight

### Insight 1：反事实构造比直接遮断更重要

早期Attention-CD直接切断Action Query到视觉token的连接，容易制造分布外negative。Recon和SHR保留token数、位置与大部分场景上下文，仅删除区域独有证据，结果更稳定。主实验Recon/SHR均显著优于Vanilla。

### Insight 2：SHR有效，但有效对象不是传统“像素目标mask”

SC、IC、Boundary、Partial四组结果一致地显示：把semantic group裁得更像紧致物体区域不会自然提升。完整semantic group中的分散token、边界token和上下文token可能共同形成动作相关证据。

### Insight 3：整体干预强度是三个量的乘积

当前实际干预可粗略理解为：

\[
\text{intervention magnitude}
\approx \frac{|G|}{256}\times\gamma\times\lambda,
\]

其中：

- $|G|/256$：mask覆盖比例；
- $\gamma$：token从原特征走向重建特征的程度；当前SHR为1；
- $\lambda$：logit residual放大强度；当前为0.5。

现有实验大量改变“选哪些token”，但尚未完成 $K\times\gamma\times\lambda$ 的系统联合扫描。这是当前最明显的实验空白。

### Insight 4：Rescue和Harm不是由action shift大小单调决定

Adaptive-SHR的99.74% episode触发率说明“变了几个action dimension”并不能判断修改是否过强。有益guidance本来就可能改变多个维度。需要考虑positive margin、candidate rank、residual方向、环境阶段和动作尺度。

### Insight 5：时间连续性应作用在身份/selector或logit controller，而非只作用于重建值

ST-SHR仅约27%使用prior，且最终action jitter没有下降。下一版若继续时间方向，应优先解决entity identity tracking、soft overlap/continuous warp，并直接约束logit residual或λ，而不是只给线性方程加一个上一帧特征项。

### Insight 6：任务异质性非常强

- drawer：最稳定、最强的SHR收益；
- pick_coke：小幅收益且对mask实例歧义敏感；
- move_near：双实体、空间关系复杂，易受component裁剪影响；
- WidowX hard tasks：OpenVLA接近floor，难以用于比较细微方法差异；
- Pi0：baseline较强，统一λ=0.5反而更容易过度干预。

因此不能仅报告9-task微平均值；必须同时报告任务级结果与floor任务。

### Insight 7：负结果正在收敛研究边界

已有证据不支持以下简单假设：

- “只要mask更紧致就会更好”；
- “动作变化多就应该降低λ”；
- “上一帧重建更平滑就能稳定动作”；
- “把与positive同向的residual删掉就能消除虚假放大”；
- “OpenVLA有效的λ可以直接迁移到Pi0”。

这使下一阶段应从更多方法命名变体，转向控制变量清晰的联合超参数和机制实验。

---

## 17. 尚未充分探索的正负分支超参数

### 17.1 Selector与mask

- KMeans $K\in\{4,8,16,32\}$；
- 每个entity Top-1/Top-2或similarity threshold；
- mask ratio上/下限；
- source与target不同权重；
- soft membership而非hard cluster；
- feature layer或多层融合；
- selector置信度和跨帧entity identity。

### 17.2 Negative reconstruction

- reconstruction strength $\gamma\in\{0.25,0.5,0.75,1\}$：
  \(V_i^-=(1-\gamma)V_i^++\gamma\hat V_i\)；
- 4-neighbor vs 8-neighbor；
- uniform vs feature/spatial weighted graph；
- boundary context半径；
- harmonic、FPS-ridge或二者混合；
- 每个entity独立强度。

### 17.3 Logit fusion

- λ系统扫描；
- 每个action dimension独立λ；
- positive/negative temperature；
- residual norm或margin归一化；
- positive Top-K/support约束；
- 按replan阶段调度λ；
- 是否guidance gripper；
- positive teacher-forced prefix vs negative自回归prefix。

### 17.4 时序

- history长度和EMA系数；
- continuous centroid warp或光流；
- soft overlap阈值；
- mask identity persistence；
- residual-level而不是feature-level temporal controller。

---

## 18. 建议的下一阶段实验

不建议直接做全部笛卡尔积。应分三阶段：

### Stage A：机制筛查

使用4核心任务、每任务30–50个canonical seeds：

- $K\in\{4,8,16,32\}$；
- $\gamma\in\{0.25,0.5,0.75,1.0\}$；
- $\lambda\in\{0.1,0.25,0.5,0.75\}$。

记录mask ratio、feature perturbation norm、residual norm、positive winner rank、各dimension action flip和闭环Success。先确定每个任务的响应面，不只选单个最高点。

### Stage B：控制机制验证

从Stage A选出少量组合，比较：

- fixed λ；
- positive-margin calibrated λ；
- Top-10或Top-20 support；
- mask-ratio calibrated λ。

重点观察Rescue保留率与Harm下降，而不是只看总成功率。

### Stage C：全量确认

只将预注册的1–2个方法扩展到9×300，并复用canonical snapshots。对floor任务单独标注，不用其0%结果夸大“non-worse task数”。

建议的首要问题是：

> 在selector保持不变时，SHR的收益/伤害是否主要由 $|G|$、$\gamma$ 和 $\lambda$ 的联合尺度决定？

这个问题比继续增加启发式component或temporal变体更直接，也最能形成一条清楚的论文机制链。

---

## 19. 关键代码与artifact索引

- 标准semantic selector：`research/semantic_token_cd/rollout_policy.py`
- SHR/ST-SHR harmonic实现：`research/semantic_token_cd/st_shr_policy.py`
- Recon实现：`research/semantic_token_cd/semantic_recon_policy.py`
- guided negative prefix：`research/semantic_token_cd/global_merge_policy.py`
- Adaptive-SHR：`research/semantic_token_cd/adaptive_shr_policy.py`
- IC-SHR：`research/semantic_token_cd/instruction_component_shr_policy.py`
- Structure-Preserving SHR：`research/semantic_token_cd/sp_shr_policy.py`
- Projected-SHR：`research/semantic_token_cd/projected_shr_policy.py`
- Pi0-SHR：`research/semantic_token_cd/pi0_shr_policy.py`
- PSC-SHR：`research/semantic_token_cd/positive_support_shr_policy.py`
- Canonical主结果：`github_results/vanilla_recon_shr_canonical_0_299_v2/FINAL_RESULTS.json`
- Adaptive结果：`adaptive_shr_experiment/statistics/summary.json`
- IC-SHR结果：`artifacts/instruction_component_shr_v1/component_stats/summary.json`
- PSC离线报告：`artifacts/positive_support_constrained_shr_v1/offline/report.md`
- SC-SHR说明：`docs/sc_shr/README.md`

---

## 20. 最终结论

当前可以有把握地声称：

1. 在严格配对的OpenVLA/SIMPLER实验中，语义选择的反事实视觉重建配合logit-level CD显著优于Vanilla；
2. SHR在总体和drawer任务上最强，但相对Recon的总体差异尚未达到统计显著；
3. SHR的有效mask不是传统实例分割mask，空间上分散的semantic token可能是方法信号的一部分；
4. 当前Adaptive、Spatial-component、Temporal和Projection增强均未稳定超过标准SHR；
5. PSC只有离线动作变化证据，没有闭环性能证据；
6. Pi0迁移显示同一思想需要针对flow policy重新校准，不能直接复用OpenVLA的固定λ；
7. 下一阶段最关键的实验是 $K\times\gamma\times\lambda$ 联合响应面，以及基于positive margin/support的有选择strength control。

简而言之：

> 已经验证“构造什么negative”非常重要；下一步需要精确回答“删多少、删多狠、放大多强”，而不是继续假设mask越像物体、时间越平滑就一定越好。
