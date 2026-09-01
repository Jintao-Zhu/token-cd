# Phase 1B 机制排查报告：任务间反向机制（Semantic-Merge vs Attention-K8）

实验：`ATTN_SEMANTIC_MERGE_K8_V1`（seeds 200–299，3 任务）。两臂共享 KMeans K=8
semantic selector，只换退化方式：
- `semantic_attn_k8_l8_15`：切断 Action Query → selected semantic visual keys 的注意通路（L8-15，mask −1e4）。
- `semantic_merge_k8_eta100`：把每个 selected group 塌缩到自身原型（η=1.0），注意通路完整。

核心问题：**为什么 move_near 必须保留 semantic-token 访问通路，而 close_drawer 反而从阻断通路中获益？**

---

## 1. 反向机制的实证签名（逐 seed，按 vanilla 结果分层）

正确分解必须**按 vanilla 结果分层**（harm 需 vanilla 成功、rescue 需 vanilla 失败，
"attn harm / merge rescue" 在同一 seed 上逻辑不可能）。

### move_near（vanilla_succ=64 / fail=33）
| | attn | merge |
|---|---|---|
| harm（打破已成功轨迹） | **19** | 9 |
| rescue（修好已失败轨迹） | 4 | **13** |
| 独杀（另一臂无伤） | **attn_harm_ONLY = 12 seed** | merge_harm_ONLY = 2 seed |

**attn 独杀的 12 个 seed 全部呈现同一签名**：`moved_correct_obj=True` 但 `near_tgt_obj=False`。
即：阻断通路后，arm **仍认识该搬哪个物体（没搬错）**，却**无法把它搬到 target 附近**。

### close_drawer（vanilla_succ=52 / fail=48）
| | attn | merge |
|---|---|---|
| harm | 12 | **16** |
| rescue | **18** | 12 |
| 独杀/独救 | attn_harm_ONLY=6 | **merge_harm_ONLY=10** |
| | **attn_rescue_ONLY=14** | |

- **merge 独杀**（10 seed）：vanilla qpos→0（关好），merge qpos→0.06–0.20（**留缝未关严**）。
- **attn 独救**（14 seed）：vanilla qpos→0.20（未关），attn qpos→0（关好）。

---

## 2. 为什么反向：任务性质 × 退化类型 的交互

selector 的选择内容决定了一切：

| 任务 | selector 选中 | 任务性质 | 任务关键信号 |
|---|---|---|---|
| move_near | **2 实体**（source+target，如 sponge+7up can），1–2 group，26–72 token | 关系型（搬 A 到 B 旁） | source→target 的**粗粒度空间关系** |
| close_drawer | **1 实体**（drawer handle），1 group，18–29 token | 局部操纵型（关抽屉） | handle 的**精细定位** |

### move_near（关系型）
- 选中 token 联合编码 source **和** target 及其相对关系。
- **阻断** = 整体移除该关系 → arm 保留物体身份（靠语言交叉注意力 + 非选中 token），但**丢失 target 定位** → `near_tgt_obj=False`（12/12）。
- **塌缩** = 每组保留质心 → source/target 的粗粒度位置关系存活 → 仍搬得到。

### close_drawer（局部操纵型）
- 选中 token 编码 drawer handle（单实体）。关键信号是 handle 的**精细空间位置**。
- **塌缩** = 抹掉组内细节 → 抓取/推拉方向偏差（div 主要在 rx,ry,rz 旋转维）→ 抽屉关不严（qpos>0）。
- **阻断** = 整体移除 handle patch → 迫使 policy 转而用抽屉面板/柜体等**周围上下文 token** 做"推关" → 反而更稳地关上。

### cos(r_attn, r_merge)@step0 的判决性证据
三个任务 mean +0.22~+0.35，`frac<0` 仅 4%–17%。**两个 counterfactual 是"同一方向、不同强度/不同精度损失"，不是相反方向**。因此反向机制**不是**"校正方向相反"，而是**任务对『完全切断访问』vs『塌缩细节』两种退化的敏感度相反**。

### 分叉的时相
move_near attn 独杀 9/12 seed 在 **t=0** 就分叉（vanilla 的 `first_moved_correct_obj`≈59–77，`first_near_tgt_obj`≈77），且每次只翻 1–2 个 action token、量级 |Δact|~0.02–0.24。即：**CD 在初始接近阶段就注入一个单维微扰，随后 80 步内复合放大**，最终表现为"搬了但没到"。

---

## 3. 结论：不稳定的来源是「可预测的」，不是噪声

CD 的收益符号是 `任务性质（关系型 vs 局部操纵型） × 退化类型（切断访问 vs 塌缩细节）`
的**可预测函数**，而非随机波动。这就解释了为什么 Phase 1A/1B 里 CD 时而正时而负：

- 关系型任务（move_near）→ 需要保持 semantic 组的**质心关系** → 塌缩（merge）无害、切断（block）致命。
- 局部操纵型任务（close_drawer）→ 需要保持 handle 的**精细定位** → 塌缩致命、切断反而去噪。

### 尚未回答（可选下一步）
close_drawer "阻断反而获益" 的**精确空间机理**（是去掉了干扰 patch，还是迫使走"推面板"的稳健策略）
需要用 replay 提取 handle 的具体位置 / 相机里被选中 token 的覆盖区域来确认。现有 flags（qpos）已证明
其行为后果（关严 vs 留缝），但未区分这两种因果路径。

---

数据：`artifacts/attn_semantic_merge_k8_v1/flip_matrix/flip_matrix.{json,csv}`
（逐 seed 全部字段：success/steps、4 类联合分类、first-divergence、分叉动作维度、分叉 token、
residual norm、cos@step0、selector coverage、vanilla phase flags）。
