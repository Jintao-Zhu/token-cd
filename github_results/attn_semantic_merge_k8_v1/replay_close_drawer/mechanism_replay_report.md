# close_drawer 反转机制 replay 分析报告

**日期**: 2026-08-30 · **任务**: `google_robot_close_drawer` · **数据**: 36 seed 全轨迹 replay
**问题**: 为什么 `semantic_attn_k8_l8_15`（block Action Query→选中 token）能 *rescue* close_drawer，
而 `semantic_merge_k8_eta100`（collapse 选中 token→原型）会 *harm* 它？

**核心结论（一句话）**: 用户假设 **"attention blocking 不是『弱化细节』，而是改变策略的操作方式"** 得到证实——
但比 "grasp→push" 更精确：blocking 把 close_drawer 的多种 *脆弱微调失败模式* 统一成一条 **粗粒度、鲁棒的"把眼前抽屉推关"** 策略；
collapse 则是 *保留但畸变* 同一个视觉区域，让策略仍做精细逼近、但落点偏 3~8cm → 留缝。

---

## Q1 — 被选中的 token 到底覆盖哪里？

**答案：选中 token 覆盖的是"目标抽屉的整个 body 区域"（前面板 + 把手 + 周边），不是精确的把手 cell。**

- 选择器（KMeans K=8, top-1 cos）选中的实体名就是目标抽屉本身：`"bottom drawer"` / `"top drawer"` / `"middle drawer"`。
- 选中 token 是**连续区域**，且其行区间**精确跟随抽屉在图像里的竖直位置**：
  | 抽屉 | handle cell | 选中行区间 (sel_rows) | 选中列区间 |
  |---|---|---|---|
  | top | [11,7] | 6–9 | 0–8 |
  | middle | [13,7] | 7–13 | 3–11 |
  | bottom | [14,7] | 12–15（有时混入 1,8–12 背景行） | 1–13 |
- **把手的精确 cell 只有 ~40% 的 seed 落在选中集合内**。attn-rescue 11 个 clean seed 里 handle_in_sel=True 只有 3 个（211/248/260），其余 8 个 handle cell 距选中团 1~3 个 cell。merge-harm 6 个 clean seed 里 3/6 True。

**⇒ "block 把手 token" 这个表述要修正**：被 block 的是"抽屉 body 区域"的视觉 token，把手只是嵌在区域里的一个小特征。
这直接决定了下面 Q2/Q3 的读法——block 掉的不是"把手细节"，而是"整块抽屉区域"。

---

## Q2 — 第一次 divergence 后 gripper 实际走向哪里？

对 attn-rescue clean seed（vanilla 失败→attn 成功），vanilla 的失败不是单一模式，而是 **3 类**：

| vanilla 失败模式 | seed | 证据 |
|---|---|---|
| **① 抓把手然后向外拉** | 211 / 248 / 260 | 都是 bottom drawer、handle_in_sel=True；vanilla 抓握 (n_close=45) 后 `drawer_qpos` 从 0.20 **升到 0.282**（拉得更开）；contact_t=None |
| **② 靠近但从不commit** | 226 / 236 / 270 / 277 / 200 | min_d_handle 0.10~0.15m 但 contact_t=None，qpos 停在 0.19~0.20 |
| **③ 接触了但只关到 ajar** | 253 / 286 | contact_t=48，但 qT=0.059（差 9mm 到阈值 0.05） |
| ④ 走错位置（很远） | 210 | min_d_handle=0.556m，qT=0.200 |

而 **attn 把这 4 类统一成一种结果**：contact（ct=39~73）→ 关到 0.000~0.054。

**关键对照（种子 211，最能说明问题）**：
- vanilla：接近 12 步后 gripper 从 0→**-1.0（抓握）**，`drawer_qpos` 0.20→**0.282（拉得更开）** —— 它把"把手"当成了"往外拉的把手"。
- attn：仍抓握 (n_close=30)，但 `drawer_qpos` 0.20→**0.0（推关）** —— 方向正确。
- merge：n_close=0、min_d_handle=0.937m、qpos 停在 0.20 —— **根本没靠近**。

→ blocking 去掉抽屉区域视觉后，策略不再"细看把手"，而用 proprioception/全局上下文判方向 → 推关。
这直接反驳 "blocking 只是让瞄准更准（a）"：种子 211 vanilla 抓握了把手（瞄准没问题是方向错了），blocking 改的是**动作方向**不是瞄准精度。

---

## Q3 — 抓 vs 推？（接触/推动轨迹）

**答案：blocking 并没有统一"grasp→push"。** 策略在 per-seed 上基本**保留**，改的是"是否 contact + 方向 + 是否关到底"：

attn-rescue clean (n=11) 的 strategy 计数：
| 臂 | grasp | push_open | neutral |
|---|---|---|---|
| vanilla | 3 | 7 | 1 |
| attn | 4 | 2 | 5 |

- grasp 的 seed（211/248/260）attn 仍是 grasp；push 的 seed（226/270）attn 仍是 push。
- 所以不是"减少 handle 逼近 → panel 接触 → 推关"的整齐切换（用户假设 (b) 的字面形式）。
- 而是：**vanilla 的 grasp 会被"把手"误导成向外拉（①），vanilla 的 push 会因过度精细对齐而悬空不接触（②）或只关到 ajar（③）；blocking 后无论抓还是推，都落到"接触并关到底"。**

aggregate 佐证（clean，Δ = arm − vanilla，负值=更近）：
- attn: min_d_handle Δ=−0.038m，min_d_front Δ=−0.051m（略更近，但主要是"接触了"而非"更近"）
- merge: min_d_handle Δ=**+0.076m**，min_d_front Δ=**+0.082m**（明显更远）

---

## 机制：为什么 block 救它、collapse 害它？（回答 a/b/c）

同一个视觉区域，两种退化做的是**相反性质**的事：

- **Blocking = 移除**抽屉区域 token → action 由全局/proprioceptive 语境驱动 → 落成一条**粗粒度、鲁棒的"把眼前抽屉推关"**策略。
  它**救** close_drawer，因为 vanilla 的病根正是**对抽屉区域的过度精细视觉对齐**——表现为犹豫不接触、误判方向（拉把手）、或差几毫米留缝。移掉它 → 粗策略反而稳。
- **Collapse = 保留但畸变**（blur 成 centroid）→ 策略**仍做精细逼近，但空间精度被破坏** → 落点偏 3~8cm（min_d_handle +0.076m）→ 抽屉留缝（qT 0.058/0.111/0.124）或根本够不到（241: min_d 0.529m）。
  它**害** close_drawer，因为 vanilla 的成功恰恰依赖"把手/前面板在哪"的精细定位。

**这印证并精化了用户的核心直觉**：
> "Attention blocking isn't 'weakening detail is good' — it's changing the policy's OPERATION STRATEGY."

具体地说：blocking 不是"减少细节 → 更好"，而是**移除整块抽屉视觉 → 退回到一条粗操作策略**。这条粗策略在 close_drawer 恰好正确（推关就对了），在 move_near 却致命（那里需要 source→target 的 relational 视觉，block 掉就丢了关系）。

而 collapse 是**第三个范畴 (c)**：不是移除、不是切换策略，而是**把精细定位信号用错了值**——这解释了为什么 merge 在 close_drawer（localized，需要定位）harm，而在 move_near（relational，只需群质心）不受害甚至更稳。

---

## 边界与 caveat

1. **8 个 canonical-mismatch seed**（202, 214, 221, 223, 258, 281, 287, 294）：今天 re-capture 的 reset 与原始 run 不一致（close_drawer reset 会随机抽 top/middle/bottom 抽屉），其中 3 个在 attn-rescue、4 个在 merge-harm。以上结论基于 **clean 子集（11 attn-rescue + 6 merge-harm）**，这些是干净的、且信号高度一致（不存在"靠 mismatched seed 才成立"）。
2. **每 seed 异构**：vanilla 的失败模式有 4 类，blocking 的救法因此也是"统一到鲁棒推关"而非单一修一个 bug——这本身就是"策略级"而非"细节级"的证据。
3. 未做统计显著性检验（n=11/6 偏小）；结论是**机制层面**的定性判断，与 flip-matrix 的 per-seed lawful 反转一致。

---

## 数据文件

- `replay_close_drawer/seed_XXX.json`（36 seed × 3 arm 全轨迹）
- `replay_close_drawer/token_coverage.json`（handle/front 投影 cell vs 选中 cell）
- `replay_close_drawer/mechanism_analysis.json`（本报告的聚合量）
- `token_overlay/seed_*.png`（选中 token 的可视化叠加，6 个代表 seed）
- 分析脚本：`research/semantic_token_cd/analyze_mechanism.py`
