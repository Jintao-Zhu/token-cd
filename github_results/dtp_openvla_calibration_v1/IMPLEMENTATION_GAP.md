# DTP 复现实现差距审计（对论文 A.2 与外部审查的逐条核实）

审计对象：`research/semantic_token_cd/dtp_openvla_policy.py`（decode_dtp / decode_dtp_fixed）、
`dtp_closed_loop_policy.py`、`dtp_paper_calibration.py`。论文：arXiv 2601.16065 §3 + A.2。
状态：实现已核对；v2 固定mask路径已实现并完成 44-episode 机制审计（结论见下）。

## 与论文的差距（按影响，均已被代码/数据证实）

1. 检测不是发生在 “model's original action generation” 上。当前 dynamic 路径每维都在
   refined prefix（前几维已被剪过的 token）上重新检测 D_j，形成
   pruning→action→attention→D 的反馈；论文对 256-token 模型（SpatialVLA）是“首个 action
   token 决定一个 mask、整步复用”。代码：`decode_dtp` 循环内 `past=result.past_key_values`。
2. 7 维各自 mask + 共用 KV cache：q=0 触发时做整段 masked prefill（全序列行都失去被剪
   key），q>=1 只屏蔽新 query 行；且 cache 里的表示是在前一 mask 环境下算的。不同维看到
   的不是统一 F(V\D)。闭环数据：剪枝臂 3652 步中 43.5% 的 step 至少剪一次、dim0 占 20.4%，
   即 q0 masked-prefill 路径不罕见。
3. 冻结配置 L11/k64/τ0.5：保护 25% vs 论文 256-token 的 k=109（42.6%）。离线 60 状态按
   dim0 检测：k64 fire 12/60、k109 仅 2/60（L11/τ0.5）。k64 落在 G109\G64 保护带内的被剪
   token 60 状态共 14 个——不是“海量误剪”，而是“频率低但单点可能错”。论文自己的
   Random-unimportant-region 消融（WidowX SpatialVLA 17.7% vs DTP 37.5%）支持
   “只剪不重要区远不够、precision 决定成败”。
4. relevance 用单层 L11/L7，论文三种模型全部是多层集合 C 平均
   （SpatialVLA C={4,6}，Nora C={12,13,21}，UniVLA C={11,12}）。单层更易出偶然热点；
   “intervention 有用”不等于“适合当 protection”。
5. τ=0.5 由 SpatialVLA 直接迁移，未在 OpenVLA 上扫 τ。论文 §4.2 对 τ 的扫描是找
   “模型偏好 attention pattern”的主要手段。
6. 论文未指定项（corner 强度/范围、Gaussian 边界、cache 语义、层权重归一化顺序）——
   我们的选择记在 IMPLEMENTATION_CHOICES.json，除 cache 语义外预计影响较小。

## 可证伪检查结果（本轮已做）

- control≈vanilla（token 级）：适配器层 offline preflight 4 状态 clean_greedy_equal=True、
  max_abs=0；同一 rollout harness 内 control vs l11/l7 的 executed action 分叉点 = 首次
  “剪枝改变动作”的 step（move 4/4 一致）。control 与 canonical-vanilla 的整条轨迹在若干
  step 后才分叉，指向 canonical vanilla 与当前 harness 的微小管线差异，而非 wrapper bug；
  待用同 harness 重跑 1–2 个 vanilla episode 复核。
- attention hook：读取 `output[1]` 的最后 query 行×视觉列 1:257，head 均值；mask 后实测
  被剪位置 attention < 1e-7（decode 内断言）。与 `outputs.attentions[l]` 的逐元素比对尚未
  独立做，列入上线前清单。
- τ→∞ 极限尚未单独跑；enabled=False（等价 D=∅ 分支）已证实与 clean greedy 逐 token 相等。

## v2：论文近路径（已实现，已跑机制审计）

新增 arm：`v2_fix_k64_t05`、`v2_fix_k109_t05`（L11 relevance，τ0.5）。
语义（decode_dtp_fixed）：先 clean 自回归生成 7 维并记录 dim0 全层 attention →
用 dim0 的 Eq.4 pattern 与保护集判定唯一 D → 在该固定 D 下从 masked prefill 重解码整步
（一致视觉条件 F(V\D)）。与 dynamic 路径唯一对照变量：检测前缀与 mask 稳定性 + k。
离线 60 状态预估：k109 仅 2/60 触发 → 该臂接近 vanilla 的 sanity reference；
若 dynamic(k64) 有损而 fixed(k109) 恢复，即坐实“过度/不稳的动态剪枝”而非 DTP 失效。

## v2：机制审计闭环结果（已完成）

同一 22 个锁定校准场景 × {v2_fix_k64_t05, v2_fix_k109_t05} = 44 episode，全部 technical_pass。
成功数：control 10、dynamic-k64 6、fixed-k64 8、fixed-k109 7（open/close_drawer 各1场景全臂失败）。

配对结论（详见 REPORT.md 与 closed_loop/V2_AUDIT.json）：

1. fixed-k64 vs dynamic-k64 = +2（rescue 2 / harm 0，恢复 pick105、move_near 131）→ dynamic
   refined-prefix 检测 / cache 反馈是两臂间主要机制差异；论文式 fixed 路径行为更稳。
2. fixed-k109 vs fixed-k64 = −1；k109 闭环触发仅 62/1826 dim0 步（3.4%）。扩保护区到 k109
   基本等价于关闭剪枝，未恢复性能 → “保护区太窄”假设在本数据上不被支持。
3. 两档固定臂（8、7）仍低于 control（10）。触发边缘统计支持用户机制猜想（刀口触发：k64 触发步
   41% 越界裕度 ≤1.1×阈值、226/292 只剪 1 token），但这些触发以扰动为主，未见稳定收益。

上线前 sanity 也已完成并通过（当前harness vanilla 与 control 逐动作 exact match；
hook 读数与 outputs.attentions 逐元素 max diff = 0.0），见 REPORT.md 与 sanity/。

## 结论边界

正式 400 已封口（400/400）：冻结 l11_k64_t05 vs vanilla 167 vs 168，无整体增益、任务分化。
v2 审计表明：实现语义差距中最有行为证据的是“dynamic refined-prefix 检测”而非保护区宽度；
但 fixed-k64/k109 在 22 个校准场景上仍低于 control。因此当前只能说“在本模型/任务集上，
L11/τ0.5 下的剪枝在闭环中成功再分配、无稳定净收益”；“τ/层集合/attention 几何本身是否适配
DTP”仍是未决项。0–99 已被多次使用，不算未见测试；v2 若进正式 400 需预先注册。
