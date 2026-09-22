# DTP 在 OpenVLA 上的论文复现与离线校准

## 当前完成度

已实现论文方法的 OpenVLA 适配、完成60个同状态样本的全层评分和剪枝统计、通过真实剪枝/自回归重生成数值检查，并完成小规模闭环校准、正式400 episode（固定0–99基准）对照、两项上线前 sanity，以及 v2 固定mask机制审计（44 episode）。协议主体四步走完：60状态离线 → 3臂闭环校准 → 冻结统一配置 → 正式400；此后按外部审查结论补充了“当前harness vanilla 一致性”与“hook 逐元素审计”两项 sanity，并完成论文 A.2 SpatialVLA 式 v2（clean dim0 → 单一固定mask）的 k64/k109 机制审计。最终结论：正式400冻结配置 l11_k64_t05 与 vanilla 基本持平（167 vs 168）；v2 审计显示“fixed 单mask 优于 dynamic 逐维 refined-prefix 检测”（配对+2），但 k64/k109 两档固定臂在22个校准场景上仍低于 control，保护区扩大并非恢复性能的杠杆。详见下文及 IMPLEMENTATION_GAP.md。

实现依据：https://arxiv.org/html/2601.16065v1 ，§3及附录A.2。官方匿名仓库过期，不能宣称与作者源码逐行一致。

## 已实现的流程

1. 使用实际完整指令、32层逐层 prompt attention、全部head均值；沿用经检查的视觉key位置1–256和指令query位置。
2. 在相关性评分上构造Top-k保护集合。论文未明确的空间处理采用：仅四角token分数置零，再以sigma=.65（patch单位）、reflect边界、truncate=4做Gaussian smoothing；另保留raw/corner-only/Gaussian-only诊断。
3. 每个动作维度单独读取全32层生成attention。层权重为该层分配给视觉token的softmax质量，跨层归一化后对原始视觉attention加权；不预先把每层视觉区间归一化。
4. 对保护区外满足 A[v] > tau*max(A[G]) 的位置实施attention key屏蔽，重新生成当前动作token；剪枝位置attention实测为零。没有harmonic reconstruction，也没有SHR的logit contrast。
5. 下一维使用本次重生成的动作作为prefix。探测与重生成使用各自复制的cache，防止探测污染。此前缓存状态保留其生成时的mask；这属于论文未明确的实现选择。
6. 共生成7个合法action token，包括gripper；论文未规定OpenVLA需要保留clean gripper，故未引入SHR的前6维限制。

源码：research/semantic_token_cd/dtp_openvla_policy.py 与 dtp_paper_calibration.py。

## 重要数值审计

- 60/60状态：开启原始逐token生成attention记录后，clean action-bin logits逐位相同，最大差值0。
- 60/60状态：重提取prompt全层评分与缓存相同，最大差值0。
- 45状态可与已有clean logits对照；15个close_drawer缓存未保存该字段，不虚报历史logit审计，但进行了当次记录前后对照。
- 初版一次性teacher-forced读出在首个状态发生动作翻转（最大logit差.671875），因此未用于本次剪枝评分。保留AUDIT_FAILURE.json及日志；后续改用实际带cache生成读出，结果通过审计。
- 4状态无剪枝自回归适配器与原始生成完全一致，最大logit差0。
- 4状态×2候选实际剪枝输出均合法、有限；遮蔽位置attention为零。选的是每任务中离线剪枝最多的状态，用于检查功能生效，不能当作随机样本推断成败。

## 相关性层筛查

复用四任务各5episode×3阶段，共60状态。已有明确新旧目标参考区域的指令切换对照为9个、覆盖3任务；close_drawer缺少相应对照，不能宣称其任务响应已得到验证。

预先以目标切换响应为正、至少60%对照响应方向为正、同义改写mask Jaccard>=.5为入口，再按跨任务平均目标响应选择最多2层，得到L11、L7。层间均值与排名一致性保存在layer_per_state.csv、layer_summary.csv、layer_rank_correlation.npy。

这些是探索候选，参考框为已有近似区域，不是正确剪枝集合的真值。固定k=109用于层响应检查；层选定后才计算预算/阈值组合。没有用成功率选层。

## 剪枝数量：主要发现是大量配置几乎不生效

下表为60状态×7 clean-prefix动作查询的离线结果；每状态只要任一维有剪枝即算触发。数值不能代替沿着剪枝后prefix重新计算的真实闭环分布。

| 层 | 保护k | tau | 每维平均剪枝数 | 完全不剪的状态 |
|---|---:|---:|---:|---:|
| L11 | 64 | 0.5 | 0.2714 | 40/60 |
| L11 | 64 | 1.0 | 0.0119 | 56/60 |
| L11 | 109 | 0.5 | 0.0357 | 54/60 |
| L11 | 109 | 1.0 | 0 | 60/60 |
| L11 | 154 | 0.5 | 0.0095 | 59/60 |
| L7 | 64 | 0.5 | 0.2476 | 40/60 |
| L7 | 109 | 0.5 | 0.0690 | 51/60 |
| L7 | 109 | 1.0 | 0.0024 | 59/60 |

解释：当保护集已经包含动作attention的最高峰时，区外token很难超过保护区内最大值。大保护集和较高tau经常使DTP退化为原策略。该结论限于本次论文重实现及样本，不能推广到作者实现或全部OpenVLA轨迹。

两个值得保留的暂定闭环候选：L11/k64/tau.5、L7/k64/tau.5。均不能称为优选成功率配置。

实际剪枝预检查还表明：很少token也能改变多个动作维度。move_near示例两候选都改变7维，open_drawer的L11改变6维。动作变化包括autoregressive prefix传播，不能解释成7个相互独立的局部效应，也不能据此宣布Harm。

## 校准场景隔离遇到的限制

按照用户要求，校准不能使用与正式0–99相同的物理初态。除字节hash外，以相同指令、sim_state绝对差<=1e-6（rtol=0）做保守排重，并逐一与全部前100状态比较，避免浮点微差制造假新场景。

| 任务 | 100–299中可用且互不重复的初态 |
|---|---:|
| open_drawer | 1（seed118） |
| close_drawer | 1（seed118） |
| pick_coke_can | 200 |
| move_near | 23 |

完整记录见CALIBRATION_SCENE_AUDIT.json。该规则保守排除相同物理场景，即使controller/RNG或微小模拟噪声不同也不把它算成独立场景；实际闭环仍需agent状态、环境配置和RGB恢复审计。

两个drawer任务现有样本不足以支撑跨任务统一配置的独立小规模校准。需要另外生成并锁定未与正式集重复的物理初态，或者明确接受校准集重叠。前者涉及扩展现有canonical场景范围，当前未自动改变环境分布或用重复seed凑数。

## 文件

- SAMPLES.json：60状态与来源/hash。
- IMPLEMENTATION_CHOICES.json：实现选择；后续修订见IMPLEMENTATION_UPDATE.json。
- layer_diagnostics.png、layer_summary.csv：逐层诊断。
- figures/：每状态每候选层的原始评分、空间处理后评分、保护/剪枝叠加图，共120张。
- pruning_summary.csv、pruning_per_dimension.csv：完整数量分布和逐维token索引。
- spatial_changes.csv：空间处理造成的集合变化。
- action_attention/：60状态的32×7×256原始attention、权重和审计。
- PREFLIGHT.json、preflight/：真实剪枝重生成检查。
- CALIBRATION_SCENE_AUDIT.json：物理初态去重与候选seed。
- sanity/：当前harness vanilla 重跑、compare、hook 逐元素审计及 COMPARE_RESULTS/HOOK_AUDIT。
- closed_loop/V2_AUDIT.json：v2 固定mask机制审计全量汇总；v2 episode summary 内含 `trace[].dim0_trace` 机制字段。

论文实现与离线/闭环校准证据见上文。正式400成功率、sanity、v2 审计见下文。实现差距与外部审查核实见 IMPLEMENTATION_GAP.md。

## 正式 400（已完成，400/400）

seeds 0–99 × 4 任务全部完成（`formal_400/episodes`），逐 episode 审计通过（technical_pass、canonical/state/rgb hash 完整）。对照 vanilla_recon_shr_canonical_0_299_v2 中同 seed 的 vanilla 固定基线（同一物理快照）。

| 任务 | l11_k64_t05 | vanilla | delta | rescue | harm | unchanged |
|---|---:|---:|---:|---:|---:|---:|
| open_drawer | 26/100 | 27/100 | −1 | 13 | 14 | 73 |
| close_drawer | 52/100 | 56/100 | −4 | 16 | 20 | 64 |
| pick_coke_can | 33/100 | 23/100 | +10 | 17 | 7 | 76 |
| move_near | 56/100 | 62/100 | −6 | 9 | 15 | 76 |
| 合计 | 167/400 | 168/400 | −1 | 55 | 56 | 289 |

- 整体无增益（167 vs 168），rescue/harm 对称（55/56），与闭环校准“剪枝净效应为负”一致但幅度更小。
- 任务方向分化稳定：pick_coke_can 是唯一持续正收益任务（+10，rescue 17 vs harm 7）；move_near（−6）与 close_drawer（−4）偏负。
- 该结果是“单层 L11/k64/τ0.5 + 逐维 refined-prefix 动态剪枝”这一迁移配置在固定基准上的验证，不是对论文原方法（多层C集合 + 整步固定mask + k109）的判定；见 IMPLEMENTATION_GAP.md。
- 全量结构化结果：`formal_400/FORMAL_RESULTS.json`、`formal_400/formal_summary.csv`。

## 小规模闭环校准（已完成，66 episode）

22 个去重校准场景（open/close_drawer 各仅1个、pick_coke_can 10个、move_near 10个）× control/l11_k64_t05/l7_k64_t05 三臂，共66 episode。所有 episode 审计通过（`technical_pass`、control 从不剪枝、RGB/state hash 完整），control=同一 DTP 适配器但禁剪枝，隔离适配器代码路径影响。

| 臂 | 成功 | 配对 rescue | 配对 harm | net |
|---|---:|---:|---:|---:|
| control（禁剪枝） | 10/22 | - | - | - |
| l11_k64_t05 | 6/22 | 0 | 4 | -4 |
| l7_k64_t05 | 7/22 | 1 | 4 | -3 |

- 分任务：pick_coke_can control 3/10、l11 1/10、l7 2/10；move_near control 7/10、l11 5/10、l7 5/10；两个 drawer 场景三臂全部失败。
- 两候选均未在成功数上超过 control；配对净效应为负。结论限于22个校准场景（其中 drawer 各只有1个物理新场景），不能推广。
- 预注册决策规则：候选未超 control 时冻结离线首选 L11，故 `closed_loop/CONFIG_LOCK.json` 锁定 `l11_k64_t05`（layer11/k64/tau0.5）。正式 400 用该统一配置跑固定 0–99 基准并对照已有 vanilla 0–99，属固定基准验证。
- 44 个 control×候选 配对视频在 `closed_loop/paired_videos/`。

## 上线前 sanity（已完成，通过）

在启动 v2 前按用户要求补两个低成本检查，脚本 `research/semantic_token_cd/dtp_sanity_checks.py`，结果存 `sanity/`。

1. 当前 harness 下重跑 vanilla：以同一物理快照、同一 rollout harness/循环（AuditedVanillaInference，非 DTP 适配器）重跑 move_near 101/112、pick_coke_can 100、open_drawer 118，与 DTP control 逐 executed action 比对：4/4 场景长度相等且逐行 exact match。结论：control（DTP 适配器禁剪枝）与当前 harness vanilla 完全等价，DTP wrapper 在禁剪枝时不引入任何动作差异。
2. hook vs outputs.attentions 逐元素比对：3 个离线状态 × {clean, 屏蔽12个key} 共6组，32层逐层比较 hook 读到的 `output[1]` 最后query行×视觉1:257（head均值）与 `outputs.attentions[l]` 同切片，max abs diff 全为 0.0。结论：attention hook 读数与模型权威 attention 一致，剪枝读数无底层风险。

附带澄清旧记录差异：move_near 101/112 的 control 与早期 canonical-vanilla 录制的分叉（step 7/3）在本轮重跑的 harness-vanilla 上同样出现（harness-vanilla vs canonical 同样 step 7/3 分叉），说明差异来自旧 harness 管线（快照/环境/相机路径），与 DTP 适配器无关；open_drawer 118、pick 100 三个方向全部 exact match。

## v2 固定mask机制审计（已完成，44 episode + 2 sanity）

目的（用户冻结，不调参）：用 `clean dim0 → 单一固定mask → 整步 masked prefill 重解码`（decode_dtp_fixed，论文 A.2 SpatialVLA 语义）回答两个机制问题：

- `fixed-k64 vs dynamic-k64`：refined-prefix 动态检测 / cache 反馈是否主要问题？
- `fixed-k64 vs fixed-k109`：保护范围太窄是否主要问题？

同一 22 个锁定校准场景 × 2 新臂 = 44 episode，全部 technical_pass（无非法/非有限动作，trace 含 mode 校验）。超参冻结 layer11/k64/k109、τ0.5。

| 任务 | control | l11_k64_t05(dynamic) | v2_fix_k64_t05 | v2_fix_k109_t05 |
|---|---:|---:|---:|---:|
| open_drawer | 0/1 | 0/1 | 0/1 | 0/1 |
| close_drawer | 0/1 | 0/1 | 0/1 | 0/1 |
| pick_coke_can | 3/10 | 1/10 | 2/10 | 1/10 |
| move_near | 7/10 | 5/10 | 6/10 | 6/10 |
| 合计 | 10/22 | 6/22 | 8/22 | 7/22 |

配对（rescue/harm/net，相对基准臂）：

| 对比 | rescue | harm | net |
|---|---:|---:|---:|
| dynamic-k64 vs control | 0 | 4 | −4 |
| fixed-k64 vs control | 1 | 3 | −2 |
| fixed-k109 vs control | 0 | 3 | −3 |
| fixed-k64 vs dynamic-k64 | 2 | 0 | +2 |
| fixed-k109 vs fixed-k64 | 1 | 2 | −1 |

dim0 触发统计与触发边缘：

| 臂 | 触发 dim0 步/总步 | 触发时 \|D\|（1/2/3/4+ token 的步数） | 触发时 a_m 均值 | 未触发时 a_m 均值 | 越界裕度（max A_unimp/(τ·a_m)） |
|---|---:|---|---:|---:|---:|
| v2_fix_k64_t05 | 292/1826（16.0%） | 226/59/6/1 | 0.0100 | 0.0171 | 41% 落在 (1,1.1] |
| v2_fix_k109_t05 | 62/1826（3.4%） | 42/20/0/0 | 0.0098 | 0.0172 | 19% 落在 (1,1.1] |

对照预注册判定：

1. `fixed-k64 > dynamic-k64`（8 vs 6，配对 net +2，恢复 pick105、move_near 131 两场景，0 次新增失败）→ **refined-prefix 动态检测 / cache 反馈是两臂间最主要的机制差异**，论文的“clean 首个动作token决定整步mask”路径在行为上更稳。
2. `fixed-k109` 未继续超过 `fixed-k64`（7 vs 8，net −1），且 k109 在闭环中仅 3.4% dim0 步触发（与离线 60 状态 2/60 一致）→ **保护区从 k64 扩到 k109 基本只是关闭剪枝，不是恢复性能的杠杆**；“保护范围太窄”假设在本数据上不被支持。
3. 两档固定臂仍低于 control（8、7 vs 10）→ 在22个校准场景上任何剪枝形态都未超过 control。逐臂 harm 场景：dynamic-k64 = pick102/107、move131/135；fixed-k64 = pick102/107、move135；fixed-k109 = pick107/109、move135——都是“control 获胜而剪枝臂在该场景触发过剪枝”的场次，与正式400“整体持平、成功在场景间再分配”一致。n=22、drawer 各1场景全臂失败，统计功效有限，该结论不宜外推为“DTP 失效”。
4. 触发边缘：k64 触发步的越界裕度 41% 落在阈值 1.1 倍以内、226/292 只剪 1 个 token → 与用户机制猜想一致：少数暴露 token 的 attention 是否跨过 τ·a_m 就决定了该步剪不剪；但这些判定多属刀口触发，行为上以扰动为主，未见稳定收益。

完整结构化数据：`closed_loop/V2_AUDIT.json`；每步 dim0 机制字段（\|D\|、a_m、τ·a_m、max A_unimportant、Σ_{v∈D}A[v]）已随 v2 episode 写入各自 `trace[].dim0_trace`。

## 当前状态

正式 400 已封口（400/400，l11_k64_t05 vs vanilla 167 vs 168）。两项 sanity 通过；v2 固定mask机制审计完成（44/44）：fixed-k64 在配对层面优于 dynamic-k64（+2），但两档固定臂在22个校准场景上均未超 control，k109 近乎关闭剪枝而未带来收益。按用户预先设定的判定，当前证据指向“dynamic/refined-prefix/cache 反馈是主要实现语义差距；保护区宽度不是主要杠杆”。是否让 fixed 路径进入新一轮正式 400（需预先注册、属新基准消耗）或先做 τ/relevance 几何检查，待用户决定。
