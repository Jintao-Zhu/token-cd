# VLA-Pruner 复现 — IMPLEMENTATION_GAP（官方 vs 我们的改动逐条登记）

状态：**源码审计完成（2026-09-09），移植未开始**。原则：最大程度复用官方语义；
凡改动必须逐条登记本表；`enabled=False/fastv_r=0` 必须逐 token 等价 Vanilla，否则停下查实现。

## A. 语义确认（不改动，仅登记）
| # | 项目 | 官方源码事实 | 出处 |
|---|---|---|---|
| A1 | fastv_r 语义 | 删除比例；keep = round(256*(1-r)) | `modeling_llama.py` :1102 |
| A2 | 剪枝时机 | 仅 prefill 一次，layer k（默认 3）进入前，用 layer k-1 attention | `fastv_forward` :1072-1127 |
| A3 | 真删除 | `hidden_states = hidden_states[:, keep, :]`，非 attention mask 屏蔽 | :1127 |
| A4 | 同步更新 | position_ids=keep（原 RoPE）、重建 causal mask、cache_position 截断 | :1128-1147 |
| A5 | 时间历史构造 | layer15 action→vision，head 均值→action token 均值，deque(maxlen=3), γ=0.8, 权重 γ^i 作用于最近第 i 步 | `modeling_prismatic.py` :615-619, 567-578 |
| A6 | 历史启动 | 只有 len(av_hist)==maxlen 才用 Temporal_Guide；之前 r 置 0（不剪） | :573-602 |
| A7 | episode reset | 每 episode 开头 `reset_av_history()` | `run_libero_eval.py` :163 |
| A8 | combine-then-filter | guide_topk ∪ current_topk，超预算用 embed 特征余弦去冗余裁回 | `modeling_llama.py` :1108-1116, `_redundancy_minization` :950 |
| A9 | 论文主方法 | prefill 准则 + temporal（`*_prefil.sh`） | README_VLA_Pruner.md |
| A10 | decode 不剪 | seq==1 分支跳过剪枝，仅续写 KV | :1073 |

## B. 必须改动的移植差异
| # | 官方 | 我们改为 | 为什么 | 是否变语义 |
|---|---|---|---|---|
| B1 | transformers 4.39.0.dev0 `LlamaModel.fastv_forward` | 在 4.40.1 `LlamaModel.forward` 上实现等价的 fastv 分支（同一实例 method 注入） | 我们 venv 是 4.40.1，无官方 monkey 版；剪枝必须发生在 generate 内部 prefill | 语义同；仅 `_update_causal_mask` 换 4.40 签名（多一个 cache_position 实参） |
| B2 | prismatic 层负责 `use_fastv`/`av_hist` 状态 | policy 层（`VlaPrunerClosedLoopInference`）维护 fastv 配置与 av_hist | checkpoint 本地 modeling_prismatic.py 为标准版，trust_remote_code 优先加载本地 | 不变；history 更新频率与官方 predict_action 一致（每 env step 一次） |
| B3 | 官方评测逐 step 由 prismatic `predict_action` 读 `results.attentions` 还原坐标 | 我们 `_forward_scores` 只开 output_scores；pruner arm 改开 output_attentions 并按 A5 还原 | harness 结构差异；还原函数体对齐 `_extract_action_modality_attentions` :640 | 不变（同一还原逻辑） |
| B4 | LIBERO 评测 loop | SIMPLER google_robot closed-loop（已审计 harness） | 目标环境 | domain shift 属已知差异（记录于 SUMMARY），非实现偏差 |
| B5 | 每 env step 后处理 | google_robot + sticky gripper（与 vanilla/control 完全一致） | 变量控制 | 不变 |

## C. 风险与验证门
| # | 风险 | 验证门（不通过则停下） |
|---|---|---|
| C1 | 4.40.1 移植引入数值偏差 | `fastv_r=0` 与 vanilla 逐 token exact match（seed 覆盖早中晚各态） |
| C2 | 剪枝坐标/历史错位 | 离线 20-60 状态记录 keep-set、保留/剪除数、layer15 向量形状与覆盖 [1:257] |
| C3 | 无实际删除（退化成 mask/空跑） | 记录每层 seq_len 变化，确认 layer>=k 的序列长度 = 保留数 |
| C4 | temporal 语义错误 | 前 3 step 不剪；第 4 step 起 guide 来自上一步 layer15 历史（打印 overlap） |
| C5 | 非法输出/NaN | 每个 action token 落合法区间，无 NaN/inf（沿用现有 audit 逻辑） |

## D. 已知客观差异（非实现缺陷，需在结果解释中保持）
- 官方 LIBERO 使用 libero-finetuned checkpoint；我们使用 base openvla-7b + SIMPLER（现有所有对照同此）。
- 官方在 LIBERO 有每 step sequence padding/长指令；SIMPLER 无 padding，指令短。prefill 全行均值准则不受影响。

## E. 移植实现修订记录（2026-09-09 r0 阶段，均已实证）
| # | 发现/修复 | 现象 | 处理 | 验证 |
|---|---|---|---|---|
| E1 | FastV 分支未按 vanilla 把输入 cache 转 `DynamicCache` | 层内 `use_cache=True` 但 `past_key_value=None` → 每步全序列重算（q_len 264→270 方阵），cache 从不累积，`pruning_info` 被最后一 decode 覆盖成 270 | 剪枝分支前照抄 4.40 vanilla：`DynamicCache.from_legacy_cache(past)` 得 `past_seen_tokens`；decode（cache 非空）仍先 delegate 到 vanilla | r0 逐 token==vanilla；decode attention 恢复 `[1,32,1,265..270]` |
| E2 | layer k 剪枝后，后续层仍用全长 `causal_mask` | prune25 崩溃 `attn_weights[...,200,...] + mask[...,264,...]` | 剪枝后保留 `pruned_mask` 给 layer>=k（此前 `elif` 分支错误回退 `causal_mask`） | prune25/50 跑通，layer>=k prefill 注意 [1,32,200/136,…] |
| E3 | 官方“还原大方阵再切片”的 attention 还原在 4.40 结构下无法 broadcast | `idx_r/idx_c`=keep 全集而 pruned 层 decode kv 仅保留集+历史 | 重写 `extract_action_vision_attentions`：按几何直接取 action 行；layer<k 用原 1:257 列，layer>=k 把 kept 视觉列按 `keep_indices` 散布回原 256 位，被剪视觉列=0（与模型实际所见一致） | 与官方语义等价；r0/剪枝层形状均验证 |
| E4 | policy 对象 `copy.copy(base)` 共享同一模型实例 | vanilla arm 在 r0 arm 之后会残留 FastV class/cfg | vanilla step 前 `detach_fastv`；pruner arm generate 后立即 `detach`；每 episode `reset()` 清空 `av_hist` | r0 sanity step/episode 双模式全过 |
| E5 | episode 双臂比较必须先各自 restore 物理初态 | `run_loop` 会 `env.step` 改变环境 | `run_episode_compare` 每臂前重放同一 snapshot | 双臂长度/成功/token/动作逐位一致 |
| E6 | r0 因果 mask 重建与 vanilla 的位一致性 | 避免 padding 语义漂移 | 重建 mask 时用 `attention_mask[:, keep]`（keep=all 时即 vanilla 输入），非官方 `None` | r0 `max_abs=0` |
