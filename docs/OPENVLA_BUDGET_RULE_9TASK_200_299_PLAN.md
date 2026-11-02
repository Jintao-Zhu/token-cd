# OpenVLA 预算策略对照实验（Matched vs Attention-TopP vs Fixed）· 计划与协议锁

*创建：2026-10-14  ·  协议 id：`OPENVLA_BUDGET_RULE_9TASK_200_299_V1`*

## 0. 要回答的问题（对应论文 IV-E）

> Does KMeans-Matched Budgeting Provide an Effective Intervention Scale Compared with
> Attention-Based and Fixed-Size Alternatives?

只改**每步预算 `m_t` 的确定方式**，其余全部锁死在现有 Ours 协议上。

## 1. 实验规模

| 预算策略 | 配置 | 组数 | Episodes |
|---|---|---|---|
| Matched（Ours，复用） | 原始 KMeans entity-matched | 1 | 900 |
| Attention-TopP（无截断） | p = 0.75 / 0.80 / 0.85 / 0.90 | 4 | 3,600 |
| Fixed Budget | m = 16 / 24 / 32 / 40 / 48 / 56 / 64 | 7 | 6,300 |
| **合计** | | **12** | **10,800**（新增 9,900） |

每组 = 9 个 SIMPLER 任务 × 100 seeds（**200–299**）。

## 2. 协议锁（与现有 Ours 完全一致，只放开预算）

| 项 | 值 | 证据字段（逐集落盘） |
|---|---|---|
| Backbone / 权重 | OpenVLA Fractal SIMPLER checkpoint | `model` / CONFIG_LOCK |
| Guidance strength | **λ = 1.0** | `lambda`, `lambda_exact` |
| Ranking | L11 prompt-attention | `attention_layers`, `layer_exact` |
| Counterfactual | Harmonic reconstruction | `counterfactual_constructor`, `constructor_exact` |
| Execution | contrast | `guidance_execution`, `execution_exact` |
| Intervention | projector | `intervention_location`(=projector) |
| 动作解码 | 与 Ours 完全一致（同一 `run_episode` / canonical decoder） | 同脚本路径 |
| RT 模式 | open_drawer / close_drawer = **rt-full depth 8, spp 32, denoiser**；其余 7 任务 = **depth 0** | `rendering`（逐集） |
| Canonical 状态 | `artifacts/vanilla_recon_shr_canonical_0_299_v2` | `canonical_snapshot_sha256`, `initial_state_sha256`, `initial_rgb_sha256` |
| Seeds | 200–299（与 Ours 复用集同一批） | `seed` |

**变化变量**：仅 `m_t` 的确定方法。

## 3. 三种预算规则

* **matched**：own-state KMeans entity-matched cluster size（Ours，均值 m̄ ≈ 36.7 tokens/step）。
* **top_p（无截断）**：`m_t = min{ k : Σ_{i≤k} ã_(π_i) ≥ p }`，`p ∈ {0.75, 0.80, 0.85, 0.90}`。
  * 现有 `prompt_attn_shr_policy` 会把 TopP 计数 clip 到 `[min_count, max_count]`；本实验把边界设为
    `[1, N_VISUAL=256]`，即 clip 失效，逐集 trace 记录 `top_p_threshold / selection_min_count /
    selection_max_count / m_raw / lower_bound_triggered / upper_bound_triggered`，可证明"无截断"。
* **fixed**：`m_t = m`，恒定，`m ∈ {16,24,32,40,48,56,64}`；逐集 trace 断言 `m_t == m`。

三类规则**都从同一个 L11 prompt-attention 排序取 top-`m_t`**，因此只改数量、不改选择准则。

## 4. 代码改动（相对 λ=1.0 Ours 路径）

1. **新脚本** `research/semantic_token_cd/openvla_budget_rule_rollout.py`：以
   `openvla_global_lambda_grid_rollout.py`（产出 Ours 的脚本）为母本，仅把预算规则的三个分支
   参数化；env、snapshot 恢复、paired-reference 校验、audit、输出 schema、arrays 落盘全部沿用。
2. **policy 白名单** `prompt_attn_shr_policy.py`：固定预算的合法集合从 `{16,24,32,48,64}` 扩到
   `{16,24,32,40,48,56,64}`（只为支持 40/56；对 matched/top_p 路径无影响）。
3. 逐集新增可读字段：`budget_rule`、`budget_param`、`min/max_selected_token_count`；`selector`
   字符串按规则动态生成。audit 增加 `fixed_budget_exact`、`top_p_threshold_exact`、
   `top_p_unclipped_bounds`、`top_p_no_clip_triggered`。

## 5. 复用审计（Matched 900 集，不重跑）

`artifacts/openvla_lambda1_extension_200_299_v1`：9 任务 × seeds 200–299 × 900 集，
`model=OpenVLA Fractal SIMPLER`、`lambda=1.0`、`attention_layers=[11]`、
`constructor=harmonic`、`execution=contrast`、`intervention=projector`、
`selector="L11 prompt attention + canonical dynamic Matched budget"`、
逐任务 RT 模式与上表一致，全部 `technical_pass`，且逐集带
`canonical_snapshot_sha256 / initial_state_sha256 / initial_rgb_sha256` → **可严格同状态配对**。

参考基线成绩：**281/900 = 31.2%**（open_drawer 40/100、close_drawer 84/100、move_near 48/100、
pick_coke_can 29/100、carrot 31/100、eggplant 27/100、spoon 14/100、stack 8/100、place_apple 0/100）。

## 6. 冒烟（1 任务 × 20 seeds × 3 臂）

见 `artifacts/openvla_budget_rule_smoke_v1/VERIFY.md`：matched 复现检查（与 Ours 逐字段一致）、
TopP 无截断边界检查、Fixed 恒定性检查。

## 7. 运行计划（GPU 1/2/3 × 4 worker = 12）

* `scripts/plan_openvla_budget_rule_run.py --write` →
  `artifacts/openvla_budget_rule_9task_200_299_v1/WORK_PLAN.json`
* 396 个 chunk（arm × task × 25-seed 块），LPT 打包到 12 个 worker（每个 33 chunk / 825 集）。
* 成本模型：每任务单集秒数取 Ours 实测（177 s place_apple … 45 s spoon），大预算臂乘 0.8–1.3
  的**配平系数**（仅用于负载均衡，不用于报告）。
* 预估：**284.6 worker-hours → 墙钟 ≈ 23.7 h**（4 worker/卡与 3 worker/卡的争用差异可能让实际落在 24–30 h）。
* 存储：Ours 900 集 = 1.6 GB → 9,900 集 ≈ 18 GB（磁盘余 25 TB）。
* 启动：`bash scripts/launch_openvla_budget_rule_tmux.sh`；监控：`scripts/monitor_openvla_budget_rule.sh`。

## 8. 统计分析（预先登记）

1. **配对**：每个新臂与 Matched 在同一 (task, seed) 上配对，报告成功率差、rescue/harm/net 与
   **exact McNemar**。
2. **多重比较**：11 个比较做 **Holm 校正**（`scripts/analyze_openvla_budget_rule_study.py`）。
3. **不做测试集选优**：报告全部 11 条结果；可以指出"表现最好的配置"，但明确标注为测试集观察。
4. **族内趋势**：Fixed（16→64）与 TopP（0.75→0.90）各做 **Cochran–Armitage 趋势检验**，
   比单点最优更难被质疑。
5. **预算分布必报**：每臂 mean / median / min / max 的 `m_t`，以及 success–budget 曲线
   （Figure 6c：横轴实测平均预算，纵轴成功率，Matched 作为单点参考）。
6. **论文呈现**：Figure 6(a) TopP 阈值 vs 九任务平均 SR（Matched 水平虚线）；6(b) Fixed 规模 vs SR；
   6(c) SR vs 实际 m̄。逐任务表进 Supplementary。

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| 4 worker/卡显存（每 worker ≈ 18 GB → 4× ≈ 72 GB/80 GB） | 启动后 10 分钟内盯 `nvidia-smi`；若 OOM 立即退回 3 worker/卡（9 worker，墙钟 ≈ 31 h） |
| Vulkan `DEVICE_LOST` | worker 脚本对技术性失败重试 1 次，配置不变 |
| 模型重复加载 | 每个 chunk 一次加载（396 次，聚合 ≈ 10 h/12 worker ≈ 3.4% 开销），可接受 |
| 中断/续跑 | 脚本按"summary+arrays 已存在"跳过，`CONFIG_LOCK` 只校验共享字段、合并 arm 字段 |

---

## 10. 更新（2026-10-14 13:20）：切块必须对齐参考集 —— 冒烟发现进程历史效应

冒烟（`artifacts/openvla_budget_rule_smoke_v1`）在技术层面全部通过：

| 检查 | 结果 |
|---|---|
| Fixed-32 恒定性 | 每步 `m_t = 32` 精确（`fixed_budget_exact`） |
| TopP-0.85 无截断 | `top_p_threshold=0.85`、`selection_min/max=1/256`、`lower/upper_bound_triggered=False`、`coverage_exact=True` |
| matched 臂 = Ours 参考集 | **前 11 个 seed 逐位一致**，但 seed 211 起分叉 |

追查后确认：**结果依赖进程内已跑过的 episode 数**（详见
`docs/OPENVLA_BUDGET_RULE_REPRODUCIBILITY_FINDING.md`）。关键证据：

| 运行 | seed 211 结果 | 与参考集 |
|---|---|---|
| 参考集（该 chunk 的第 1 集，新进程） | succ=True, m_t=36.25 | — |
| 冒烟 smoke1（一个进程连续 200–219，第 12 集） | succ=False, m_t=39.93 | 不一致（step 1 起分叉） |
| 冒烟 smoke2（**新进程只跑 211–220**，第 1 集） | succ=True, m_t=36.25 | **逐位一致（211–218 全同）** |

**因此正式 run 改为对齐参考集的 per-process 切块**：每任务 `[200]`、`[201-210]`、`[211-220]` …
`[291-299]`（= `--chunk-mode reference`），共 **1,089 次调用 / 9,900 集**，12 worker（GPU1/2/3 × 4）。
切块对齐后，历史 900 集 matched 参考**可继续复用**（smoke2 已证明同结构新进程可逐位复现）。

额外开销：调用次数从 396 增到 1,089，模型加载聚合 ≈ +22 h / 12 worker ≈ **+1.9 h 墙钟**，
故预估总时长 **≈ 26 h**（纯 episode 计算 23.7 h）。

启动时间：2026-10-14 13:21，12 个 tmux 会话 `br_openvla_br_g{1,2,3}_w{1..4}`；
显存 71.4 / 69.7 / 36.7 GB（上限 81.6，需盯 OOM，必要时退回 3 worker/卡）。
