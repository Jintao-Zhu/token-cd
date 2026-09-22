# VLA-Pruner (OpenVLA) 官方源码分析 — 决定移植语义与缺口清单

- 仓库：`/home/leju-suzhou/zjt_ws/VLA-Pruner`（`MINT-SJTU/VLA-Pruner`，commit `84d4b71`）
- 作用：作为 OpenVLA 上“真删除型视觉 token 剪枝 + 动作注意力时间引导”的**独立实现**，
  回答此前 DTP 掩码式复现未回答的问题：同 checkpoint/harness 下 VLA-Pruner-only 是否不劣于 Vanilla。
- 本文件结论经逐行核对官方源码，行号以 clone 内文件为准。

---

## 1. 官方实现地图（本回合逐行核对）

| 文件 | 关键位置 | 内容 |
|---|---|---|
| `src/openvla/transformers/src/transformers/models/llama/modeling_llama.py` | `LlamaModel._redundancy_minization` :950 | 余弦去冗余贪心选择（保留恰好 num_keep） |
| 同文件 | `LlamaModel.fastv_forward` :973 | 前向内单次“边界压缩”的真删除剪枝 |
| 同文件 | `LlamaForCausalLM.fastv_forward` :1438 | LM head 包装，把 `pruning_info` 冒泡给上层 |
| `src/openvla/prismatic/extern/hf/modeling_prismatic.py` | `__init__/attrs` :254-260 | `use_fastv`、`fastv_k/r`、`use_temporal` 等默认值 |
| 同文件 | 语言模型调用点 :406-418 | `use_fastv=True` 时 prefill 走 `language_model.fastv_forward(...)`，否则普通 forward |
| 同文件 | `_generate_with_fastv_forward` :518 | **只是再包一层标准 `generate`**，不重写解码循环 |
| 同文件 | `predict_action` :558 | 每 env step 一次 generate；维护 `av_hist` 时间引导 |
| 同文件 | `_extract_action_modality_attentions` :640 | 把压缩坐标下 decode attention 映射回原始 256 视觉坐标 |
| `src/openvla/experiments/robot/openvla_utils.py` | `get_vla` :285-311 | 官方评测注入 fastv/temporal 配置（config + 实例属性 + deque） |
| `src/openvla/experiments/robot/libero/run_libero_eval.py` | :158-163 | 每 episode 先 `model.reset_av_history()` |

## 2. 已核实的方法语义（必须原样保留，不得简化）

### 2.1 删除比例 `fastv_r` 的语义
- 源码：`num_keep = round(FASTV_image_token_length * (1 - FASTV_r))`
  （`modeling_llama.py` prefill 分支 :1102；`image_token_length=256`）。
- 官方评测脚本注释与之一致：`token retention ratio = 1 - fastv_r`。
- 结论：**`fastv_r` 是删除比例**。prune25% → `fastv_r=0.25`（keep≈192）；prune50% → `fastv_r=0.50`（keep=128）。
  `fastv_r=0` → `num_keep=256`，topk 选中全部 → 理论退化为 Vanilla（r=0 sanity arm）。

### 2.2 剪枝时机与真删除位置
- 剪枝只发生在 **prefill（`hidden_states.shape[1] != 1`）一次**；在 `layer_idx == FASTV_k`
  （默认 k=3）**进入该层之前**用第 `k-1=2` 层 attention 决策。
- 真删除：`hidden_states = hidden_states[:, keep_indexs, :]`（:1127），后续层与 KV cache 只见到保留 token。
- `keep_indexs = cat(arange(1), top_ranked_image_idx+1, arange(image_token_end, seq_len))` 后 `sort()`（:1117-1120）；
  同步更新 `position_ids = keep_indexs.unsqueeze(0)`（保留原 RoPE 位置）、重建 causal mask、
  `cache_position = cache_position[:new_seq_length]`、`seq_length_with_past = new_seq_length`。
- decode（seq==1 的逐 action token）不剪枝，在压缩 KV 上按原 RoPE 位置续写。
  （关键正确性：保留 token 保持原始递增顺序 ⇒ 按 cache 序号做 causal 等价于按原位置 causal。）

### 2.3 三种准则（decision criterion）
`modeling_llama.py` :1082-1123：
- `use_text_vision_selection`（SparseVLM rater，本实验 False）：文本行 258:-1 平均 → 视觉列；
- `use_prefil_attention`：**全部 query 行平均**（`last_layer_attention_prefill = avg.mean(dim=0)`）；
- 默认（两者皆 False，FastV 谱系）：**最后一行 query** 的视觉 attention（`avg[-1]`）。
- VLA-Pruner temporal 模式（`use_temporal=True` 且给入 `Temporal_Guide`）：
  1. `guide_topk = Temporal_Guide.topk(num_keep)`；
  2. `current_topk = last_layer_attention_avg_last_tok_image.topk(num_keep)`；
  3. 并集 → 若超出 num_keep，用当前输入图像的 embed 特征做 `_redundancy_minization`（:950）裁回 num_keep。
  即 combine-then-filter，不是简单交集/并集后截断。

### 2.4 时间引导 `av_hist` 的构造（predict_action :567-635）
- 每 env step 把 **layer=15**（0-based 第 16 层）的 action→vision attention 入历史：
  `vec = action_vision_attentions[15].float().mean(dim=0).mean(dim=0)` → 先对 head、再对 action token 平均，
  得 256 维向量；`deque(maxlen=av_hist_w=3)`。
- 加权：`guided = Σ_i γ^i · av_hist[-1-i]`，γ=0.80，i=0 为最近步；再除以 sum（对 topk 无影响但官方保留）。
  即 `A_hist ≈ A_t + 0.8·A_{t-1} + 0.64·A_{t-2}`（历史来自**上一个 env step**，不是同一步解码阶段）。
- **启动条件**：`use_temporal` 且 `len(av_hist)==maxlen` 才给 Temporal_Guide；
  不足时 fastv_r 强制置 0 → **每个 episode 前 3 个 env step 不剪枝**。
- 每 episode 开始必须 `reset_av_history()`（run_libero_eval :163 逐 episode 调用）。

### 2.5 论文主方法的选择
- README 与 run 脚本目录说明：**`*_prefil.sh` 即“按论文”的 VLA-Pruner 配置**；
  非 prefil 变体是“FastV last-token + temporal”，在某些 benchmark 更好但非论文主方法。
- 因此本轮锁定：`use_fastv=True, fastv_k=3, use_temporal=True, temporal_w=3, temporal_gamma=0.80,
  use_prefil_attention=True, use_text_vision_selection=False, SparseVLM=False, image_token_start_index=1,
  image_token_length=256`，只扫 `fastv_r ∈ {0.25, 0.50}`，另加 `fastv_r=0` 等价性 arm。

## 3. 与我们环境的差异审计（决定移植面）

| 维度 | 官方 | 我们 | 影响 |
|---|---|---|---|
| Llama 实现 | vendored transformers `4.39.0.dev0`，自带 `fastv_forward` | venv `openvla-ar-h100` transformers `4.40.1`，无 `fastv_forward` | 需移植 ~150 行 Llama 剪枝逻辑到 4.40.1 forward 语义 |
| 接入类 | 官方 prismatic `modeling_prismatic.py`（含 `use_fastv`/`av_hist`） | checkpoint 本地 `modeling_prismatic.py`（标准版，无 fastv/temporal） | 不能直接 trust_remote_code 使用官方 prismatic 覆写（会与本地 auto_map 冲突）；历史/引导逻辑放在 policy 层 |
| 生成方式 | `predict_action`（标准 `generate` + `output_attentions`） | 同款 `OpenVLAForActionPrediction.generate`（`_forward_scores` 现只用 `output_scores`） | pruner arm 需开 `output_attentions=True` 并仿照 `_extract_action_modality_attentions` 还原 256 维 action→vision |
| DynamicCache | 4.39 vendored append 语义 | 4.40.1 `DynamicCache.update` 也是 append 语义（已读源码） | 兼容：prefill 后层只 append 保留 token，decode 续写不产生空洞 |
| `_update_causal_mask` 签名 | `(attention_mask, input_tensor, past_seen_tokens)` | `(attention_mask, input_tensor, cache_position, past_seen_tokens)` | 移植剪枝分支时按 4.40 签名补 cache_position 实参 |
| 环境 | LIBERO（seq padding、指令较长） | SIMPLER google_robot（每步单帧，指令短） | prefill 全行均值准则不受文本位置假设影响；仅 `text_vision` 准则有影响（本实验不用） |
| 动作后处理 | LIBERO 评测自身 | google_robot + sticky gripper（已审计 harness） | 保持 harness 完全一致，唯一变量=是否开 VLA-Pruner |

### 移植路线结论：方案 B（把官方 Llama 剪枝语义移植进我们当前 harness 的 LlamaModel forward），理由：
1. 我们的闭环验证目标要求“与已审计 closed-loop harness 完全同构”；整栈切到官方 prismatic/LIBERO 反而引入
   env/processor/unnorm 差异，破坏变量控制。
2. checkpoint 本地 `modeling_prismatic.py` 会被 `trust_remote_code` 优先加载，官方 prismatic 模块无法直接替换
   （auto_map/本地文件冲突），强行 PYTHONPATH 顶替还需连带解决 modeling/processor 版本一致性。
3. 剪枝语义集中在 Llama 内部一处（prefill 边界真删除）+ policy 层一处（layer15 action→vision 历史），
   两处都能按官方源码逐行对齐，并可用 “官方栈子进程等价性测试” 或 r=0==Vanilla 做客观校验。
4. 若后续希望零偏差核对，可在 GPU2/3 子进程用官方 vendored transformers 加载**同一 checkpoint**，
   对同状态对比 logits/keep-set（作为移植正确性 gold reference，属可选加强审计）。

## 4. 未决实现点（写进 IMPLEMENTATION_GAP.md 并逐条记录取舍）
见 `IMPLEMENTATION_GAP.md`（官方 vs 我们改动清单与理由）。关键取舍预告：
- 每 env step 的调用入口：保持我们 `_forward_scores` 式 `generate(...)`；在 prefill 第一次 forward 内触发剪枝，
  需把 `fastv_r / historical_attention` 等以实例属性挂到 LlamaModel 上（disabled → 原 forward，bitwise == Vanilla）。
- action→vision 时间历史：policy 侧维护 `deque(maxlen=3)` + γ=0.80 加权，函数体对齐 `_extract_action_modality_attentions`
  的坐标还原（否则 layer15 向量会因压缩坐标错位）。
- 严格逐条以“官方是什么 → 我们改成什么 → 为什么 → 是否变语义”记录。
