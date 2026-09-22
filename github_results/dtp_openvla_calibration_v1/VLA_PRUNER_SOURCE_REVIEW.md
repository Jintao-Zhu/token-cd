# VLA-Pruner (OpenVLA) 源码审查 — 与 DTP 复现的关系

仓库：`/home/leju-suzhou/zjt_ws/VLA-Pruner`（MINT-SJTU/VLA-Pruner，arXiv 2511.16449v5）。
注意：VLA-Pruner **不是** DTP 论文（arXiv 2601.16065）的官方实现；它是独立的 training-free
VLA 视觉 token 剪枝方法（FastV 谱系 + 时间平滑），但其 `src/openvla/` 是**带真实剪枝接入的
OpenVLA 完整实现**，是核对 OpenVLA 序列布局、剪枝机制与 action-attention 读取的现成参考。

## 仓库地图（src/openvla）

- `transformers/src/transformers/models/llama/modeling_llama.py`：vendored HF Llama，剪枝全部发生在此。
  - `LlamaModel.fastv_forward`（:973）：前向内做“单层边界处压缩序列”。
  - `LlamaForCausalLM.fastv_forward`（:1438）：LM head 包装，供 prismatic 调用。
- `prismatic/extern/hf/modeling_prismatic.py`：OpenVLA 推理接入。
  - `OpenVLAForActionPrediction.predict_action`（:558）：每步生成动作 + 维护时间引导历史。
  - `_extract_action_modality_attentions`（:640）：从 generate 的 attention 拆出
    action→vision、text→vision、prefill attention，并按 kept_indices 把压缩坐标映射回原始坐标。
- `experiments/robot/openvla_utils.py`：加载 openvla-7b（eager attention），注入 fastv config（:285-314）。
- `experiments/robot/libero/run_libero_eval.py`：LIBERO 评测入口。默认
  `fastv_k=3`、`fastv_r=0.50`（run 脚本用 0.75）、`image_token_start_index=1`、`image_token_length=256`、
  `temporal_w=3`、`temporal_gamma=0.80`（:52-64）。

## 方法管线（每 env step，fresh prefill + 7 action tokens 一次 generate）

1. 预填充（image+instruction，序列长 >1）在 Llama 层边界处做一次**真删除型压缩**：
   在 layer `FASTV_k-1` 之后（`pruning_layer==FASTV_k`，默认 k=3，用第 2 层 attention 作准则），
   只保留 image 前导 token + 选中的 image token + 全部后续文本，`hidden_states=hidden_states[:,keep,:]`
   （:1127-1147）；同步改写 position_ids（保留原始 RoPE 位置）、重建 causal mask 与 cache_position。
   之后的层和 KV cache 都只见到压缩序列 → 这是 FastV 式“剪 token 省算力”，不是 DTP 的“attention mask 屏蔽”。
2. 准则三选一（:1082-1123）：
   - FastV：layer k-1 最后一行 query 的视觉 attention，top-(1-r) 保留；
   - prefill-attention（`use_prefil_attention`）：同一层 `mean(dim=0)`（所有行平均）的视觉 attention；
   - VLA-Pruner temporal：`guide_topk ∪ current_topk` 后接
     `_redundancy_minization`（:950，特征余弦距离贪心去冗余）裁回正好 `num_keep=round(256*(1-r))`。
3. 时间引导：`predict_action` 维护 `av_hist`（deque，maxlen=3，decay=0.8，:567-578）；每步取
   layer 15 的 action→vision attention（head 均值再 action token 均值，:613-619）入历史；
   下一步预填充用它做 guide_topk。即“最近动作解码的动作相关性”跨步平滑。
4. 动作解码（序列长=1 分支）不剪枝，只在已压缩的 KV cache 上逐 token 生成。

## 序列布局假设（与我们实现核对一致）

OpenVLA 在 Llama 侧：`image_token_start_index=1`、长度 256 → 视觉 key 位置 1..256；
其代码约定 `text_start=258`（即在 256 个 image token 后还有至少 1 个非文本占位，确切身份未写死），
动作 tokens 是最后 7 个生成 token。这与我们
`attention_access` 读取 `output[1][..., -1, 1:257]`、`top(spatial(...),k)` 的视觉列一致；
VLA-Pruner 还证实可用 generate 的 `output_attentions`（而非 hook）读 action attention——我们已做过
hook vs outputs.attentions 逐元素 max-diff=0 的等价性审计。

## 与 DTP 复现的关键差异（机制层面）

| 维度 | VLA-Pruner/FastV（本仓库） | 我们的 DTP 复现 |
|---|---|---|
| 剪枝方式 | 预填充中途删 hidden state/不进 KV cache（省算力） | 全序列保留，attention mask 把 key 置 -inf（软屏蔽，不省算力） |
| 决策时机 | 每 env step 一次，prefill 层 3 边界 | 逐动作 token 解码时（v1 dynamic / v2 fixed dim0） |
| 预算 | 固定删除 `round(256*(1-r))`，r=0.5-0.75 必剪 | `τ·a_m` 阈值 + Top-k 保护区，通常大量 step 不剪 |
| 重要度来源 | 浅层(2) prefill/最后行 attention + layer15 动作相关性时间平滑 | L11/L7 prompt 相关性层构造保护区 G，动作 attention 超阈值触发 |
| 保护区 | 无（保留下 top-k 即保护其余被删） | G（Top-k 空间处理后相关性），删 G 外超过阈值的 |

含义：两方法都在处理“语义 vs 动作相关性”错配，但实现完全两路。VLA-Pruner 在 OpenVLA 上
（LIBERO）以高删除率保性能，说明“OpenVLA 上按正确准则做真删除剪枝”可以无害/有利；
我们的 DTP 迁移在 22 校准场景/正式 400 上未超 vanilla，反映的是**DTP 式阈值+保护区语义**
在 OpenVLA 上的表现，不能外推为“OpenVLA 剪枝无效”。

## 可直接借用的后续实验选项（如需）

1. 在闭环节奏里加一个 FastV/VLA-Pruner 式固定预算真删除臂（如 k=3、r=0.25/0.5，准则：
   最后行 vs prefill 均值 vs layer15 动作相关时间平滑），作为“剪枝机制本身在 OpenVLA 是否有效”
   的对照——这回答的是与 DTP 正交的问题，需用户确认是否进入范围。
2. 其 kept_indices 坐标映射思路可用于核对/简化我们 hook 读取在压缩场景下的正确性（当前我们无压缩）。
3. 默认超参与 DTP 参数（L11 vs 其 k=3 层、τ 阈值 vs 固定 r）提示：DTP 迁移参数没有用
   “OpenVLA 自身偏好浅层注意力”的证据，τ 校准仍是未决项。
