# H100 实验交接 — Semantic Token-CD（Attention-mask Contrastive Decoding）

> 这是一份给 H100 服务器上 Claude 会话的交接文档。目标：把 Semantic Token-CD
> 实验线在 H100 上跑通，并验证 CD 干预机制正确（技术审计通过）。

---

## 0. 一句话总览

在 **OpenVLA-7B**（VLA 机器人策略，Llama-2-7B 主干）上做 **token 级 contrastive
decoding（CD）**：把 attention 里「Action Query → 选定视觉 key」的连接 mask 掉，
得到 negative 分支，再 `final = (1+λ)·clean − λ·negative`（λ=0.5）。视觉 key 的
「选定」由语义 selector 决定（KMeans 聚类 + 语言 grounding）。当前主线是
**窄带（K=8, layers 8-15）vs 全带（K=32, layers 8-31）** 的对比。

代码在 GitHub：`https://github.com/Jintao-Zhu/token-cd`（`research/` 包在根目录，
导入为 `research.<subpackage>...`）。本仓库根目录要加入 `PYTHONPATH`。

---

## 1. 实验线状态（结论速查）

| 线 | 结论 | 关键数字 |
|---|---|---|
| Semantic Token-CD 主线（kmeans 对象组） | PASS_TO_ROLLOUT（首个通过） | kmeans 对象组 cos 0.703 vs attention 0.319 vs random 0.119 |
| Phase-1/1.5/1.6/2A/2B selector | 逐步 NO_GO，瓶颈=语言-视觉对齐判别力 | entity_set 0.586 迄今最佳 selector |
| Action-grounded / 多信号融合 | STOP（三入口全验证完） | action 0.505 / 融合 0.575 < language 0.586 |
| Semantic Entity-CD Rollout Pilot | **PASS_TO_ROLLOUT**（跨闭环） | vanilla 0.133 → cd_050 0.211，Rescue/Harm 9:2 |
| LIBERO-Object 泛化 | Phase-0 PASS，Rollout NO_GO | align 0.628；Gate D 失守（semantic 0.75 < random 0.765） |
| **Attention-CD 窄带 K8 L8-15（当前主线）** | **窄带 ≥ 全带 ≥ vanilla（Google Robot 单调）** | close_drawer p=0.032（Rescue 17:7）；stack_cube 首非零 3/50 |

**当前主线（窄带 Attention-CD）的确定性结论**：6 个确定性任务各 50 seeds：

| task | vanilla | K32 L8-31 | K8 L8-15 |
|---|---|---|---|
| close_drawer | 26/50 | 33/50 | **36/50** |
| move_near | 31/50 | 32/50 | **35/49** |
| open_drawer | 5/50 | 9/50 | **9/50** |
| pick_coke_can | 12/50 | 14/50 | **16/50** |
| carrot_on_plate | 1/50 | 2/50 | 0/50 |
| stack_cube | 0/50 | 0/50 | **3/50** |

含义：高层（16-31）对 Google Robot 是纯噪声，砍掉反而更好；WidowX 硬任务仍
卡 0-3%，瓶颈在 selector 判别力不在带宽。

---

## 2. 环境要求

### 2.1 必须有的三块东西

1. **`research/` 代码**（GitHub 已传）：`git clone https://github.com/Jintao-Zhu/token-cd.git`。
2. **PCD 环境目录**（含 15G checkpoint，**GitHub 没传**）：RTX 上是
   `official-reproductions/pcd_openvla_simpler_box_31b027e/source/PCD`，内含：
   - `pretrained/openvla-7b/`（**15G** OpenVLA-7B checkpoint）
   - `simpler_env/`（SIMPLER 环境）
   - `properties.py`、`parallel_inference.py`、`utils.py`、`contrast_policies/`、`contrast_utils/`
   - 这个目录必须整体在 H100 上有一份（从 RTX scp/rsync，或从 huggingface 重新拉 checkpoint）。
3. **Python venv**：Python 3.10，`torch 2.7.1+cu128`，`transformers 4.40.1`，`simpler_env` 的依赖（svulkan2、maniskill2、sapien 等）。

### 2.2 依赖版本（RTX 实测可用）

```
python 3.10.12
torch 2.7.1+cu128
transformers 4.40.1
setuptools 84.0.0 (RTX) / 80.10.2 (H100, 原生 pkg_resources)
```

> **shim_site 说明**：RTX 上 `task1/shim_site` 是 vendored 的 `pkg_resources` shim
> （因为 setuptools 84 移除了 pkg_resources）。**H100 是 setuptools 80.10.2，自带
> pkg_resources，不需要 shim_site**。

---

## 3. 关键路径与硬编码修复（⚠️ H100 必做）

`research/semantic_token_cd/distractor_rollout.py` 第 15 行把 repo 根**硬编码**成：

```python
REPO_ROOT = Path("/data/docker/dev_zjt/data/code")
```

共 **26 个文件**硬编码了这个路径。H100 上二选一：

- **方案 A（推荐，不改代码）**：在 H100 上建立同样的路径
  `/data/docker/dev_zjt/data/code`（symlink 也行），把 repo 和 PCD 环境放进去。
- **方案 B（改代码）**：全局替换 `research/semantic_token_cd/` 下所有
  `/data/docker/dev_zjt/data/code` 为 H100 实际路径。

涉及导入链（跑通必需）：

```
from properties import get_policy_config              # 来自 PCD_SOURCE
from simpler_env.policies.openvla.openvla_model import OpenVLAInference  # 来自 PCD_SOURCE
from parallel_inference import ...                     # 来自 PCD_SOURCE
from research.semantic_token_cd.distractor_policy import AuditedVanillaInference
from research.semantic_token_cd.spatial_grid_policy import SpatialGridAttentionCDInference
from research.semantic_token_cd.spatial_grid_rollout import make_environment, run_episode
from research.semantic_token_cd.distractor_rollout import PCD_SOURCE, capture_snapshot, restore_snapshot, snapshot_sha
```

`distractor_rollout.py` 顶部的 sys.path 插入顺序（H100 去掉 shim_site 那行）：

```python
for path in (REPO_ROOT / "task1/shim_site", REPO_ROOT, PCD_SOURCE):  # H100: 删掉 shim_site
    sys.path.insert(0, str(path))
```

即 `PYTHONPATH` 需要三个入口：`<repo_root>`、`<PCD_SOURCE>`（+ RTX 独有的 shim_site）。

---

## 4. 协议事实（写实验/读结果必背）

### 4.1 OpenVLA-7B 架构
- Llama-2-7B 主干，**32 层**（0-31），每步 **256 visual tokens + 7 action tokens**。
- Action token 位置 **[31744, 31999]**；Action Query 在 prompt 的 **281-287** 位置。
- 视觉 token 的 KEY_MASK 逐位一致（bit-identical）——这是 CD 减法的前提。

### 4.2 Attention-CD 干预
- `final = (1+λ)·clean − λ·negative`，**λ=0.5**。
- negative 分支 = 把 Action Query → 选定 visual key 的 attention 连接 mask 成 `-1e4`。
- **窄带**：K=8, layers 8-15 → `EXPECTED_HOOK_CALLS = 7 × (16−8) = 56`。
- **全带**：K=32, layers 8-31 → `7 × (32−8) = 168`。
- 技术审计两项：`all_visual_features_bit_identical`（clean/negative 视觉特征逐位一致）
  + `attention_mask.hook_calls == EXPECTED_HOOK_CALLS`（mask 精确命中）。

### 4.3 Semantic selector
- KMeans `n_clusters=8, seed=0` → entity_set 并集；`selection_mode="semantic"`，
  `spatial_selection_mode="semantic_hard"`。

### 4.4 两种 pairing 协议（关键区别）
- **单臂 hash-paired**（确定性任务用）：跨进程和冻结 reference 的
  `(state_sha256, rgb_sha256)` 配对，复用已存的 per-seed 快照。
- **双臂 capture-once-reuse**（非确定任务用）：每 seed 只 reset 一次抓快照，
  同一快照上先后跑 vanilla + CD 两臂，进程内 bit-identical 配对。

### 4.5 reset 确定性图谱（⚠️ 重要）
- **确定性**（可 hash-pair）：close_drawer、move_near、open_drawer、pick_coke_can、
  carrot_on_plate、stack_cube。
- **非确定**（同 seed 每次 reset 场景都不同，hash-pair 不适用）：place_apple、
  eggplant（put_eggplant）、spoon_on_towel。这 3 个必须用双臂 capture-once-reuse。

### 4.6 评测指标
- Rescue = vanilla 失败 + CD 成功；Harm = vanilla 成功 + CD 失败；McNemar binomtest。

---

## 5. 跑通烟雾测试（H100 第一步）

新增了 `research/semantic_token_cd/smoke_test_attn_cd.py`，**自包含**（不依赖任何
冻结 reference，直接抓新快照），跑 `vanilla + attn_semantic(K8 L8-15)` 两臂 N 个 seed。

```bash
cd <repo_root>
export PYTHONPATH="<repo_root>:<PCD_SOURCE>"      # 必要时加 shim_site
export HF_HUB_OFFLINE=1
PY=<H100 venv python>                               # e.g. /path/to/venv/bin/python
"$PY" research/semantic_token_cd/smoke_test_attn_cd.py \
    --task google_robot_close_drawer --seeds 3 --gpu 0
```

**成功判据**（脚本末尾会打印）：
- 每 seed 两臂都跑完、有 JSON 输出；
- 最后打印 `{"SMOKE_TEST_PASS": true, "audit_failures": []}`（exit 0）。
- 若 `SMOKE_TEST_PASS: false`，看 `audit_failures`：`feature_equal=false` 说明视觉特征
  非逐位一致；`hook_pass=false` 说明 attention mask 的 hook_calls 不等于 56。

**推荐先用确定性任务** `google_robot_close_drawer`（vanilla 26/50 有正信号，环境必能跑通），
再换 `widowx_stack_cube` 或 `google_robot_place_apple_in_closed_top_drawer` 验证 WidowX/非确定路径。

---

## 6. 正式实验怎么跑（跑通后）

双臂 capture-once-reuse（非确定任务三件套，已有脚本）：

```bash
"$PY" research/semantic_token_cd/attn_semantic_k8_l8_16_3nondet_50_v1.py \
    --artifact artifacts/attn_semantic_k8_l8_16_3nondet_50_v1 --gpu 0
```

单臂 hash-paired（确定性任务，需要先有冻结 reference，见 `attn_semantic_k32_l8_8task_50_v1`）：

```bash
"$PY" research/semantic_token_cd/attn_semantic_k8_l8_16_widowx_stack_cube_50_v1.py \
    --artifact artifacts/attn_semantic_k8_l8_16_widowx_stack_cube_50_v1 --gpu 0
```

> 单臂 hash-paired 脚本依赖 `artifacts/attn_semantic_k32_l8_8task_50_v1/` 里的冻结
> reference（**未传到 GitHub**，在 RTX 的 artifacts 里）。H100 上若没有该 reference，
> 先用双臂脚本或先重跑 K32 8task reference 生成冻结快照。

---

## 7. 常见坑

1. **`ModuleNotFoundError: torch`** → 用错了 python（系统 python 没装 torch），必须用 venv 的 python。
2. **`GLFW error: X11: DISPLAY ... missing` / `Continue without GLFW`** → 无害，headless 渲染正常。
3. **`Frozen pairing mismatch` / `state=... rgb=...`** → 非确定任务却用了 hash-paired 脚本，或 reference 没冻结好。改用双臂脚本。
4. **`pkg_resources` 相关报错** → RTX 才需要 shim_site；H100 的 setuptools 80 自带，别加 shim。
5. **checkpoint 加载慢/失败** → 确认 `PCD_SOURCE/pretrained/openvla-7b`（15G）完整，且 `HF_HUB_OFFLINE=1`。
6. **GPU 争抢** → H100 若共享，先 `nvidia-smi` 看显存；单卡 OpenVLA-7B rollout 约需 18-20G 显存。
