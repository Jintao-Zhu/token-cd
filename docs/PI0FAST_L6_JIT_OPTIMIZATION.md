# pi0-FAST L6-Matched JIT 优化交接

本文整理 pi0-FAST 上的 L6-Matched 对比引导实现、固定动作头 JIT 优化和 SIMPLER 对照入口。仓库只保存代码与复现说明；模型权重、仿真资产和逐 episode 输出不提交。

## 做了什么

方法仍然执行完整的 L6-Matched 对比引导：从 prompt-to-visual attention 选 token，以动态 Matched 预算控制选区，在 projector 特征上做 harmonic reconstruction，并用同一动作 token 前缀计算 clean/negative 分支。JIT 优化没有删去负分支或改变引导公式。

优化点是把固定的 `Action:` 前缀 token 的 clean/negative 解码步骤编译起来，避免这些固定步骤走 Python eager dispatch。FAST 动作负载仍经过校验；当引导分支的解码未通过 OpenPI 官方解码一致性检查时，该 replan 使用并记录 Vanilla fallback。

## 代码入口

| 文件 | 用途 |
|---|---|
| `research/semantic_token_cd/pi0_fast_matched_cd.py` | L6 attention、Matched KMeans 预算、harmonic negative、FAST token 对比解码及 JIT/cache 开关。 |
| `research/semantic_token_cd/pi0_fast_l6_simpler_server.py` | OpenPI 推理服务；每个服务进程独立加载一个 checkpoint。 |
| `research/semantic_token_cd/pi0_fast_simpler_l6_matched_rollout.py` | SIMPLER reset、渲染、Google Robot 与 Bridge/WidowX 输入输出适配和 episode 记录。 |
| `research/semantic_token_cd/pi0_fast_jit_header_closed_loop.py` | 同 seed 的 eager-header 与 JIT-header 闭环对照及动作/token 一致性记录。 |
| `research/semantic_token_cd/pi0_fast_vanilla_vs_jit_9task.py` | 9 个 SIMPLER 任务的 Vanilla 与 JIT L6-Matched 配对 rollout。 |
| `scripts/run_pi0fast_l6_server.sh` | 单个推理服务的环境启动示例；路径和 Conda 环境可通过环境变量覆盖。 |

## 推荐运行配置

```bash
export PI0FAST_WORKSPACE_ROOT=/path/to/workspace
export PI0FAST_REPO_ROOT=/path/to/token-cd
export PI0FAST_OPENPI_ROOT=/path/to/openpi
export PI0FAST_CHECKPOINT=/path/to/pi0-fast-checkpoint
export PI0FAST_CD_KMEANS=sklearn
export PI0FAST_CD_TAP_MODE=separate
export PI0FAST_CD_DECODE=verified
export PI0FAST_CD_JIT_HEADER=1
export PI0FAST_CD_CACHE_STATIC_VIEWS=1
export PI0FAST_CD_EMBED_JIT=0
export PI0FAST_CD_BATCH=0
```

`PI0FAST_WORKSPACE_ROOT` 下需准备 SIMPLER/OpenPI 运行所需的 checkpoint 与资产。Google Robot 使用 Fractal checkpoint；WidowX 对照使用 Bridge checkpoint 和对应 Bridge normalization statistics。Bridge checkpoint 并非 WidowX-SIMPLER 微调权重，相关成功率应解读为跨任务迁移结果。

服务端依赖 OpenPI、JAX 和相应设备运行时；rollout 端还依赖 SIMPLER/SAPIEN。Git 不包含这些外部仓库、模型权重、数据或仿真资产。当前 rollout helper 默认在 `~/zjt_ws` 下查找这些依赖，也可用 `PI0FAST_WORKSPACE_ROOT`、`SIMPLER_ENV_ROOT`、`SPATIALVLA_SIMPLER_ROOT`、`SPATIALVLA_MANISKILL_ROOT` 和 `PCD_SIMPLER_SOURCE` 指定位置。

在准备好服务端 Conda 环境、OpenPI 与 checkpoint 后，可先启动一个服务：

```bash
PI0FAST_WORKSPACE_ROOT=/path/to/workspace \
PI0FAST_OPENPI_ROOT=/path/to/openpi \
PI0FAST_CHECKPOINT=/path/to/fractal-checkpoint \
bash scripts/run_pi0fast_l6_server.sh 1 18761 jit
```

rollout 端按实际任务选择 `--family google`（Fractal）或 `--family widowx`（Bridge），启动 `research/semantic_token_cd/pi0_fast_vanilla_vs_jit_9task.py` 并传入 `--job TASK:SEED_START-SEED_END`。每个 worker 应连接独立的推理服务；多 worker 共用一个同步 WebSocket 服务会排队。RT-full 配置锁为 depth 8、32 spp、denoiser 开启。

## 开关语义

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `PI0FAST_CD_JIT_HEADER` | `1` | 编译固定 `Action:` 前缀步骤；设为 `0` 可运行 eager 对照。 |
| `PI0FAST_CD_TAP_MODE` | `separate` | attention tap 与正式解码 cache 分开，避免 tap 对 cache 的数值影响。 |
| `PI0FAST_CD_DECODE` | `verified` | 使用已校验的 FAST 解码循环。 |
| `PI0FAST_CD_CACHE_STATIC_VIEWS` | `1` | 缓存重复静态视图的冻结视觉 embedding。 |
| `PI0FAST_CD_EMBED_JIT` | `0` | 默认不 JIT 整个 observation embedding；切换前需验证 token/action 等价。 |
| `PI0FAST_CD_BATCH` | `0` | 默认分开计算 clean/negative cache；batch=2 路径需独立做数值等价验证。 |

## 复现与解释边界

`pi0_fast_jit_header_closed_loop.py` 用配对 seed 检查 JIT 与 eager header 的 selected tokens、action chunks 和闭环结果。`pi0_fast_vanilla_vs_jit_9task.py` 比较 Vanilla 与固定-header JIT L6-Matched；配对时记录 initial state、RGB、成功、步数、完整 episode wall time 和 fallback 数。

总 episode wall time 同时包含模型推理和仿真/渲染；按 step 归一化的耗时可辅助区分模型开销与 episode 长度变化。未完成的 rollout 不应当作最终成功率或最终速度结论。每次完整实验的配置、seed、checkpoint 和 renderer 应写入 run manifest，并保留在独立实验结果目录中，不要把大型原始 rollout 写入 Git。
