# LIBERO-90 OpenVLA Migration Smoke Report

## 1. 迁移目的

将当前 `L11 instruction-attention + KMeans matched budget + harmonic reconstruction + contrastive guidance` 工程迁移到 `VQ-VLA/openvla-7b-finetuned-libero-90`，只做 checkpoint 下载、加载适配、正确性检查和最多 5 任务 × 2 arm 的 smoke validation。

本轮没有启动完整 LIBERO-90 评测、参数搜索、query/layer/head 搜索、训练或微调。

## 2. Checkpoint 来源、revision 和实际架构

- 仓库：`VQ-VLA/openvla-7b-finetuned-libero-90`
- revision：`794ef81b7be928ea9270e81ca1ef5b60ffa9420f`
- 本地目录：`/home/leju-suzhou/zjt_ws/checkpoints/libero/vq-vla-openvla-7b-libero90`
- `architectures`：`OpenVLAForActionPrediction`
- `model_type`：`openvla`
- action bins：256
- 权重：4 个 safetensors 分片，索引引用的文件全部存在，SHA256 与 Hugging Face LFS 元数据一致
- `dataset_statistics.json`：只包含 `libero_90_no_noops`

## 3. 与 Spatial 实现的兼容性差异

1. 该 checkpoint 的 `config.json` 没有 `libero_spatial` 或 `libero_90_no_noops` 动作统计项。
2. LIBERO-90 统计位于独立的 `dataset_statistics.json`，需要通过显式加载适配注入模型。
3. LIBERO-90 包含直接 `put ... on/in ...`、`stack ...`、`open/close ...`、`turn on/off ...` 等指令，不能直接使用 Spatial 的 source/target 解析器。
4. LIBERO-90 官方最大控制步数为 400、warmup 为 10，不能沿用 Spatial 的 220 步设置。
5. 本次没有替换成 VQ-VLA 专用动作解码器；checkpoint 配置仍是原版 `OpenVLAForActionPrediction` 路径。

## 4. 代码修改及理由

- `research/ar_token_counterfactual/libero_runtime.py`：增加显式 `attach_action_statistics(...)`，并在 `load_policy(...)` 中支持 `dataset_statistics_path` 与 `unnorm_key`。
- `research/semantic_token_cd/libero_policy.py`：增加 `extract_source_target_entities_libero90(...)`，不修改原 Spatial parser。
- `research/semantic_token_cd/libero_matched_rollout.py`：支持 `--suite`、`--unnorm-key`、`--dataset-statistics`、`source_target_libero90` 和可选视频保存。
- `research/semantic_token_cd/libero_vanilla_rollout.py`：支持相同 suite、动作统计、步数和视频接口。
- `research/semantic_token_cd/run_libero90_migration_smoke.sh`：按 GPU worker 运行 smoke validation。
- `research/semantic_token_cd/analyze_libero90_migration_smoke.py`：汇总成功、步数、paired initial-state hash 和 matched 元数据。

## 5. 动作统计加载方式

加载路径为：

```text
原始 config.norm_stats
  + 显式读取 dataset_statistics.json 中的 libero_90_no_noops
  -> model.norm_stats['libero_90_no_noops']
```

检查结果：

- `q01` 和 `q99` 均为 7 维；
- `mask` 为 `[true, true, true, true, true, true, false]`；
- 没有修改或覆盖下载的 `config.json`；
- vanilla 和 matched 都使用同一个 `libero_90_no_noops` 反归一化 key。

## 6. 同状态检查结果

- 总体：`True`
- λ=0 对 vanilla 动作最大差异：`0.0`
- λ=0 token 完全一致：`True`
- attention 提取前后 vanilla token 一致：`True`
- attention 提取前后动作最大差异：`0.0`
- identity reconstruction 动作 token 一致：`True`
- identity reconstruction action-bin logit 最大差异：`1.0`，来源是 bf16 下 `generate()` 与 teacher-forced forward 的数值差异
- 非空 mask：`True`
- mask token 唯一：`True`
- 重建特征发生变化：`True`
- guidance 公式 token 对照：`True`
- 首个状态 matched budget：`13`

## 7. Smoke Test 实际数据

任务与 init state 均记录在 `TASK_MANIFEST.json`。所有五个任务的 vanilla/matched initial-state SHA256 均一致。

| Task | Init state | Arm | 正常结束 | Success | Steps | Mean m | 备注 |
|---|---:|---|---|---|---:|---:|---|
| `KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it` | 0 | vanilla | True | True | 174 | - | paired init state verified |
| `KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it` | 0 | matched | True | True | 169 | 21.22 | paired init state verified; entities=['butter at the back', 'top drawer of the cabinet'] |
| `KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet` | 0 | vanilla | True | True | 205 | - | paired init state verified |
| `KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet` | 0 | matched | True | True | 144 | 28.22 | paired init state verified; entities=['black bowl', 'top of the cabinet'] |
| `LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket` | 0 | vanilla | True | True | 146 | - | paired init state verified |
| `LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket` | 0 | matched | True | False | 400 | 56.05 | paired init state verified; entities=['tomato sauce', 'basket'] |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate` | 0 | vanilla | True | False | 400 | - | paired init state verified |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate` | 0 | matched | True | True | 131 | 45.78 | paired init state verified; entities=['white mug', 'plate'] |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy` | 0 | vanilla | True | False | 400 | - | paired init state verified |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy` | 0 | matched | True | True | 241 | 39.03 | paired init state verified; entities=['book', 'front compartment of the caddy'] |

结果解释仅限工程验收：所有 10 个 episode 都完成到成功或合法超时，没有悬空进程、NaN/Inf 或环境接口错误。不能据此评价方法优劣。

## 8. 未解决问题或采用的假设

- 论文中的 PCD 自然语言任务与 LIBERO-90 task ID 的 scene 对应关系尚未由 PCD 元数据正式确认；本轮只标记为 smoke-test candidates。
- 当前 LIBERO-90 parser 对选中的 5 个任务解析成功，但完整 90 个任务中仍包含 `open/close`、`turn on/off`、`stack`、`under` 等不同句式；完整评测前需要逐类补齐或明确回退规则。
- 本轮没有执行完整 LIBERO-90，也没有比较 matched 与 vanilla 的统计显著性。
- 没有改变 harmonic reconstruction、guidance 公式、L11 层号含义、lambda 基值或 KMeans 设置。

## 9. 可复现命令

```bash
cd /home/leju-suzhou/zjt_ws/token-cd
source scripts/activate_libero_openvla.sh

python artifacts/libero90_openvla_migration_smoke_v1/verify_checkpoint.py
python artifacts/libero90_openvla_migration_smoke_v1/checkpoint_load_check.py
python artifacts/libero90_openvla_migration_smoke_v1/same_state_sanity.py

bash research/semantic_token_cd/run_libero90_migration_smoke.sh 1
bash research/semantic_token_cd/run_libero90_migration_smoke.sh 2
bash research/semantic_token_cd/run_libero90_migration_smoke.sh 3

python research/semantic_token_cd/analyze_libero90_migration_smoke.py \
  --root artifacts/libero90_openvla_migration_smoke_v1 \
  --manifest artifacts/libero90_openvla_migration_smoke_v1/TASK_MANIFEST.json
```

## 10. 停止状态

已完成 checkpoint 下载、统计适配、vanilla 验证、方法接线和 10 个 episode 以内的 smoke validation。没有启动完整 LIBERO-90、seed 扩展、参数搜索或后续消融。
