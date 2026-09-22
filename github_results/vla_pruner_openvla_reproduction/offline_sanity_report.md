# VLA-Pruner r0 离线 sanity 报告（2026-09-09）

状态：**r0 门全过（36/36）；prune25/prune50 尚未按 22 场景闭环（下一阶段，未启动）**。

## 结论（一句话）
方案 B 的 `fastv_r=0` 适配器在当前 OpenVLA+SIMPLER harness 中**严格退化为 vanilla**：
7 个 action token、raw action、executed action 全部逐位一致（`max_abs=0`），
且 `pruned_indices==∅`、kept image==256、每个 episode 开头 history 清空。

## 通过矩阵（mode=step 为同初态单步，mode=episode 为整条闭环）
| task | step seeds | episode seeds | 结果 |
|---|---|---|---|
| google_robot_pick_coke_can | 100-104,150,200,250 | 100,101,102,103 | ALL PASS |
| google_robot_open_drawer | 100-102,150,200 | 100,101 | ALL PASS |
| google_robot_close_drawer | 100-102,150 | 100 | ALL PASS |
| google_robot_move_near | 100-102,150 | – | ALL PASS（step） |
| google_robot_place_apple_in_closed_top_drawer | 100-102,150 | – | ALL PASS（step） |
| widowx_carrot_on_plate（widowx 分支） | 100-101 | 100 | ALL PASS |

合计 36 组 task×mode×seed 对比，0 失败（r0_summary.json）。GPU2/GPU3 并行。

每次 PASS 断言：token_ids_equal / raw_action_equal / executed_equal / raw_action_max_abs==0 /
r0_pruned_empty / r0_kept_count_eq_original / r0_kept_image_count==256 / r0_pruning_layer==3 /
r0_history_reset_first（episode）。

## 修掉的实现偏差（详见 IMPLEMENTATION_GAP.md 第 E 节）
1. **E1** FastV 分支未转 `DynamicCache` → 每步全序列重算、cache 从不累积、pruning_info 被最后 decode 覆盖成 270。
   修复后 decode 恢复缓存路径（attention `[1,32,1,265..270]`），r0 与 vanilla 逐 token 一致。
2. **E2** 剪枝后 layer>=k 仍用全长 causal mask → prune25 崩溃；保留 pruned-mask 后 layer>=k 真实变短。
3. **E3** 官方“还原大方阵”的 attention 还原无法在 4.40 结构 broadcast；改为按 kept 列散布回原 256 视觉位。
4. **E4/E5** 共享模型实例的 attach/detach 卫生；episode 双臂各自 restore 物理初态。
5. **E6** r0 mask 重建用 `attention_mask[:, keep]`，keep=all 即 vanilla 输入，保证位一致。

## 剪枝路径（实现级验证，非闭环成功率）
GPU2 开发诊断（单状态、无 temporal warm-up）：
- r0：`orig_seq=264, kept=264, pruned=0`；layer<3 cache 264，layer>=3 cache 264；decode q=1。
- prune25：`num_keep=192, kept_total=200, pruned=64`（全为视觉）；layer0..2 全长，layer>=3 prefill `[1,32,200,200]`，decode kv 200→206。
- prune50：`num_keep=128, kept_total=136, pruned=128`；layer>=3 `[1,32,136,…]`。
- 无 NaN；7 个 action token 合法。
- 注意：剪枝发生在 layer k=3 边界真删除，层>=3 cache 短于层<3；decode 由 HF 原生按层各自续写 KV，
  4.40.1 DynamicCache/legacy 往返已验证兼容。

## 尚未执行（遵守 r0 卡点）
- 22 个校准场景（CALIBRATION_MANIFEST）prune25 / prune50 闭环 —— 等 r0 全量（含 widowx）通过后由下一阶段执行。
- 冻结配置后正式 400 episode 成功率验证。
