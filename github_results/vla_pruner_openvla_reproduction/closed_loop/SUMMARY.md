# VLA-Pruner reproduction — 22-scene closed-loop calibration (VLA_PRUNER_OPENVLA_REPRODUCTION_V1)

完成 66/66 个 episode(22 场景 × vanilla/prune25/prune50),全部 technical_pass。prune 前 3 step 严格不剪、且与同 obs vanilla 参考逐 token 一致。

## 总览(22 场景)

| arm | n | success | prune激活steps | 剪枝token总数 | 平均kept(激活步) | token flip总数 | rescue | harm | net |
|---|---|---|---|---|---|---|---|---|---|
| vanilla | 22 | 10 | - | - | - | - | - | - | - |
| vla_pruner_prune25 | 22 | 9 | 1760 | 112640 | 192.0 | 3295 | 3 | 4 | -1 |
| vla_pruner_prune50 | 22 | 10 | 1760 | 225280 | 128.0 | 4782 | 1 | 1 | 0 |

## 分任务 success

| task | vanilla | prune25 | prune50 |
|---|---|---|---|
| google_robot_open_drawer | 0 | 0 | 0 |
| google_robot_close_drawer | 0 | 1 | 0 |
| google_robot_pick_coke_can | 3 | 2 | 3 |
| google_robot_move_near | 7 | 6 | 7 |

## 说明

- flips/L1/L2 是 prune arm 每 env step 与同 obs 的 vanilla 参考前向对比;参考向前不影响被执行的轨迹。
- 首次 3 步为 temporal warm-up(fastv_r 强制 0),audit 要求与 vanilla 完全一致。
- 本阶段只报告,不冻结配置。正式配置需结合 Rescue/Harm 与任务分布决定。
