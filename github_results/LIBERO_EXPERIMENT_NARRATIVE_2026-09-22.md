# LIBERO-Spatial 实验逻辑记录（2026-09-22）

## 统一说明

- benchmark：LIBERO-Spatial
- 任务数：10
- 官方协议：每个任务 50 个 unique init states
- 官方单 arm 总 episode：500
- env_seed：0
- warmup：10 步 no-op
- max policy steps：220
- 旧版 custom：最多 300 步，episode 会循环 50 个 init states，因此 100/task 实际是 50 个状态重复两次；旧版只作为历史结果，不和新协议混用。

---

# 实验 1：旧版 custom 协议下 matched vs vanilla

## 实验目的
验证 L11 + matched 方法从 SIMPLER 迁移到 LIBERO-Spatial 后，是否能整体优于 vanilla。

## 预期结果
matched 应保持 SIMPLER 上的正向收益，整体成功率高于 vanilla。

## 实际结果数据
| Arm | Success |
|---|---:|
| L11 + matched（all entity） | 761 / 1000 |
| Vanilla | 782 / 1000 |

## 实验结论
旧版 custom 协议下，matched 没有超过 vanilla，反而低 21 个成功。该结果说明迁移存在明显问题，但不能直接和官方 LIBERO 结果比较，因为协议不同且状态有重复。

## 下一步实验
对齐官方协议：50 unique states/task、10 步 warmup、220 步。

---

# 实验 2：source/target-only 诊断（T1/T2/T4，旧版 custom）

## 实验目的
检验 all-entity matched 是否因为把关系物体也并入预算，导致 m 过大、over-masking。

## 预期结果
去掉 ramekin/cookie box 等关系物体后，m 缩小，成功数上升。

## 实际结果数据
| 任务 | all-entity matched | source/target-only | vanilla |
|---|---:|---:|---:|
| T1 between plate / ramekin | 84 / 100 | 85 / 100 | 79 / 100 |
| T2 next to ramekin | 71 / 100 | 77 / 100 | 86 / 100 |
| T4 on cookie box | 89 / 100 | 91 / 100 | 92 / 100 |
| Total | 244 / 300 | 253 / 300 | 257 / 300 |

## 实验结论
去掉关系物体后 m 明显下降，成功数有提升，说明 over-masking 确实存在。但 T2 仍然明显低于 vanilla，说明 m 过大不是唯一原因。

## 下一步实验
对齐官方协议后重跑 matched vs vanilla。

---

# 实验 3：官方协议正式 matched vs vanilla

## 实验目的
在标准 LIBERO-Spatial 协议下比较 current L11 matched 和 vanilla。

## 预期结果
若方法有效，matched 应该高于 vanilla；若迁移失败，matched 低于 vanilla。

## 实际结果数据
| Arm | Success |
|---|---:|
| Vanilla | 423 / 500 |
| Current L11 matched | 410 / 500 |

逐任务：

| 任务 | matched | vanilla | Δ |
|---|---:|---:|---:|
| T1 between plate / ramekin | 45 | 47 | −2 |
| T2 next to ramekin | 36 | 48 | −12 |
| T3 from table center | 45 | 40 | +5 |
| T4 on cookie box | 43 | 49 | −6 |
| T5 in top drawer | 35 | 35 | 0 |
| T6 on ramekin | 46 | 43 | +3 |
| T7 next to cookie box | 48 | 45 | +3 |
| T8 on stove | 41 | 46 | −5 |
| T9 next to plate | 35 | 39 | −4 |
| T10 on wooden cabinet | 36 | 31 | +5 |
| Total | 410 | 423 | −13 |

配对统计：

```text
rescue = 57
harm = 44
net = +13（vanilla 更好）
p = 0.2323
```

## 实验结论
官方协议下 matched 总体仍低于 vanilla，整体差异不显著。T2 是明显负结果，配对 rescue/harm = 12/0，是主要失分点。

## 下一步实验
检查 L11 attention 是否真的定位到目标 bowl，并检查 query、layer、head 设计。

---

# 实验 4：L11 attention 热力图与目标定位检查

## 实验目的
确认 L11 attention 是否关注到正确的目标 bowl。

## 预期结果
若 L11 可用，attention 高响应应集中在正确目标 bowl 附近。

## 实际结果数据
T2 场景：

```text
pick up the black bowl next to the ramekin and place it on the plate
```

GT 中：
- 红色目标 bowl：ramekin 旁边的 bowl；
- 橙色干扰 bowl：另一个 bowl；
- 蓝色 plate：放置目标。

L11 热力图观察：
- 高 attention 主要落在机器人、背景、柜子、白色物体和图像边缘；
- 目标红色 bowl 没有稳定高响应。

## 实验结论
L11 在当前 LIBERO checkpoint 和 prompt 下，并不能稳定定位空间关系指定的目标 bowl。仅调整 m 无法修复位置排序错误。

## 下一步实验
尝试不同 query grouping、layer 和 attention heads，并做离线目标区分度诊断。

---

# 实验 5：role / source-relation / destination 离线诊断

## 实验目的
比较不同 prompt query 分组方式，寻找更合理的关系定位信号。

## 预期结果
完整关系短语优于把 relation 和 reference 拆开；destination 可能只有弱辅助作用。

## 实际结果数据
Dev：10 tasks × 20 states = 200 states。

Top-16 target discrimination：

| layer | full | role | SR | SR + plate ×1.0 | SR + plate ×0.5 | SR + plate ×0.25 |
|---|---:|---:|---:|---:|---:|---:|
| L11 | 0.1265 | 0.1613 | 0.1548 | 0.1366 | 0.1526 | 0.1607 |
| mean 23–25 | 0.0031 | 0.0045 | 0.0062 | 0.0056 | 0.0054 | 0.0048 |

Relation-switch coverage response：

| layer | full | role | SR | SR+D1 | SR+D0.5 | SR+D0.25 |
|---|---:|---:|---:|---:|---:|---:|
| L11 | 0.0626 | 0.0190 | 0.0277 | 0.0553 | 0.0473 | 0.0451 |
| mean 23–25 | 0.1762 | 0.1736 | 0.1722 | 0.1689 | 0.1675 | 0.1684 |

## 实验结论
- 将 `next to` 和 `ramekin` 拆开不合理；
- 完整 source-relation 短语更合理；
- `plate` 不能与 SR 等权，等权会降低目标区分度；
- L11 目标区分度较好但关系切换响应弱；
- 深层对关系切换响应强，但目标/干扰区分度很差。

## 下一步实验
在闭环中比较 role 和 deep 版本，并完整扫描所有 layer 和 heads。

---

# 实验 6：Role L11 / Role deep 闭环

## 实验目的
验证 query role aggregation 和深层 attention 是否能改善闭环。

## 预期结果
role 或 deep 至少应优于 current L11，理想情况下超过 vanilla。

## 实际结果数据
Dev states 0–9，100 episodes / arm：

| Arm | Success |
|---|---:|
| Vanilla | 80 / 100 |
| Current L11 | 79 / 100 |
| Role L11 | 81 / 100 |
| Role deep 23–25 | 83 / 100 |

Held-out states 20–49，300 episodes / arm：

| Arm | Success |
|---|---:|
| Vanilla | 260 / 300 |
| Current L11 | 250 / 300 |
| Role L11 | 251 / 300 |
| Role deep 23–25 | 248 / 300 |
| Full deep 23–25 | 241 / 300 |

## 实验结论
Dev 上 role/deep 有弱正向提升，但 held-out 没有复现：Role L11 只比 current 多 1，Role deep 反而少 2，Full deep 明显更差。旧 role 方案不能作为最终方法。

## 下一步实验
放弃继续围绕旧 role 调参，做完整 layer/head 扫描和 endpoint query。

---

# 实验 7：完整 layer 0–31 扫描

## 实验目的
确认 L11 是否真的是最佳层，是否存在更适合目标定位或关系响应的层。

## 预期结果
可能存在比 L11 更好的层，或深层在关系响应上更强。

## 实际结果数据
Target discrimination top-16 最优：
- L11 role：0.1613
- L11 SR+plate0.25：0.1607
- L11 SR：0.1548

Target discrimination top-32 最优：
- L14 role：0.1977
- L14 SR+plate0.5：0.1943
- L14 SR：0.1893

Relation-switch response 最优层：
- L23、L4、L30、L22、L24、L25

L11 relation-switch 仅约：
```text
0.0428
```

## 实验结论
- L11 不是所有指标上的最佳层；
- L14 在 top-32 目标区分度上优于 L11；
- 深层对关系变化更敏感，但对目标/干扰区分差；
- 不存在单一层同时最优。

## 下一步实验
选择少数定位 heads，而不是继续用整层平均。

---

# 实验 8：Head selection

## 实验目的
验证整层平均是否掩盖了少数定位 heads。

## 预期结果
选出的少数 heads 应同时保留目标区分度和关系切换响应。

## 实际结果数据
Selection：states 0–9；Validation：states 10–19。

Top-3：

| layer | head | visual mass | target discrimination | switch response |
|---:|---:|---:|---:|---:|
| 14 | 5 | 0.382 | 0.158 | 0.155 |
| 7 | 14 | 0.268 | 0.091 | 0.171 |
| 9 | 12 | 0.363 | 0.081 | 0.180 |

Top-5 额外加入：

| layer | head |
|---:|---:|
| 19 | 11 |
| 21 | 2 |

Validation 上指标稳定。

## 实验结论
少数定位 heads 的候选是有效的，值得进行闭环验证。

## 下一步实验
构建 endpoint query + selected heads 的 ranking repair matrix。

---

# 实验 9：Ranking Repair Matrix

## 实验目的
验证末端 query 和 selected heads 是否能修复 L11 ranking。

## 预期结果
至少有一个 arm 能稳定超过 current/vanilla。

## 实际结果数据
States 20–49，300 episodes / arm：

| Arm | Success |
|---|---:|
| Vanilla | **260 / 300** |
| selected5_endpoint | **259 / 300** |
| selected3_instruction | 253 / 300 |
| sr_d025_L11 | 252 / 300 |
| sr_d025_selected3 | 252 / 300 |
| current L11 | 250 / 300 |
| endpoint_L11 | 250 / 300 |
| selected3_full_endpoint | 250 / 300 |
| endpoint_L14 | 248 / 300 |
| selected3_endpoint | 243 / 300 |

selected5 vs current：
```text
rescue = 30
harm = 21
net = +9
p = 0.2624
```

selected5 vs vanilla：
```text
rescue = 27
harm = 28
net = -1
p = 1.0000
```

selected5 逐任务：

| 任务 | Vanilla | current | selected5 |
|---|---:|---:|---:|
| T1 between plate / ramekin | 28 | 27 | 27 |
| T2 next to ramekin | 28 | 22 | 21 |
| T3 from table center | 26 | 27 | 28 |
| T4 on cookie box | 30 | 26 | 27 |
| T5 in top drawer | 23 | 19 | 27 |
| T6 on ramekin | 26 | 28 | 29 |
| T7 next to cookie box | 28 | 29 | 30 |
| T8 on stove | 29 | 28 | 25 |
| T9 next to plate | 24 | 20 | 23 |
| T10 on wooden cabinet | 18 | 24 | 22 |

## 实验结论
selected5 基本追平 vanilla，但未超过；它比 current 改善 9 个成功，但仍有大量 rescue/harm 抵消。T2 仍未修复，T8 出现新增退化。

## 下一步实验
分析 rescue/harm，检查干预强度是否过大。

---

# 实验 10：Rescue / Harm 轨迹分析

## 实验目的
解释 selected5 为什么几乎追平但无法超过 vanilla。

## 预期结果
Harm 组的动作修正应比 Rescue 组更强。

## 实际结果数据
selected5 vs vanilla，300 paired：

| Group | n | First divergence | Action L2 | Guided change | mean m |
|---|---:|---:|---:|---:|---:|
| Rescue | 27 | 0.33 | 0.156 | 0.217 | 34.15 |
| Harm | 28 | 0.36 | 0.210 | 0.329 | 35.89 |
| Both success | 232 | 0.60 | 0.155 | 0.216 | 32.92 |
| Both fail | 13 | 0.38 | 0.161 | 0.332 | 35.44 |

## 实验结论
Harm 组的 guided change 和首步动作变化明显更大，说明当前负结果很可能受到过大干预强度影响。

## 下一步实验
固定 ranking，扫描 lambda。

---

# 实验 11：Lambda sweep（正在跑，部分结果）

## 实验目的
检验 current L11 和 selected5 是否因为 lambda 过大导致 harm。

## 预期结果
较小 lambda 应减少 harm，可能保留 rescue，从而超过 vanilla。

## 实际结果数据（快照 09:59）
| Arm | Completed | Success |
|---|---:|---:|
| current λ=0.25λ₀ | 113 / 300 | 99 |
| current λ=0.50λ₀ | 114 / 300 | 102 |
| selected5 λ=0.25λ₀ | 102 / 300 | 90 |
| selected5 λ=0.50λ₀ | 103 / 300 | 90 |

早期共同配对中：
- current λ=0.50λ₀ 相对 current 有约 +6
- 但仍低于 vanilla 约 2
- selected5 低 λ 也有改善，但暂时没超过 vanilla

## 实验结论（暂定）
降低 lambda 有减少 harm 的趋势，但还没有净收益，不能定为成功。

## 下一步实验
等完整 300 跑完后看最终结果；随后做：
1. selected5 head ablation；
2. GT 目标区域 / 目标+参照物区域诊断；
3. 阶段干预诊断；
4. 最后再复核 matched 预算。

---

# 当前总体结论

1. 旧版 custom 和官方协议下，matched 都未超过 vanilla。
2. over-masking 存在，但不是全部原因。
3. L11 attention 不能稳定定位 LIBERO-Spatial 的空间关系目标。
4. L11 不是唯一可用层；L14 在 top-32 目标区分度更好，深层关系响应更强，但目标区分差。
5. selected5_endpoint 是目前最好的修复版，基本追平 vanilla，但没有稳定超过。
6. Rescue/Harm 分析显示 harm 组干预更强，lambda sweep 正在验证是否是引导强度问题。
7. 如果 GT 正确区域 + 小 lambda 仍不能超过 vanilla，则应停止继续调 query/layer/head，承认当前方法在 LIBERO 上没有可靠增益。
