# TopP90 vs matched 判决实验：λ=1.0，seeds 100–199（独立种子段）

> **结论口径（2026-10-16 补注）**：本文件只报告 100–199 这一段的 400 对。**该对比的正式结论应当用合并两段的 800 对**：
> TopP90 **403/800 (50.4%)** vs matched **389/800 (48.6%)**，差值 **+1.75pp**，95% CI **[−1.88, +5.50]**，p=0.398，
> 代价是 **+73% 的干预 token**（TopP90 平均 m_t 57.9 vs matched 33.4）。
> 两段之间确有异质性（+6.25pp vs −2.75pp，差 9.00pp，z=2.35，p=0.019），但调节变量分析
> （[OPENVLA_TOPP90_MODERATORS.md](OPENVLA_TOPP90_MODERATORS.md)）未能用场景复杂度或预算比解释它，
> 分层效应呈 **+13 / −11 / −6 / +11pp 的非单调摆动** —— 说明 ±10pp 就是这个样本量下的噪声尺度。
> 另注：400 对的配对检验只能可靠检出 **≥8pp** 的差异；2pp 需要约 5,900 对。


生成 2026-10-16T19:23:36+0800。两臂在同一进程内成对运行。
覆盖完整：**True**  {'close_drawer': [100, 100], 'open_drawer': [100, 100], 'pick_coke_can': [100, 100], 'move_near': [100, 100]}

## 本次结果

| 条件 | 对数 | TopP90 | matched | 净 | 救/坏 | 差值 | bootstrap 95% CI | McNemar p |
|---|---|---|---|---|---|---|---|---|
| 合并 | 400 | 177 (44.2%) | 188 (47.0%) | -11 | 53/64 | -2.75pp | [-9.00, +3.50] | 0.3553 |
| close_drawer | 100 | 73 | 83 | -10 | 11/21 | -10.00pp | — | 0.1102 |
| open_drawer | 100 | 35 | 40 | -5 | 14/19 | -5.00pp | — | 0.4869 |
| pick_coke_can | 100 | 31 | 25 | +6 | 16/10 | +6.00pp | — | 0.3269 |
| move_near | 100 | 38 | 40 | -2 | 12/14 | -2.00pp | — | 0.8450 |

实测预算：TopP90 平均 m_t **57.9** vs matched **33.4**

## 与先前三次测量的对照

| 设置 | TopP90 | matched | 差值 | p | 说明 |
|---|---|---|---|---|---|
| λ=1.0, seeds 200-299, 4 Google 任务 | 226/400 | 201/400 | +6.25pp | 0.0274 | 网格实验（原始 p；Holm(8)=0.164）—— 9 任务实验的 4 任务子集，属同一份证据 |
| λ=1.0, seeds 200-299, 全部 9 任务 | 293/900 | 281/900 | +1.33pp | 0.4604 | 加入 WidowX 与 apple 后差异缩小到 +1.33pp |
| λ=0.5, seeds 100-199, 4 Google 任务 | 191/400 | 213/400 | -5.50pp | 0.0651 | 旧管线（TopP 无 m_t 记录），方向相反：TopP90 明显更差 |
| **本次（seeds 100–199, λ=1.0）** | 177/400 | 188/400 | **-2.75pp** | 0.3553 | 独立种子段 |

## 判据

- If TopP90 wins again here with a CI excluding zero, the seeds-200-299 advantage is corroborated. If the difference is near zero or negative, the advantage was multiplicity noise.

## 限制

- Seeds 100-199 are an independent seed range for THIS contrast at lambda=1.0, but the same states were used by older lambda=0.5 studies; that does not affect this test.
- Both arms were run in one process per chunk, so the pairs share process history.
- A single seed range still cannot separate 'TopP90 is better on Google tasks' from 'this seed range happens to favour it'; the verdict combines three measurements.
