# Budget estimator comparison (offline)

状态数: **800**（4 任务 × 50 seeds × 4 progress）

- 校准目标: 让 Relative / Spectral 的平均预算 ≈ Matched 的 **34.02**
- **gamma\* = 1.750** → 平均 m = 34.34
- **tau\* = 0.900** → 平均 m = 60.07

## 三个估计量的分布

| 估计量 | mean | std | median | min | max | P10 | P90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Matched | 34.02 | 16.60 | 34.0 | 2 | 89 | 12 | 53 |
| Relative | 34.34 | 6.67 | 34.0 | 18 | 54 | 26 | 43 |
| Spectral | 60.07 | 4.37 | 60.5 | 50 | 70 | 54 | 66 |

## 与 Matched 的 Spearman 相关

| 范围 | n | rho(Matched, Relative) | rho(Matched, Spectral) |
|---|---:|---:|---:|
| **总体** | 800 | **0.024** | **-0.290** |
| open_drawer | 200 | 0.048 | -0.240 |
| close_drawer | 200 | -0.218 | -0.305 |
| pick_coke_can | 200 | 0.068 | -0.182 |
| move_near | 200 | -0.101 | -0.269 |

## episode 内中心化（去掉任务/场景造成的假相关）

| 对比 | rho | p |
|---|---:|---:|
| Matched vs Relative | **-0.101** | 0.0041 |
| Matched vs Spectral | **-0.124** | 0.0004 |

## 每条 episode 内的相关（逐条算再汇总）

| 对比 | 逐条 rho 均值 | 中位 | 正相关比例 |
|---|---:|---:|---:|
| Matched vs Relative | -0.048 | 0.080 | 51% |
| Matched vs Spectral | -0.103 | -0.258 | 43% |
