# Phase A: Matched 预算 vs 有效反事实预算

状态总数: **800**，其中有明确 knee 的: **695**

## 总相关（pooled）

- Spearman(ρ) = **0.042** (p = 0.2651), n = 695

## episode 内中心化（真正的机制检验）

- Spearman(Δmatched, Δknee) = **0.081** (p = 0.0336)
- episode-cluster bootstrap(5000): mean = 0.082, 95% CI = [-0.026, 0.188]

## 逐任务

| 任务 | n | Spearman(matched, knee) | mean |matched−knee| |
|---|---:|---:|---:|
| google_robot_open_drawer | 166 | 0.267 | 16.21 |
| google_robot_close_drawer | 171 | -0.225 | 19.50 |
| google_robot_pick_coke_can | 175 | 0.128 | 12.78 |
| google_robot_move_near | 183 | 0.002 | 19.90 |

## 方法对比：|m_method − m_knee|

| 方法 | mean | median | Spearman | |
|---|---:|---:|---:|---|
| matched | 17.13 | 16.00 | 0.042 | |
| TopP80 | 13.68 | 12.00 | 0.111 | |
| TopP85 | 18.31 | 17.00 | 0.088 | |
| fixed32 | 11.89 | 15.00 | nan | |
| m80(robust) | 20.90 | 16.00 | -0.067 | |
