# SPATIALVLA_RTfull9_300_SEEDS_V1 — 数据集正式记录

**这是本工作区认可的正式数据集，不是临时拼凑。** 下面的组成即该数据集的定义；
以后引用/复算直接按本文档，无需再回溯各个 artifact 的来历。

- **名称**：`SPATIALVLA_RTfull9_300_SEEDS_V1`
- **规模**：9 任务 × 300 seeds（0–299）× 2 arms = **5400 集**
- **模型**：SpatialVLA 4B SFT —— `checkpoints/spatialvla-4b-224-sft-fractal`（google_robot 5 任务）、`checkpoints/spatialvla-4b-224-sft-bridge`（widowx 4 任务）
- **臂**：`vanilla`（Base） vs `l10_matched`（Ours，λ=0.5，L10 prompt-attention，16×16 四邻 harmonic 负样本 β=0，greedy `(1+0.5)·pos − 0.5·neg`，action_chunk=4）
- **渲染（全数据集一致）**：SAPIEN RT **full，path_depth = 8，spp = 32，denoiser = on**
- **配对**：同 task/seed 同 canonical snapshot（`vanilla_recon_shr_canonical_0_299_v2`），初帧 RGB/状态哈希可校验
- **复算脚本**：`token-cd/scripts/build_spatialvla_rtfull9_300_table.py`（强制 `renderer.depth == 8`，任一任务 n≠300 会 WARN）
- **结果表**：`token-cd/docs/spatialvla_rtfull9_300_seed_table.md`
- **机器可读锁**：`token-cd/docs/spatialvla_rtfull9_300_seed_dataset_lock.json`

## 1. 组成（这是数据集的定义，8 条记录，全部 depth=8）

| # | 来源 artifact | 任务 | 臂 | seed 段 | n | depth |
|---|---|---|---|---|---:|---:|
| 1 | `spatialvla_l10_vanilla_rtfull7_to300_v1` | 非抽屉 7 个 | vanilla | 0–99, 200–299 | 1400 | 8 |
| 2 | `spatialvla_l10_vanilla_rtfull7_to300_v1` | 非抽屉 7 个 | l10_matched | 0–99, 200–299 | 1400 | 8 |
| 3 | `spatialvla_remaining7_rtfull_vanilla_100_199_v1` | 非抽屉 7 个 | vanilla | 100–199 | 700 | 8 |
| 4 | `spatialvla_l10_matched_rtfull7_n100_199` | 非抽屉 7 个 | l10_matched | 100–199 | 700 | 8 |
| 5 | `spatialvla_l10_vanilla_expand9_to300_v1` | 抽屉 2 个 | vanilla | 0–99, 200–299 | 400 | 8 |
| 6 | `spatialvla_l10_vanilla_expand9_to300_v1` | 抽屉 2 个 | l10_matched | 0–99, 200–299 | 400 | 8 |
| 7 | `spatialvla_simpler_l11matched_4task_n100` | 抽屉 2 个 | vanilla | 100–199 | 200 | 8 |
| 8 | `spatialvla_l10_vs_l11_4task_n100` | 抽屉 2 个 | l10_matched | 100–199 | 200 | 8 |

**任务列表**（9）：`google_robot_open_drawer`、`google_robot_close_drawer`、`google_robot_move_near`、
`google_robot_pick_coke_can`、`google_robot_place_apple_in_closed_top_drawer`、
`widowx_carrot_on_plate`、`widowx_put_eggplant_in_basket`、`widowx_spoon_on_towel`、`widowx_stack_cube`

**按臂核算**：vanilla = 1400+700+400+200 = **2700** ✅ ；l10_matched = 1400+700+400+200 = **2700** ✅
（合并时另丢弃 4000 条 depth0 / 非目标臂记录，见 §3）

## 2. 结果（n=300/任务，λ=0.5）

| Task | Base | +Ours | Δ | rescue | harm | net | counts |
|---|---:|---:|---:|---:|---:|---:|---|
| open_drawer | 50.7% | 63.0% | +12.3 | 41 | 4 | +37 | 152→189 /300 |
| close_drawer | 66.0% | 80.0% | +14.0 | 53 | 11 | +42 | 198→240 /300 |
| move_near | 37.3% | 61.0% | +23.7 | 87 | 16 | +71 | 112→183 /300 |
| pick_coke_can | 37.3% | 79.3% | +42.0 | 139 | 13 | +126 | 112→238 /300 |
| place_apple_in_closed_top_drawer | 0.0% | 4.3% | +4.3 | 13 | 0 | +13 | 0→13 /300 |
| widowx_carrot_on_plate | 27.3% | 34.7% | +7.3 | 37 | 15 | +22 | 82→104 /300 |
| widowx_put_eggplant_in_basket | 99.3% | 95.3% | −4.0 | 2 | 14 | −12 | 298→286 /300 |
| widowx_spoon_on_towel | 22.0% | 32.7% | +10.7 | 48 | 16 | +32 | 66→98 /300 |
| widowx_stack_cube | 33.0% | 37.0% | +4.0 | 15 | 3 | +12 | 99→111 /300 |
| **MEAN** | **41.4%** | **54.1%** | **+12.7** | 435 | 92 | +343 | 1119→1462 /2700 |

## 3. 明确排除的 depth0 数据（名字相近，勿混用）

| artifact | 任务 | 臂 | n | depth | 处置 |
|---|---|---|---:|---:|---|
| `spatialvla_l10_vanilla_expand9_to300_v1` | 非抽屉 7 个 | vanilla / l10_matched | 各 1400 | 0 | 排除（只取抽屉子集） |
| `spatialvla_simpler_l11matched_4task_n100` | move_near / pick_coke | vanilla / l11_matched | 各 200 | 0 | 排除 |
| `spatialvla_l10_vs_l11_4task_n100` | move_near / pick_coke | l10_matched / l11_matched | 各 200 | 0 | 排除 |
| **`spatialvla_drawers_rt_depth0_vanilla_l10_100_199_v1`** | 抽屉 2 个 | vanilla / l10_matched | 各 200 | 0 | **整份排除**（名字含 "rt" 但 depth=0） |

> 注意：`expand9_to300` 指 "扩展到 9 任务 / 300 seed"，**不代表全 depth8** —— 它 3600 条里只有抽屉的 800 条是 depth8，其余 2800 条是 depth0。
> 抽屉 100–199 的 vanilla 取自名字带 `l11matched` 的 artifact（取其 **vanilla 臂**，depth8）；l10 取自 `l10_vs_l11_4task`（取其 **l10_matched 臂**，depth8）。

## 4. 校验方式

```bash
cd /home/leju-suzhou/zjt_ws/token-cd
python scripts/build_spatialvla_rtfull9_300_table.py     # 应输出 5400 kept, 4000 dropped，且各任务 n=300
```

出现 `WARNING: ... n≠300` 或 MEAN 不等于 41.4 / 54.1，说明某个来源 artifact 被改动，需重新核对 §1。
