# SpatialVLA: 9 tasks x 300 seeds x 2 arms, all SAPIEN RT-full (depth=8)

Dataset **`SPATIALVLA_RTfull9_300_SEEDS_V1`** - canonical record: `docs/SPATIALVLA_RTfull9_300_SEEDS_V1_DATASET.md`, lock: `docs/spatialvla_rtfull9_300_seed_dataset_lock.json`.

Arms: `vanilla` (Base) vs `l10_matched` (Ours, lambda=0.5). Every episode below has `renderer.depth == 8`, spp=32, denoiser on.

Episodes merged: 5400 kept, 4000 dropped (wrong renderer / wrong arm / unreadable).

| Task | Base | +Ours | Δ | rescue | harm | net | counts |
|---|---:|---:|---:|---:|---:|---:|---|
| open_drawer | 50.7% | 63.0% | +12.3 | 41 | 4 | +37 | 152→189 / 300 |
| close_drawer | 66.0% | 80.0% | +14.0 | 53 | 11 | +42 | 198→240 / 300 |
| move_near | 37.3% | 61.0% | +23.7 | 87 | 16 | +71 | 112→183 / 300 |
| pick_coke_can | 37.3% | 79.3% | +42.0 | 139 | 13 | +126 | 112→238 / 300 |
| place_apple_in_closed_top_drawer | 0.0% | 4.3% | +4.3 | 13 | 0 | +13 | 0→13 / 300 |
| widowx_carrot_on_plate | 27.3% | 34.7% | +7.3 | 37 | 15 | +22 | 82→104 / 300 |
| widowx_put_eggplant_in_basket | 99.3% | 95.3% | -4.0 | 2 | 14 | -12 | 298→286 / 300 |
| widowx_spoon_on_towel | 22.0% | 32.7% | +10.7 | 48 | 16 | +32 | 66→98 / 300 |
| widowx_stack_cube | 33.0% | 37.0% | +4.0 | 15 | 3 | +12 | 99→111 / 300 |
| **MEAN** | **41.4%** | **54.1%** | **+12.7** | | | | 1119→1462 / 2700 |

Source artifacts (all filtered to `depth == 8`):

* `spatialvla_l10_vanilla_rtfull7_to300_v1`
* `spatialvla_remaining7_rtfull_vanilla_100_199_v1`
* `spatialvla_l10_matched_rtfull7_n100_199`
* `spatialvla_l10_vanilla_expand9_to300_v1`
* `spatialvla_simpler_l11matched_4task_n100`
* `spatialvla_l10_vs_l11_4task_n100`
