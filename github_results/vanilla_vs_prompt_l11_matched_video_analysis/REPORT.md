# Vanilla vs Prompt-L11-Matched 配对视频

左侧为 Vanilla，右侧为当前 Prompt-Attention（单层 L11、Matched 覆盖量）。视频按控制步同步；一侧轨迹较短时保留其最后一帧。每个案例提供完整 10 FPS 视频和关键分歧附近的 3 FPS 慢放。

这些视频不是重新推理得到的另一批 rollout：它们从相同 canonical snapshot 出发，回放原实验保存的 `executed_actions`。每对的 canonical snapshot、初始 simulator state 和初始 RGB 三重 hash 均一致。由于抓取接触存在微小物理敏感性，候选按 seed 从小到大检查，只保留保存成败在回放中也完全复现的前两例；不按行为解释挑案例。

## open_drawer

| 类型 | Seed | 完整视频 | 关键慢放 | 实际过程 |
|---|---:|---|---|---|
| Rescue | 001 | [全程](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/open_drawer/rescue_seed_001_paired.mp4>) | [慢放](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/open_drawer/rescue_seed_001_key_slow.mp4>) | Vanilla 到达抽屉前但始终没有形成有效拉动，直到第111步才有不足1 cm的微小位移；Prompt 在第43步开始拉动，第52步完成。 |
| Rescue | 017 | [全程](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/open_drawer/rescue_seed_017_paired.mp4>) | [慢放](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/open_drawer/rescue_seed_017_key_slow.mp4>) | Vanilla 多次在把手附近运动但没有带动抽屉；Prompt 第50步形成有效拉动，持续扩大开度并在第103步完成。 |
| Harm | 013 | [全程](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/open_drawer/harm_seed_013_paired.mp4>) | [慢放](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/open_drawer/harm_seed_013_key_slow.mp4>) | Vanilla 第46步形成有效拉动并在第58步完成；Prompt 虽到达抽屉前方，却全程没有使目标抽屉产生可测位移。 |
| Harm | 042 | [全程](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/open_drawer/harm_seed_042_paired.mp4>) | [慢放](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/open_drawer/harm_seed_042_key_slow.mp4>) | Vanilla 第55步开始拉动、第64步完成；Prompt 到第87步才产生轻微位移，之后没有持续拉开，最终开度不足。 |

## close_drawer

抽屉的“首次运动”是第一次产生1 cm有效关闭位移，可视为首次有效接触/推动节点。

| 类型 | Seed | 完整视频 | 关键慢放 | 实际过程 |
|---|---:|---|---|---|
| Rescue | 000 | [全程](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/close_drawer/rescue_seed_000_paired.mp4>) | [慢放](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/close_drawer/rescue_seed_000_key_slow.mp4>) | 两组都在第23步开始推动；Prompt 把抽屉持续推到 qpos≈0.017 并于第30步完成，Vanilla 停在≈0.070，属于“接触成功但推进不足”。 |
| Rescue | 001 | [全程](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/close_drawer/rescue_seed_001_paired.mp4>) | [慢放](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/close_drawer/rescue_seed_001_key_slow.mp4>) | Vanilla 接近后没有带动抽屉；Prompt 第33步开始有效推动，第40步完全关闭。 |
| Harm | 009 | [全程](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/close_drawer/harm_seed_009_paired.mp4>) | [慢放](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/close_drawer/harm_seed_009_key_slow.mp4>) | Prompt 第57步只有短暂小位移，之后停在 qpos≈0.177；Vanilla 虽较晚接触，但随后持续推动并在第99步完成。 |
| Harm | 049 | [全程](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/close_drawer/harm_seed_049_paired.mp4>) | [慢放](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/close_drawer/harm_seed_049_key_slow.mp4>) | 两组接触时间接近；Vanilla 第57步开始并于第66步关完，Prompt 的有效推进很迟且很慢，最终停在 qpos≈0.152。 |

## pick_coke_can

视频状态栏中的 `near` 对应夹爪首次进入目标邻域，`grasp` 对应首次稳定抓取。

| 类型 | Seed | 完整视频 | 关键慢放 | 实际过程 |
|---|---:|---|---|---|
| Rescue | 007 | [全程](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/pick_coke_can/rescue_seed_007_paired.mp4>) | [慢放](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/pick_coke_can/rescue_seed_007_key_slow.mp4>) | 两组都接触并抓住红色 Coke；Prompt 更早接近（14 vs 25步）、更早抓取（21 vs 29步），随后形成足够提升并在第28步成功；Vanilla 保持低位，未满足完成条件。 |
| Rescue | 013 | [全程](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/pick_coke_can/rescue_seed_013_paired.mp4>) | [慢放](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/pick_coke_can/rescue_seed_013_key_slow.mp4>) | Vanilla 基本没有向红色 Coke 建立有效接近；Prompt 在第52步进入目标邻域、第60步抓住，并在第62步完成提升。 |
| Harm | 041 | [全程](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/pick_coke_can/harm_seed_041_paired.mp4>) | [慢放](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/pick_coke_can/harm_seed_041_key_slow.mp4>) | Vanilla 第29步接近红色 Coke、第33步抓取并马上完成；Prompt 明显转向桌面中央的苹果/海绵区域，全程没有进入 Coke 邻域。 |
| Harm | 049 | [全程](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/pick_coke_can/harm_seed_049_paired.mp4>) | [慢放](</home/leju-suzhou/zjt_ws/token-cd/artifacts/vanilla_vs_prompt_l11_matched_video_analysis/pick_coke_can/harm_seed_049_key_slow.mp4>) | Vanilla 对准红色 Coke，第45步抓取、第46步成功；Prompt 曾靠近目标，但始终停在约12–18 cm距离，没有形成抓取。 |

## 这12对直接说明什么

- Rescue 不止一种：既有“原本完全没形成有效接触”，也有“双方都接触，但 Prompt 让操作持续到完成”。
- Harm 也不止一种：既有明确走向干扰物，也有目标正确但接触/持续操作失败。
- 因此，Prompt-L11-Matched 不是统一地“增强目标定位”或统一地“加大动作”。它会改变接近与接触后的控制过程；下一步若分析原因，必须回到关键分歧前的同状态 clean/guided 动作。

逐帧事件与物理量保存在各 `seed_XXX_analysis.json` 中。未入选但已生成的回放只用于技术复现筛选，不作为这12对证据的一部分。
