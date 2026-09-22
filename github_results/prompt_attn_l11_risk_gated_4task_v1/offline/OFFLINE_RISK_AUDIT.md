# L11 Risk-Gated Lambda — Offline Discovery Audit

Discovery set: four tasks, seeds 0--49, 200 fixed-L11 episodes.

- Category counts: `{'rescue': 34, 'stable_success': 65, 'fixed_cd_harm': 10, 'stable_fail': 91}`
- Harm-vs-L11-success R90 AUC: **0.5803**
- Bootstrap 95% interval: **[0.3959, 0.7818]**
- Selected threshold: `b0.7_g-0.2_m1.5`
- Harm hit rate: **90.0%**
- L11-success hit rate: **81.8%**
- Stable-success hit rate: **76.9%**
- Timestep trigger rate: **2.2%**
- Implied mean lambda: **0.4945**
- Go/no-go: **STOP**

## Leave-one-task-out threshold selections

- close_drawer: `b0.5_g0_m1.5`
- pick_coke_can: `b0.7_g-0.2_m1.5`
- move_near: `b0.7_g-0.2_m1.5`
- carrot_on_plate: `b0.5_g0_m1.5`
