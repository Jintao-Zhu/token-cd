# L11 matched-budget causal pilot

Seeds 100–124, four tasks. Matched is reused; 700 candidate episodes are newly run.

| Arm | Matched | Candidate | Rescue | Harm | Net | Exact p | Mean count |
|---|---:|---:|---:|---:|---:|---:|---:|
| global_shuffle | 51/100 | 48/100 | 15 | 18 | -3 | 0.728332 | 34.14 |
| within_task_shuffle | 51/100 | 47/100 | 15 | 19 | -4 | 0.607591 | 33.97 |
| episode_fixed | 51/100 | 44/100 | 11 | 18 | -7 | 0.264931 | 28.60 |
| matched_scale_050 | 51/100 | 41/100 | 8 | 18 | -10 | 0.0755187 | 16.86 |
| matched_scale_075 | 51/100 | 42/100 | 9 | 18 | -9 | 0.122078 | 24.92 |
| matched_scale_125 | 51/100 | 42/100 | 10 | 19 | -9 | 0.136046 | 42.50 |
| matched_scale_150 | 51/100 | 53/100 | 18 | 16 | +2 | 0.864166 | 51.56 |

## Frozen screening decision

**EXTEND** to the full 100-seed study.

## Mechanism decomposition

The contrasts have distinct causal meanings:

- `matched > global_shuffle`: the budget is not explained by its marginal distribution alone.
- `matched > within_task_shuffle`: the budget must remain paired with its own state, beyond task identity.
- `matched > episode_fixed`: within-episode budget changes matter.
- A peak at scale 1.0: the absolute KMeans-derived intervention dose is locally calibrated.

Task-centered episode-level correlations trace how count changes propagate through the negative branch:

- count vs feature perturbation: 0.9696
- count vs centered logit residual: 0.5782
- feature perturbation vs centered logit residual: 0.6246

These correlations establish the intervention-strength pathway; paired success contrasts determine whether that pathway is beneficial.

Detailed evidence is saved in `EPISODE_MECHANISMS.csv` and `PHASE_MECHANISMS.csv`.
