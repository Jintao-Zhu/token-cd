# Positive-Support Constrained SHR — offline report

PSC is evaluated on stored float16 action-vocabulary logits. A changed token is not a
counterfactual success label; PSC Rescue/Harm requires closed-loop rollout.

| Task | Method | Changed dims | Change rate | Changed episodes | Episode rate | Mean support |
|---|---|---:|---:|---:|---:|---:|
| close_drawer | k10 | 662 | 0.2790% | 222 | 74.00% | 10.00 |
| close_drawer | k20 | 50 | 0.0211% | 47 | 15.67% | 20.00 |
| close_drawer | k50 | 0 | 0.0000% | 0 | 0.00% | 50.00 |
| close_drawer | rel010 | 11393 | 4.8011% | 300 | 100.00% | 2.14 |
| open_drawer | k10 | 644 | 0.2714% | 245 | 81.67% | 10.00 |
| open_drawer | k20 | 66 | 0.0278% | 64 | 21.33% | 20.00 |
| open_drawer | k50 | 0 | 0.0000% | 0 | 0.00% | 50.00 |
| open_drawer | rel010 | 12632 | 5.3232% | 300 | 100.00% | 2.28 |
| pick_coke_can | k10 | 204 | 0.1214% | 126 | 42.00% | 10.00 |
| pick_coke_can | k20 | 7 | 0.0042% | 7 | 2.33% | 20.00 |
| pick_coke_can | k50 | 0 | 0.0000% | 0 | 0.00% | 50.00 |
| pick_coke_can | rel010 | 4193 | 2.4958% | 298 | 99.33% | 1.76 |
| move_near | k10 | 687 | 0.4089% | 258 | 86.00% | 10.00 |
| move_near | k20 | 22 | 0.0131% | 22 | 7.33% | 20.00 |
| move_near | k50 | 0 | 0.0000% | 0 | 0.00% | 50.00 |
| move_near | rel010 | 9943 | 5.9185% | 300 | 100.00% | 2.58 |

## Interpretation guardrail

Existing Vanilla/SHR outcomes are used only to stratify which old trajectories are touched.
They are not relabeled as PSC outcomes. Run paired PSC rollouts to measure Rescue/Harm.
