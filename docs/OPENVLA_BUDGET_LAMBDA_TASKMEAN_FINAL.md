# Reloaded budget-rule analysis: lambda x rule interaction and the task-mean ablation

Generated 2026-10-16T11:54:35+0800 by `scripts/analyze_openvla_budget_lambda_taskmean.py`.
Design: 4 Google tasks, seeds 200-299, 5 budget rules, lambda 0.5 and 1.0; every contrast is paired on identical (task, seed).

## Pooled success rates (successes/denominator, percent, Wilson 95% CI)

| rule | lambda | successes/denominator | percent | Wilson 95% | complete |
|---|---|---|---|---|---|
| Matched | 0.5 | 205/400 | 51.25% | [46.4, 56.1] | True |
| Matched | 1.0 | 201/400 | 50.25% | [45.4, 55.1] | True |
| Fixed 16 | 0.5 | 165/400 | 41.25% | [36.5, 46.1] | True |
| Fixed 16 | 1.0 | 206/400 | 51.50% | [46.6, 56.4] | True |
| Fixed 48 | 0.5 | 199/400 | 49.75% | [44.9, 54.6] | True |
| Fixed 48 | 1.0 | 202/400 | 50.50% | [45.6, 55.4] | True |
| Random 48 | 0.5 | 157/400 | 39.25% | [34.6, 44.1] | True |
| Random 48 | 1.0 | 189/400 | 47.25% | [42.4, 52.1] | True |
| TopP 0.90 | 0.5 | 201/400 | 50.25% | [45.4, 55.1] | True |
| TopP 0.90 | 1.0 | 226/400 | 56.50% | [51.6, 61.3] | True |

## Each rule against Matched, within each lambda

Rescue = the rule succeeds where Matched fails. Holm is applied over the four non-matched rules within each lambda.

| lambda | rule | pairs | rule | matched | rescue | harm | diff (pp) | bootstrap 95% CI | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | Fixed 16 | 400 | 205 | 165 | 89 | 49 | +10.00 | [+3.25, +16.75] | 0.000836 | 0.002508 |
| 0.5 | Fixed 48 | 400 | 205 | 199 | 50 | 44 | +1.50 | [-5.50, +8.00] | 0.6063 | 1 |
| 0.5 | Random 48 | 400 | 205 | 157 | 88 | 40 | +12.00 | [+5.25, +18.50] | 2.663e-05 | 0.0001065 |
| 0.5 | TopP 0.90 | 400 | 205 | 201 | 60 | 56 | +1.00 | [-5.50, +8.00] | 0.7807 | 1 |
| 1.0 | Fixed 16 | 400 | 201 | 206 | 49 | 54 | -1.25 | [-7.50, +5.25] | 0.6937 | 1 |
| 1.0 | Fixed 48 | 400 | 201 | 202 | 58 | 59 | -0.25 | [-6.75, +6.00] | 1 | 1 |
| 1.0 | Random 48 | 400 | 201 | 189 | 64 | 52 | +3.00 | [-3.50, +9.50] | 0.3071 | 0.9213 |
| 1.0 | TopP 0.90 | 400 | 201 | 226 | 47 | 72 | -6.25 | [-12.75, +0.00] | 0.02738 | 0.1095 |

## Within-rule lambda effect (lambda 1.0 minus lambda 0.5)

| rule | pairs | lambda 1.0 | lambda 0.5 | rescue | harm | diff (pp) | bootstrap 95% CI | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| Matched | 400 | 201 | 205 | 63 | 67 | -1.00 | [-7.50, +5.50] | 0.7926 | 1 |
| Fixed 16 | 400 | 206 | 165 | 82 | 41 | +10.25 | [+3.50, +16.75] | 0.0002756 | 0.001378 |
| Fixed 48 | 400 | 202 | 199 | 61 | 58 | +0.75 | [-5.75, +7.50] | 0.8546 | 1 |
| Random 48 | 400 | 189 | 157 | 73 | 41 | +8.00 | [+1.25, +14.75] | 0.003505 | 0.01402 |
| TopP 0.90 | 400 | 226 | 201 | 73 | 48 | +6.25 | [-0.50, +13.00] | 0.0287 | 0.08611 |

## The interaction itself

I = (Matched - rule) at lambda 0.5 minus (Matched - rule) at lambda 1.0, computed on the same task/seed cells and bootstrapped within task. Positive means the rule's disadvantage relative to Matched is smaller at higher lambda.

| rule | I (pp) | bootstrap 95% CI | bootstrap p | Holm p |
|---|---|---|---|---|
| Fixed 16 | +11.25 | [+3.75, +18.75] | 0.0052 | 0.0208 |
| Fixed 48 | +1.75 | [-5.50, +9.00] | 0.661 | 0.661 |
| Random 48 | +9.00 | [+1.50, +16.25] | 0.0192 | 0.0576 |
| TopP 0.90 | +7.25 | [-0.50, +15.00] | 0.0648 | 0.1296 |

## Task-mean constant instead of per-step adaptation

| setting | pairs | task-mean | matched | rescue | harm | diff (pp) | bootstrap 95% CI | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| lambda 0.5 (seeds 200-299) | 400 | 182 | 205 | 54 | 77 | -5.75 | [-12.50, +1.00] | 0.05417 | 0.1083 |
| lambda 1.0 (seeds 200-299) | 400 | 199 | 201 | 63 | 65 | -0.50 | [-7.00, +5.75] | 0.9296 | 0.9296 |
| lambda 0.5 (seeds 100-199, older protocol) | 400 | 185 | 213 | 44 | 72 | -7.00 | [-13.75, -0.25] | 0.01185 | n/a (separate replication) |

## Per-task task-mean contrasts (descriptive; no separate correction claimed)

**lambda 0.5**: open_drawer 34-50 (net -16, p=0.011); close_drawer 65-71 (net -6, p=0.286); pick_coke_can 32-32 (net +0, p=1.000); move_near 51-52 (net -1, p=1.000)
**lambda 1.0**: open_drawer 53-40 (net +13, p=0.072); close_drawer 73-84 (net -11, p=0.061); pick_coke_can 28-29 (net -1, p=1.000); move_near 45-48 (net -3, p=0.648)

## Caveats that must travel with these numbers

- The interaction is estimated directly as a paired contrast; it is never inferred from one comparison being significant and another not.
- Non-significance is not equivalence: every non-significant contrast is reported with its observed difference and a task-stratified paired bootstrap interval.
- The task-mean arm uses constants computed from the lambda=1.0 Matched reference on the SAME seeds 200-299 and reuses them at lambda=0.5. It is therefore a post-hoc mean-matched ablation, not an independently calibrated baseline, and at lambda=0.5 it is not a same-lambda mean-matched control.
- Random 48 changes both the count rule and the token content, so its contrast with Matched mixes both; content is isolated only by Fixed 48 versus Random 48.
- Closed-loop m_t values come from divergent trajectories and are not same-state causal evidence.
- The 100-199 replication runs on an older renderer/protocol than the 200-299 grid for pick and move, so it is a directional replication of the lambda=0.5 task-mean effect, not a bitwise continuation of the same experiment.

Figure: `docs/figures/openvla_budget_lambda_taskmean.png`.
