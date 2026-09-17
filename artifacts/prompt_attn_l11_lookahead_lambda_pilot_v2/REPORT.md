# Training-Free L11 Lambda Lookahead Pilot

Date: 2026-09-14

## Design

- Candidate lambdas: 0, .25, and .5.
- Default: lambda=.5.
- Every 10 actions, restore the same current simulator state and run each candidate for 10 actions.
- Score candidates with privileged task-native physical progress and a terminal-success bonus.
- Approach phases require a .025 score advantage to switch; manipulation phases require .002.
- Near-tied candidates prefer the lambda closest to .5.
- The selected branch's complete simulator state is committed exactly; all state-commit audits passed.

This is a targeted diagnostic, not an unbiased success-rate estimate. The ten seeds were selected from known lambda=.25-only, lambda=.5-only, or both-success fixed-arm outcomes.

## Paired outcomes

| Task | Seed | lambda=0 | lambda=.25 | lambda=.5 | Lookahead | Switches |
|---|---:|---:|---:|---:|---:|---:|
| close_drawer | 0 | 0 | 1 | 1 | 1 | 1 |
| close_drawer | 1 | 0 | 0 | 1 | 0 | 5 |
| close_drawer | 49 | 1 | 1 | 0 | 1 | 3 |
| close_drawer | 54 | 0 | 1 | 0 | 0 | 6 |
| close_drawer | 55 | 0 | 1 | 0 | 1 | 3 |
| open_drawer | 0 | 0 | 1 | 0 | 1 | 2 |
| open_drawer | 1 | 0 | 0 | 1 | 0 | 4 |
| open_drawer | 2 | 0 | 0 | 1 | 0 | 3 |
| open_drawer | 5 | 0 | 1 | 0 | 0 | 5 |
| open_drawer | 12 | 0 | 1 | 0 | 0 | 0 |

## Summary

| Task | N | lambda=.5 | Lookahead | Fixed-arm Oracle | Rescue | Harm | Net |
|---|---:|---:|---:|---:|---:|---:|---:|
| close_drawer | 5 | 2 | 3 | 5 | 2 | 1 | +1 |
| open_drawer | 5 | 2 | 1 | 5 | 1 | 2 | -1 |
| Overall | 10 | 4 | 4 | 10 | 3 | 3 | 0 |

## Decision

**STOP before an unbiased large-scale rollout.**

The implementation proves that exact same-state branching and privileged progress-based lambda selection are technically feasible. It also rescues some lambda=.25-only episodes. However, ten-step physical progress is not selective enough: it destroys as many lambda=.5 successes as it rescues lambda=.5 failures, while missing three lambda=.25-only successes. The targeted set has a 100% retrospective fixed-arm Oracle by construction, but lookahead reaches only 40%, equal to fixed lambda=.5 on these cases.

The result rejects the current short-horizon progress score, not every possible lookahead method. Approaching the fixed-arm Oracle would require a more predictive long-horizon verifier, substantially longer rollouts, or a learned value/success model.

