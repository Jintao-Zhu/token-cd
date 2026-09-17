# Offline Layer Shortlist

## Scope

- Data: 90 cached same-image states from 30 episodes.
- Locked split: 20 exploration episodes / 60 states; 10 validation episodes / 30 states.
- Layers: L0-L31.
- No manually annotated image boxes are used.
- Every layer uses the same state-matched intervention budget `m_t`.
- This report is an offline screen. It does not replace closed-loop evaluation.

## What Was Measured

1. **Action relevance**: whether masking a layer's Prompt Top-`m_t` tokens changes action logits in the same direction as changing the task prompt.
2. **Random control advantage**: action relevance minus the result from a Random-`m_t` mask.
3. **Synonym consistency**: whether equivalent prompt wording produces a similar action-logit intervention.
4. **Target selectivity**: whether changing the task target produces a different intervention. Reported as synonym residual cosine minus changed-target residual cosine.
5. **Mask selectivity**: synonym-mask Jaccard minus changed-target-mask Jaccard.

## Main Result

The expanded offline tests do **not** prove that L11 is the unique best layer. They support a narrower conclusion:

> L11 is a strong and robust prompt-conditioned action-intervention layer, and it remains one of three credible offline candidates: L9, L11, and L14.

L11 is especially strong at distinguishing an equivalent prompt from a genuinely changed target:

| Split | Action alignment | Alignment rank | Semantic separation | Separation rank | Mask separation rank |
|---|---:|---:|---:|---:|---:|
| Exploration | 0.3410 | 5 / 32 | 0.1182 | 2 / 32 | 1 / 32 |
| Validation | 0.2961 | 3 / 32 | 0.1141 | 1 / 32 | 1 / 32 |

Its synonym residual cosine is also high in absolute terms (0.9184 / 0.9244), although several shallow layers are even more invariant because they react weakly to both synonym and target changes.

## Candidate Comparison

| Layer | Exp action alignment | Val action alignment | Exp synonym cosine | Val synonym cosine | Exp semantic separation | Val semantic separation |
|---|---:|---:|---:|---:|---:|---:|
| L9 | 0.3509 | 0.2777 | 0.9229 | 0.9293 | 0.0851 | 0.0743 |
| **L11** | **0.3410** | **0.2961** | **0.9184** | **0.9244** | **0.1182** | **0.1141** |
| L14 | 0.3460 | 0.2838 | 0.9172 | 0.9328 | 0.1547 | 0.0980 |

A descriptive, post-hoc screen required all of the following in both splits:

- action alignment in the top quartile;
- synonym residual cosine at least 0.90;
- semantic separation at least 0.05.

Only **L9, L11, and L14** passed. This screen was not preregistered and should be treated as a shortlist, not a statistical proof.

## Pairwise Uncertainty

Paired episode-bootstrap confidence intervals all include zero:

| Split | Comparison | Metric difference | Mean | 95% CI |
|---|---|---|---:|---:|
| Exploration | L11 - L9 | Action alignment | -0.0099 | [-0.0536, 0.0294] |
| Exploration | L11 - L9 | Semantic separation | +0.0332 | [-0.0203, 0.0841] |
| Exploration | L11 - L14 | Action alignment | -0.0050 | [-0.0365, 0.0253] |
| Exploration | L11 - L14 | Semantic separation | -0.0365 | [-0.1028, 0.0220] |
| Validation | L11 - L9 | Action alignment | +0.0184 | [-0.0241, 0.0590] |
| Validation | L11 - L9 | Semantic separation | +0.0398 | [-0.0017, 0.0861] |
| Validation | L11 - L14 | Action alignment | +0.0124 | [-0.0306, 0.0540] |
| Validation | L11 - L14 | Semantic separation | +0.0161 | [-0.0470, 0.0778] |

Therefore the current offline sample cannot resolve a statistically reliable winner among L9, L11, and L14.

## Closed-Loop Boundary

- Pure **L11** has direct closed-loop evidence.
- **L11+L14** has direct closed-loop evidence and is worse than pure L11 in the existing comparison.
- Pure **L9** and pure **L14** have not been tested in closed loop.
- Consequently, the existing result proves that adding L14 by equal-weight score fusion hurts L11; it does **not** prove that L14 alone is worse than L11.

## Honest Conclusion

The evidence currently supports this statement:

> L11 was not selected arbitrarily. It has strong offline prompt-conditioned action relevance and unusually strong target selectivity, and it is the best single-layer selector among the arms actually tested in closed loop. However, an exhaustive offline scan does not establish L11 as the unique best layer; L9 and L14 remain unresolved single-layer candidates.

## Recommended Next Filter

Before a larger closed-loop run, test budget robustness for L9, L11, and L14 using identical cached states and budget scales such as 0.75, 1.0, and 1.25. If the shortlist remains stable, run a small same-seed closed-loop comparison of the three single layers. Do not infer pure-L14 performance from the failed L11+L14 fusion.
