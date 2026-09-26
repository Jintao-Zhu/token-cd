# Current Best Answer

The leading explanation is that L11-Matched transfers poorly because its counterfactual action correction has episode- and task-dependent value in LIBERO: it rescues some runs and harms nearly as many, while the existing evidence does not yet identify whether harmonic reconstruction or downstream action guidance causes that variability.

## Confidence

PARTIALLY SUPPORTED

## Primary Bottleneck

1. **Counterfactual action value / action guidance** is the leading proximal candidate. The frozen method produces many Rescue and Harm outcomes; action-margin analyses show different Rescue/Harm patterns, but they are observational and sensitive to trace/render fidelity.
2. **Harmonic negative construction** remains a secondary candidate. One tightly controlled feature-kNN replacement barely changed episode-level top-two action-margin effects, so this specific alternative does not explain or fix the transfer gap.

This is not a causal identification of either candidate. SIMPLER and LIBERO also differ in benchmark, tasks, simulator, and checkpoint, so the contrast is not an isolated domain-transfer factor.

## Evidence Chain

**Observation** → SIMPLER historical L11-Matched succeeds in 213/400 (53.2%); LIBERO yields 32 Rescue / 34 Harm in the 250-pair five-task study and 44 Rescue / 57 Harm in the audited 500-pair Spatial study.

**Mechanism candidate** → In LIBERO, the negative branch's effect on the clean winner-versus-runner-up margin varies across episodes. On historical trace-valid states, the harmonic guidance-margin Rescue-higher AUC is .698 (95% episode-bootstrap CI [.550,.831]); the clean-pair flip fraction instead associates with Harm (Rescue-higher AUC .261 [.135,.402]). These describe association, not whether the intervention caused the whole episode outcome.

**Behavioral consequence** → Matched guidance helps some episodes and harms others, largely cancelling in aggregate. A 53-case fresh replay negative-branch probe found 46 episodes with at least one historically exact `m`/ID state (21 Rescue / 25 Harm; 353 states). Within the same fresh frame and clean logits, feature-kNN replacement changed some negative logits but barely changed the guided top-two margin (mean paired change +.0047 Rescue, −.0016 Harm; Rescue-vs-Harm AUC .518 [.347,.690]).

**Transfer failure** → The current data support a variable action-value boundary in LIBERO, but they do not yet show why its distribution differs from SIMPLER. The comparison is confounded by checkpoint, task set, and simulator, and fresh replay RGB hashes do not match historical hashes.

## Failure Map

| Component | SIMPLER | LIBERO | Bottleneck? | Evidence |
|---|---|---|---|---|
| L11 selector | Historical ablations report selector contribution. | L11-Matched has Rescue and Harm, but no same-`m`/same-negative selector replacement establishes selector failure. | UNRESOLVED | Relevance evidence does not establish action value. |
| Matched budget | Historical ablations report a matched-budget contribution. | Frozen recipe gives 182/250 vs Vanilla 184/250; budget itself was not isolated in the transfer comparison. | UNRESOLVED | No controlled budget-only test. |
| Harmonic negative | Effective as part of the historical SIMPLER recipe. | One local feature-kNN negative replacement changes some negative logits but scarcely moves the action margin; no closed-loop benefit shown. | UNRESOLVED | Constrained offline alternative only; not proof harmonic is irrelevant. |
| Action guidance | Historical recipe has clear aggregate gain. | Rescue/Harm are frequent and action-margin effects associate with outcome in trace-valid historical states. | UNRESOLVED | Strongest proximal candidate, but observational and replay-sensitive. |
| Recoverability | SIMPLER tasks/checkpoint show higher historical method success. | Vanilla is already strong on measured LIBERO subsets (184/250 and 423/500), but action-support recoverability is not isolated. | UNRESOLVED | Aggregate success alone does not test whether failed states are recoverable. |
| Long-horizon effect | Not established as the cross-benchmark cause. | Early/Late renderer ablation did not support a simple “late intervention alone causes Full RT Harm” explanation. | UNRESOLVED | Does not rule out task-stage interaction generally. |

## Rejected Hypotheses

- **“L11-Matched has no effect in LIBERO.”** Rejected: LIBERO contains substantial Rescue and Harm discordance.
- **“Full RT harm is simply caused by continuing intervention late.”** Not supported by the tested Full/Depth0 Early/Late ablation.
- **“A single visible robot surface swap or arena-diffuse component explains the relative Full/Depth0 gain.”** Not supported by the tested swap and renderer ablation results.
- **“One rank-1 renderer nuisance direction can be removed to recover L11 benefit.”** Not supported: removal degraded Vanilla and L11 together.
- **“The tested feature-kNN local negative clearly fixes Harm.”** Rejected for this alternative: the offline guided-margin change is near zero and has no reliable Rescue/Harm discrimination.

The mild-view reliability hypothesis is **not** rejected as useless. Same-renderer mild photometric views have already been tested; LIBERO primary AUC was .451 (95% CI [.313,.593]), with trace-valid sensitivity around .49. Low historical `m`/ID fidelity and RGB replay mismatch make this inconclusive for a general stability claim. No gate should be trained from it.

## Strongest Experiment

The strongest variable-isolation probe so far is the 53-case paired offline negative-branch probe: for each eligible state it held RGB, clean logits, selector, `m`, selected IDs, and lambda fixed and changed only the selected-token replacement. It rules out a clear action-margin improvement from this particular feature-kNN replacement, but it is not a closed-loop causal validation and cannot settle the harmonic-negative hypothesis.

## Remaining uncertainty

1. Does harmonic reconstruction itself cause a harmful action correction on the true in-trajectory LIBERO observation, or is the variability mainly due to task/checkpoint-specific action support and state distribution?
2. How much of the apparent Rescue/Harm margin separation survives with historical RGB and exact per-step intervention traces, given that fresh replay RGB hashes fail to match history?

## Next research decision

**KEEP L11-Matched as a fixed baseline and improve trace-faithful negative/action-value diagnosis.** Do not train a stability gate or replace harmonic reconstruction based on the current local-kNN result. The next causal pilot should wait until one alternative negative produces a clear held-out offline prediction—reduce harmful margin changes while retaining rescue-like changes—then compare only that negative against harmonic in a 30–50 paired closed-loop pilot with all per-step states, RGB hashes, masks, logits, actions, TCP/gripper/object state, and success events logged.
