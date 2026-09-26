# SIMPLER → LIBERO transfer diagnosis

**Status: PARTIALLY SUPPORTED.** The frozen L11-Matched recipe has clear historical SIMPLER gains, but no clear average gain on the available paired LIBERO evaluations. Current evidence localizes the failure to the *action value of the intervention being heterogeneous across episodes and task stages*; it does not isolate whether selector, matched budget, or harmonic negative construction is the primary cause. Mild-view guidance-direction stability, as the candidate reliability signal, does not explain Rescue/Harm on the LIBERO sample.

## Evidence at a glance

| Evidence | Result | What it establishes |
|---|---|---|
| Historical SIMPLER four-task result | L11-Matched 213/400 (53.2%) | The historical method has a positive benchmark result. |
| LIBERO frozen SIMPLER-config, five tasks | Vanilla 184/250 (73.6%), L11-Matched 182/250 (72.8%); 32 Rescue / 34 Harm; paired difference −0.8 pp, 95% CI [−7.2, +5.6], exact McNemar p=.902 | There is no clear average benefit or harm in this subset; both beneficial and harmful episode effects occur. |
| LIBERO Spatial, audited 500 pairs | Vanilla 423/500, Matched 410/500; 44 Rescue / 57 Harm; net −13 | A larger, separate paired evaluation also does not show an aggregate gain. The original report had rescue/harm reversed and the wrong net sign; use the raw-pair audit. |
| Mild-view continuous direction stability | 66 LIBERO discordant episodes (32/34), 3,913 states. Primary episode AUC .451 (95% CI [.313, .593]); late AUC .420; task-macro AUC .495 (CI [.313, .663]). | Sign stability under ±5% brightness/contrast and gamma .95/1.05 does not predict Rescue on this sample. |
| Stricter all-six-view agreement | Overall AUC .557 (CI [.414, .699]); task-macro .615 (CI [.413, .814]) | Weakly positive point estimates, but uncertain and inconsistent by task; not validated evidence for a gate. |
| Trace-valid sensitivity | Restricting to episodes with at least one state where current `m` and selected IDs match historical gives 27 Rescue / 26 Harm; all-state AUC .490 (CI [.330, .650]), late AUC .435 (CI [.278, .602]); task-macro .524 (CI [.356, .688]). | Low trace fidelity does not conceal a clear primary-score signal in the trace-valid subset. |

The SIMPLER mild-view pilot is also exploratory: 17 episodes (5 Rescue / 12 Harm), primary AUC .700 (CI [.333, 1.000]). Its apparent signal has not replicated in LIBERO. A cross-benchmark contrast cannot be attributed to benchmark alone because task set, renderer, checkpoint, and replay fidelity all change together.

## Failure map by method component

### 1. L11 prompt selector: unresolved

SIMPLER evidence shows L11 prompt attention can select useful task-related tokens. LIBERO offline attention diagnostics show some target discrimination, but target relevance is not action utility. Existing selector repair and endpoint controls do not establish a stable matched-`m`, same-reconstruction behavioral advantage for L11 in LIBERO. The mild-view replay further exposes a reproducibility concern: physical reset hashes and episode outcomes match history 66/66, but current identity-view `m` matches historical `m` in 1,133/3,913 states (29.0%), and selected IDs match in 735/3,913 (18.8%). This flags renderer/selector sensitivity in the replay path; it does not by itself prove the historical experiment used a bad selector.

### 2. Matched budget: not shown to transfer as a benefit; cause unresolved

The frozen-config LIBERO study uses the historical L11/Matched recipe and yields 182/250 versus Vanilla's 184/250. Its episode-mean matched budget is about 37.1 tokens (median 42.5); budgets vary substantially over states. Existing dynamic-budget / selector controls do not show a robust alternative that restores a positive average result. Since selector fidelity and negative reconstruction vary together, this does not isolate the budget as the failure point.

### 3. Harmonic negative and contrastive action effect: leading unresolved mechanism

In the five-task diagnosis, Rescue and Harm have similar feature-perturbation magnitude (episode mean 0.360 vs 0.365; p=.667), so magnitude alone is not the separator. Harm episodes have more changed action dimensions (2.235 vs 1.932, p=.00016) and more fragmented masks (6.324 vs 5.325 components, p=.0057); these are associations, not causal proofs. Most notably, 13/34 harms are dropped/lost-target failures and 7/34 are placement/subtask failures after engagement, while 18/32 rescues repair approach/grasp failures. This points to task- and stage-dependent action consequences, not a universally bad residual norm or a single wrong-object selector story.

The mild-view primary metric fixes the clean winner/runner-up and checks whether the continuous `z+ − z−` effect on their action margin keeps its sign. Its LIBERO AUC is .451, and it remains near chance in the trace-valid sensitivity. Thus simple photometric direction consistency does not identify when the harmonic negative produces useful action evidence. We have not yet isolated the negative construction itself from the selector and budget.

### 4. Base-policy recoverability: not the sole explanation

Vanilla succeeds in 184/250 episodes, and L11 rescues 32 of its failures in the paired subset. That establishes a meaningful set of locally recoverable failures. It also creates 34 harms among Vanilla successes. Some remaining failures may be planning or long-horizon failures that local logit reordering cannot repair, but base-policy recoverability has not been measured in a way that explains the net result. It cannot account for the whole transfer gap by itself.

## Mild-view experiment integrity and limits

- Exactly 66 unique manifest cases were collected: 32 Rescue, 34 Harm.
- Initial simulator state hashes matched history in 66/66; replayed final success labels matched in 66/66.
- 3,913 sampled physical states each have identity plus six photometric views and paired positive/negative action-bin logits.
- Every completed case passed output shape, view count, outcome-label, and replay validation checks. GPUs 4 and 5 were not used.
- Views are transformations of the same live replay RGB at each state. The experiment does not change geometry or renderer and does not run new closed-loop episodes.
- The classifier unit is the episode; state/view records are averaged within an episode. Bootstrap resamples whole episodes. Task macro estimates only include tasks with both Rescue and Harm; one task has a single Harm and no Rescue.
- Replay state and outcome fidelity are exact, but the live RGB/selector trace often differs from the historical intervention. Results describe current replayed L11 behavior under mild views and require that caveat for historical attribution.

## Research decision

The corrected view-domain interpretation stands: Full RT versus Depth0 is too large a visual shift to test nuisance reliability. On the appropriate mild-view test, however, the primary stability score fails to separate LIBERO Rescue from Harm, including in the trace-valid subset. The stricter all-view score remains inconclusive. **Do not train or tune a reliability gate. Do not conclude that reliability is useless.**

The next smallest discriminative diagnosis should keep L11 and matched budget fixed and examine the negative branch's action-margin effects on trace-valid states, stratified by Rescue/Harm and by approach/grasp versus post-grasp/placement phase. This tests whether harmonic reconstruction changes the clean decision in different ways without reopening selector, budget, or renderer searches. Keep task and checkpoint as explicit strata.

```yaml
Hypothesis: mild-view stability of guidance direction predicts L11 benefit in LIBERO
Evidence for: SIMPLER pilot primary AUC 0.700, but only 5 Rescue episodes
Evidence against: LIBERO primary AUC 0.451; late 0.420; trace-valid AUC 0.490; all intervals span chance
Decision: do not use it as a gate; retain as unvalidated, and inspect the negative-branch action effect next
```

## Supporting records

- [Known evidence ledger](/home/leju-suzhou/zjt_ws/token-cd/KNOWN_EVIDENCE.md)
- [Mild-view report](/home/leju-suzhou/zjt_ws/token-cd/artifacts/libero_transfer_diagnostics/MILD_VIEW_TRANSFER_REPORT.md)
- [Full mild-view episode and bootstrap data](/home/leju-suzhou/zjt_ws/token-cd/artifacts/libero_transfer_diagnostics/mild_view_transfer_analysis.json)
- [Five-task diagnosis](/home/leju-suzhou/zjt_ws/token-cd/artifacts/libero90_five_task_simpler_config_v1/DIAGNOSIS_REPORT.md)
- [Five-task final report](/home/leju-suzhou/zjt_ws/token-cd/artifacts/libero90_five_task_simpler_config_v1/FINAL_REPORT.md)
- [Corrected 500-pair audit](/home/leju-suzhou/zjt_ws/token-cd/artifacts/libero_official_compare_500_v1/AUDITED_COMPARE_REPORT.md)
- [Reproducible mild-view analyzer](/home/leju-suzhou/zjt_ws/token-cd/research/semantic_token_cd/analyze_mild_view_transfer.py)

## Follow-up: isolate the negative branch (v4)

**Hypothesis:** If harmonic reconstruction is specifically creating harmful action corrections in LIBERO, replacing only the negative feature construction with a more local feature-space counterfactual should selectively improve Harm-state action margins while retaining Rescue-state effects.

**Experiment:** Completed 53 replay cases; among exact historical `m`/selected-ID states, compared harmonic against 4-nearest-unselected-feature mean replacement. Held current RGB, clean logits, selector, matched `m`, selected IDs, and lambda=.5 fixed. Every replay initial-state hash and outcome matched history; no fresh RGB hash matched historical RGB.

**Result:** 46 episodes / 353 states were trace-valid (21 Rescue / 25 Harm). Negative logits changed in some action dimensions, but the paired change in guided clean top-two margin was near zero and not Rescue/Harm-discriminative (AUC .518, 95% CI [.347,.690]). The current-frame harmonic guidance-margin AUC was .560, while historical RGB logits for the same 46 episodes gave .693; this is descriptive evidence that RGB replay mismatch can move the apparent association.

**Evidence for:** Counterfactual action effect remains heterogeneous; replacing the tested local feature reconstruction is not an obvious fix.

**Evidence against:** This single feature-kNN alternative does not selectively improve Harm; no closed-loop validation supports a negative-construction fix.

**Confidence:** Low-to-moderate for rejecting this specific feature-kNN candidate as an offline action-margin fix; low for identifying the cross-benchmark bottleneck.

**Decision:** Keep L11 and Matched fixed; do not run a closed-loop feature-kNN pilot and do not conclude all negative construction is irrelevant. The mild-view study is already completed and remains inconclusive under its trace-fidelity limits; the Full RT/Depth0 shift should not be treated as a mild-stability test.

**Next minimal test:** Wait for an alternative negative construction to show a pre-specified held-out offline prediction—selective harmful-margin reduction with rescue preservation—before a 30–50 paired closed-loop comparison. Preserve historical RGB and per-step intervention traces to remove the replay ambiguity.

See [negative-branch report](/home/leju-suzhou/zjt_ws/token-cd/artifacts/libero_transfer_diagnostics/negative_variant_probe_v4/NEGATIVE_VARIANT_PROBE_V4_REPORT.md) and [final report](/home/leju-suzhou/zjt_ws/token-cd/AUTONOMOUS_LIBERO_TRANSFER_FINAL_REPORT.md).
