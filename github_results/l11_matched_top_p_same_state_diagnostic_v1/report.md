# Why Matched beats cumulative Top-p: targeted diagnostic

## Scope

No new adaptive rule was introduced. The analysis used 60 pre-existing identical observations (15 per task), plus 30 fully replay-verified open-drawer states from 10 outcome-stratified episodes. Matched, TopP80 and TopP85 always used the same L11 ranking; only the cutoff differed.

## Main findings

1. **Global mean count hides large state-level errors.** Across the 60 common states, TopP80 differed from Matched by 16.0 tokens on average in absolute value despite a signed mean of only +3.1. It selected fewer tokens in 46.7% of states and more in 53.3%. TopP85 differed by 18.3 tokens in absolute value, selecting fewer in 31.7% and more in 66.7%.
2. **Drawer mismatch is phase-dependent, not a fixed under-selection.** In open-drawer, TopP80 was below Matched by 6.6 tokens on average in middle states but above it by 6.8 in late states. TopP85 was below Matched in 53.3% of all sampled open-drawer states, despite a positive overall mean difference.
3. **The differing rank tail often contains structured context, but it is not uniformly beneficial.** Visual inspection shows drawer fronts/edges, adjacent drawer bands, robot/contact vicinity, and occasional background. Seed 122 is a Matched-only case where the extra tail covers drawer structure and changes one action dimension. Seed 124 is the opposite: Matched extends over several drawer bands, robot and background, while the smaller TopP85 set is more concentrated and the full TopP85 rollout succeeds.
4. **Budget mismatch frequently reaches the policy.** On the 60 identical states, TopP85 and Matched produced different guided actions in 56.7% of states (1.17 differing dimensions on average). On the 30 outcome-aligned open-drawer states, they differed in 53.3% (1.0 dimension on average).
5. **Count difference alone does not predict action difference or success.** For the 60 states, correlation between signed TopP85 count difference and number of changed dimensions was approximately zero (-0.006). In the outcome sample, TopP85-only states tended to use fewer tokens, while both-success states used substantially more; therefore neither 'more is better' nor 'less is better' explains Matched.
6. **One-step causal replacement was null in the technically valid subset.** 6 of 8 continuations reproduced the Matched control outcome; none changed success after replacing one selected action with TopP85 and then returning to Matched. The remaining 2 were excluded because the restored Matched continuation did not reproduce its original outcome. This suggests the closed-loop difference is accumulated over multiple replans, or that the sampled single step was not decisive.

## Conclusion

The data support the narrow conclusion that **attention concentration is not a sufficient proxy for the useful intervention budget**. They do not yet prove that KMeans group size directly estimates the correct causal evidence extent. Matched appears to provide a different, phase-sensitive budget signal; the extra or omitted rank segment can contain task-relevant structure, but also irrelevant context. Its benefit is therefore conditional rather than a simple preference for larger masks.

The next justified experiment would manipulate only the identified rank segment over a short multi-step window on replayable states. No such new rule or full closed-loop arm was run here.
