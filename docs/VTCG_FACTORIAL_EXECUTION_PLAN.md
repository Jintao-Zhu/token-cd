# VTCG factorial + counterfactual controls: execution plan

Status: implementation / CPU audit; no new GPU rollout launched yet.
Authorized devices: physical GPUs 1,2,3 ONLY. No pi0/pi05 experiments.

## Scientific scope
OpenVLA only; 4 SIMPLER tasks: open_drawer, close_drawer, pick_coke_can, move_near.
Nine arms: attn_matched, attn_fixed, attn_topp, kmeans_matched, kmeans_fixed,
kmeans_topp, vanilla, random_matched, norm_noise. Formal target 100 cases/task,
3600 episodes total. Engineering target two cases/task, 72 episodes; disjoint.

Lambda=.5; L11 instruction attention; KMeans K8 random_state0 n_init10; projector
intervention; harmonic beta0; motion guidance only; shared CLEAN greedy action
prefix (verified global_merge_policy.py:117-152). Existing guided_prefix metadata
is not evidence of guided-autoregressive decoding.

## Budget baseline selection provenance
Proposed global Fixed48 and TopP .85 clipped16..64 inherit the historical
lambda.5 four-task sweep (docs/MATCHED_EVIDENCE_TABLE.md:G1). They are the best
observed global Fixed and clipped-TopP configurations there. This is historical
exploratory selection, NOT an independent validation-set result. They will be
frozen before new outcomes are inspected and shared by both ranking rows.
Historical pick/move rendering differs; disclose rather than merge old outcomes.
TopP count always comes from current CLEAN attention, regardless of ranking row.

## KMeans full ranking is a NEW well-defined comparator
Legacy kmeans_matched selects only the entity-winning cluster union; it is NOT a
full ranking and cannot be crossed with Fixed/TopP without modification.
New comparator orders clusters by max entity-centroid cosine; within a cluster,
orders by token-centroid cosine; deterministic token-index tie handling. It need
not reproduce legacy union at union cardinality. Name and report it explicitly.

## Engineering isolation
Do not alter existing shared policy source or locked studies. New private policy
module/helper. New study outputs only. No reuse of old success labels as formal
paired arms. Known process-history effects mean equal initial hashes are
necessary checks, not trajectory-equivalence guarantees.

Prefer each case/arm in fresh simulator process with same reset/restore history.
Independent workers may reuse model only if safe simulator isolation is validated;
otherwise pay model reload cost. Arm order precomputed balanced rotation; GPU
assignment balanced by case, never by method. Resume only COMPLETE audited cases,
not arbitrary partial arm sets with different process histories.

## Data / seed issue (pending inspection)
Historical 0..299 have already been explored. Do not claim any 0..299 unseen.
Formal500..599 / engineering400..401. Bounded history scan found300..399 already
used, as well as engineering400/401 (deliberately reused only for debugging).
500..599 not found among174870 standard episode filenames and8687 metadata
files within bounded artifacts/reports index; this is not an exhaustive unseen
certificate. Source reads confirm seed passed to RandomState without modulo300,
but MoveNear samples one of60 object layouts. New integers are not proof of new
scenes. Audit state/RGB duplicates without silently filtering based on outcomes.
Generate fresh snapshots and in-run paired reference, not historical success labels.

## Logging
Vanilla/full videos all formal cases; all arms initial/final RGB, actions,
step-info events, masks/counts/rankings, per-token perturbation-norm audits,
float32 logits and source/config hashes. Prespecified subset of controls videos.
Efficiency end-to-end measurement occurs later on a single otherwise idle GPU;
concurrent throughput timing is not publication latency.

## Analysis commitments
Exact task/seed/arm membership and duplicates checks; fail on missing/nonfinite
or pairing violations. Episode is sample unit, not steps. Paired stratified
bootstrap within task with same resampled indices across arms; explicit contrast
families for Holm. Factorial interaction via paired difference-of-differences,
not ranks. Main global result includes all four tasks regardless of outcomes.
Failure annotations use blinded method labels where feasible, discordant pairs
and task-stratified concordant sample; distinguish sampled from population rates.

Next: CPU helper tests; policy equivalence; seed/environment inspection;
freeze manifest; 72-episode preflight; examine memory/throughput then launch
formal queue with 2 workers/card initially (max4/card only if measured useful).
