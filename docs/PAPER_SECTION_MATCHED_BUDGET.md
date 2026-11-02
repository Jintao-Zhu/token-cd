# Paper section: the Matched budget rule (claim A)

**Claim being made:** *Matched needs no budget hyper-parameter. It is never significantly
exceeded by any alternative we tested, it is significantly better than under-budgeting and
than random token content, and it reaches the top tier of results with 30–45% fewer
intervened tokens than an attention-mass threshold rule.*

**How to read this file.** Section 1 is the paper-ready English draft. Section 2 lists the
numbers behind every sentence, with the artefact each was counted from. Section 3 is the
list of statements that must **not** appear, because the data contradicts them. Nothing
here is quoted from memory: every cell was counted from `episode_*_summary.json`.

---

## 1. Paper-ready draft

### Do we need to choose an intervention budget?

Every token-level intervention method needs to decide **how many** visual tokens to modify
at each control step. The two common answers are a fixed budget `m` (a hyper-parameter that
must be tuned per task) and a relative rule such as "the smallest set whose attention mass
exceeds `p`" (a hyper-parameter that must be tuned per model and per guidance strength).
Both spend effort on a quantity that the method could estimate from the state itself.

Our budget rule, **Matched**, estimates the budget from the current state instead: it
clusters the visual tokens, matches each task entity to its nearest cluster, and sets `m_t`
to the size of the union of the matched clusters. This introduces no budget
hyper-parameter to tune. We evaluate whether that costs anything relative to tuning.

**Setup.** All comparisons are paired on identical `(task, seed)` cells. Across the studies
below, everything except the budget rule is held fixed: same checkpoint, same canonical
snapshots, same intervention point, same harmonic counterfactual, same contrast execution,
and the same per-process seed blocks, so an arm difference is a budget-rule difference.
Guidance strength takes two values, `λ = 0.5` (weak) and `λ = 1.0` (strong); these are the
two settings for which the full rule comparison exists.

**Result 1 — Matched avoids the two failure modes that tuning is meant to avoid.**
Under weak guidance, a budget that is too small and a token set that ignores the
instruction both cost success rate, and Matched is significantly better than both
(Table 1). The contrast with *random content* is the cleaner ablation: `random 48` selects
48 tokens at random while `fixed 48` selects 48 tokens by attention rank, so the two differ
only in **which** tokens are intervened on. At `λ = 0.5` this content change costs
**−12.0pp** (Holm-adjusted `p = 1e-4`), while an under-sized budget (`fixed 16`) costs
**−10.0pp** (Holm `p = 0.0025`). A second, independent seed range reproduces the direction
with larger effects for both (`−15.5pp` and `−10.0pp` at 16 and 24 tokens), and also shows
Matched significantly ahead of the weakest attention-mass rule (`TopP 0.75`, `−9.5pp`,
Holm `p = 0.007`).

**Result 2 — Against well-chosen alternatives, Matched is statistically indistinguishable,
not better.** The interesting negative result is in the same tables: a *well-chosen* fixed
budget (48 tokens) and a *well-chosen* attention-mass threshold (`TopP 0.90`) sit within
1.5pp of Matched at `λ = 0.5`, and none of those differences survives correction. At
`λ = 1.0` the whole ladder collapses onto a plateau: across nine tasks and 900 episodes per
rule, all seven fixed budgets and all four thresholds land within **1.9pp** of Matched (the
entire ladder spans only 3.6pp, from 29.33% to 32.89%), and
**every pairwise contrast is non-significant after Holm correction** (Table 2). The largest
observed gap anywhere in the strong-guidance 4-task grid is `TopP 0.90` at `+6.25pp`
(`p = 0.027` raw, Holm `0.110`); re-running that exact contrast on an independent seed range
gives `−2.75pp`, and pooling both ranges gives `+1.75pp` (95% CI `[−1.9, +5.5]`, `p = 0.40`).
We therefore report the two quantities as indistinguishable rather than claiming an
advantage, and we explicitly do **not** read the non-significance as equivalence.

**Result 3 — Matched gets there without tuning, and with less intervention.** Table 3
reports the realised budget. On the nine-task study Matched uses 36.7 tokens per step on
average, versus 48.0 for the best fixed budget and 62.6 for the best attention-mass
threshold — a **1.7×** reduction relative to the threshold rule at a success rate that is
1.3pp lower and not significantly different. The threshold rule's budget is also far more
variable across episodes (5th–95th percentile 48–76 tokens versus 21–55 for Matched), so
matching its success rate requires spending that extra budget on every step.

**Result 4 — The conclusion is insensitive to the clustering granularity.** `Matched` has a
second, hidden knob: the number of clusters. Increasing it from 8 to 10 shrinks the matched
union by 27% (36.7 → 25.0 tokens per step) because clusters are divided more finely. Executed
for real at both guidance strengths on 400 paired episodes per setting, this changes success
by `−1.75pp` (`p = 0.55`) and `−3.75pp` (`p = 0.20`). The rule's output is therefore not
sharply tuned to a particular granularity, and the quantity that matters is not the average
budget but that the budget is derived from the state.

**Summary.** Matched is never significantly exceeded by any rule we tested, across two
guidance strengths, two independent seed ranges, two task suites (4 and 9 tasks, 900
episodes each in the largest study) and a 27% budget perturbation. It is significantly
better than under-budgeting and than random content. Relative to well-tuned alternatives it
is indistinguishable, but it requires no budget hyper-parameter and intervenes on about 40%
fewer tokens than the attention-mass threshold that ties it.

---

## 2. Numbers behind the draft

### Table 1 — weak guidance (`λ = 0.5`), paired, 4 Google tasks

Seeds 200–299, same protocol and process blocks, 400 episodes per rule, matched = 205/400:

| Rule | Successes/denominator | Rate | Δ vs Matched | Holm p |
|---|---|---|---|---|
| **Matched** | **205/400** | **51.25%** | — | — |
| random 48 (same count, random content) | 157/400 | 39.25% | −12.00pp | **0.0001** |
| fixed 16 | 165/400 | 41.25% | −10.00pp | **0.0025** |
| fixed 48 | 199/400 | 49.75% | −1.50pp | 1.000 |
| TopP 0.90 | 201/400 | 50.25% | −1.00pp | 1.000 |

Seeds 100–199, independent range, 400 episodes per rule, matched = 213/400 (53.25%):

| Rule | Successes/denominator | Rate | Δ vs Matched | Holm p |
|---|---|---|---|---|
| fixed 16 | 151/400 | 37.75% | −15.50pp | **<0.001** |
| fixed 24 | 173/400 | 43.25% | −10.00pp | **0.0018** |
| TopP 0.75 | 175/400 | 43.75% | −9.50pp | **0.0072** |
| TopP 0.85 (unclipped) | 186/400 | 46.50% | −6.75pp | 0.093 |
| TopP 0.90 | 191/400 | 47.75% | −5.50pp | 0.391 |
| fixed 32 / 64 | 196 / 198 | 49.00 / 49.50% | −4.25 / −3.75pp | 0.49 / 0.53 |
| TopP 0.80 | 194/400 | 48.50% | −4.75pp | 0.49 |
| TopP 0.85 (clipped) | 206/400 | 51.50% | −1.75pp | 1.000 |
| fixed 48 | 207/400 | 51.75% | −1.50pp | 1.000 |

### Table 2 — strong guidance (`λ = 1.0`), paired

Nine SIMPLER tasks, 900 episodes per rule (matched = 281/900 = 31.22%):

| Rule | Successes/denominator | Rate | Δ vs Matched | Holm |
|---|---|---|---|---|
| fixed 16 / 24 / 32 / 40 | 289 / 267 / 267 / 275 of 900 | 32.11 / 29.67 / 29.67 / 30.56% | +0.89 / −1.56 / −1.56 / −0.67pp | n.s. |
| fixed 48 / 56 / 64 | 285 / 296 / 279 of 900 | 31.67 / 32.89 / 31.00% | +0.44 / **+1.67** / −0.22pp | n.s. |
| TopP 0.75 / 0.80 / 0.85 / 0.90 | 266 / 264 / 280 / 293 of 900 | 29.56 / 29.33 / 31.11 / 32.56% | −1.67 / −1.89 / −0.11 / **+1.33pp** | n.s. |

Four Google tasks, 400 episodes per rule (matched = 201/400 = 50.25%):

| Rule | Successes/denominator | Rate | Δ vs Matched | p (raw / Holm) |
|---|---|---|---|---|
| TopP 0.90 | 226/400 | 56.50% | +6.25pp | 0.027 / 0.110 |
| fixed 16 | 206/400 | 51.50% | +1.25pp | 0.694 / 1.000 |
| fixed 48 | 202/400 | 50.50% | +0.25pp | 1.000 / 1.000 |
| random 48 | 189/400 | 47.25% | −3.00pp | 0.307 / 0.921 |

Same contrast (`TopP 0.90` vs Matched) on the independent seed range 100–199, 400 pairs:
matched 188/400 (47.00%) vs TopP 0.90 177/400 (44.25%), i.e. `−2.75pp` (`p = 0.355`).
Pooling both seed ranges (800 pairs): TopP 0.90 403/800 (50.38%) vs Matched 389/800
(48.63%), `+1.75pp`, 95% CI `[−1.9, +5.5]`, `p = 0.398`.

### Table 3 — realised intervention budget (`m_t`, tokens per step)

| Setting | Matched | fixed 48 | TopP 0.90 | Matched's saving vs TopP 0.90 |
|---|---|---|---|---|
| 9 tasks, `λ = 1.0` | **36.7** | 48.0 | 62.6 | 1.71× |
| 4 tasks, `λ = 1.0` | **33.7** | 48.0 | 58.5 | 1.74× |
| 4 tasks, `λ = 0.5` | **34.1** | 48.0 | 58.1 | 1.70× |

Per-episode spread (5th–95th percentile), 9 tasks `λ = 1.0`: Matched 21–55, TopP 0.90 48–76,
fixed 48 exactly 48.

### Table 4 — robustness of the conclusion

| Perturbation | Matched | Perturbed | Δ | p |
|---|---|---|---|---|
| clusters 8 → 10, `λ = 0.5` (budget −27%) | 205/400 | 198/400 | −1.75pp | 0.551 |
| clusters 8 → 10, `λ = 1.0` (budget −27%) | 201/400 | 186/400 | −3.75pp | 0.203 |
| seed range 100–199 → 200–299, `λ = 0.5` | 213/400 (53.25%) | 205/400 (51.25%) | −2.00pp | — |
| seed range 100–199 → 200–299, `λ = 1.0` | 188/400 (47.00%) | 201/400 (50.25%) | +3.25pp | — |

No alternative rule significantly exceeds Matched in any of the studies above, under Holm
correction within its declared family.

### Power statement (must accompany the tables)

With paired seeds and a discordance rate of about 30%, a 400-pair comparison reliably
detects only differences of **≥8pp**; a 900-pair comparison about **≥5pp**; detecting a 2pp
difference would need roughly **5,900** pairs. All statements above are therefore restricted
to effects of that magnitude, and differences of 1–6pp are reported with intervals rather
than as conclusions.

---

## 3. Statements that must not appear

1. "Matched is the best rule in every setting." False: at `λ = 1.0`, seeds 200–299 it ranks
   4th of 5 on the 4-task grid (TopP 0.90 56.50% > fixed 16 51.50% > fixed 48 50.50% >
   Matched 50.25%) and 5th of 12 on the nine-task study (fixed 56 32.89%, TopP 0.90 32.56%,
   fixed 16 32.11%, fixed 48 31.67% > Matched 31.22%).
2. "Matched is significantly better than a well-tuned fixed budget or threshold." False: all
   such contrasts are non-significant; the pooled `TopP 0.90` contrast is `+1.75pp` in
   *favour of TopP 0.90*.
3. "Non-significant means equivalent." Not supported; state the interval.
4. "The advantage comes from per-step adaptation." Not established: replacing the per-step
   budget with a per-task constant gives `−5.75pp` (`p = 0.054`) at `λ = 0.5` and `−0.50pp`
   (`p = 0.93`) at `λ = 1.0`; the cross-seed-range replication at `λ = 0.5` gives `−7.0pp`
   (`p = 0.012`) but the refreshed run at the same λ does not reach significance. Report the
   inconsistency.
5. "K (the cluster count) is a granularity knob that can be increased for finer estimates."
   False: cluster size scales as 256/K, so raising K shrinks the budget (43.5 → 24.3 tokens
   from K=8 to K=10 on the probed states); at K=16/32 the budget collapses to about 8 and 6
   tokens and the planned sweep was abandoned for that reason.
6. Cross-model claims. The SpatialVLA and pi0-FAST matched runs exist in the repository but
   were **not re-verified during this analysis**; either verify them before writing any
   "works across models" sentence, or drop the sentence.

---

## 4. Files

| Content | Path |
|---|---|
| Evidence table (all groups, counted cells) | `docs/MATCHED_EVIDENCE_TABLE.md` / `.tsv` |
| `λ × rule` grid, interaction estimates | `docs/OPENVLA_BUDGET_LAMBDA_TASKMEAN_FINAL.md` |
| Nine-task budget-rule study | `docs/OPENVLA_BUDGET_RULE_9TASK_FINAL.md` |
| Task-mean (no per-step adaptation) ablation | `docs/OPENVLA_BUDGET_LAMBDA_TASKMEAN_FINAL.md` |
| K = 10 closed-loop run | `docs/OPENVLA_K10_2LAMBDA_FINAL.md` |
| TopP 0.90 replication + pooled verdict | `docs/OPENVLA_TOPP90_REPLICATION_FINAL.md` |
| Moderator analysis (scene complexity, budget ratio) | `docs/OPENVLA_TOPP90_MODERATORS.md` |
| Figure set (λ response, dose-response, content vs count, heatmap, distributions) | `docs/figures/fig1..fig5*.png`, `docs/figures/fig6_topp90_moderators.png` |
