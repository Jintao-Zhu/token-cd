# OpenVLA KMeans granularity × existing budget controls

## Locked scope

- OpenVLA, lambda=1.0, nine SIMPLER tasks, seeds200–299 inclusive (100/task).
- Add only Matched-K16 and Matched-K32: 1,800 main episodes.
- Reuse Matched-K8 (900) from `artifacts/openvla_lambda1_extension_200_299_v1`.
- Reuse Fixed16/24/32/40/48/56/64 and unclipped TopP75/80/85/90 (9,900) from `artifacts/openvla_budget_rule_9task_200_299_v1`.
- Exclude the budget artifact's 20 diagnostic `lambda_100` episodes; never merge them into K8.
- Combined table: 14 methods ×9 tasks ×100 seeds =12,600 main evaluations, of which1,800 new.
- New K configurations are exploratory sensitivity analyses on an already observed evaluation set. Do not claim a test-selected K is independently validated.

## What changes and what stays fixed

Only KMeans K changes (8→16/32). Same projector visual features, random_state=0, n_init=10, entity-to-one-cluster matching with duplicate cluster removal, matched union count m_t, L11 attention top-m selection, harmonic reconstruction and contrast execution.

K is the number of clusters, NOT a fixed token count. More clusters may shrink matching coverage or split an entity. Record each step's K, seed, selected/matched groups, reference union IDs, m_t and selected IDs. Audit union size and selected count exactly. Do not describe a smaller average m_t as computational savings without timing evidence.

Drawers: RT path depth8, spp32, denoiser. Other seven tasks: depth0, spp32, denoiser. Same canonical snapshots, initial state/RGB SHA and instruction as old controls.

## Process-history identity

For each arm/task use separate fresh-process blocks: [200], [201–210], [211–220], …, [281–290], [291–299]. Exactly198 new process calls. Never interleave arms in one process. Same starting SHA alone is insufficient for trajectory reproducibility. Never skip completed seeds in a partial block; failures stop that worker for explicit review. No automatic retry.

## Gates and resource scheduling

Only GPUs1/2/3, maximum4 concurrent rollout workers per GPU. No GPU0/4/5/6/7 usage, no killing other experiments.

1. Audit reusable control coverage, top-level success booleans, technical checks and paired identity.
2. CPU tests and real-policy wrapper test without constructing model/environment.
3. Immutable experiment locks plus code hash manifest; drift stops work before the next chunk.
4. Queue waits until all existing `ovtm_*` taskmean sessions have ended AND GPUs1/2/3 have no compute processes and each<1024MiB used.
5. Nine separate smoke processes: close drawer/pick coke/carrot × K8/K16/K32, each fresh seed200. K8 outcome/steps/selected IDs/action token IDs/executed actions must reproduce the historical reference; K16/32 must pass strict audits. Logit differences are diagnostics, not efficacy.
6. Only passing smoke gate permits12-worker main launch. Gate or chunk failure is visible in status, no automatic partial resume. An exclusive supervisor lock prevents duplicate launches.

Smoke lives outside the main artifact and is never counted among the1,800 main episodes.

## Analysis plan

- Task rows; method columns; every cell successes/denominator and percentage. Full reporting may split wide columns into K/TopP and K/Fixed panels with identical task rows.
- Primary pooled paired comparisons: each of K16/K32 vs K8, TopP85, TopP90 =six tests; Holm across six.
- Secondary pooled comparisons: each of K16/K32 vs all seven Fixed budgets and TopP75/80 =18 tests; label exploratory and Holm across18. Do not select only favorable fixed budgets.
- Preserve task/seed pairing, report rescue/harm, paired success-rate difference and task-stratified paired bootstrap confidence intervals. Per-task effects descriptive unless a separate multiplicity family is declared.
- Report all nine tasks, plus predeclared Google-five/WidowX-four summaries. No deleting low-success tasks.
- Budget diagnostics: task-wise episode-balanced m_t summaries and step distributions with weighting explicitly labeled; group counts and runtimes. Closed-loop budgets are on divergent trajectories, not same-state causal evidence.
- No non-significance-as-equivalence; no inference that K sensitivity alone proves better semantic localization or universal optimality. If selecting a preferred K, confirm on independent data before confirmatory claims.

## Artifacts

- Main: `artifacts/openvla_kmeans_budget_9task_200_299_v1`
- Smoke: `artifacts/openvla_kmeans_budget_smoke_v1`
- Immutable protocol: `EXPERIMENT_LOCK.json`; process schedule: `WORK_PLAN.json`
- Gates: `CONTROL_REUSE_AUDIT.json`, `STATIC_VALIDATION.json`, smoke `SMOKE_VALIDATION.json`
- Live queue status: main `STATUS.json`, `STATUS_HISTORY.jsonl`, `queue_workers/`

No main K16/K32 efficacy results existed when this plan was written.
