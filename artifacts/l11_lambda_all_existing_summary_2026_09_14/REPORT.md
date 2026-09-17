# L11 Lambda Experiments: Consolidated Existing Results

Date: 2026-09-14

## 1. Main Pure L11-Matched Fixed-Lambda Sweep

Same four tasks, canonical seeds 0--99, 100 episodes per task and arm.

| Task | lambda=0 | lambda=.25 | lambda=.50 | lambda=.60 | lambda=.75 | Best observed |
|---|---:|---:|---:|---:|---:|---:|
| open_drawer | 27/100 | 50/100 | 50/100 | 38/100 | 30/100 | .25/.50: 50/100 |
| close_drawer | 56/100 | 71/100 | 82/100 | 77/100 | 76/100 | .50: 82/100 |
| pick_coke_can | 23/100 | 28/100 | 44/100 | 40/100 | 37/100 | .50: 44/100 |
| move_near | 53/100 | 56/100 | 62/100 | 63/100 | 53/100 | .60: 63/100 |
| Overall | 159/400 | 205/400 | 238/400 | 218/400 | 196/400 | .50: 238/400 |

- Five directly comparable lambda values were tested: 0, .25, .50, .60, and .75.
- This table contains 2,000 real closed-loop episode-arm outcomes.
- Per-task post-hoc best selection gives 239/400, only +1 success over a single global lambda=.50.
- The empirical three-arm oracle over lambda in {0,.25,.5} is 299/400, but this is retrospective and not deployable.

## 2. Lambda=.50 Large-Scale Result

Nine tasks, seeds 0--299, 300 episodes per task.

| Task | Vanilla | SHR | L11-Matched lambda=.50 |
|---|---:|---:|---:|
| open_drawer | 71/300 | 138/300 | 144/300 |
| close_drawer | 154/300 | 209/300 | 230/300 |
| pick_coke_can | 83/300 | 97/300 | 119/300 |
| move_near | 184/300 | 197/300 | 182/300 |
| place_apple_in_closed_top_drawer | 0/300 | 0/300 | 1/300 |
| carrot_on_plate | 14/300 | 4/300 | 0/300 |
| put_eggplant_in_basket | 1/300 | 0/300 | 12/300 |
| spoon_on_towel | 0/300 | 1/300 | 4/300 |
| stack_cube | 0/300 | 0/300 | 8/300 |
| Overall | 507/2700 | 646/2700 | 700/2700 |

Four-task lambda=.50 seed blocks:

| Seeds | open | close | pick | move | Overall |
|---|---:|---:|---:|---:|---:|
| 0--99 | 50/100 | 82/100 | 44/100 | 62/100 | 238/400 |
| 100--199 | 44/100 | 76/100 | 33/100 | 60/100 | 213/400 |
| 200--299 | 50/100 | 72/100 | 42/100 | 60/100 | 224/400 |
| 0--299 | 144/300 | 230/300 | 119/300 | 182/300 | 675/1200 |

## 3. Temporal Adaptive Lambda Development

Tasks are close_drawer, pick_coke_can, move_near, and carrot_on_plate; seeds 0--99.

| Arm | close | pick | move | carrot | Overall | Mean episode lambda |
|---|---:|---:|---:|---:|---:|---:|
| fixed .25 | 71/100 | 28/100 | 56/100 | 0/100 | 155/400 | .250 |
| Adaptive Raw | 72/100 | 42/100 | 56/100 | 0/100 | 170/400 | .261 |
| Adaptive EMA | 73/100 | 37/100 | 63/100 | 7/100 | 180/400 | .236 |
| fixed .50 reference | 82/100 | 44/100 | 62/100 | 0/100 | 188/400 | .500 |

- Adaptive Raw maps temporal correction cosine directly to lambda in [.1,.5].
- Adaptive EMA smooths that cosine before applying the same mapping.
- Neither adaptive arm beat fixed lambda=.50 overall.
- A stopped confirmation attempt left 55 incomplete close_drawer episode files: 27 EMA and 28 fixed-.25. They are not a valid balanced result and should not enter the paper table.

## 4. DTP-Positive + L11-Matched-Negative Lambda Sweep

This is a different positive branch and must not be merged with the pure L11-Matched curve.

| Task | lambda=.10 | lambda=.25 | lambda=.50 | lambda=.60 | lambda=.75 |
|---|---:|---:|---:|---:|---:|
| open_drawer | 38/100 | 40/100 | 48/100 | 43/100 | 32/100 |
| close_drawer | 52/100 | 71/100 | 76/100 | 79/100 | 81/100 |
| pick_coke_can | 32/100 | 35/100 | 38/100 | 42/100 | 47/100 |
| move_near | 65/100 | 57/100 | 60/100 | 54/100 | 56/100 |
| Overall | 187/400 | 203/400 | 222/400 | 218/400 | 216/400 |

- Five lambda values and 2,000 real closed-loop episode-arm outcomes.
- Overall best is lambda=.50, although individual tasks prefer different values.

## 5. Early Non-Matched Semantic Selector Sweep

This predates the current L11-Matched selector and is not directly comparable. It used semantic_hard K8, layers 8--15, move_near only, seeds 100--199.

| Arm | Success |
|---|---:|
| Vanilla | 56/100 |
| lambda=.10 | 53/100 |
| lambda=.25 | 51/100 |
| lambda=.50 | 44/100 |
| lambda=1.00 | 45/100 |
| lambda=2.00 | 45/100 |

## 6. Other Hyperparameter Experiments at Fixed Lambda=.50

These are not lambda sweeps:

- Visual Top-p selector sweep: p=.75/.80/.85/.90 on four tasks, seeds 100--199. Matched was best overall at 213/400; TopP75/80/85/90 obtained 175/194/206/191.
- Token-count sweep: K=16/24/32/48/64 on four tasks, seeds 100--199. Matched obtained 213/400; candidates obtained 151/173/196/207/198.
- Positive-boost eta sweep: eta=.5/1.0/reverse at downstream lambda=.50 on seeds 0--99. Correct L11 was 238/400; candidates were 217/212/213.
- Held-out confidence-gated positive boost: matched 213/400, ungated 188/400, gated 199/400 on seeds 100--199.
- Risk/confidence gates were stopped after offline audits; no valid closed-loop lambda result was produced.

## 7. Counts and Paper Interpretation

- Main current pure L11-Matched sweep: five lambda values, 2,000 comparable outcomes on the four-task development set.
- Pure L11-Matched fixed-lambda unique outcomes including the full nine-task lambda=.50 evaluation: 4,300. The 400 lambda=.50 development outcomes are already contained in the 2,700 full evaluation and are counted only once here.
- Temporal adaptive complete outcomes: 800 dynamic-arm episodes; fixed-.25 references overlap the fixed sweep.
- DTP-positive lambda family: 2,000 outcomes, reported separately because the positive branch changed.
- Early non-matched semantic sweep: 500 guided outcomes plus 100 Vanilla outcomes, reported only as historical evidence.

The clean paper conclusion from the current matched data is that lambda=.50 is the best global setting. Selecting each task's best lambda post hoc changes 238/400 to only 239/400, so the current data do not justify a strong claim that per-task lambda tuning materially improves the four-task aggregate.
