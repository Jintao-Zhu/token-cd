# Intervention-dose response analysis

800 states, each scanned with 5 relative budgets plus Fixed34 and Entity-TopP.

Metrics: `D_feat` visual feature change, `D_action` action JS divergence, `D_support` clean-action log-probability drop.

## Matched-relative response

| r = m/m0 | Small | Middle | Large |
|---|---:|---:|---:|
### D_feat

| r | Small | Middle | Large |
|---|---:|---:|---:|
| 0.5 | 0.1483 | 0.2190 | 0.2650 |
| 0.75 | 0.1821 | 0.2700 | 0.3301 |
| 1 | 0.2116 | 0.3157 | 0.3857 |
| 1.25 | 0.2374 | 0.3572 | 0.4352 |
| 1.5 | 0.2616 | 0.3946 | 0.4803 |

### D_action

| r | Small | Middle | Large |
|---|---:|---:|---:|
| 0.5 | 0.2531 | 0.3195 | 0.3979 |
| 0.75 | 0.2946 | 0.3492 | 0.4134 |
| 1 | 0.3212 | 0.3715 | 0.4219 |
| 1.25 | 0.3419 | 0.3770 | 0.4251 |
| 1.5 | 0.3522 | 0.3798 | 0.4309 |

### D_support

| r | Small | Middle | Large |
|---|---:|---:|---:|
| 0.5 | 2.6250 | 3.8503 | 5.7227 |
| 0.75 | 3.2826 | 4.6039 | 6.4919 |
| 1 | 3.8267 | 5.1860 | 6.7769 |
| 1.25 | 4.3059 | 5.5755 | 6.8940 |
| 1.5 | 4.6461 | 5.8166 | 7.0887 |

## Fixed34 / Entity vs Matched scale

| Budget | Metric | Overall mean | Overall std | Small mean | Middle mean | Large mean |
|---|---:|---:|---:|---:|---:|---:|
| Matched r=1 | D_feat | 0.3044 | 0.0857 | 0.2116 | 0.3157 | 0.3857 |
| Matched r=1 | D_action | 0.3716 | 0.2693 | 0.3212 | 0.3715 | 0.4219 |
| Matched r=1 | D_support | 5.2650 | 4.5949 | 3.8267 | 5.1860 | 6.7769 |
| Fixed34 | D_feat | 0.3152 | 0.0270 | 0.3234 | 0.3164 | 0.3058 |
| Fixed34 | D_action | 0.3851 | 0.2732 | 0.3774 | 0.3712 | 0.4067 |
| Fixed34 | D_support | 5.6609 | 4.6715 | 5.4782 | 5.2762 | 6.2277 |
| Entity-TopP | D_feat | 0.3125 | 0.0472 | 0.3216 | 0.3202 | 0.2959 |
| Entity-TopP | D_action | 0.3858 | 0.2743 | 0.3829 | 0.3726 | 0.4018 |
| Entity-TopP | D_support | 5.6512 | 4.6248 | 5.5252 | 5.3755 | 6.0524 |

## r=1 by task and matched-m tercile

| Task | Size | m0 mean | D_feat | D_action | D_support |
|---|---:|---:|---:|---:|---:|
| open_drawer | small | 16.8 | 0.1966 | 0.2899 | 3.3885 |
| open_drawer | middle | 40.6 | 0.3315 | 0.4032 | 5.7191 |
| open_drawer | large | 49.3 | 0.3625 | 0.5445 | 8.6232 |
| close_drawer | small | 16.8 | 0.2135 | 0.3997 | 4.5551 |
| close_drawer | middle | 34.1 | 0.3113 | 0.2554 | 3.9476 |
| close_drawer | large | 49.1 | 0.3742 | 0.2765 | 5.1511 |
| pick_coke_can | small | 12.0 | 0.1955 | 0.2097 | 2.3870 |
| pick_coke_can | middle | 24.0 | 0.2822 | 0.2914 | 4.2750 |
| pick_coke_can | large | 43.4 | 0.3631 | 0.2864 | 4.5394 |
| move_near | small | 21.4 | 0.2460 | 0.4626 | 5.5843 |
| move_near | middle | 37.6 | 0.3402 | 0.5009 | 6.7171 |
| move_near | large | 62.1 | 0.4313 | 0.5370 | 8.2158 |

## Budget-vs-state trend

| Budget | Metric | Spearman rho vs m0 |
|---|---:|---:|
| Fixed34 | D_feat | -0.2373 |
| Fixed34 | D_action | 0.0665 |
| Fixed34 | D_support | 0.0958 |
| Entity-TopP | D_feat | -0.1810 |
| Entity-TopP | D_action | 0.0449 |
| Entity-TopP | D_support | 0.0736 |
