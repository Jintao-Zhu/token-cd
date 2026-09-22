# Matched dynamic-budget mechanism analysis

All analyses reuse existing states/episodes. No new rollout episodes.

## Phase 1: What distinguishes small-m and large-m states?

Spearman rho between matched m and each state metric. Computed per task.

| Metric | open | close | pick | move |
|---|---:|---:|---:|---:|
| bbox_area | 0.831 | 0.899 | 0.804 | 0.871 |
| mean_spatial_distance | 0.856 | 0.861 | 0.677 | 0.780 |
| components | -0.360 | -0.251 | 0.457 | 0.691 |
| density | 0.321 | 0.205 | -0.308 | -0.466 |
| cluster_intra_cosine_distance | 0.551 | 0.775 | 0.398 | 0.183 |
| cluster_separation_margin | 0.025 | 0.368 | -0.095 | 0.213 |
| semantic_centroid_cosine | -0.499 | -0.493 | -0.328 | -0.103 |
| semantic_margin | 0.192 | 0.341 | -0.210 | -0.308 |
| cluster_size_entropy | 0.249 | 0.249 | 0.461 | 0.108 |
| matched_cluster_size_rank_mean | -0.898 | -0.940 | -0.924 | -0.705 |

### Small / Middle / Large states: metric means

Within each task, Small = lowest 25% m, Large = highest 25% m.

#### bbox_area

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 26.180 | 67.130 | 88.600 |
| close_drawer | 26.460 | 59.120 | 83.680 |
| pick_coke_can | 38.440 | 92.120 | 211.800 |
| move_near | 53.580 | 122.870 | 216.160 |

#### mean_spatial_distance

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 1.847 | 2.618 | 2.886 |
| close_drawer | 1.932 | 2.494 | 2.905 |
| pick_coke_can | 2.172 | 3.474 | 5.496 |
| move_near | 2.346 | 3.872 | 5.098 |

#### components

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 2.520 | 1.500 | 1.520 |
| close_drawer | 1.880 | 1.690 | 1.420 |
| pick_coke_can | 2.020 | 3.440 | 7.140 |
| move_near | 1.820 | 3.180 | 4.960 |

#### density

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 0.476 | 0.627 | 0.604 |
| close_drawer | 0.532 | 0.625 | 0.632 |
| pick_coke_can | 0.463 | 0.399 | 0.276 |
| move_near | 0.477 | 0.371 | 0.316 |

#### cluster_intra_cosine_distance

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 0.166 | 0.329 | 0.335 |
| close_drawer | 0.178 | 0.270 | 0.328 |
| pick_coke_can | 0.214 | 0.302 | 0.296 |
| move_near | 0.260 | 0.295 | 0.297 |

#### cluster_separation_margin

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 0.117 | 0.125 | 0.124 |
| close_drawer | 0.086 | 0.116 | 0.159 |
| pick_coke_can | 0.118 | 0.096 | 0.088 |
| move_near | 0.072 | 0.124 | 0.103 |

#### semantic_centroid_cosine

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 0.079 | 0.070 | 0.066 |
| close_drawer | 0.078 | 0.073 | 0.066 |
| pick_coke_can | 0.038 | 0.037 | 0.032 |
| move_near | 0.050 | 0.055 | 0.046 |

#### semantic_margin

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 0.016 | 0.023 | 0.021 |
| close_drawer | 0.015 | 0.020 | 0.023 |
| pick_coke_can | 0.007 | 0.007 | 0.004 |
| move_near | 0.023 | 0.021 | 0.015 |

#### cluster_size_entropy

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 1.948 | 2.011 | 1.999 |
| close_drawer | 1.956 | 2.001 | 1.990 |
| pick_coke_can | 1.912 | 1.952 | 1.976 |
| move_near | 1.955 | 1.972 | 1.975 |

#### matched_cluster_size_rank_mean

| Task | Small | Middle | Large |
|---|---:|---:|---:|
| open_drawer | 7.480 | 2.650 | 1.440 |
| close_drawer | 7.340 | 3.800 | 1.580 |
| pick_coke_can | 7.680 | 5.530 | 2.000 |
| move_near | 7.240 | 5.910 | 4.340 |

## Phase 2: Does matched m cover the KMeans task-related support?

For each state, `G` is the matched KMeans token set and `S_k` is full-L11 Top-k.

| Method | Recall | Precision |
|---|---:|---:|
| matched | 0.378 | 0.378 |
| fixed34 | 0.415 | 0.381 |
| entity_top_p | 0.414 | 0.378 |
| shuffle | 0.391 | 0.370 |

### By matched-m quartile

| Task | Size | Matched recall | Fixed34 recall | Entity recall | Shuffle recall |
|---|---:|---:|---:|---:|---:|
| open_drawer | small | 0.315 | 0.605 | 0.570 | 0.539 |
| open_drawer | middle | 0.585 | 0.501 | 0.489 | 0.501 |
| open_drawer | large | 0.595 | 0.435 | 0.404 | 0.504 |
| close_drawer | small | 0.356 | 0.661 | 0.657 | 0.514 |
| close_drawer | middle | 0.396 | 0.405 | 0.474 | 0.378 |
| close_drawer | large | 0.509 | 0.346 | 0.329 | 0.422 |
| pick_coke_can | small | 0.191 | 0.363 | 0.357 | 0.239 |
| pick_coke_can | middle | 0.178 | 0.237 | 0.229 | 0.170 |
| pick_coke_can | large | 0.077 | 0.067 | 0.062 | 0.057 |
| move_near | small | 0.440 | 0.615 | 0.643 | 0.560 |
| move_near | middle | 0.522 | 0.496 | 0.506 | 0.509 |
| move_near | large | 0.368 | 0.258 | 0.262 | 0.304 |

## Phase 3: Does shuffle harm grow with budget mismatch?

`D = mean |m_shuffle - m_matched| / m_matched`. Episode-level matched-vs-shuffle outcomes.

| D bin | n | Matched | Shuffle | Matched-only | Shuffle-only | Net (matched-only − shuffle-only) |
|---|---:|---:|---:|---:|---:|---:|
| D < 20% | 11 | 4/11 | 3/11 | 2 | 1 | +1 |
| 20–40% | 126 | 74/126 | 57/126 | 25 | 8 | +17 |
| D ≥ 40% | 263 | 135/263 | 138/263 | 46 | 49 | -3 |

### Under- vs over-budget direction

| Direction | n | Matched-only | Shuffle-only | Net |
|---|---:|---:|---:|---:|
| under (< −10%) | 122 | 24 | 18 | +6 |
| balanced | 186 | 29 | 20 | +9 |
| over (> +10%) | 92 | 20 | 20 | +0 |
