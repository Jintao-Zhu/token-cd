# L11 Current-step Confidence × Intervention Audit

Discovery set: four tasks, seeds 0--49, 200 fixed-L11 episodes.

Category counts: `{'rescue': 34, 'stable_success': 65, 'fixed_cd_harm': 10, 'stable_fail': 91}`

| Metric | Aggregation | Harm-vs-Rescue AUC | Bootstrap 95% interval |
|---|---|---:|---:|
| positive_negative_js | mean | 0.6853 | [0.4382, 0.9088] |
| intervention_magnitude | mean | 0.5971 | [0.3412, 0.8412] |
| confidence_x_magnitude | p90 | 0.5735 | [0.3176, 0.8176] |
| intervention_magnitude | p90 | 0.5735 | [0.3206, 0.8119] |
| confidence_x_magnitude | mean | 0.5588 | [0.3265, 0.7912] |
| margin_x_magnitude | mean | 0.5471 | [0.3088, 0.7854] |
| margin_x_magnitude | p90 | 0.5235 | [0.3029, 0.7529] |
| positive_negative_js | p90 | 0.5235 | [0.3176, 0.7265] |
| intervention_magnitude | max | 0.4647 | [0.2441, 0.6912] |
| confidence_x_magnitude | max | 0.4471 | [0.2147, 0.6765] |
| margin_x_magnitude | max | 0.3912 | [0.2029, 0.5912] |
| positive_negative_js | max | 0.3676 | [0.1912, 0.5765] |
| positive_confidence | mean | 0.3529 | [0.1323, 0.5853] |
| positive_margin | mean | 0.3353 | [0.1147, 0.5854] |
| positive_confidence | p90 | 0.2853 | [0.0999, 0.5059] |
| positive_margin | p90 | 0.2824 | [0.0882, 0.5001] |
| positive_margin | max | 0.2471 | [0.0735, 0.4529] |
| positive_confidence | max | 0.2441 | [0.0706, 0.4530] |

Primary decision: **STOP**.
Best primary P90 metric: `confidence_x_magnitude` with AUC **0.5735**.
