# Lambda transition / rescue-harm intersection analysis

paired=300

## current: base λ0.5 vs vanilla, then lower λ

### current -> lambda_0.25
| original group | n | low success | low fail | rate |
|---|---:|---:|---:|---:|
| both_success | 227 | 211 | 16 | 0.930 |
| original_harm | 33 | 20 | 13 | 0.606 |
| original_rescue | 23 | 18 | 5 | 0.783 |
| both_fail | 17 | 6 | 11 | 0.353 |

### current -> lambda_0.125
| original group | n | low success | low fail | rate |
|---|---:|---:|---:|---:|
| both_success | 227 | 209 | 18 | 0.921 |
| original_harm | 33 | 28 | 5 | 0.848 |
| original_rescue | 23 | 18 | 5 | 0.783 |
| both_fail | 17 | 6 | 11 | 0.353 |

## selected5: base λ0.5 vs vanilla, then lower λ

### selected5 -> lambda_0.25
| original group | n | low success | low fail | rate |
|---|---:|---:|---:|---:|
| both_success | 232 | 217 | 15 | 0.935 |
| both_fail | 13 | 6 | 7 | 0.462 |
| original_rescue | 27 | 21 | 6 | 0.778 |
| original_harm | 28 | 14 | 14 | 0.500 |

### selected5 -> lambda_0.125
| original group | n | low success | low fail | rate |
|---|---:|---:|---:|---:|
| both_success | 232 | 206 | 26 | 0.888 |
| both_fail | 13 | 4 | 9 | 0.308 |
| original_rescue | 27 | 21 | 6 | 0.778 |
| original_harm | 28 | 21 | 7 | 0.750 |

## Best-config intersections: current λ0.125 vs selected5 λ0.5

### vanilla_fail
{
  "n": 40,
  "both_rescue": 20,
  "only_a_rescue": 4,
  "only_b_rescue": 7,
  "neither": 9
}

### vanilla_success
{
  "n": 260,
  "both_harm": 4,
  "only_a_harm": 19,
  "only_b_harm": 24,
  "both_preserved": 213
}

