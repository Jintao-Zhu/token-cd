# TIES-style L11 reliability diagnostic

- Cached full-layer states: 105 from 35 episodes.
- Primary unit: episode mean across early/middle/late states (no frame pseudo-replication).
- Primary hypothesis: L11 Harm episodes have higher inter-layer rank rigidity than Rescue episodes.

## Primary result

| Metric | Harm | Rescue | Difference | AUC | one-sided exact p |
|---|---:|---:|---:|---:|---:|
| all adjacent layers | 0.8224 | 0.8166 | +0.0058 | 0.600 | 0.2831 |

Decision: **does not support** using TIES rigidity as a Harm gate under the locked exploratory rule.

## Per-task primary metric

| Task | Rescue n/mean | Harm n/mean | Harm-Rescue |
|---|---:|---:|---:|
| close_drawer | 2/0.8023 | 2/0.8002 | -0.0022 |
| move_near | 3/0.8395 | 0/nan | +nan |
| open_drawer | 6/0.8036 | 0/nan | +nan |
| pick_coke_can | 2/0.8355 | 3/0.8372 | +0.0017 |

## Guardrail

These cached episodes were selected for earlier layer diagnostics and are not a random confirmatory sample. A positive signal justifies a separately locked threshold test; it does not validate a closed-loop gate by itself.
