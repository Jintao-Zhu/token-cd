# L11 Fixed-Lambda Heterogeneity Discovery

| Pattern (0,.25,.5) | Count |
|---|---:|
| 000 | 101 |
| 100 | 16 |
| 010 | 31 |
| 001 | 56 |
| 110 | 14 |
| 101 | 22 |
| 011 | 53 |
| 111 | 107 |

- Fixed λ=.5: **238/400**
- Oracle: **299/400**
- Oracle gap: **15.3pp**

| Feature | OOF router | Recovery | Rescue/Harm | Row AUC |
|---|---:|---:|---:|---:|
| task_only | 238/400 | 0.0% | 0/0 | 0.667 |
| prompt_context | 239/400 | 1.6% | 19/18 | 0.826 |
| prompt_action | 242/400 | 6.6% | 20/16 | 0.824 |

- Decision: **STOP**
