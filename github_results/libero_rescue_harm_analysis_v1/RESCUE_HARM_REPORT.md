# selected5 rescue/harm analysis

paired=300

| group | n | first divergence step | action L2 | selected steps | vanilla steps | selected m | feature perturbation | guided change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rescue | 27 | 0.33 | 0.15591 | 110.59 | 220.00 | 34.15 | 0.29966 | 0.2167 |
| harm | 28 | 0.36 | 0.21029 | 220.00 | 115.93 | 35.89 | 0.31185 | 0.3291 |
| both_success | 232 | 0.60 | 0.15547 | 103.43 | 106.37 | 32.92 | 0.29941 | 0.2155 |
| both_fail | 13 | 0.38 | 0.16139 | 220.00 | 220.00 | 35.44 | 0.31042 | 0.3316 |
