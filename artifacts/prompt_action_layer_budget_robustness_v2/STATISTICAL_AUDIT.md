# Statistical audit

## Direct answer

The exact matched scale `1.0` is not uniquely optimal for the offline proxies. Reducing the budget to `0.75×` consistently strengthens L11's target selectivity, while action alignment remains similar or slightly better. However, the 10-episode validation split is too small for most paired confidence intervals to exclude zero.

## L11 budget differences

Values are paired episode means relative to scale 1.0.

| Split | Scale | Metric | Difference | 95% bootstrap CI |
|---|---:|---|---:|---:|
| exploration | 0.75 | top_alignment | +0.0209 | [-0.0116, +0.0573] |
| exploration | 0.75 | semantic_separation | +0.0581 | [+0.0298, +0.0883] |
| exploration | 0.75 | mask_jaccard_separation | +0.0304 | [+0.0043, +0.0568] |
| exploration | 1.25 | top_alignment | +0.0071 | [-0.0167, +0.0343] |
| exploration | 1.25 | semantic_separation | +0.0211 | [-0.0246, +0.0652] |
| exploration | 1.25 | mask_jaccard_separation | -0.0278 | [-0.0560, +0.0003] |
| validation | 0.75 | top_alignment | +0.0080 | [-0.0136, +0.0324] |
| validation | 0.75 | semantic_separation | +0.0185 | [-0.0319, +0.0620] |
| validation | 0.75 | mask_jaccard_separation | +0.0134 | [-0.0304, +0.0560] |
| validation | 1.25 | top_alignment | +0.0311 | [+0.0035, +0.0596] |
| validation | 1.25 | semantic_separation | -0.0257 | [-0.0920, +0.0348] |
| validation | 1.25 | mask_jaccard_separation | -0.0229 | [-0.0607, +0.0173] |

## Layer comparison at 0.75×

| Split | Comparison | Metric | Difference | 95% bootstrap CI |
|---|---|---|---:|---:|
| exploration | L11 - L9 | top_alignment | +0.0670 | [+0.0290, +0.1064] |
| exploration | L11 - L9 | semantic_separation | +0.0899 | [+0.0230, +0.1576] |
| exploration | L11 - L14 | top_alignment | +0.0228 | [-0.0163, +0.0640] |
| exploration | L11 - L14 | semantic_separation | -0.0035 | [-0.0530, +0.0442] |
| validation | L11 - L9 | top_alignment | +0.0194 | [-0.0272, +0.0684] |
| validation | L11 - L9 | semantic_separation | +0.0305 | [-0.0200, +0.0880] |
| validation | L11 - L14 | top_alignment | +0.0245 | [-0.0153, +0.0633] |
| validation | L11 - L14 | semantic_separation | +0.0213 | [-0.1261, +0.1708] |

## Interpretation

- L11 is the only shortlisted layer for which `0.75×` improves action alignment, target separation, and mask separation in both splits.
- L14 at `0.75×` gains semantic separation but does not improve action alignment consistently.
- L9 is less stable: its exploration action alignment drops strongly at `0.75×`.
- L11 at `1.25×` raises validation action alignment but loses semantic and mask selectivity, suggesting that larger masks increasingly include prompt-insensitive context.
- These results strengthen L11 as the most budget-robust offline candidate, but still do not prove superior closed-loop success.
