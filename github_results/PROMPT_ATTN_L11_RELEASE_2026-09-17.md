# Prompt-Attention / L11 experiment release — 2026-09-17

This release archives the experiment code and lightweight result records developed after the 2026-09-05 repository checkpoint.

## Included experiment families

- Prompt-to-visual attention diagnostics and all-layer selection scans.
- L11 matched-budget, fixed-count, Top-p, and Top-p90 comparisons.
- Nine-task, 300-seed L11-Matched evaluation.
- Dense guidance-strength (`lambda`) scans on all nine tasks.
- Adaptive-lambda, risk-gate, static-router, and lookahead investigations.
- L11/L14 spatial diagnosis and exhaustive L11 partner-layer scans.
- Prompt/action attention complement, reranking, and causal-layer analyses.
- DTP and VLA-Pruner reproduction/calibration studies.
- Matched-budget encoding, shuffle/scale pilot, and budget-provenance audits.
- Semantic-sensitivity layer scan used to compare L11 with all other layers.

## Result-file policy

The commit includes compact Markdown, JSON, CSV, and PNG summaries from completed artifacts. Raw per-episode arrays, simulator snapshots, model caches, videos, logs, and other regenerable rollout data remain excluded by `.gitignore`.

The `l11_matched_budget_provenance_closed_loop_100_199_v1` rollout was still running when this release was created, so its partial outputs are intentionally not included. Its runner and analyzer source code are included.
