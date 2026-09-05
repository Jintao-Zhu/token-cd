# SHR follow-up experiment release

This directory contains the compact, GitHub-friendly release of the completed
SHR follow-up experiments.  The full experiment rationale, implementation
details, cross-method tables, and audited conclusions are in
[`docs/SHR_COMPREHENSIVE_REPORT_2026-09-05.md`](../docs/SHR_COMPREHENSIVE_REPORT_2026-09-05.md).

## Released data

| Directory | Scope | Episode rows | Contents |
|---|---:|---:|---|
| `adaptive_shr_experiment/` | SIMPLER 9 tasks × 300 seeds | 2,700 | episode data, paired CSV, config, report, aggregate statistics |
| `instruction_component_shr_v1/` | 4 tasks × 300 seeds | 1,200 | episode data, paired CSV, component statistics, report, representative mask visualizations |
| `projected_shr_5task_0_299_v1/` | 5 tasks × 300 seeds × 2 projected arms | 3,000 | episode data, config, regenerated aggregate statistics |
| `sp_shr_boundary_partial_3task_0_99_v1/` | 3 tasks × 100 seeds × 2 SP arms | 600 | episode data, config, regenerated aggregate statistics |
| `pi0_shr_5task_0_299_v1/` | 5 tasks × 300 seeds × 2 arms | 3,000 | episode data, config, regenerated aggregate statistics |
| `positive_support_constrained_shr_v1/` | offline diagnostic | — | offline analysis report and summary; no rollout was launched |

The original Vanilla/Recon/SHR canonical 9 × 300 release and the ST-SHR
release remain in their existing sibling directories.

## Data format

`EPISODES.jsonl` contains one JSON object per episode/arm.  Rows retain the
task, seed, success outcome, pairing hashes, integrity flags, method
hyperparameters, runtime, and compact method-specific diagnostics.  Large
step-level traces, rendered videos, model checkpoints, caches, and NPZ tensors
are intentionally excluded.

`FINAL_RESULTS.json` is either the analyzer's original aggregate output or a
deterministically regenerated task/overall summary.  Generated summaries
include success counts, paired Rescue/Harm/Net counts, and within-experiment
paired comparisons where multiple arms share the same seed.

## Reproducing the compact export

From the repository root, after placing the raw rollout directories in the
paths used by the experiment launchers:

```bash
python research/semantic_token_cd/export_shr_followup_release.py
```

Raw artifacts are ignored by Git.  This keeps the repository practical to
clone while preserving enough information to reproduce all reported tables
and audit snapshot pairing.
