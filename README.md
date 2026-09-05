# token-cd

OpenVLA token-level contrastive-decoding (CD) research.

## Layout

- `research/semantic_token_cd/` — attention-mask contrastive decoding (Semantic Token-CD line).
- `research/ar_*` — action-token / counterfactual probing experiments.
- `research/coreact_*` — coreact (core-activation) probing experiments.
- `research/token_pcd_*`, `research/cw_lpcd`, `research/latent_pcd_*` — token PCD variants.
- `research/causal_failure_gate` — causal failure-gating experiments.

## Usage

The package imports as `research.<subpackage>...`. Add this repository's root
to `PYTHONPATH` (the environment must provide `simpler_env`, `properties`, and
the OpenVLA-7B checkpoint under `PCD_SOURCE/pretrained/openvla-7b`).

## SHR experiment status

- [Comprehensive SHR methods and results report](docs/SHR_COMPREHENSIVE_REPORT_2026-09-05.md)
- [Compact follow-up experiment data release](github_results/SHR_FOLLOWUP_RELEASE.md)

The compact release contains episode-level outcomes and pairing metadata; raw
rollout traces, checkpoints, caches, and large tensor dumps remain excluded.
