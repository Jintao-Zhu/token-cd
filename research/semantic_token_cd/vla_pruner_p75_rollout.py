"""Paper-v5 reproduction runner: code-L15 vs paper-L16:31 at 75% pruning.

This deliberately reuses the audited formal rollout machinery while replacing
only the protocol constants and arms.  Vanilla is not rerun here; its verified
formal_1200 summaries are joined by the analysis script using snapshot hashes.
"""
from __future__ import annotations

from research.semantic_token_cd import vla_pruner_formal_rollout as runner

runner.PROTOCOL = "VLA_PRUNER_OPENVLA_PAPER_V5_P75"
runner.FORMAL_STAGE = "p75_code_vs_paper"
runner.ARMS = (
    "vla_pruner_prune75_code_l15",
    "vla_pruner_prune75_paper_l16_31",
)
runner.ARM_TARGET_R = {arm: 0.75 for arm in runner.ARMS}
runner.ARM_IMAGE_KEEP = {arm: 64 for arm in runner.ARMS}


if __name__ == "__main__":
    runner.main()
