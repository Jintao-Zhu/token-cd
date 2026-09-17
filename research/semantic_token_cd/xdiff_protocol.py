"""Locked protocol for full-prompt semantic-difference Prompt-Attn-SHR."""
from __future__ import annotations

from pathlib import Path

from research.semantic_token_cd.xswap_protocol import (
    ATTENTION_LAYERS, CANONICAL, LAMBDA, N_VISUAL, PCD_SOURCE, REPO_ROOT,
    TASKS, atomic_json, build_scene_manifest, instruction_set,
    resolve_present_phrases, swap_for_scene,
)

PROTOCOL = "PROMPT_ATTN_SEMANTIC_DIFFERENCE_V1"
SOURCE_ARTIFACT = REPO_ROOT / "artifacts/prompt_attn_instr_swap_v1"
ARTIFACT = REPO_ROOT / "artifacts/prompt_attn_semantic_difference_v1"
ARMS = (
    "vanilla", "correct", "semantic_0p5", "semantic_1p0",
    "paraphrase_0p5", "reverse_0p5",
)
NEW_ARMS = ARMS[2:]
EPSILON = 1e-8
ARM_CONFIG = {
    "semantic_0p5": {"eta": 0.5, "contrast_kind": "semantic"},
    "semantic_1p0": {"eta": 1.0, "contrast_kind": "semantic"},
    "paraphrase_0p5": {"eta": 0.5, "contrast_kind": "paraphrase"},
    "reverse_0p5": {"eta": -0.5, "contrast_kind": "semantic"},
}


def selector_config(arm: str, instruction_plan: dict) -> dict:
    cfg = ARM_CONFIG[arm]
    contrast = (instruction_plan["swapped"] if cfg["contrast_kind"] == "semantic"
                else instruction_plan["paraphrase"])
    if not contrast:
        raise RuntimeError(f"missing {cfg['contrast_kind']} contrast for {arm}")
    return {**cfg, "contrast_instruction": contrast, "epsilon": EPSILON}

