"""Locked protocol for positive-only target-specific attention boosting."""
from __future__ import annotations

from pathlib import Path

from research.semantic_token_cd.target_specific_protocol import (
    CANONICAL, PCD_SOURCE, REPO_ROOT, TASKS, generic_instructions,
)
from research.semantic_token_cd.xswap_protocol import atomic_json

PROTOCOL = "TARGET_POSITIVE_BOOST_SHR_V1"
ARTIFACT = REPO_ROOT / "artifacts/target_positive_boost_shr_v1"
SOURCE_OFFLINE = REPO_ROOT / "artifacts/target_specific_attention_shr_v1/offline"
ARMS = ("correct", "positive_boost_0p5", "positive_boost_1p0", "reverse_boost_1p0")
NEW_ARMS = ARMS[1:]
ATTENTION_LAYERS = (11,)
LAMBDA = 0.5
ARM_CONFIG = {
    "positive_boost_0p5": {"formula": "positive_boost", "eta": 0.5},
    "positive_boost_1p0": {"formula": "positive_boost", "eta": 1.0},
    "reverse_boost_1p0": {"formula": "reverse_boost", "eta": 1.0},
}


def selector_config(task: str, arm: str) -> dict:
    if arm == "correct":
        return {"formula": None, "eta": None, "contrast_instruction": None}
    return {**ARM_CONFIG[arm], "contrast_instruction": generic_instructions(task)[0]}


def write_lock() -> None:
    atomic_json(ARTIFACT / "CONFIG_LOCK.json", {
        "protocol": PROTOCOL, "source_offline": str(SOURCE_OFFLINE), "tasks": list(TASKS),
        "seeds": "0-99", "arms": list(ARMS), "new_closed_loop_episodes": 1200,
        "reused_correct_episodes": 400, "attention_layers": [11], "lambda": .5,
        "generic_instructions": {task: list(generic_instructions(task)) for task in TASKS},
        "formulae": {
            "positive_boost_0p5": "P + 0.5*max(P-Q,0)",
            "positive_boost_1p0": "P + max(P-Q,0)",
            "reverse_boost_1p0": "P + max(Q-P,0)",
        },
        "coverage": "own-state Standard-SHR matched m; all arms exact m",
        "decode": "real instruction, harmonic beta0 gamma1, shared clean prefix, lambda0.5 dims0..5",
        "gate": "eta0 exact Correct; exact m; entered-token direction; generic paraphrase stability",
    })

