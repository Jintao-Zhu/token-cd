"""Locked held-out protocol for confidence-gated Target Positive-Boost."""
from __future__ import annotations

from pathlib import Path

from research.semantic_token_cd.target_specific_protocol import CANONICAL, PCD_SOURCE, TASKS, generic_instructions
from research.semantic_token_cd.xswap_protocol import atomic_json

PROTOCOL = "TARGET_BOOST_CONFIDENCE_GATE_V1"
REPO = Path("/home/leju-suzhou/zjt_ws/token-cd")
ARTIFACT = REPO / "artifacts/target_positive_boost_confidence_gate_v1"
BASELINE = REPO / "artifacts/prompt_attn_l11_token_count_v1"
ARMS = ("positive_boost_0p5", "confidence_gate_0p4")
SEEDS = tuple(range(100, 200))
LAMBDA = 0.5
ETA = 0.5
CONFIDENCE_THRESHOLD = 0.4


def arm_config(task: str, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    return {
        "formula": "confidence_gated_positive_boost" if arm == "confidence_gate_0p4" else "positive_boost",
        "eta": ETA,
        "confidence_threshold": CONFIDENCE_THRESHOLD if arm == "confidence_gate_0p4" else None,
        "contrast_instruction": generic_instructions(task)[0],
    }


def write_lock() -> None:
    atomic_json(ARTIFACT / "CONFIG_LOCK.json", {
        "protocol_id": PROTOCOL, "tasks": list(TASKS), "seeds": "100-199",
        "arms": ["l11_matched_reused", *ARMS], "nominal_arm_episodes": 1200,
        "reused_l11_matched": 400, "new_episodes": 800,
        "baseline_artifact": str(BASELINE), "canonical_snapshots": str(CANONICAL),
        "attention_layer": 11, "eta": ETA, "confidence_threshold": CONFIDENCE_THRESHOLD,
        "confidence": "(P-Q)/(P+1e-8)",
        "gated_score": "P + 0.5*max(P-Q,0)*I[(P-Q)/(P+eps)>=0.4]",
        "ungated_score": "P + 0.5*max(P-Q,0)",
        "generic_instructions": {task: generic_instructions(task)[0] for task in TASKS},
        "matched_budget": "own-state Standard-SHR KMeans count; token identity from L11 score",
        "locked_downstream": "harmonic beta0 gamma1; shared clean prefix; lambda0.5 dims0..5; gripper clean",
        "selection_provenance": "threshold chosen on seeds0-99; evaluated once on held-out seeds100-199",
    })
