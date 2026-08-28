"""Attention-CD arm, K=8, layers 8-15 (inclusive), widowx_put_eggplant_in_basket, 50 seeds.

Extends the narrow-band (K8, L8-15) cell to widowx_put_eggplant_in_basket, the
last remaining WidowX task with a nonzero full-band signal (vanilla 0/50,
K32-L8-31 1/50). Everything (selector, seed, lambda, layer interval) is identical
to the K8 L8-15 cells on the other tasks; only the task differs.

Hash-paired against the frozen ATTN_SEMANTIC_K32_L8_8TASK_50_V1 attn_semantic
arm so each episode reuses the exact per-seed snapshots already captured.

Pairing gate: initial state (sim_state) + RGB are required to match the frozen
reference per seed. The canonical snapshot hash is recorded but NOT gated, because
on WidowX the agent/RNG components of the snapshot are launch-non-deterministic
(different seeds flip across launches) while sim_state + RGB are bit-stable. The
episode `step` is deterministic (no RNG use after reset), so the flaky field does
not affect the rollout.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from research.semantic_token_cd.distractor_rollout import (
    PCD_SOURCE,
    capture_snapshot,
    jsonable,
    restore_snapshot,
    snapshot_sha,
)
from research.semantic_token_cd.spatial_grid_policy import (
    SpatialGridAttentionCDInference,
)
from research.semantic_token_cd.spatial_grid_rollout import (
    make_environment,
    run_episode,
    write_arrays,
)


TASK = "widowx_put_eggplant_in_basket"
ARM = "attn_semantic"
SEEDS = tuple(range(50))
KMEANS_K = 8
LAYER_START = 8
LAYER_END = 16  # layers 8..15 inclusive
EXPECTED_HOOK_CALLS = 7 * (LAYER_END - LAYER_START)  # 7 x 8 = 56
REFERENCE_ROOT = Path(
    "artifacts/attn_semantic_k32_l8_8task_50_v1/episodes/"
    "widowx_put_eggplant_in_basket/attn_semantic"
)


def build_policy(base):
    policy = copy.copy(base)
    policy.__class__ = SpatialGridAttentionCDInference
    policy.alpha = 0.5
    policy.lambd = 0.5
    policy.kmeans_K = KMEANS_K
    policy.kmeans_seed = 0
    policy.selection_mode = "semantic"
    policy.spatial_selection_mode = "semantic_hard"
    policy.attention_layer_start = LAYER_START
    policy.attention_layer_end = LAYER_END
    policy.attention_mask_value = -1e4
    policy._selector_instr = None
    policy._entities = []
    policy._entity_emb = []
    policy._emb_cache = {}
    policy._episode_trace = []
    policy._episode_logits = []
    policy._episode_seed = 0
    policy._selector_step = 0
    return policy


def expected_reference(seed: int) -> dict:
    path = REFERENCE_ROOT / f"episode_{seed:03d}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen reference summary: {path}")
    return json.loads(path.read_text())


def locked_config() -> dict:
    return {
        "experiment": "ATTN_SEMANTIC_K8_L8_16_WIDOWX_PUT_EGGPLANT_50_V1",
        "benchmark": "SIMPLER widowx_put_eggplant_in_basket (WidowX)",
        "task": TASK,
        "seeds": list(SEEDS),
        "arm": ARM,
        "lambda": 0.5,
        "kmeans_k": KMEANS_K,
        "attention_layers": [LAYER_START, LAYER_END],
        "attention_mask_value_requested": -10000.0,
        "selector": "entity_set; per-entity top-1 KMeans group; union",
        "intervention": (
            "action-to-visual attention block; Action Query -> selected visual "
            "keys on layers [8, 16)"
        ),
        "reference_root": str(REFERENCE_ROOT.resolve()),
        "pairing": (
            "initial state (sim_state) and RGB must match the frozen reference per "
            "seed; canonical snapshot hash recorded but not gated (WidowX agent/RNG "
            "state is launch-non-deterministic while state+RGB are bit-stable)"
        ),
    }


def write_config(artifact: Path, config: dict) -> None:
    path = artifact / "CONFIG_LOCK.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError(f"CONFIG_LOCK differs from requested cell: {path}")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def audit_trace(trace: list[dict], expected_calls: int) -> dict:
    if not trace:
        raise RuntimeError("Attention arm produced no trace")
    feature_equal = all(step["feature_equal"] for step in trace)
    hook_pass = all(
        step.get("negative_truncated", False)
        or step["attention_mask"]["hook_calls"] == expected_calls
        for step in trace
    )
    audit = {
        "all_visual_features_bit_identical": feature_equal,
        "all_attention_hook_audits_pass": hook_pass,
        "technical_pass": feature_equal and hook_pass,
    }
    if not audit["technical_pass"]:
        raise RuntimeError(f"Technical audit failed: {audit}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from properties import get_policy_config
    from simpler_env.policies.openvla.openvla_model import OpenVLAInference

    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    config = locked_config()
    write_config(artifact, config)
    task_root = artifact / "episodes" / TASK / ARM
    task_root.mkdir(parents=True, exist_ok=True)

    env, environment_id = make_environment(TASK)
    checkpoint = str(PCD_SOURCE / "pretrained/openvla-7b")
    policy_config = get_policy_config("openvla", checkpoint, TASK, {}, False)
    policy = build_policy(OpenVLAInference(**policy_config))
    completed = []

    for seed in SEEDS:
        summary_path = task_root / f"episode_{seed:03d}_summary.json"
        arrays_path = task_root / f"episode_{seed:03d}_arrays.npz"
        if summary_path.exists() and arrays_path.exists():
            summary = json.loads(summary_path.read_text())
            if not summary.get("technical_pass"):
                raise RuntimeError(f"Existing seed failed audit: {summary_path}")
            completed.append(seed)
            print(json.dumps({"seed": seed, "skip_complete": True}), flush=True)
            continue

        reference = expected_reference(seed)
        snapshot = capture_snapshot(env, seed)
        canonical = snapshot_sha(snapshot)
        obs, state_sha, rgb_sha = restore_snapshot(env, seed, snapshot)
        if (state_sha, rgb_sha) != (
            reference["initial_state_sha256"],
            reference["initial_rgb_sha256"],
        ):
            raise RuntimeError(
                f"Frozen pairing mismatch at seed {seed}: "
                f"state={state_sha}, rgb={rgb_sha}"
            )
        canonical_match = canonical == reference["canonical_snapshot_sha256"]
        if not canonical_match:
            print(
                json.dumps({"seed": seed, "canonical_mismatch": True}),
                flush=True,
            )

        instruction = env.unwrapped.get_language_instruction()
        if instruction != reference["instruction"]:
            raise RuntimeError(f"Instruction mismatch at seed {seed}")
        policy.reset(instruction, seed=seed)
        policy._episode_trace = []
        policy._episode_logits = []
        result, steps, reason, actions, jitter = run_episode(
            env, policy, instruction, obs
        )
        write_arrays(arrays_path, policy._episode_logits, actions)
        trace = jsonable(policy._episode_trace)
        audit = audit_trace(trace, EXPECTED_HOOK_CALLS)
        summary = {
            "protocol_id": config["experiment"],
            "environment_id": environment_id,
            "task": TASK,
            "seed": seed,
            "episode_id": seed,
            "arm": ARM,
            "instruction": instruction,
            "success": bool(result["success"]),
            "failure_reason": reason,
            "control_steps": steps,
            "action_jitter_index": jitter,
            "first_step_residual_norm": float(trace[0]["residual_norm"]),
            "lambda": 0.5,
            "kmeans_k": KMEANS_K,
            "attention_layers": [LAYER_START, LAYER_END],
            "canonical_snapshot_sha256": canonical,
            "canonical_match": canonical_match,
            "initial_state_sha256": state_sha,
            "initial_rgb_sha256": rgb_sha,
            "reference_protocol_id": reference["protocol_id"],
            "arrays_file": arrays_path.name,
            "selector_trace": trace,
            "result": jsonable(result),
            **audit,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        completed.append(seed)
        (artifact / "task_progress.json").write_text(
            json.dumps(
                {"completed_seeds": completed, "n_completed": len(completed)},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "success": summary["success"],
                    "steps": steps,
                    "technical_pass": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    manifest = {
        "protocol_id": config["experiment"],
        "task": TASK,
        "seeds": list(SEEDS),
        "n_seeds": len(SEEDS),
        "all_reference_state_rgb_match": True,
        "all_technical_audits_pass": True,
    }
    (artifact / "PAIRING_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"DONE": True, "n_seeds": len(SEEDS)}), flush=True)


if __name__ == "__main__":
    main()
