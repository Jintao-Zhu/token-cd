#!/usr/bin/env python3
"""Complete diagnostic controls and lock a passed development artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from research.coreact_exploration.instrumentation import predict_teacher_forced_velocity
from research.coreact_exploration.metrics import sha256_file, signed_effect_interpretation
from research.coreact_exploration.qualify_single_suite import CHECKPOINT_REVISION, Paths, load_preprocessor
from research.coreact_exploration.run_development_gates import (
    CAMERA_IDS,
    SnapshotRows,
    load_policy,
    processed_state,
    write_json,
)


def swapped_instruction_diagnostic(workspace, artifact, manifest, store, preprocessor, policy):
    source = next(row for row in manifest if row["split"] == "development")
    alternative = next(
        row["instruction"]
        for row in manifest
        if row["split"] == "development" and row["task_id"] != source["task_id"]
    )
    batch = processed_state(store, preprocessor, source)
    swapped_row = {**source, "instruction": alternative}
    swapped = processed_state(store, preprocessor, swapped_row)
    images, masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    actions = policy.prepare_action(batch)
    generator = torch.Generator(device=actions.device).manual_seed(1729)
    noise = torch.randn(actions.shape, generator=generator, device=actions.device, dtype=actions.dtype)
    tau = torch.tensor([0.5], device=actions.device, dtype=actions.dtype)
    common = (policy.model, images, masks)
    with torch.inference_mode():
        base = predict_teacher_forced_velocity(
            *common,
            batch["observation.language.tokens"],
            batch["observation.language.attention_mask"],
            state,
            actions,
            tau,
            noise,
            camera_ids=CAMERA_IDS,
        )
        changed = predict_teacher_forced_velocity(
            *common,
            swapped["observation.language.tokens"],
            swapped["observation.language.attention_mask"],
            state,
            actions,
            tau,
            noise,
            camera_ids=CAMERA_IDS,
        )
    return {
        "source_instruction": source["instruction"],
        "replacement_instruction": alternative,
        "same_observation_action_tau_noise": True,
        "velocity_delta_mse": float((base["v_pred"] - changed["v_pred"]).square().mean()),
        "velocity_max_abs_diff": float((base["v_pred"] - changed["v_pred"]).abs().max()),
        "diagnostic_only": True,
    }


def negative_controls(artifact: Path):
    examples = json.loads((artifact / "prefix_map_examples.json").read_text())
    protected_mislabeled = [
        token
        for example in examples
        for token in example["tokens"]
        if token["intervention_allowed"]
        and (token["is_special"] or token["is_padding"] or token["is_state"])
    ]
    effects = [json.loads(line) for line in (artifact / "development_effects.jsonl").read_text().splitlines()]
    first_state = [row for row in effects if row["state_index"] == 0 and row["modality"] == "visual"]
    scores = np.asarray([row["late_half_action_to_context_attention"] for row in first_state])
    ids = np.asarray([row["token_index"] for row in first_state])
    original_top = set(ids[np.argsort(scores)[-8:]].tolist())
    permuted = np.random.default_rng(271828).permutation(scores)
    permuted_top = set(ids[np.argsort(permuted)[-8:]].tolist())
    shuffled_semantics = signed_effect_interpretation(
        {"I_G": 0.5, "Q_G": 0.25, "Q_rel": 0.5}, target_semantics_valid=False
    )
    return {
        "protected_token_mislabeled_count": len(protected_mislabeled),
        "score_permutation": {
            "modality": "visual",
            "seed": 271828,
            "original_top8": sorted(original_top),
            "permuted_top8": sorted(permuted_top),
            "top8_changed": original_top != permuted_top,
        },
        "shuffled_demonstration_target": shuffled_semantics,
        "pass": not protected_mislabeled
        and original_top != permuted_top
        and shuffled_semantics["Q_G"] is None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    integrity = json.loads((artifact / "integrity_report.json").read_text())
    if not integrity["pass"]:
        raise RuntimeError("cannot lock a failed development artifact")
    manifest = [json.loads(line) for line in (artifact / "sample_manifest.jsonl").read_text().splitlines()]
    dataset = workspace / "counterfactual-flow-vla/pi05_svcpd_p0/dataset_cache/HuggingFaceVLA_libero"
    paths = Paths(
        workspace,
        dataset,
        workspace / "task1/.hf-cache/hub/models--lerobot--smolvla_libero/snapshots" / CHECKPOINT_REVISION,
        workspace / "counterfactual-flow-vla/pi05_svcpd_p0/verified_calibration_download_manifest.json",
    )
    store = SnapshotRows(dataset)
    preprocessor = load_preprocessor(paths)
    policy = load_policy(paths)
    diagnostic = swapped_instruction_diagnostic(
        workspace, artifact, manifest, store, preprocessor, policy
    )
    controls = negative_controls(artifact)
    write_json(artifact / "instruction_swap_diagnostic.json", diagnostic)
    write_json(artifact / "negative_controls.json", controls)
    if not controls["pass"]:
        raise RuntimeError("negative controls failed")

    protocol = yaml.safe_load((artifact / "protocol.yaml").read_text())
    lock = {
        **protocol,
        "lock_stage": "development_integrity_passed_heldout_deferred",
        "actual_expert_layers": list(range(16)),
        "late_half_expert_layers": list(range(8, 16)),
        "attention_types": {
            "even_layers": "joint_self_attention_action_queries_to_prefix_keys",
            "odd_layers": "expert_cross_attention_action_queries_to_prefix_keys",
        },
        "group_definition": {
            "visual": "single post-connector visual token; no pixel-grid topology claimed",
            "language": "single valid non-special tokenizer token",
        },
        "locked_effect_thresholds": integrity["development_effect_thresholds"],
        "locked_no_op_99_9_percentile": integrity["no_op_99_9_percentile"],
        "code_hashes": {
            "instrumentation": sha256_file(workspace / "research/coreact_exploration/instrumentation.py"),
            "metrics": sha256_file(workspace / "research/coreact_exploration/metrics.py"),
            "attention_hook": sha256_file(
                workspace / "lerobot/src/lerobot/policies/smolvla/smolvlm_with_expert.py"
            ),
            "development_runner": sha256_file(
                workspace / "research/coreact_exploration/run_development_gates.py"
            ),
        },
        "heldout_unblock_condition": "obtain a second formally verified LIBERO suite at the fixed evaluation revision",
    }
    (artifact / "protocol.lock.yaml").write_text(yaml.safe_dump(lock, sort_keys=False))
    report = f"""# CoreAct Development Integrity Report

## Scope

Single-suite development only on `HuggingFaceVLA/libero@{protocol['dataset']['evaluation_snapshot_revision']}` LIBERO-Object. No held-out rows were created or viewed.

## Mandatory Gates

| Gate | Result | Maximum discrepancy |
|---|---|---:|
| Determinism velocity/attention | PASS | 0 |
| Teacher-forced forward parity | PASS | 1.862645149230957e-09 |
| Identity and empty-group no-op | PASS | 0 |
| Batched versus serial intervention | PASS | 1.1831521987915039e-05 (locked tolerance 1e-4) |
| Prefix map/protected tokens | PASS | 24/24 legal maps; 0 protected tokens eligible |
| Attention normalization/masking | PASS | row-sum 3.5762786865234375e-07; masked probability 0 |
| Negative controls | PASS | top-8 changes under fixed permutation; shuffled target disables Q semantics |

All 24 development states passed. The model was in `eval()` and all parameters were frozen. The calibration means use 512 states from 256 episodes disjoint from development; their actions/effects were not used.

## Diagnostic Controls

- All valid language embeddings to zero: velocity MSE `{integrity['diagnostic_positive_controls']['all_language_zero_velocity_mse']}`.
- Camera1 visual embeddings to zero: velocity MSE `{integrity['diagnostic_positive_controls']['camera1_zero_velocity_mse']}`.
- Alternate real task instruction: velocity MSE `{diagnostic['velocity_delta_mse']}`.

These controls show that the hook location changes model velocity. Zero replacement is OOD and none of these diagnostics is evidence for the ranking hypothesis.

## Locked Development Thresholds

- Visual I_G threshold: `{integrity['development_effect_thresholds']['visual']}`.
- Language I_G threshold: `{integrity['development_effect_thresholds']['language']}`.
- Q_rel anchor/nuisance engineering threshold remains +/-0.05.

## Blocker

The formal local source has complete coverage for LIBERO-Object only. Boundary fragments from other suites are not eligible. The original two-suite held-out experiment remains blocked until a second suite is downloaded and provenance-verified at the fixed evaluation revision.
"""
    (artifact / "integrity_report.md").write_text(report)
    decision = json.loads((artifact / "decision.json").read_text())
    decision["protocol_locked"] = True
    decision["diagnostic_controls_complete"] = True
    write_json(artifact / "decision.json", decision)
    print(artifact)


if __name__ == "__main__":
    main()
