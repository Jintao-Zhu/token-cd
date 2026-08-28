#!/usr/bin/env python3
"""Run mandatory real-model parity and protection gates before guided development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from research.coreact_closed_loop.guidance import GuidanceConfig, sample_coreact_actions, tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare


TOLERANCE = 1e-6


def batch_sha256(batch: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            continue
        digest.update(key.encode("utf-8"))
        digest.update(tensor_sha256(value).encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = args.artifact.resolve()
    qualification = json.loads((artifact / "qualification_gate.json").read_text())
    if not qualification.get("pass"):
        raise RuntimeError("backbone qualification did not pass")

    cfg, policy, preprocessor, _ = load_policy_and_processors(workspace)
    env, env_preprocessor, _ = make_task_env("libero_spatial", 0, cfg)
    try:
        inner = env.envs[0]
        inner.init_state_id = 0
        observation, _ = env.reset(seed=42_000_001)
        language = inner.task_description
        prepared = prepare(policy, preprocessor, env_preprocessor, observation, language)
    finally:
        env.close()
    means = torch.load(
        workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",
        weights_only=True,
        map_location="cpu",
    )
    generator = torch.Generator(device=prepared["state"].device).manual_seed(42_000_002)
    noise = torch.randn(
        (1, cfg.chunk_size, cfg.max_action_dim),
        generator=generator,
        dtype=prepared["state"].dtype,
        device=prepared["state"].device,
    )
    args_common = (
        policy.model,
        prepared["images"],
        prepared["image_masks"],
        prepared["lang_tokens"],
        prepared["lang_masks"],
        prepared["state"],
    )
    with torch.inference_mode():
        native_a = policy.model.sample_actions(*args_common[1:], noise=noise)
        native_b = policy.model.sample_actions(*args_common[1:], noise=noise)
        zero_a, zero_trace_a = sample_coreact_actions(
            *args_common,
            noise,
            means["visual_position_mean"],
            config=GuidanceConfig(selection="top", guidance_scale=0.0),
            selection_seed=42_000_003,
        )
        zero_b, zero_trace_b = sample_coreact_actions(
            *args_common,
            noise,
            means["visual_position_mean"],
            config=GuidanceConfig(selection="top", guidance_scale=0.0),
            selection_seed=42_000_003,
        )
        guided_a, guided_trace_a = sample_coreact_actions(
            *args_common,
            noise,
            means["visual_position_mean"],
            config=GuidanceConfig(selection="top"),
            selection_seed=42_000_003,
        )
        guided_b, guided_trace_b = sample_coreact_actions(
            *args_common,
            noise,
            means["visual_position_mean"],
            config=GuidanceConfig(selection="top"),
            selection_seed=42_000_003,
        )
        random_a, random_trace_a = sample_coreact_actions(
            *args_common,
            noise,
            means["visual_position_mean"],
            config=GuidanceConfig(selection="random"),
            selection_seed=42_000_004,
        )
        random_b, random_trace_b = sample_coreact_actions(
            *args_common,
            noise,
            means["visual_position_mean"],
            config=GuidanceConfig(selection="random"),
            selection_seed=42_000_004,
        )

    native_determinism = float((native_a - native_b).abs().max())
    zero_native_parity = float((zero_a - native_a).abs().max())
    zero_determinism = float((zero_a - zero_b).abs().max())
    guided_determinism = float((guided_a - guided_b).abs().max())
    random_determinism = float((random_a - random_b).abs().max())
    traces = [zero_trace_a, zero_trace_b, guided_trace_a, guided_trace_b, random_trace_a, random_trace_b]
    protected_ok = all(
        token["modality"] == "visual"
        and token["intervention_allowed"]
        and not token["is_special"]
        and not token["is_padding"]
        and not token["is_state"]
        for trace in traces
        for token in trace["selected_tokens"]
    )
    exactly_eight = all(len(trace["selected_indices"]) == 8 for trace in traces)
    virtual_guidance_zero = all(
        step["virtual_guidance_norm"] == 0.0 for trace in traces for step in trace["step_traces"]
    )
    random_repeat_same_selection = random_trace_a["selected_indices"] == random_trace_b["selected_indices"]
    top_repeat_same_selection = guided_trace_a["selected_indices"] == guided_trace_b["selected_indices"]
    all_finite = all(bool(torch.isfinite(value).all()) for value in (native_a, zero_a, guided_a, random_a))
    report = {
        "gate": "closed_loop_sampler_integrity",
        "real_batch": {
            "suite": "libero_spatial",
            "task_id": 0,
            "init_state_id": 0,
            "canonical_language": language,
            "batch_sha256": batch_sha256(prepared["batch"]),
            "noise_sha256": tensor_sha256(noise),
            "real_camera_count": len(prepared["images"]),
            "camera_masks_all_valid": [bool(mask.all()) for mask in prepared["image_masks"]],
            "state_input_dim": int(prepared["batch"]["observation.state"].shape[-1]),
            "state_model_dim": int(prepared["state"].shape[-1]),
            "language_valid_tokens": int(prepared["lang_masks"].sum()),
        },
        "checks": {
            "policy_eval": not policy.training,
            "all_parameters_frozen": not any(parameter.requires_grad for parameter in policy.parameters()),
            "native_determinism_max_abs": native_determinism,
            "lambda_zero_native_parity_max_abs": zero_native_parity,
            "lambda_zero_determinism_max_abs": zero_determinism,
            "guided_determinism_max_abs": guided_determinism,
            "random_determinism_max_abs": random_determinism,
            "protected_tokens_untouched": protected_ok,
            "exactly_eight_visual_groups": exactly_eight,
            "virtual_action_guidance_zero": virtual_guidance_zero,
            "top_selection_repeatable": top_repeat_same_selection,
            "random_selection_repeatable": random_repeat_same_selection,
            "attention_layer_count": guided_trace_a["attention_layer_count"],
            "all_outputs_finite": all_finite,
            "guided_differs_from_native": bool((guided_a - native_a).abs().max() > TOLERANCE),
            "random_differs_from_native": bool((random_a - native_a).abs().max() > TOLERANCE),
        },
        "tolerance": TOLERANCE,
        "output_hashes": {
            "native": tensor_sha256(native_a),
            "lambda_zero": tensor_sha256(zero_a),
            "guided_top": tensor_sha256(guided_a),
            "guided_random": tensor_sha256(random_a),
        },
    }
    checks = report["checks"]
    report["pass"] = (
        report["real_batch"]["real_camera_count"] == 2
        and report["real_batch"]["camera_masks_all_valid"] == [True, True]
        and report["real_batch"]["state_input_dim"] == 8
        and report["real_batch"]["state_model_dim"] == 32
        and report["real_batch"]["language_valid_tokens"] > 0
        and checks["policy_eval"]
        and checks["all_parameters_frozen"]
        and native_determinism <= TOLERANCE
        and zero_native_parity <= TOLERANCE
        and zero_determinism <= TOLERANCE
        and guided_determinism <= TOLERANCE
        and random_determinism <= TOLERANCE
        and checks["protected_tokens_untouched"]
        and checks["exactly_eight_visual_groups"]
        and checks["virtual_action_guidance_zero"]
        and checks["top_selection_repeatable"]
        and checks["random_selection_repeatable"]
        and checks["attention_layer_count"] == 16
        and checks["all_outputs_finite"]
        and checks["guided_differs_from_native"]
        and checks["random_differs_from_native"]
    )
    (artifact / "integrity_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Closed-loop Sampler Integrity",
        "",
        f"Overall: {'PASS' if report['pass'] else 'FAIL'}",
        "",
        "| Check | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in checks.items())
    (artifact / "integrity_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()
