from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.coreact_closed_loop.guidance import GuidanceConfig, sample_coreact_actions, tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_ensemble_control.methods import sample_dual_seed_average


def max_abs(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first - second).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if (artifact / "integrity_report.json").exists():
        raise FileExistsError("refusing to overwrite integrity report")
    specs = [json.loads(line) for line in (artifact / "episode_manifest.jsonl").read_text().splitlines()]
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    toward_top = GuidanceConfig(selection="top", group_count=8, guidance_scale=0.5, trust_region_kappa=0.25, action_dim=7, num_steps=10, direction="toward")
    toward_random = GuidanceConfig(selection="random", group_count=8, guidance_scale=0.5, trust_region_kappa=0.25, action_dim=7, num_steps=10, direction="toward")
    task_reports = []
    for task_id in (4, 7):
        spec = next(row for row in specs if row["task_id"] == task_id and row["init_state_id"] == 0)
        env, env_preprocessor, _ = make_task_env(spec["suite"], task_id, config)
        try:
            env.envs[0].init_state_id = 0
            observation, _ = env.reset(seed=spec["reset_seed"])
            batch = prepare(policy, preprocessor, env_preprocessor, observation, spec["language"])
        finally:
            env.close()
        shape = (1, config.chunk_size, config.max_action_dim)
        generator = torch.Generator(device=batch["state"].device)
        noise1 = torch.randn(shape, generator=generator.manual_seed(spec["action_noise_seed"] * 1000), device=batch["state"].device, dtype=batch["state"].dtype)
        noise2 = torch.randn(shape, generator=generator.manual_seed(spec["second_action_noise_seed"] * 1000), device=batch["state"].device, dtype=batch["state"].dtype)

        def native():
            return policy.model.sample_actions(batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise=noise1)

        def top():
            return sample_coreact_actions(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise1, means["visual_position_mean"], config=toward_top, selection_seed=spec["selection_seed"] * 1000)

        def random():
            return sample_coreact_actions(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise1, means["visual_position_mean"], config=toward_random, selection_seed=spec["selection_seed"] * 1000)

        def ensemble():
            return sample_dual_seed_average(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise1, noise2, matched_step=True)

        with torch.inference_mode():
            native1, native2 = native(), native()
            top1, top_trace1 = top(); top2, top_trace2 = top()
            random1, random_trace1 = random(); random2, random_trace2 = random()
            ensemble1, ensemble_trace1 = ensemble(); ensemble2, ensemble_trace2 = ensemble()
        checks = {
            "frozen_eval_model": not policy.training and not any(parameter.requires_grad for parameter in policy.parameters()),
            "native_repeat": max_abs(native1, native2) <= 1e-6,
            "top_repeat": max_abs(top1, top2) <= 1e-6,
            "random_repeat": max_abs(random1, random2) <= 1e-6,
            "ensemble_repeat": max_abs(ensemble1, ensemble2) <= 1e-6,
            "top_exactly_8_changed": len(top_trace1["selected_indices"]) == 8 and len(top_trace1["changed_indices"]) == 8,
            "random_exactly_8_changed": len(random_trace1["selected_indices"]) == 8 and len(random_trace1["changed_indices"]) == 8,
            "top_selected_equals_changed": sorted(top_trace1["selected_indices"]) == sorted(top_trace1["changed_indices"]),
            "random_selected_equals_changed": sorted(random_trace1["selected_indices"]) == sorted(random_trace1["changed_indices"]),
            "protected_tokens_untouched": top_trace1["protected_tokens_untouched"] and random_trace1["protected_tokens_untouched"],
            "shared_first_noise": top_trace1["step_traces"][0]["finite"] and tensor_sha256(noise1) == ensemble_trace1["first_noise_sha256"],
            "arm_c_zero_masked_tokens": ensemble_trace1["masked_token_count"] == 0 and ensemble_trace1["selected_indices"] == [] and ensemble_trace1["changed_indices"] == [],
            "random_overlap_recorded": random_trace1["overlap_with_attention_top"] == len(set(random_trace1["selected_indices"]) & set(random_trace1["attention_top_indices"])),
            "all_finite": bool(torch.isfinite(torch.stack([native1, top1, random1, ensemble1])).all()),
            "trace_repeat": top_trace1["selected_indices"] == top_trace2["selected_indices"] and random_trace1["selected_indices"] == random_trace2["selected_indices"],
        }
        task_reports.append(
            {
                "task_id": task_id,
                "checks": checks,
                "repeat_max_abs": {
                    "native": max_abs(native1, native2),
                    "top": max_abs(top1, top2),
                    "random": max_abs(random1, random2),
                    "ensemble": max_abs(ensemble1, ensemble2),
                },
                "noise_hashes": {"first": tensor_sha256(noise1), "second": tensor_sha256(noise2)},
                "prefix_hashes": {"top_native": top_trace1["prefix_sha256"], "top_masked": top_trace1["negative_prefix_sha256"], "random_native": random_trace1["prefix_sha256"], "random_masked": random_trace1["negative_prefix_sha256"]},
                "top_indices": top_trace1["selected_indices"],
                "random_indices": random_trace1["selected_indices"],
                "random_attention_overlap": random_trace1["overlap_with_attention_top"],
                "ensemble_trace": ensemble_trace1,
            }
        )
    passed = all(all(report["checks"].values()) for report in task_reports)
    result = {"pass": passed, "repeat_tolerance": 1e-6, "tasks": task_reports}
    (artifact / "integrity_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("mandatory integrity gate failed")
    (artifact / "status/integrity.pass").touch(exist_ok=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
