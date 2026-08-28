from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_self_guidance.sampler import sample_self_guided_actions, select_ref_top8_actions
from research.coreact_self_guidance.relative_sampler import sample_relative_self_guided_actions
from research.coreact_self_guidance.timestep_sampler import sample_timestep_shift_actions
from research.coreact_closed_loop.guidance import tensor_sha256


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workspace", type=Path, required=True); parser.add_argument("--artifact", type=Path, required=True); args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    if not (artifact / "protocol.lock.yaml").exists():
        raise RuntimeError("phase0 has not locked delta")
    protocol = __import__("yaml").safe_load((artifact / "protocol.lock.yaml").read_text())
    relative_mode = "locked_alpha" in protocol
    timestep_mode = "locked_shift" in protocol
    delta = float(protocol.get("locked_delta", 0.0))
    alpha = float(protocol.get("locked_alpha", 0.0))
    shift = float(protocol.get("locked_shift", 0.0))
    save_vectors = bool(protocol.get("save_velocity_correction_vectors", False))
    config, policy, preprocessor, _ = load_policy_and_processors(workspace)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")
    manifest = [json.loads(line) for line in (artifact / "episode_manifest.jsonl").read_text().splitlines()]
    reports = []
    task_ids = [int(task["id"]) for task in protocol.get("tasks", [{"id": 4}, {"id": 7}])]
    for task_id in task_ids:
        spec = next(row for row in manifest if row["task_id"] == task_id and row["init_state_id"] == 0)
        env, env_preprocessor, _ = make_task_env("libero_spatial", task_id, config)
        try:
            env.envs[0].init_state_id = 0; observation, _ = env.reset(seed=spec["reset_seed"]); batch = prepare(policy, preprocessor, env_preprocessor, observation, spec["language"])
        finally: env.close()
        shape = (1, config.chunk_size, config.max_action_dim); gen = torch.Generator(device=batch["state"].device)
        noise = torch.randn(shape, generator=gen.manual_seed(spec["action_noise_seed"] * 1000), device=batch["state"].device, dtype=batch["state"].dtype)
        def native(): return policy.model.sample_actions(batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise=noise)
        def self_guided(w, pure=False):
            if timestep_mode:
                return sample_timestep_shift_actions(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise, shift=shift, w=w, pure_negative=pure, record_correction_vectors=save_vectors)
            if relative_mode:
                return sample_relative_self_guided_actions(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise, alpha=alpha, w=w, pure_negative=pure)
            return sample_self_guided_actions(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise, delta=delta, w=w, pure_negative=pure)
        def ref(): return select_ref_top8_actions(policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise, means, selection_seed=spec["selection_seed"] * 1000, record_correction_vectors=save_vectors)
        with torch.inference_mode():
            native1, native2 = native(), native()
            w1, w1_trace = self_guided(1.0); w1_repeat, w1_repeat_trace = self_guided(1.0)
            n0, n0_trace = self_guided(0.0, True); n0_repeat, n0_repeat_trace = self_guided(0.0, True)
            w05, w05_trace = self_guided(0.5); w05_repeat, w05_repeat_trace = self_guided(0.5)
            w15, w15_trace = self_guided(1.5); w15_repeat, w15_repeat_trace = self_guided(1.5)
            w20, w20_trace = self_guided(2.0); w20_repeat, w20_repeat_trace = self_guided(2.0)
            ref1, ref_trace1 = ref(); ref2, ref_trace2 = ref()
        def diff(a,b): return float((a-b).abs().max())
        checks = {
            "frozen_eval_model": not policy.training and not any(parameter.requires_grad for parameter in policy.parameters()),
            "native_repeat": diff(native1,native2) <= 1e-6,
            "w1_reproduces_vanilla_bitwise": diff(w1,native1) <= 1e-6,
            "w1_repeat": diff(w1, w1_repeat) <= 1e-6,
            "n0_repeat": diff(n0, n0_repeat) <= 1e-6,
            "w05_repeat": diff(w05, w05_repeat) <= 1e-6,
            "w15_repeat": diff(w15, w15_repeat) <= 1e-6,
            "w20_repeat": diff(w20, w20_repeat) <= 1e-6,
            "ref_repeat": diff(ref1, ref2) <= 1e-6,
            "self_zero_visual_changes": all(trace["masked_token_count"] == 0 and trace["changed_indices"] == [] for trace in (w1_trace,n0_trace,w05_trace,w15_trace,w20_trace)),
            "protected_tokens_untouched": all(trace["protected_tokens_untouched"] for trace in (w1_trace,n0_trace,w05_trace,w15_trace,w20_trace,ref_trace1)),
            "ref_exactly_8_changed": len(ref_trace1["changed_indices"]) == 8 and sorted(ref_trace1["selected_indices"]) == sorted(ref_trace1["changed_indices"]),
            "shared_noise": all(trace["noise_sha256"] == tensor_sha256(noise) for trace in (w1_trace,n0_trace,w05_trace,w15_trace,w20_trace)) and ref_trace1.get("noise_sha256") is None,
            "back_projection_deterministic": w15_trace["reference_final_sha256"] == w15_repeat_trace["reference_final_sha256"],
            "all_finite": bool(torch.isfinite(torch.stack([native1,w1,n0,w05,w15,w20,ref1])).all()),
            "boundary_logged": all(("active_timestep_shift_bool" if timestep_mode else "active_earlier_self_bool" if relative_mode else "skipped_step_due_to_boundary_bool") in step for step in w15_trace["step_traces"]),
            "correction_vectors_recorded": (all(len(trace["step_traces"]) == 10 and all("_correction_vector" in step and "_applied_correction_vector" in step and step["_correction_vector"].shape[-1] == 7 and torch.isfinite(step["_correction_vector"]).all() and torch.isfinite(step["_applied_correction_vector"]).all() for step in trace["step_traces"]) for trace in (n0_trace,w05_trace,w15_trace,w20_trace,ref_trace1)) if save_vectors else True),
        }
        reports.append({"task_id": task_id, "checks": checks, "max_abs": {"native_repeat":diff(native1,native2),"w1_native":diff(w1,native1),"ref_repeat":diff(ref1,ref2)}, "delta":delta, "alpha":alpha, "shift":shift, "ref_selected_indices":ref_trace1["selected_indices"], "active_timestep_shift_fraction":float(sum(s.get("active_timestep_shift_bool", False) for s in w15_trace["step_traces"])/len(w15_trace["step_traces"])) if timestep_mode else None, "active_earlier_self_fraction":float(sum(s.get("active_earlier_self_bool", False) for s in w15_trace["step_traces"])/len(w15_trace["step_traces"])) if relative_mode else None, "skipped_fraction":float(sum(s.get("skipped_step_due_to_boundary_bool", False) for s in w15_trace["step_traces"])/len(w15_trace["step_traces"])) if not relative_mode else 0.0})
    passed = all(all(report["checks"].values()) for report in reports)
    result = {"pass":passed,"repeat_tolerance":1e-6,"tasks":reports,"locked_delta":delta,"locked_alpha":alpha,"locked_shift":shift}
    (artifact/"integrity_report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    if not passed: raise RuntimeError("self-guidance mandatory integrity gate failed")
    (artifact/"status/integrity.pass").touch(exist_ok=False)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__ == "__main__": main()
