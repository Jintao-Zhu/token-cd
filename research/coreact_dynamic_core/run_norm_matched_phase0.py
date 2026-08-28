#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import yaml

WS = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WS / "LIBERO"), str(WS / "lerobot/src"), str(WS)]

from research.coreact_closed_loop.guidance import tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_dynamic_core.dynamic_guidance import sample_dynamic_core_actions


SELECTORS = ("random", "attention", "instability")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    protocol = yaml.safe_load((artifact / "protocol.lock.yaml").read_text())
    target = float(protocol["operator"]["target_applied_correction_rms"])
    os.environ.setdefault("HF_HOME", str(workspace / "task1/.hf-cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(workspace / "task1/.hf-cache/hub"))
    os.environ["MUJOCO_GL"] = "egl"
    cfg, policy, pre, _ = load_policy_and_processors(workspace)
    means = torch.load(workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt", weights_only=True, map_location="cpu")["visual_position_mean"]
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    language = suite.get_task(4).language
    rows = []
    repeat_pass = vanilla_parity = True
    for init in protocol["init_state_ids"]:
        env, env_pre, _ = make_task_env("libero_spatial", 4, cfg)
        try:
            env.envs[0].init_state_id = init
            obs, _ = env.reset(seed=160004000 + init)
            batch = prepare(policy, pre, env_pre, obs, language)
            generator = torch.Generator(device=batch["state"].device).manual_seed(202608540000 + init * 1000)
            noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim), generator=generator, device=batch["state"].device, dtype=batch["state"].dtype)
            state_rows = []
            for selector in SELECTORS:
                output, trace = sample_dynamic_core_actions(
                    policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise, means,
                    selector=selector, selection_seed=20260814 + 40000 + init * 100, norm_target_rms=target,
                )
                steps = trace["step_traces"]
                state_rows.append({
                    "selector": selector,
                    "selected_indices": trace["selected_indices"],
                    "changed_indices": trace["changed_indices"],
                    "postclip_rms": math.sqrt(statistics.mean(step["applied_guidance_norm"] ** 2 for step in steps)),
                    "base_rms": math.sqrt(statistics.mean(step["postclip_base_norm"] ** 2 for step in steps)),
                    "alpha_median": statistics.median(step["alpha"] for step in steps),
                    "alpha_max": max(step["alpha"] for step in steps),
                    "under_matched_fraction": statistics.mean(step["under_matched"] for step in steps),
                    "clipping_fraction": statistics.mean(step["clipped"] for step in steps),
                    "clean_flow_final_sha256": trace["clean_flow_final_sha256"],
                    "output_sha256": tensor_sha256(output),
                    "finite": trace["all_output_finite"],
                    "protected": trace["protected_tokens_untouched"],
                    "step_traces": steps,
                })
            if init == 0:
                vanilla = policy.model.sample_actions(batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise=noise)
                vanilla_parity = state_rows[0]["clean_flow_final_sha256"] == tensor_sha256(vanilla)
                output2, trace2 = sample_dynamic_core_actions(
                    policy.model, batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"], noise, means,
                    selector="instability", selection_seed=20260814 + 40000, norm_target_rms=target,
                )
                reference = state_rows[2]
                repeat_pass = reference["selected_indices"] == trace2["selected_indices"] and reference["output_sha256"] == tensor_sha256(output2)
            payload = {"task_id": 4, "init_state_id": init, "selectors": state_rows}
            (artifact / "phase0_states" / f"task04__init{init:02d}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            rows.append(payload)
            print(json.dumps({"phase0_complete": len(rows), "planned": 50, "init": init}), flush=True)
        finally:
            env.close()

    flat = [entry for row in rows for entry in row["selectors"]]
    ratios = {}
    for selector in ("random", "instability"):
        ratios[selector] = statistics.median(
            next(x for x in row["selectors"] if x["selector"] == selector)["postclip_rms"]
            / next(x for x in row["selectors"] if x["selector"] == "attention")["postclip_rms"]
            for row in rows
        )
    per_selector = {
        selector: {
            "postclip_rms_median": statistics.median(x["postclip_rms"] for x in flat if x["selector"] == selector),
            "alpha_median": statistics.median(x["alpha_median"] for x in flat if x["selector"] == selector),
            "under_matched_fraction": statistics.mean(x["under_matched_fraction"] for x in flat if x["selector"] == selector),
            "clipping_fraction": statistics.mean(x["clipping_fraction"] for x in flat if x["selector"] == selector),
        }
        for selector in SELECTORS
    }
    gate = {
        "states_complete": len(rows) == 50,
        "exact8_protected": all(len(x["selected_indices"]) == 8 and sorted(x["selected_indices"]) == sorted(x["changed_indices"]) and x["protected"] for x in flat),
        "finite": all(x["finite"] for x in flat),
        "clean_hash_identity": all(len({x["clean_flow_final_sha256"] for x in row["selectors"]}) == 1 for row in rows),
        "vanilla_clean_parity": vanilla_parity,
        "deterministic_repeat": repeat_pass,
        "median_postclip_rms_ratios_to_attention": ratios,
        "structured_magnitude_pass": 0.95 <= ratios["instability"] <= 1.05,
        "random_under_match_reported": True,
        "alpha_cap_pass": all(x["alpha_max"] <= 1.0 for x in flat),
        "per_selector": per_selector,
    }
    gate["pass"] = all(gate[key] for key in ("states_complete", "exact8_protected", "finite", "clean_hash_identity", "vanilla_clean_parity", "deterministic_repeat", "structured_magnitude_pass", "random_under_match_reported", "alpha_cap_pass"))
    decision = "NORM_MATCH_PHASE0_PASS_READY_FOR_ROLLOUT" if gate["pass"] else "NORM_MATCH_PHASE0_FAILED_NO_ROLLOUT"
    (artifact / "phase0_summary.json").write_text(json.dumps({"gate": gate, "states": rows}, indent=2, sort_keys=True) + "\n")
    (artifact / "decision.json").write_text(json.dumps({"decision": decision, "gate": gate, "rollout_started": False}, indent=2) + "\n")
    (artifact / "status" / ("phase0.pass" if gate["pass"] else "phase0.fail")).write_text(decision + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
