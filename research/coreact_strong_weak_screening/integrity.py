#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_strong_weak_screening.sampler import (
    ARMS,
    GUIDANCE_LAMBDA,
    sample_strong_weak_actions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    protocol = yaml.safe_load((artifact / "protocol.lock.yaml").read_text())
    if protocol["stage"] != "closed_loop_screening_not_confirmation":
        raise RuntimeError("unexpected protocol stage")
    config, policy, preprocessor, _postprocessor = load_policy_and_processors(workspace)
    task = json.loads((artifact / "task_manifest.json").read_text())["0"]
    env, env_preprocessor, _env_postprocessor = make_task_env(
        "libero_spatial", 0, config
    )
    try:
        env.envs[0].init_state_id = 0
        observation, _ = env.reset(seed=410000000)
        batch = prepare(
            policy, preprocessor, env_preprocessor, observation, task["language"]
        )
    finally:
        env.close()
    noise = torch.randn(
        (1, config.chunk_size, config.max_action_dim),
        generator=torch.Generator(device=batch["state"].device).manual_seed(420000000),
        device=batch["state"].device,
        dtype=batch["state"].dtype,
    )
    outputs = {}
    traces = {}
    rerun_errors = {}
    for arm in ARMS:
        chunk, trace = sample_strong_weak_actions(
            policy.model,
            batch["images"],
            batch["image_masks"],
            batch["lang_tokens"],
            batch["lang_masks"],
            batch["state"],
            noise,
            arm=arm,
            return_debug=True,
        )
        repeat, _ = sample_strong_weak_actions(
            policy.model,
            batch["images"],
            batch["image_masks"],
            batch["lang_tokens"],
            batch["lang_masks"],
            batch["state"],
            noise,
            arm=arm,
        )
        outputs[arm] = chunk
        traces[arm] = trace
        rerun_errors[arm] = float((repeat - chunk).abs().max())

    native = policy.model.sample_actions(
        batch["images"],
        batch["image_masks"],
        batch["lang_tokens"],
        batch["lang_masks"],
        batch["state"],
        noise=noise,
    )
    native_parity = float((native - outputs["Vanilla"]).abs().max())
    checks = {
        "native_vanilla_exact_parity": native_parity == 0.0,
        "deterministic_all_arms": all(value == 0.0 for value in rerun_errors.values()),
        "all_outputs_finite": all(torch.isfinite(value).all() for value in outputs.values()),
        "w3_last_one_half": traces["W3_CFG"]["_debug"]["scales"][0, -1].item()
        == 0.5
        and bool((traces["W3_CFG"]["_debug"]["scales"][0, :-1] == 1).all()),
        "w4_last_two_half": bool(
            (traces["W4_CFG"]["_debug"]["scales"][0, -2:] == 0.5).all()
        )
        and bool((traces["W4_CFG"]["_debug"]["scales"][0, :-2] == 1).all()),
        "w4_only_is_direct_weak": torch.equal(
            traces["W4_only"]["_debug"]["used"],
            traces["W4_only"]["_debug"]["weak"],
        ),
        "cfg_formula_w3": torch.equal(
            traces["W3_CFG"]["_debug"]["used"],
            traces["W3_CFG"]["_debug"]["strong"]
            + GUIDANCE_LAMBDA
            * (
                traces["W3_CFG"]["_debug"]["strong"]
                - traces["W3_CFG"]["_debug"]["weak"]
            ),
        ),
        "cfg_formula_w4": torch.equal(
            traces["W4_CFG"]["_debug"]["used"],
            traces["W4_CFG"]["_debug"]["strong"]
            + GUIDANCE_LAMBDA
            * (
                traces["W4_CFG"]["_debug"]["strong"]
                - traces["W4_CFG"]["_debug"]["weak"]
            ),
        ),
        "corrections_nonzero": traces["W3_CFG"]["step_traces"][0][
            "strong_minus_weak_norm"
        ]
        > 0
        and traces["W4_CFG"]["step_traces"][0]["strong_minus_weak_norm"] > 0,
        "no_ev_gate_no_clipping": all(
            trace["no_ev_gate"] and trace["no_clipping"] for trace in traces.values()
        ),
    }
    report = {
        "pass": all(checks.values()),
        "checks": checks,
        "native_vanilla_max_abs": native_parity,
        "deterministic_rerun_max_abs": rerun_errors,
        "first_step_correction_norm": {
            arm: traces[arm]["step_traces"][0]["strong_minus_weak_norm"]
            for arm in ("W3_CFG", "W4_CFG")
        },
    }
    (artifact / "integrity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    status = artifact / "status" / ("integrity.pass" if report["pass"] else "integrity.fail")
    status.write_text(("PASS" if report["pass"] else "FAIL") + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise RuntimeError("Strong-Weak dry-run integrity failed")


if __name__ == "__main__":
    main()
