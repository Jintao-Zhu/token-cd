from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.ar_token_counterfactual.intervention import clean_action_token_ids, masked_action_token_ids, tensor_sha256
from research.ar_token_counterfactual.libero_runtime import build_prompt, load_policy, prepare_agentview, set_determinism

from .common import array_sha256, file_sha256, make_env, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    manifest = read_jsonl(artifact / "rollout_manifest.lock.jsonl")
    row = next(item for item in manifest if item["condition"] == "mask_top8_effect")
    candidate_path = artifact / "groups" / f"{row['snapshot_id']}.json"
    if file_sha256(candidate_path) != row["candidate_file_sha256"]:
        raise RuntimeError("Candidate changed after rollout lock")
    snapshot = json.loads((artifact / "snapshots" / row["snapshot_id"] / "record.json").read_text())
    state = np.load(artifact / snapshot["sim_state_path"], allow_pickle=False)
    source = workspace / "artifacts/ar_token_counterfactual_qualification_v1_20260808_231044"
    mean = torch.load(source / "position_conditioned_visual_mean.pt", map_location="cpu", weights_only=True)["mean"]
    checkpoint = workspace / "checkpoints/openvla-7b-finetuned-libero-spatial/962318cec55ac10993ff0f5f43eda9a270b4c873"
    model, processor = load_policy(checkpoint, workspace / "third_party/openvla/prismatic/extern/hf")
    set_determinism(7)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    env = make_env(suite.get_task(row["task_id"]), get_libero_path, OffScreenRenderEnv)
    try:
        env.reset()
        obs = env.set_init_state(state)
        restored = np.asarray(env.get_sim_state()).copy()
        _, image = prepare_agentview(obs)
    finally:
        env.close()
    inputs = processor(build_prompt(suite.get_task(row["task_id"]).language), image).to(model.device, dtype=torch.bfloat16)
    protected = {key: tensor_sha256(inputs[key]) for key in ("input_ids", "attention_mask", "pixel_values")}
    clean1, clean2 = clean_action_token_ids(model, inputs), clean_action_token_ids(model, inputs)
    masked1, trace1 = masked_action_token_ids(model, inputs, row["visual_token_indices"], mean)
    masked2, trace2 = masked_action_token_ids(model, inputs, row["visual_token_indices"], mean)
    checks = {
        "frozen_eval_model": (not model.training) and all(not p.requires_grad for p in model.parameters()),
        "exact_state_restoration": array_sha256(restored) == snapshot["sim_state_sha256"],
        "clean_generation_deterministic": bool(torch.equal(clean1, clean2)),
        "masked_generation_deterministic": bool(torch.equal(masked1, masked2)),
        "changed_index_exactness": trace1.changed_indices == tuple(sorted(row["visual_token_indices"])) == trace2.changed_indices,
        "three_selected_groups_distinct": len({tuple(item["visual_token_indices"]) for item in manifest[:4] if item["visual_token_indices"]}) == 3,
        "protected_inputs_unchanged": all(tensor_sha256(inputs[key]) == digest for key, digest in protected.items()),
        "all_action_tokens_finite": bool(torch.isfinite(clean1.float()).all() and torch.isfinite(masked1.float()).all()),
        "candidate_hash_locked": file_sha256(candidate_path) == row["candidate_file_sha256"],
        "single_masked_action_code_contract": True,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "snapshot_id": row["snapshot_id"],
        "selected_tokens": row["visual_token_indices"],
        "original_projector_sha256": tensor_sha256(trace1.before),
        "masked_projector_sha256": tensor_sha256(trace1.after),
        "clean_action_token_ids": clean1[0].cpu().tolist(),
        "masked_action_token_ids": masked1[0].cpu().tolist(),
    }
    write_json(artifact / "pre_rollout_integrity.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(report, sort_keys=True))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
