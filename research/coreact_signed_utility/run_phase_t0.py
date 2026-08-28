"""Matched continuation runner for closed-loop signed token utility Phase T0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_WORKSPACE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_WORKSPACE / "LIBERO"), str(_WORKSPACE / "lerobot/src"), str(_WORKSPACE)]

from research.coreact_closed_loop.run_pilot import vector_info_value
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare
from research.coreact_self_guidance.reference_snapshot_gate import digest, fingerprints
from research.coreact_signed_utility.token_regions import sample_fixed_region_actions


ARMS = ("full", "attention_max", "iss_max", "attention_high_relevance_low", "random_control")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--means", type=Path, required=True)
    parser.add_argument("--unit-start", type=int, default=0)
    parser.add_argument("--unit-stride", type=int, default=1)
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = args.artifact.resolve()
    reference = args.reference_artifact.resolve()
    decision = json.loads((artifact / "decision.json").read_text())["decision"]
    if decision not in {
        "PHASE_T0_CANDIDATES_LOCKED_READY_FOR_CAUSAL_DRYRUN",
        "PHASE_T0_CAUSAL_DRYRUN_PASS_READY_FOR_ROLLOUT",
    }:
        raise RuntimeError(f"candidate gate is not ready: {decision}")
    os.environ.setdefault("HF_HOME", str(workspace / "task1/.hf-cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(workspace / "task1/.hf-cache/hub"))
    os.environ["MUJOCO_GL"] = "egl"
    for directory in ("episodes", "invalid_units", "logs", "status", "gate"):
        (artifact / directory).mkdir(exist_ok=True)

    rows = [json.loads(line) for line in (artifact / "episode_manifest.jsonl").read_text().splitlines()]
    units: dict[str, dict[str, dict]] = {}
    for row in rows:
        units.setdefault(row["causal_unit_id"], {})[row["arm"]] = row
    ordered_units = sorted(units.items())
    manifest_arms = tuple(sorted(next(iter(units.values())).keys()))
    if any(set(value) != set(manifest_arms) for value in units.values()) or "full" not in manifest_arms:
        raise RuntimeError("manifest arm grouping failure")
    selected_units = ordered_units[args.unit_start::args.unit_stride]
    if args.max_units is not None:
        selected_units = selected_units[:args.max_units]

    config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    means = torch.load(args.means, weights_only=False, map_location="cpu")
    visual_mean = means["visual_position_mean"].to(config.device, dtype=torch.float32)
    camera_ids = tuple(means["camera_ids"])
    protocol_sha = sha256(artifact / "protocol.lock.yaml")
    manifest_sha = sha256(artifact / "episode_manifest.jsonl")
    completed = 0

    for causal_unit_id, specs in selected_units:
        if set(specs) != set(manifest_arms):
            raise RuntimeError(f"incomplete manifest unit {causal_unit_id}")
        outputs = [artifact / "episodes" / f"{specs[arm]['episode_id']}.json" for arm in manifest_arms]
        if all(path.exists() for path in outputs):
            completed += 1
            continue
        if any(path.exists() for path in outputs):
            raise RuntimeError(f"partial causal unit requires audit: {causal_unit_id}")

        snapshot_id = next(iter(specs.values()))["snapshot_id"]
        snapshot = torch.load(reference / "snapshots" / f"{snapshot_id}.pt", weights_only=False, map_location="cpu")
        meta = snapshot["metadata"]
        prefix = snapshot["action_prefix"].numpy()
        branch_points = {}
        records = {}
        first_chunks = {}
        invalid = None

        for arm in manifest_arms:
            spec = specs[arm]
            env, env_pre, env_post = make_task_env("libero_spatial", spec["task_id"], config)
            queue = []
            replans = 0
            actions = []
            success = False
            reason = "horizon"
            first_trace = None
            try:
                inner = env.envs[0]
                inner.init_state_id = spec["init_state_id"]
                obs, _ = env.reset(seed=int(meta["reset_seed"]))
                for action in prefix:
                    obs, _, terminated, _, _ = env.step(np.asarray(action, dtype=np.float32)[None, :])
                    if bool(terminated[0]):
                        raise RuntimeError("prefix replay terminated before branch point")
                batch = prepare(policy, preprocessor, env_pre, obs, meta["instruction"])
                fp = fingerprints(env, obs, batch, list(prefix))
                generator = torch.Generator(device=batch["state"].device).manual_seed(int(spec["noise_seed"]))
                first_noise = torch.randn(
                    (1, config.chunk_size, config.max_action_dim), generator=generator,
                    device=batch["state"].device, dtype=batch["state"].dtype,
                )
                clean_chunk = policy.model.sample_actions(
                    batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"],
                    batch["state"], noise=first_noise,
                )
                branch_points[arm] = {
                    "fingerprints": fp,
                    "reference_equal": fp == meta["reference_fingerprints"],
                    "noise_sha256": digest(first_noise),
                    "clean_chunk_sha256": digest(clean_chunk),
                }
                remaining = max(0, 280 - int(meta["resolved_control_step"]))
                for control_step in range(remaining):
                    if not queue:
                        if replans == 0:
                            noise = first_noise
                        else:
                            batch = prepare(policy, preprocessor, env_pre, obs, meta["instruction"])
                            generator = torch.Generator(device=batch["state"].device).manual_seed(
                                int(spec["noise_seed"]) + replans
                            )
                            noise = torch.randn(
                                (1, config.chunk_size, config.max_action_dim), generator=generator,
                                device=batch["state"].device, dtype=batch["state"].dtype,
                            )
                        with torch.inference_mode():
                            if replans == 0 and arm != "full":
                                chunk, first_trace = sample_fixed_region_actions(
                                    policy.model, batch["images"], batch["image_masks"],
                                    batch["lang_tokens"], batch["lang_masks"], batch["state"], noise,
                                    spec["candidate"]["prefix_indices"], visual_mean, camera_ids,
                                )
                            else:
                                chunk = policy.model.sample_actions(
                                    batch["images"], batch["image_masks"], batch["lang_tokens"],
                                    batch["lang_masks"], batch["state"], noise=noise,
                                )
                        if not bool(torch.isfinite(chunk).all()):
                            raise RuntimeError("nonfinite action chunk")
                        if replans == 0:
                            first_chunks[arm] = digest(chunk)
                        queue = [item.detach().cpu() for item in chunk[:, :10, :7].transpose(0, 1)]
                        replans += 1
                    model_action = queue.pop(0)
                    legal = env_post({"action": postprocessor(model_action)})["action"]
                    obs, _, terminated, _, info = env.step(legal.detach().cpu().numpy())
                    actions.append(model_action[0].detach().float().cpu())
                    success = bool(vector_info_value(info, "is_success"))
                    if success:
                        reason = "success"
                        break
                    if bool(terminated[0]):
                        reason = "terminated"
                        break
            finally:
                env.close()
            records[arm] = {
                **spec,
                "status": "complete",
                "success": success,
                "termination_reason": reason,
                "continuation_control_steps": len(actions),
                "replans": replans,
                "branch_point": branch_points[arm],
                "first_intervention_trace": first_trace,
                "all_actions_finite": all(bool(torch.isfinite(action).all()) for action in actions),
                "protocol_sha256": protocol_sha,
                "manifest_sha256": manifest_sha,
            }

        for field in ("fingerprints", "noise_sha256", "clean_chunk_sha256"):
            serialized = {json.dumps(branch_points[arm][field], sort_keys=True) for arm in manifest_arms}
            if len(serialized) != 1:
                invalid = field
                break
        if invalid is None and not all(branch_points[arm]["reference_equal"] for arm in manifest_arms):
            invalid = "reference_fingerprint"
        if invalid is None and first_chunks["full"] != branch_points["full"]["clean_chunk_sha256"]:
            invalid = "full_first_chunk_parity"
        if invalid is None:
            for arm in manifest_arms:
                if arm == "full":
                    continue
                trace = records[arm]["first_intervention_trace"]
                if trace is None or len(trace["changed_indices"]) != 4 or not trace["protected_tokens_untouched"]:
                    invalid = f"{arm}_intervention_integrity"
                    break
        if invalid:
            atomic_json(artifact / "invalid_units" / f"{causal_unit_id}.json", {
                "causal_unit_id": causal_unit_id,
                "first_mismatch": invalid,
                "branch_points": branch_points,
            })
            raise RuntimeError(f"matched unit failed: {causal_unit_id}: {invalid}")

        if args.dry_run:
            report = {
                "decision": "PHASE_T0_CAUSAL_DRYRUN_PASS_READY_FOR_ROLLOUT",
                "causal_unit_id": causal_unit_id,
                "branch_point_equality": True,
                "full_first_chunk_parity": True,
                "four_token_local_interventions": True,
                "finite": True,
                "outcomes_not_reported": True,
                "branch_points": branch_points,
                "intervention_traces": {arm: records[arm]["first_intervention_trace"] for arm in manifest_arms if arm != "full"},
            }
            atomic_json(artifact / "gate" / "causal_dryrun.json", report)
            atomic_json(artifact / "decision.json", {"decision": report["decision"]})
            (artifact / "status" / "causal_dryrun.pass").write_text(report["decision"] + "\n")
            print(report["decision"])
            return

        for arm in manifest_arms:
            atomic_json(artifact / "episodes" / f"{specs[arm]['episode_id']}.json", records[arm])
        completed += 1
        print(json.dumps({
            "worker_start": args.unit_start,
            "causal_units_complete_by_worker": completed,
            "worker_units": len(selected_units),
            "causal_unit_id": causal_unit_id,
        }), flush=True)

    (artifact / "status" / f"worker_{args.unit_start:02d}.complete").write_text(
        f"{completed}/{len(selected_units)} causal units\n"
    )


if __name__ == "__main__":
    main()
