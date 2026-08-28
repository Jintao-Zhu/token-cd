"""Prepare outcome-blind candidates and the locked rollout manifest for Phase T0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_WORKSPACE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_WORKSPACE / "LIBERO"), str(_WORKSPACE / "lerobot/src"), str(_WORKSPACE)]

from research.coreact_closed_loop.guidance import _full_velocity, tensor_sha256
from research.coreact_closed_loop.runtime import load_policy_and_processors
from research.coreact_exploration.instrumentation import attention_ranking_scores, build_prefix_span_map
from research.coreact_self_guidance.reference_snapshot_gate import digest
from research.coreact_signed_utility.token_regions import (
    local_2x2_regions,
    region_features,
    select_four_candidates,
)


ARMS = ("full", "attention_max", "iss_max", "attention_high_relevance_low", "random_control")


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value)}")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--means", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-ids", default="0,1")
    parser.add_argument("--rollout-arms", default=",".join(ARMS))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    reference = args.reference_artifact.resolve()
    output = args.output.resolve()
    init_ids = tuple(int(value) for value in args.init_ids.split(",") if value)
    rollout_arms = tuple(value for value in args.rollout_arms.split(",") if value)
    if not init_ids or not set(rollout_arms).issubset(ARMS) or "full" not in rollout_arms:
        raise ValueError("invalid init IDs or rollout arms")
    if output.exists():
        raise FileExistsError(output)
    if json.loads((reference / "decision.json").read_text())["decision"] != \
            "REFERENCE_SNAPSHOT_GATE_PASS_READY_FOR_MATCHED_CAUSAL_ROLLOUT":
        raise RuntimeError("reference snapshot gate is not valid")

    os.environ.setdefault("HF_HOME", str(workspace / "task1/.hf-cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(workspace / "task1/.hf-cache/hub"))
    os.environ["MUJOCO_GL"] = "egl"
    for directory in ("candidates", "features", "gate", "episodes", "invalid_units", "logs", "status"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    config, policy, _, _ = load_policy_and_processors(workspace)
    means = torch.load(args.means, weights_only=False, map_location="cpu")
    visual_mean = means["visual_position_mean"].to(device=config.device, dtype=torch.float32)
    camera_ids = tuple(means["camera_ids"])
    snapshots = []
    for task_id in range(10):
        for init_id in init_ids:
            for bin_id in (0, 1):
                path = reference / "snapshots" / f"task{task_id:02d}__init{init_id:02d}__bin{bin_id:02d}.pt"
                snapshots.append(path)
    expected_states = 10 * len(init_ids) * 2
    if len(snapshots) != expected_states or not all(path.exists() for path in snapshots):
        raise RuntimeError("the preregistered snapshot subset is incomplete")

    protocol = {
        "experiment_name": "coreact_closed_loop_signed_token_utility_phase_t0_v1",
        "stage": "outcome_blind_candidate_construction",
        "research_question": "can a non-trained pre-outcome signal predict signed closed-loop region utility",
        "model_family": "SmolVLA flow",
        "reference_artifact": str(reference),
        "reference_gate_sha256": file_sha256(reference / "reference_gate.lock.yaml"),
        "tasks": list(range(10)),
        "init_state_ids": list(init_ids),
        "progress_bins": [0.25, 0.65],
        "states": expected_states,
        "matched_noise_seeds_per_state": 5,
        "arms": list(rollout_arms),
        "planned_causal_units": expected_states * 5,
        "planned_episodes": expected_states * 5 * len(rollout_arms),
        "intervention": {
            "unit": "one local 2x2 region on one camera 8x8 connector-token grid",
            "token_count": 4,
            "operator": "camera-and-position-conditioned visual embedding mean replacement",
            "timing": "branch-point replan only; all later replans vanilla",
            "persistent_intervention": False,
            "guidance_or_cfg": False,
        },
        "candidate_selectors": {
            "attention_max": "maximum mean late-half action-to-context attention",
            "iss_max": "maximum full 10-step flow action-chunk RMS after replacement, non-overlapping",
            "attention_high_relevance_low": "maximum rank(attention)*(1-rank(frozen embedding cosine relevance)), non-overlapping",
            "random_control": "deterministic uniform region from remaining non-overlapping regions",
        },
        "pre_outcome_features": [
            "attention_mean", "attention_sum", "iss_action_rms", "iss_action_l2",
            "task_relevance_cosine", "nuisance_score", "iss_x_relevance", "nuisance_x_iss",
            "task_id", "target_progress", "robot_state", "region_camera_row_col",
        ],
        "temporal_persistence": "unavailable in validated reference artifacts; not imputed",
        "ground_truth": "U_j(s)=mean_seed(success_full-success_perturb_j)",
        "outcome_blinding": "all candidates/features and hashes locked before continuation outcomes",
        "noise_rule": "202608120000 + task*100000 + init*10000 + bin*1000 + continuation_seed",
        "candidate_probe_noise_rule": "202608129000 + task*100 + init*10 + bin",
        "invalid_rule": "any branch-point mismatch invalidates the full five-arm matched unit",
        "forbidden": ["predictor training", "critic", "progress model", "new CFG", "guidance", "partial-outcome tuning"],
        "analysis": {
            "cluster_unit": "snapshot",
            "signals": ["attention", "ISS", "task relevance", "ISS x relevance", "nuisance x ISS"],
            "report_taskwise_direction_consistency": True,
            "no_trained_classifier": True,
        },
        "checkpoint_revision": "31d453f7edd78c839a8bbc39744a292686daf0de",
        "means_path": str(args.means.resolve()),
        "means_sha256": file_sha256(args.means.resolve()),
    }
    (output / "protocol.lock.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False))

    candidate_manifest = []
    episode_manifest = []
    for snapshot_path in snapshots:
        snapshot = torch.load(snapshot_path, weights_only=False, map_location="cpu")
        meta = snapshot["metadata"]
        prepared = snapshot["reference_prepared"]
        device = torch.device(config.device)
        images = [tensor.to(device) for tensor in prepared["images"]]
        image_masks = [tensor.to(device) for tensor in prepared["image_masks"]]
        lang_tokens = prepared["lang_tokens"].to(device)
        lang_masks = prepared["lang_masks"].to(device)
        state = prepared["state"].to(device)
        prefix, pads, atts = policy.model.embed_prefix(images, image_masks, lang_tokens, lang_masks, state=state)
        span_map = build_prefix_span_map(
            policy.model, images, image_masks, lang_tokens, lang_masks, pads, camera_ids=camera_ids
        )[0]
        regions = local_2x2_regions(span_map, camera_ids)
        probe_seed = 202608129000 + meta["task_id"] * 100 + meta["init_state_id"] * 10 + int(meta["target_progress"] > 0.5)
        generator = torch.Generator(device=device).manual_seed(probe_seed)
        noise = torch.randn((1, config.chunk_size, config.max_action_dim), generator=generator,
                            device=device, dtype=state.dtype)
        first_time = torch.ones((1,), device=device, dtype=torch.float32)
        _, traces = _full_velocity(policy.model, prefix, pads, atts, noise, first_time, record_attention=True)
        scores = attention_ranking_scores(traces, prefix.shape[1])["late_half_action_to_context_attention"]
        rows, clean_action = region_features(
            policy.model, prefix, pads, atts, span_map, regions, scores, noise,
            visual_mean, camera_ids,
        )
        candidates = select_four_candidates(rows, probe_seed)
        snapshot_id = meta["snapshot_id"]
        state_row = {
            "snapshot_id": snapshot_id,
            "snapshot_file": str(snapshot_path),
            "task_id": meta["task_id"],
            "init_state_id": meta["init_state_id"],
            "target_progress": meta["target_progress"],
            "resolved_control_step": meta["resolved_control_step"],
            "reference_fingerprints": meta["reference_fingerprints"],
            "probe_seed": probe_seed,
            "probe_noise_sha256": tensor_sha256(noise),
            "clean_action_sha256": tensor_sha256(clean_action),
            "prefix_sha256": tensor_sha256(prefix),
            "eligible_regions": len(regions),
            "candidates": candidates,
            "robot_state": snapshot["reference_observation"].get("robot_state", {}),
        }
        torch.save({"state": state_row, "all_region_features": rows}, output / "features" / f"{snapshot_id}.pt")
        write_json(output / "candidates" / f"{snapshot_id}.json", state_row)
        candidate_manifest.append(state_row)
        candidate_by_id = {item["candidate_id"]: item for item in candidates}
        for continuation_seed in range(5):
            noise_seed = 202608120000 + meta["task_id"] * 100000 + meta["init_state_id"] * 10000 + int(meta["target_progress"] > 0.5) * 1000 + continuation_seed
            for arm in rollout_arms:
                row = {
                    "episode_id": f"{snapshot_id}__seed{continuation_seed:02d}__{arm}",
                    "causal_unit_id": f"{snapshot_id}__seed{continuation_seed:02d}",
                    "snapshot_id": snapshot_id,
                    "task_id": meta["task_id"],
                    "init_state_id": meta["init_state_id"],
                    "target_progress": meta["target_progress"],
                    "continuation_seed": continuation_seed,
                    "noise_seed": noise_seed,
                    "arm": arm,
                    "candidate": None if arm == "full" else candidate_by_id[arm],
                }
                episode_manifest.append(row)
        print(json.dumps({"prepared": len(candidate_manifest), "planned": expected_states, "snapshot_id": snapshot_id}), flush=True)

    (output / "candidate_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True, default=json_default) + "\n" for row in candidate_manifest))
    (output / "episode_manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in episode_manifest))
    checks = {
        "states": len(candidate_manifest),
        "episodes": len(episode_manifest),
        "exactly_four_candidates": all(len(row["candidates"]) == 4 for row in candidate_manifest),
        "four_unique_regions": all(len({x["region_id"] for x in row["candidates"]}) == 4 for row in candidate_manifest),
        "all_regions_four_tokens": all(len(x["prefix_indices"]) == 4 for row in candidate_manifest for x in row["candidates"]),
        "all_finite": all(x["finite"] for row in candidate_manifest for x in row["candidates"]),
        "protocol_sha256": file_sha256(output / "protocol.lock.yaml"),
        "candidate_manifest_sha256": file_sha256(output / "candidate_manifest.jsonl"),
        "episode_manifest_sha256": file_sha256(output / "episode_manifest.jsonl"),
    }
    checks["pass"] = len(candidate_manifest) == expected_states and len(episode_manifest) == expected_states * 5 * len(rollout_arms) and all(
        checks[key] for key in ("exactly_four_candidates", "four_unique_regions", "all_regions_four_tokens", "all_finite")
    )
    write_json(output / "candidate_gate.json", checks)
    decision = "PHASE_T0_CANDIDATES_LOCKED_READY_FOR_CAUSAL_DRYRUN" if checks["pass"] else "PHASE_T0_CANDIDATE_GATE_FAILED_NO_ROLLOUT"
    write_json(output / "decision.json", {"decision": decision, **checks})
    (output / "status" / ("candidates.pass" if checks["pass"] else "candidates.fail")).write_text(decision + "\n")
    print(decision)


if __name__ == "__main__":
    main()
