#!/usr/bin/env python3
"""Run resumable, append-only shards of the preregistered closed-loop pilot."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.utils.io_utils import write_video
from research.coreact_closed_loop.guidance import GuidanceConfig, sample_coreact_actions
from research.coreact_closed_loop.runtime import load_policy_and_processors, make_task_env, prepare


MAX_CONTROL_STEPS = 280
EXECUTED_PREFIX = 10
CAMERA_IDS = ("camera1", "camera2")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def vector_info_value(info: dict, key: str, index: int = 0, default=False):
    """Read Gymnasium vector-info fields, honoring the optional presence mask."""
    presence = info.get(f"_{key}")
    if presence is not None and not bool(presence[index]):
        return default
    values = info.get(key)
    return default if values is None else values[index]


def condition_config(condition: str) -> GuidanceConfig | None:
    if condition == "vanilla":
        return None
    mapping = {"coreact_top8": "top", "random8": "random", "bottom8": "bottom"}
    if condition not in mapping:
        raise ValueError(f"unknown condition {condition}")
    return GuidanceConfig(selection=mapping[condition])


def run_episode(
    workspace: Path,
    artifact: Path,
    policy_config,
    policy,
    preprocessor,
    postprocessor,
    means: dict,
    spec: dict,
) -> dict:
    episode_dir = artifact / "episodes" / spec["episode_id"]
    record_path = episode_dir / "episode.json"
    if record_path.exists():
        return json.loads(record_path.read_text())
    episode_dir.mkdir(parents=True, exist_ok=True)
    env, env_preprocessor, env_postprocessor = make_task_env(
        spec["suite"], spec["task_id"], policy_config
    )
    frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    step_rows: list[dict] = []
    replan_traces: list[dict] = []
    queue: list[torch.Tensor] = []
    replans = 0
    success = False
    nonfinite_action_count = 0
    normalized_bound_violation_count = 0
    policy_seconds: list[float] = []
    chunk_boundaries: list[float] = []
    config = condition_config(spec["condition"])
    try:
        inner = env.envs[0]
        inner.init_state_id = spec["init_state_id"]
        observation, _ = env.reset(seed=spec["reset_seed"])
        torch.cuda.reset_peak_memory_stats()
        for control_step in range(MAX_CONTROL_STEPS):
            frames.append(np.asarray(env.render())[0].copy())
            if not queue:
                prepared = prepare(
                    policy, preprocessor, env_preprocessor, observation, spec["language"]
                )
                generator = torch.Generator(device=prepared["state"].device).manual_seed(
                    spec["action_noise_seed"] * 1000 + replans
                )
                noise = torch.randn(
                    (1, policy_config.chunk_size, policy_config.max_action_dim),
                    generator=generator,
                    dtype=prepared["state"].dtype,
                    device=prepared["state"].device,
                )
                started = time.perf_counter()
                with torch.inference_mode():
                    if config is None:
                        chunk = policy.model.sample_actions(
                            prepared["images"],
                            prepared["image_masks"],
                            prepared["lang_tokens"],
                            prepared["lang_masks"],
                            prepared["state"],
                            noise=noise,
                        )
                        guidance_trace = None
                    else:
                        chunk, guidance_trace = sample_coreact_actions(
                            policy.model,
                            prepared["images"],
                            prepared["image_masks"],
                            prepared["lang_tokens"],
                            prepared["lang_masks"],
                            prepared["state"],
                            noise,
                            means["visual_position_mean"],
                            camera_ids=CAMERA_IDS,
                            config=config,
                            selection_seed=spec["selection_seed"] * 1000 + replans,
                        )
                torch.cuda.synchronize()
                policy_seconds.append(time.perf_counter() - started)
                model_prefix = chunk[:, :EXECUTED_PREFIX, :7].transpose(0, 1)
                if actions:
                    chunk_boundaries.append(
                        float(torch.linalg.vector_norm(model_prefix[0, 0].cpu() - actions[-1]))
                    )
                queue.extend(model_prefix)
                if guidance_trace is not None:
                    replan_traces.append({"replan": replans, **guidance_trace})
                replans += 1

            model_action = queue.pop(0)
            finite_model = bool(torch.isfinite(model_action).all())
            nonfinite_action_count += int(not finite_model)
            if not finite_model:
                raise RuntimeError("nonfinite model action")
            normalized_bound_violation_count += int(bool((model_action.abs() > 1.000001).any()))
            physical_action = postprocessor(model_action)
            legal_action = env_postprocessor({"action": physical_action})["action"]
            observation, _, terminated, _, info = env.step(legal_action.detach().cpu().numpy())
            action_cpu = model_action[0].detach().float().cpu()
            actions.append(action_cpu)
            success = bool(vector_info_value(info, "is_success"))
            step_rows.append(
                {
                    "control_step": control_step,
                    "replan": replans - 1,
                    "model_action": action_cpu.tolist(),
                    "physical_action": physical_action[0].detach().float().cpu().tolist(),
                    "legal_action": legal_action[0].detach().float().cpu().tolist(),
                    "success": success,
                }
            )
            if bool(terminated[0]) or success:
                frames.append(np.asarray(env.render())[0].copy())
                break

        action_array = torch.stack(actions) if actions else torch.empty((0, 7))
        differences = action_array[1:] - action_array[:-1]
        total_variation = float(torch.linalg.vector_norm(differences, dim=1).sum()) if len(actions) > 1 else 0.0
        video_path = episode_dir / "rollout.mp4"
        steps_path = episode_dir / "steps.jsonl"
        write_video(video_path, np.asarray(frames), fps=20)
        with steps_path.open("w") as stream:
            for row in step_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        record = {
            **spec,
            "status": "complete",
            "success": success,
            "control_steps": len(step_rows),
            "replans": replans,
            "action_total_variation": total_variation,
            "mean_chunk_boundary_discontinuity": float(np.mean(chunk_boundaries))
            if chunk_boundaries
            else 0.0,
            "nonfinite_action_count": nonfinite_action_count,
            "normalized_bound_violation_count": normalized_bound_violation_count,
            "policy_seconds_total": float(sum(policy_seconds)),
            "policy_seconds_median_per_replan": float(np.median(policy_seconds)),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "guidance_fallback_count": sum(trace["fallback_count"] for trace in replan_traces),
            "guidance_activation_count": sum(len(trace["step_traces"]) for trace in replan_traces),
            "all_guidance_outputs_finite": all(
                trace["all_output_finite"] for trace in replan_traces
            ),
            "replan_traces": replan_traces,
            "video": str(video_path.relative_to(artifact)),
            "step_log": str(steps_path.relative_to(artifact)),
        }
        if not math.isfinite(total_variation):
            raise RuntimeError("nonfinite action total variation")
        write_json(record_path, record)
        return record
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split", choices=("qualification", "development", "confirmation"), required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    artifact = args.artifact.resolve()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    if args.split in ("development", "confirmation"):
        qualification_gate = artifact / "qualification_gate.json"
        integrity_gate = artifact / "integrity_report.json"
        if not qualification_gate.exists() or not json.loads(qualification_gate.read_text()).get("pass"):
            raise RuntimeError("guided stages are locked until backbone qualification passes")
        if not integrity_gate.exists() or not json.loads(integrity_gate.read_text()).get("pass"):
            raise RuntimeError("guided stages are locked until sampler integrity passes")
    if args.split == "confirmation":
        lock = artifact / "protocol.lock.yaml"
        development_gate = artifact / "development_gate.json"
        if not lock.exists() or not development_gate.exists():
            raise RuntimeError("confirmation is locked until development integrity passes")
        if not json.loads(development_gate.read_text()).get("pass"):
            raise RuntimeError("development integrity gate did not pass")

    rows = [row for row in read_jsonl(artifact / "episode_manifest.jsonl") if row["split"] == args.split]
    rows = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    policy_config, policy, preprocessor, postprocessor = load_policy_and_processors(workspace)
    means = torch.load(
        workspace / "artifacts/coreact_exploration_v8_20260807_002951/modality_means.pt",
        weights_only=True,
        map_location="cpu",
    )
    for ordinal, spec in enumerate(rows, start=1):
        record = run_episode(
            workspace,
            artifact,
            policy_config,
            policy,
            preprocessor,
            postprocessor,
            means,
            spec,
        )
        print(
            f"[{args.split}:{args.shard_index}] {ordinal}/{len(rows)} {spec['episode_id']} "
            f"success={record['success']} steps={record['control_steps']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
