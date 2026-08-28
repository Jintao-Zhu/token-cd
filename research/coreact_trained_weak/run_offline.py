from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from lerobot.envs.factory import make_env_pre_post_processors
from libero.libero.envs.utils import postprocess_model_xml
from research.coreact_closed_loop.runtime import env_config
from research.coreact_expert_direction.direction_audit import prepare_demo_state, rewrite_demo_xml
from research.coreact_quality_negative_branch.quality_branch import (
    FLOW_TIMES,
    point_metrics,
    tensor_sha256,
)
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from research.coreact_region.segmented_runtime import make_segmented_env
from research.coreact_trained_weak.runtime import load_locked_pair
from research.coreact_trained_weak.sampler import applied_correction


SOURCE_MANIFEST = Path(
    "artifacts/coreact_selective_cfg_ev_validity_phase1_v1_20260811_203336/unit_manifest.jsonl"
)
LAMBDAS = (0.1, 0.25, 0.5)


def repeat(value: torch.Tensor, count: int) -> torch.Tensor:
    return value.expand(count, *value.shape[1:])


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


@torch.no_grad()
def velocity_from_embeddings(model, prefix, prefix_pad, prefix_att, x_t, timestep):
    suffix, suffix_pad, suffix_att = model.embed_suffix(x_t, timestep)
    pads = torch.cat([prefix_pad, suffix_pad], dim=1)
    atts = torch.cat([prefix_att, suffix_att], dim=1)
    mask = make_att_2d_masks(pads, atts)
    positions = torch.cumsum(pads, dim=1) - 1
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=mask,
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
        fill_kv_cache=False,
    )
    suffix_out = suffix_out[:, -model.config.chunk_size :].to(dtype=torch.float32)
    return model.action_out_proj(suffix_out)


@torch.no_grad()
def evaluate(strong_model, weak_model, batch: dict, units: list[dict]):
    actions = batch["actions"]
    noises = torch.cat([
        torch.randn(
            actions.shape,
            generator=torch.Generator(device=actions.device).manual_seed(unit["noise_seed"]),
            device=actions.device,
            dtype=actions.dtype,
        )
        for unit in units
    ])
    actions3 = repeat(actions, 3)
    strong_prefix, strong_pad, strong_att = strong_model.embed_prefix(
        batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], state=batch["state"]
    )
    weak_prefix, weak_pad, weak_att = weak_model.embed_prefix(
        batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], state=batch["state"]
    )
    if not torch.equal(strong_pad, weak_pad) or not torch.equal(strong_att, weak_att):
        raise RuntimeError("Strong/Weak prefix masks differ")
    valid = (~batch["action_is_pad"]).expand(3, -1).unsqueeze(-1).expand(-1, -1, 7)
    rows = []
    repeat_max_abs = 0.0
    for flow_step, tau_value in enumerate(FLOW_TIMES):
        timestep = torch.full((3,), tau_value, device=actions.device, dtype=actions.dtype)
        time_expanded = timestep[:, None, None]
        x_t = time_expanded * noises + (1 - time_expanded) * actions3
        target = noises - actions3
        strong = velocity_from_embeddings(
            strong_model, repeat(strong_prefix, 3), repeat(strong_pad, 3), repeat(strong_att, 3), x_t, timestep
        )
        weak = velocity_from_embeddings(
            weak_model, repeat(weak_prefix, 3), repeat(weak_pad, 3), repeat(weak_att, 3), x_t, timestep
        )
        if flow_step == 0:
            repeated = velocity_from_embeddings(
                strong_model, repeat(strong_prefix, 3), repeat(strong_pad, 3), repeat(strong_att, 3), x_t, timestep
            )
            repeat_max_abs = float((repeated - strong).abs().max())
        for noise_index, unit in enumerate(units):
            metrics = point_metrics(
                strong[noise_index, :, :7], weak[noise_index, :, :7],
                target[noise_index, :, :7], valid[noise_index],
            )
            strong_valid = strong[noise_index, :, :7][valid[noise_index]]
            weak_valid = weak[noise_index, :, :7][valid[noise_index]]
            calibration = {}
            for lambda_value in LAMBDAS:
                correction, clip_scale = applied_correction(
                    strong_valid, weak_valid, lambda_value, action_dim=None
                )
                calibration[f"applied_ratio_lambda_{lambda_value:g}"] = float(
                    torch.linalg.vector_norm(correction) /
                    (torch.linalg.vector_norm(strong_valid) + 1e-12)
                )
                calibration[f"clip_scale_lambda_{lambda_value:g}"] = float(clip_scale)
            rows.append({
                "point_id": f"{unit['unit_id']}__flow{flow_step:02d}",
                "state_id": unit["state_id"],
                "unit_id": unit["unit_id"],
                "task_id": unit["task_id"],
                "demo_ordinal": unit["demo_ordinal"],
                "noise_ordinal": unit["noise_ordinal"],
                "noise_seed": unit["noise_seed"],
                "flow_step": flow_step,
                "timestep": tau_value,
                "valid_action_steps": int(valid[noise_index, :, 0].sum()),
                "noise_sha256": tensor_sha256(noises[noise_index]),
                "x_t_sha256": tensor_sha256(x_t[noise_index]),
                "target_sha256": tensor_sha256(target[noise_index]),
                **metrics,
                **calibration,
            })
    return rows, {
        "finite": all(np.isfinite(v) for row in rows for v in row.values() if isinstance(v, float)),
        "deterministic_repeat_max_abs": repeat_max_abs,
        "strong_prefix_sha256": tensor_sha256(strong_prefix),
        "weak_prefix_sha256": tensor_sha256(weak_prefix),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--max-states", type=int, default=20)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    source = workspace / SOURCE_MANIFEST
    all_units = [json.loads(line) for line in source.read_text().splitlines()]
    units = [
        unit for unit in all_units
        if unit["task_id"] in args.task_ids and unit["demo_ordinal"] < args.max_states
    ]
    if len(units) != 3 * args.max_states * len(args.task_ids):
        raise RuntimeError(f"unexpected unit count {len(units)}")
    output_dir = artifact / "trained_weak_offline_raw"
    output_dir.mkdir(exist_ok=True)
    os.chdir(workspace / "LIBERO")
    _, strong, weak = load_locked_pair(artifact)
    config, strong_policy, preprocessor, _ = strong
    _, weak_policy, _, _ = weak
    for task_id in args.task_ids:
        task_units = [unit for unit in units if unit["task_id"] == task_id]
        env_preprocessor, _ = make_env_pre_post_processors(
            env_cfg=env_config("libero_spatial", task_id), policy_cfg=config
        )
        env = make_segmented_env("libero_spatial", task_id)
        try:
            with h5py.File(task_units[0]["demo_path"], "r") as handle:
                for ordinal in range(args.max_states):
                    group = [unit for unit in task_units if unit["demo_ordinal"] == ordinal]
                    output = output_dir / f"task{task_id:02d}__demo{ordinal:02d}.json"
                    if output.exists():
                        continue
                    row = group[0]
                    episode = handle["data"][row["demo_id"]]
                    states, actions = np.asarray(episode["states"]), np.asarray(episode["actions"])
                    xml = episode.attrs["model_file"]
                    xml = xml.decode() if isinstance(xml, bytes) else xml
                    env._env.reset()
                    env._env.reset_from_xml_string(postprocess_model_xml(
                        rewrite_demo_xml(xml, workspace), {}, demo_generation=False
                    ))
                    env._env.env.sim.reset()
                    raw = env._env.regenerate_obs_from_state(states[row["resolved_frame"]])
                    batch = prepare_demo_state(
                        strong_policy, preprocessor, env_preprocessor, env, raw,
                        row["language"], actions, row["resolved_frame"],
                    )
                    rows, integrity = evaluate(strong_policy.model, weak_policy.model, batch, group)
                    atomic_json(output, {"rows": rows, "integrity": integrity})
                    print(json.dumps({"task": task_id, "state": ordinal, "points": len(rows)}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
