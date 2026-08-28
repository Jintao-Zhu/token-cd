from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch
from lerobot.envs.factory import make_env_pre_post_processors

from research.coreact_closed_loop.runtime import env_config, prepare
from research.coreact_quality_negative_branch.quality_branch import velocity_from_embeddings
from research.coreact_self_guidance.reference_snapshot_gate import digest, fingerprints
from research.coreact_trained_weak.runtime import load_policy


ACTIVE_STEPS = tuple(range(1, 9))
ELIGIBLE = tuple(range(1, 16))
EPS = 1e-12


def masks(candidate: str, seed: int) -> list[tuple[int, ...]]:
    rng = np.random.default_rng(seed)
    if candidate == "S1":
        return [(int(value),) for value in rng.permutation(ELIGIBLE)[:12]]
    combinations = list(itertools.combinations(ELIGIBLE, 2))
    selected = rng.choice(len(combinations), size=12, replace=False)
    return [combinations[int(index)] for index in selected]


def scales_for(mask_rows: list[tuple[int, ...]], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    output = torch.ones(len(mask_rows), 16, device=device, dtype=dtype)
    for row, dropped in enumerate(mask_rows):
        output[row, list(dropped)] = 0
    return output


def consensus(directions: torch.Tensor) -> dict:
    vectors = directions[..., :7].reshape(len(directions), -1).float()
    norms = torch.linalg.vector_norm(vectors, dim=1)
    normalized = vectors / (norms[:, None] + EPS)
    matrix = normalized @ normalized.T
    triangle = matrix[torch.triu_indices(len(vectors), len(vectors), offset=1).unbind()]
    mean = vectors.mean(dim=0)
    mean_norm = torch.linalg.vector_norm(mean)
    to_mean = (vectors @ mean) / (norms * mean_norm + EPS)
    return {
        "pairwise_cosine_median": float(torch.median(triangle)),
        "pairwise_cosine_mean": float(torch.mean(triangle)),
        "consensus_ratio": float(mean_norm / (torch.mean(norms) + EPS)),
        "positive_to_consensus_fraction": float(torch.mean((to_mean > 0).float())),
        "residual_norms": norms.cpu().tolist(),
        "to_consensus_cosines": to_mean.cpu().tolist(),
        "mean_vector": mean,
    }


def atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", required=True)
    parser.add_argument("--max-states", type=int)
    args = parser.parse_args()
    workspace, artifact = args.workspace.resolve(), args.artifact.resolve()
    os.environ.setdefault("MUJOCO_GL", "egl")
    checkpoint = Path(json.loads((artifact / "protocol.lock.json").read_text())["strong_checkpoint"])
    config, policy, preprocessor, _ = load_policy(checkpoint)
    model = policy.model
    snapshots = [path for path in sorted((artifact / "snapshots").glob("*.pt")) if torch.load(path, weights_only=False, map_location="cpu")["metadata"]["task_id"] in args.task_ids]
    if args.max_states:
        snapshots = snapshots[:args.max_states]
    for ordinal, path in enumerate(snapshots):
        snapshot = torch.load(path, weights_only=False, map_location="cpu")
        meta, prefix_actions = snapshot["metadata"], snapshot["action_prefix"].numpy()
        output = artifact / "s0" / f"{meta['snapshot_id']}.json"
        if output.exists():
            continue
        cfg = env_config("libero_spatial", meta["task_id"])
        env_pre, _ = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=config)
        env = cfg.create_envs(n_envs=1, use_async_envs=False)["libero_spatial"][meta["task_id"]]
        try:
            inner = env.envs[0]
            inner.init_state_id = meta["init_state_id"]
            observation, _ = env.reset(seed=meta["reset_seed"])
            for action in prefix_actions:
                observation, _, terminated, _, _ = env.step(np.asarray(action, np.float32)[None, :])
                if bool(terminated[0]):
                    raise RuntimeError("prefix terminated")
            batch = prepare(policy, preprocessor, env_pre, observation, meta["instruction"])
            state_fp = fingerprints(env, observation, batch, list(prefix_actions))
            prefix, pads, atts = model.embed_prefix(batch["images"], batch["image_masks"], batch["lang_tokens"], batch["lang_masks"], state=batch["state"])
            noise_seed = 202608190000 + meta["task_id"] * 1000 + meta["init_state_id"] * 10 + int(meta["target_progress"] * 10)
            generator = torch.Generator(device=batch["state"].device).manual_seed(noise_seed)
            x_t = torch.randn((1, config.chunk_size, config.max_action_dim), generator=generator, device=batch["state"].device, dtype=batch["state"].dtype)
            rows = []
            all_finite, all_ones_parity = True, 0.0
            for step in range(10):
                tau = 1.0 - 0.1 * step
                timestep = torch.full((1,), tau, device=x_t.device, dtype=torch.float32)
                strong = velocity_from_embeddings(model, prefix, pads, atts, x_t, timestep)
                if step == 0:
                    same = velocity_from_embeddings(model, prefix, pads, atts, x_t, timestep, expert_residual_scales=torch.ones(1, 16, device=x_t.device, dtype=prefix.dtype))
                    all_ones_parity = float((same - strong).abs().max())
                if step in ACTIVE_STEPS:
                    w1_scale = torch.ones(1, 16, device=x_t.device, dtype=prefix.dtype)
                    w1_scale[:, 15] = 0
                    w1 = velocity_from_embeddings(model, prefix, pads, atts, x_t, timestep, expert_residual_scales=w1_scale)
                    d_w1 = (strong - w1)[0, :, :7].reshape(-1).float()
                    for candidate in ("S1", "S2"):
                        mask_seed = 202608180000 + meta["task_id"] * 1_000_000 + meta["init_state_id"] * 10_000 + int(meta["target_progress"] * 10) * 1000 + step * 100 + (1 if candidate == "S2" else 0)
                        mask_rows = masks(candidate, mask_seed)
                        scales = scales_for(mask_rows, device=x_t.device, dtype=prefix.dtype)
                        weak = velocity_from_embeddings(model, prefix.expand(12, -1, -1), pads.expand(12, -1), atts.expand(12, -1), x_t.expand(12, -1, -1), timestep.expand(12), expert_residual_scales=scales)
                        directions = strong.expand_as(weak) - weak
                        metrics = consensus(directions)
                        mean = metrics.pop("mean_vector")
                        w1_cosine = float(torch.dot(mean, d_w1) / (torch.linalg.vector_norm(mean) * torch.linalg.vector_norm(d_w1) + EPS))
                        all_finite = all_finite and bool(torch.isfinite(weak).all()) and all(np.isfinite(value) for value in metrics["residual_norms"])
                        rows.append({"candidate": candidate, "flow_step": step, "tau": tau, "mask_seed": mask_seed, "masks": [list(mask) for mask in mask_rows], "consensus_to_fixed_w1_cosine": w1_cosine, **metrics})
                x_t = x_t - 0.1 * strong
            integrity = {"finite": all_finite and bool(torch.isfinite(x_t).all()), "all_ones_strong_parity_max_abs": all_ones_parity, "prefix_fingerprint": state_fp, "noise_seed": noise_seed, "noise_sha256": digest(torch.randn((1, config.chunk_size, config.max_action_dim), generator=torch.Generator(device=batch["state"].device).manual_seed(noise_seed), device=batch["state"].device, dtype=batch["state"].dtype)), "rows": len(rows), "num_expert_layers": model.vlm_with_expert.num_expert_layers}
            if not integrity["finite"] or all_ones_parity >= 1e-6 or len(rows) != 16:
                atomic(artifact / "invalid_units" / f"{meta['snapshot_id']}.json", {"metadata": meta, "integrity": integrity})
                raise RuntimeError(f"S0 integrity failure {meta['snapshot_id']}")
            atomic(output, {"metadata": meta, "rows": rows, "integrity": integrity})
            print(json.dumps({"task": meta["task_id"], "snapshot": meta["snapshot_id"], "rows": len(rows)}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
